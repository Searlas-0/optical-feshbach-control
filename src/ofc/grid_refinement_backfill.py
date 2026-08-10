"""Backfill queryable 2N-1 grid-refinement diagnostics for stored runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3

from .device import configure_jax_environment

configure_jax_environment()

import jax
import jax.numpy as jnp
import numpy as np

from .grid_refinement import grid_refinement_diagnostics
from .storage import ResultStore, decode_array, parameter_database_path


DEFAULT_LOOSE_TOLERANCE = 1e-2
DEFAULT_STRICT_TOLERANCE = 1e-3
DEFAULT_Y_FLOOR = 1e-12
METRIC_NAME = "best_grid_refinement_relative_error"


def _run_id_bounds(database: Path, shard_index: int, shard_count: int):
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        minimum, maximum = connection.execute(
            "SELECT MIN(run_id), MAX(run_id) FROM runs"
        ).fetchone()
    if minimum is None:
        return None
    span = int(maximum) - int(minimum) + 1
    lower = int(minimum) + span * shard_index // shard_count
    upper = int(minimum) + span * (shard_index + 1) // shard_count - 1
    return lower, upper


def _pending_run_ids(database: Path, lower: int, upper: int, overwrite: bool):
    metric_clause = "" if overwrite else "AND metric.run_id IS NULL"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        return [
            int(row[0])
            for row in connection.execute(
                f"""SELECT r.run_id
                      FROM runs r
                      LEFT JOIN physical_values metric
                        ON metric.run_id=r.run_id AND metric.name=?
                     WHERE r.run_id BETWEEN ? AND ? {metric_clause}
                     ORDER BY r.run_id""",
                (METRIC_NAME, lower, upper),
            )
        ]


def _eligible_rows(database: Path, lower: int, upper: int, overwrite: bool):
    metric_clause = "" if overwrite else "AND metric.run_id IS NULL"
    sql = f"""
        SELECT r.run_id,
               CAST(n.numeric_value AS INTEGER) AS N,
               duration.numeric_value AS t_interval,
               background.numeric_value AS r_bg,
               objective.numeric_value AS best_objective,
               u.values_blob AS u_blob,
               v.values_blob AS v_blob
          FROM runs r
          JOIN physical_values n
            ON n.run_id=r.run_id AND n.name='N'
          JOIN physical_values duration
            ON duration.run_id=r.run_id AND duration.name='t_interval'
          JOIN physical_values background
            ON background.run_id=r.run_id AND background.name='r_bg'
          JOIN physical_values objective
            ON objective.run_id=r.run_id AND objective.name='best_objective'
          JOIN control_arrays u
            ON u.run_id=r.run_id AND u.kind='best' AND u.name='u'
          JOIN control_arrays v
            ON v.run_id=r.run_id AND v.kind='best' AND v.name='v'
          LEFT JOIN physical_values metric
            ON metric.run_id=r.run_id AND metric.name=?
         WHERE r.run_id BETWEEN ? AND ? {metric_clause}
         ORDER BY r.run_id
    """
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(sql, (METRIC_NAME, lower, upper)).fetchall()


def _parameter_settings(database: Path, run_ids: list[int]):
    if not run_ids:
        return {}
    parameter_database = parameter_database_path(database)
    settings = {}
    with sqlite3.connect(
        f"file:{parameter_database}?mode=ro", uri=True
    ) as connection:
        connection.row_factory = sqlite3.Row
        for start in range(0, len(run_ids), 900):
            chunk = run_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"""SELECT run_id, parameters_json FROM run_parameters
                     WHERE run_id IN ({placeholders})""",
                chunk,
            ).fetchall()
            for row in rows:
                values = json.loads(row["parameters_json"])
                tolerance = values.get("grid_refinement_tol")
                if tolerance is None:
                    name = str(values.get("config_name", "")).lower()
                    j_tolerance = values.get("J_tol")
                    strict = "strict" in name or (
                        isinstance(j_tolerance, (int, float))
                        and not isinstance(j_tolerance, bool)
                        and float(j_tolerance) <= 1e-6
                    )
                    tolerance = (
                        DEFAULT_STRICT_TOLERANCE
                        if strict
                        else DEFAULT_LOOSE_TOLERANCE
                    )
                settings[int(row["run_id"])] = (
                    float(tolerance),
                    float(values.get("grid_refinement_y_floor", DEFAULT_Y_FLOOR)),
                )
    return settings


def _record(row, settings):
    run_id = int(row["run_id"])
    tolerance, y_floor = settings.get(
        run_id, (DEFAULT_LOOSE_TOLERANCE, DEFAULT_Y_FLOOR)
    )
    return {
        "run_id": run_id,
        "N": int(row["N"]),
        "t_interval": float(row["t_interval"]),
        "r_bg": float(row["r_bg"]),
        "best_objective": float(row["best_objective"]),
        "u": decode_array(row["u_blob"]),
        "v": decode_array(row["v_blob"]),
        "tolerance": tolerance,
        "y_floor": y_floor,
    }


def _calculate(records: list[dict], *, use_jit: bool = True):
    N = records[0]["N"]
    if any(record["N"] != N for record in records):
        raise ValueError("A diagnostic batch must contain one base grid size.")
    controls = {
        name: np.stack([record[name] for record in records]) for name in ("u", "v")
    }
    diagnostics = jax.device_get(
        grid_refinement_diagnostics(
            controls,
            np.asarray([record["best_objective"] for record in records]),
            N=N,
            r_bg=np.asarray([record["r_bg"] for record in records]),
            t_interval=np.asarray(
                [record["t_interval"] for record in records]
            ),
            tolerance=np.asarray([record["tolerance"] for record in records]),
            y_floor=np.asarray([record["y_floor"] for record in records]),
            dtype=jnp.float64,
            use_jit=use_jit,
        )
    )
    return {
        record["run_id"]: {
            **{
                name: np.asarray(values)[index].item()
                for name, values in diagnostics.items()
            },
            "best_grid_refinement_status": "computed",
        }
        for index, record in enumerate(records)
    }


def backfill_database(
    database: str | Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    batch_size: int = 32,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Calculate all missing eligible metrics in one numeric run-ID shard."""

    database = Path(database).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count).")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    bounds = _run_id_bounds(database, shard_index, shard_count)
    if bounds is None:
        return {"pending": 0, "computed": 0, "unavailable": 0, "failed": 0}
    lower, upper = bounds
    pending = _pending_run_ids(database, lower, upper, overwrite)
    rows = list(_eligible_rows(database, lower, upper, overwrite))
    if limit is not None:
        rows = rows[:limit]
        pending = pending[:limit]
    settings = _parameter_settings(database, [int(row["run_id"]) for row in rows])
    grouped = defaultdict(list)
    failed = {}
    for row in rows:
        try:
            record = _record(row, settings)
            if (
                record["N"] < 2
                or not math.isfinite(record["best_objective"])
                or record["tolerance"] <= 0.0
                or record["y_floor"] <= 0.0
            ):
                raise ValueError("invalid stored grid diagnostic input")
            grouped[record["N"]].append(record)
        except Exception as error:
            failed[int(row["run_id"])] = {
                "best_grid_refinement_status": (
                    f"error:{type(error).__name__}:{error}"
                )
            }

    store = ResultStore(database)
    computed_count = 0
    for N, records in sorted(grouped.items()):
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            try:
                values = _calculate(batch)
            except Exception:
                values = {}
                for record in batch:
                    try:
                        values.update(_calculate([record]))
                    except Exception as error:
                        failed[record["run_id"]] = {
                            "best_grid_refinement_status": (
                                f"error:{type(error).__name__}:{error}"
                            )
                        }
            if values:
                store.save_physical_values(values)
                computed_count += len(values)
            print(
                f"{database.name} | shard {shard_index + 1}/{shard_count} | "
                f"N={N} | computed {min(start + len(batch), len(records))}/"
                f"{len(records)}",
                flush=True,
            )

    eligible_ids = {int(row["run_id"]) for row in rows}
    unavailable_ids = set(pending) - eligible_ids
    unavailable = {
        run_id: {
            "best_grid_refinement_status": (
                "unavailable:no_finite_best_objective_or_best_controls"
            )
        }
        for run_id in unavailable_ids
    }
    if unavailable:
        store.save_physical_values(unavailable)
    if failed:
        store.save_physical_values(failed)
    summary = {
        "pending": len(pending),
        "computed": computed_count,
        "unavailable": len(unavailable),
        "failed": len(failed),
    }
    print(
        f"DONE {database} | shard {shard_index + 1}/{shard_count} | "
        + " | ".join(f"{name}={value}" for name, value in summary.items()),
        flush=True,
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    backfill_database(
        arguments.database,
        shard_index=arguments.shard_index,
        shard_count=arguments.shard_count,
        batch_size=arguments.batch_size,
        overwrite=arguments.overwrite,
        limit=arguments.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
