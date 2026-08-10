"""Resolve data-backed Fourier intensity centers in bounded control space."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np

from .results import Results
from .storage import parameter_database_path


AUTO_PRIOR_SCHEMA_VERSION = 1
AUTO_PRIOR_WEIGHT = 0.3
AUTO_GLOBAL_LIMIT = 10
AUTO_EXACT_CAP_LIMIT = 50
AUTO_PRIOR_TABLE = "fourier_intensity_priors"


@dataclass(frozen=True)
class IntensityPrior:
    """One ranked solution summarized by its mean bounded intensity control."""

    source_key: str
    source_database: str
    source_run_id: int
    t_interval: float
    r_bg: float
    u_max: float
    best_objective: float
    mean_u: float


@dataclass(frozen=True)
class AutoIntensityCenter:
    """Resolved auto-center and enough provenance to reproduce the calculation."""

    bounded_center: float
    fraction: float
    source_count: int
    source_keys: tuple[str, ...]
    global_source_count: int
    exact_cap_source_count: int

    def metadata(self) -> dict:
        return {
            "fourier_u_center_mode": "auto",
            "fourier_u_center": self.bounded_center,
            "fourier_u_center_fraction": self.fraction,
            "fourier_u_center_prior_fraction": AUTO_PRIOR_WEIGHT,
            "fourier_u_center_source_count": self.source_count,
            "fourier_u_center_global_source_count": self.global_source_count,
            "fourier_u_center_exact_cap_source_count": self.exact_cap_source_count,
            "fourier_u_center_source_keys": list(self.source_keys),
        }


def _ranked(priors: Iterable[IntensityPrior], limit: int) -> tuple[IntensityPrior, ...]:
    unique = {}
    for prior in priors:
        if not math.isfinite(prior.best_objective) or not math.isfinite(prior.mean_u):
            continue
        existing = unique.get(prior.source_key)
        if existing is None or prior.best_objective > existing.best_objective:
            unique[prior.source_key] = prior
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.best_objective, item.source_key),
            reverse=True,
        )[:limit]
    )


def priors_from_results(
    database: str | Path,
    *,
    t_interval: float,
    r_bg: float,
    u_max: float | None,
    limit: int,
) -> tuple[IntensityPrior, ...]:
    """Load ranked complete solutions from a normal result database pair."""

    database = Path(database).expanduser().resolve()
    if not database.is_file() or not parameter_database_path(database).is_file():
        return ()
    filters = {
        "status": "complete",
        "t_interval": float(t_interval),
        "r_bg": float(r_bg),
    }
    if u_max is not None:
        filters["u_max"] = float(u_max)
    results = Results(database)
    # Ask for spare rows because old or interrupted records may lack best controls.
    rows = results.search(
        limit=max(limit * 3, limit),
        order_by="best_objective",
        descending=True,
        **filters,
    )
    priors = []
    for row in rows:
        objective = row.get("best_objective")
        if objective is None:
            continue
        try:
            controls = results.controls(int(row["run_id"]), "best")
        except KeyError:
            continue
        mean_u = float(np.mean(np.asarray(controls["u"], dtype=float)))
        run_id = int(row["run_id"])
        source_database = str(database)
        priors.append(
            IntensityPrior(
                source_key=f"{source_database}#{run_id}",
                source_database=source_database,
                source_run_id=run_id,
                t_interval=float(row["t_interval"]),
                r_bg=float(row["r_bg"]),
                u_max=float(row["u_max"]),
                best_objective=float(objective),
                mean_u=mean_u,
            )
        )
    return _ranked(priors, limit)


def priors_from_snapshot(
    database: str | Path,
    *,
    t_interval: float,
    r_bg: float,
    u_max: float | None,
    limit: int,
) -> tuple[IntensityPrior, ...]:
    """Load ranked summaries from a portable auto-center prior database."""

    database = Path(database).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Auto intensity prior database is missing: {database}")
    clauses = ["t_interval=?", "r_bg=?"]
    arguments: list[float | int] = [float(t_interval), float(r_bg)]
    if u_max is not None:
        clauses.append("u_max=?")
        arguments.append(float(u_max))
    arguments.append(int(limit))
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            version = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if version is None or int(version[0]) != AUTO_PRIOR_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported auto intensity prior database version in {database}."
                )
            rows = connection.execute(
                f"""SELECT source_key, source_database, source_run_id,
                           t_interval, r_bg, u_max, best_objective, mean_u
                    FROM {AUTO_PRIOR_TABLE}
                    WHERE {' AND '.join(clauses)}
                    ORDER BY best_objective DESC, source_key DESC
                    LIMIT ?""",
                arguments,
            ).fetchall()
    except sqlite3.DatabaseError as error:
        raise RuntimeError(
            f"Invalid auto intensity prior database {database}: {error}"
        ) from error
    return tuple(IntensityPrior(*row) for row in rows)


def write_prior_snapshot(
    database: str | Path,
    priors: Iterable[IntensityPrior],
    *,
    metadata: dict | None = None,
) -> Path:
    """Atomically write a compact, portable database of intensity summaries."""

    database = Path(database).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_suffix(database.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with sqlite3.connect(temporary) as connection:
        connection.executescript(
            f"""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE {AUTO_PRIOR_TABLE}(
                source_key TEXT PRIMARY KEY,
                source_database TEXT NOT NULL,
                source_run_id INTEGER NOT NULL,
                t_interval REAL NOT NULL,
                r_bg REAL NOT NULL,
                u_max REAL NOT NULL,
                best_objective REAL NOT NULL,
                mean_u REAL NOT NULL
            );
            CREATE INDEX idx_auto_prior_conditions
                ON {AUTO_PRIOR_TABLE}(t_interval, r_bg, u_max, best_objective);
            """
        )
        records = [asdict(prior) for prior in priors]
        connection.executemany(
            f"""INSERT INTO {AUTO_PRIOR_TABLE}(
                    source_key, source_database, source_run_id, t_interval,
                    r_bg, u_max, best_objective, mean_u
                ) VALUES(
                    :source_key, :source_database, :source_run_id, :t_interval,
                    :r_bg, :u_max, :best_objective, :mean_u
                )""",
            records,
        )
        stored_metadata = {
            "schema_version": str(AUTO_PRIOR_SCHEMA_VERSION),
            "formula": (
                "u_max * (0.3 + sum(unique top-10 matching T,r_bg mean_u/source_u_max) "
                "+ sum(unique top-50 exact-cap mean_u/source_u_max)) / "
                "(1 + unique source count)"
            ),
            "row_count": str(len(records)),
            **{
                str(name): json.dumps(value, sort_keys=True)
                for name, value in (metadata or {}).items()
            },
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES(?, ?)",
            stored_metadata.items(),
        )
    temporary.replace(database)
    return database


def resolve_auto_intensity_center(
    *,
    output_database: str | Path,
    prior_database: str | Path | None,
    t_interval: float,
    r_bg: float,
    u_max: float,
) -> AutoIntensityCenter:
    """Apply the requested prior/global/exact-cap weighted center formula.

    The normal output database is included so later configs can learn from
    results produced earlier in the same queue. The optional compact snapshot
    supplies cross-cap context without copying full result histories.
    """

    global_candidates = list(
        priors_from_results(
            output_database,
            t_interval=t_interval,
            r_bg=r_bg,
            u_max=None,
            limit=AUTO_GLOBAL_LIMIT,
        )
    )
    exact_candidates = list(
        priors_from_results(
            output_database,
            t_interval=t_interval,
            r_bg=r_bg,
            u_max=u_max,
            limit=AUTO_EXACT_CAP_LIMIT,
        )
    )
    if prior_database is not None:
        global_candidates.extend(
            priors_from_snapshot(
                prior_database,
                t_interval=t_interval,
                r_bg=r_bg,
                u_max=None,
                limit=AUTO_GLOBAL_LIMIT,
            )
        )
        exact_candidates.extend(
            priors_from_snapshot(
                prior_database,
                t_interval=t_interval,
                r_bg=r_bg,
                u_max=u_max,
                limit=AUTO_EXACT_CAP_LIMIT,
            )
        )

    global_priors = _ranked(global_candidates, AUTO_GLOBAL_LIMIT)
    exact_priors = _ranked(exact_candidates, AUTO_EXACT_CAP_LIMIT)
    selected = {prior.source_key: prior for prior in global_priors}
    selected.update({prior.source_key: prior for prior in exact_priors})
    fraction = (
        AUTO_PRIOR_WEIGHT
        + sum(prior.mean_u / prior.u_max for prior in selected.values())
    ) / (len(selected) + 1)
    minimum_fraction = 1e-7
    fraction = float(
        np.clip(fraction, minimum_fraction, 1.0 - minimum_fraction)
    )
    return AutoIntensityCenter(
        bounded_center=fraction * float(u_max),
        fraction=fraction,
        source_count=len(selected),
        source_keys=tuple(sorted(selected)),
        global_source_count=len(global_priors),
        exact_cap_source_count=len(exact_priors),
    )
