"""Read-only access to physical results joined to flexible run methodology.

The primary database contains only calculated physical values and arrays. The
adjacent ``*.parameters.sqlite3`` database is attached read-only when config or
optimizer provenance is needed. ``run_id`` is the sole cross-database link.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

import numpy as np

from .storage import decode_array, parameter_database_path


RUN_COLUMNS = {
    "run_id": "r.run_id",
    "status": "rp.status",
}
PHYSICAL_ALIASES = {
    "best_obj": "best_objective",
    "final_obj": "final_objective",
    "max_grad_u": "best_max_abs_du_dt",
    "max_grad_v": "best_max_abs_dv_dt",
}
PHYSICAL_VALUES = {
    "N",
    "t_interval",
    "r_bg",
    "u_max",
    "v_max",
    "completed_steps",
    "best_score",
    "best_objective",
    "best_penalty",
    "best_step",
    "final_score",
    "final_objective",
    "final_penalty",
    "best_max_abs_du_dt",
    "best_max_abs_dv_dt",
    "best_max_abs_d2u_dt2",
    "best_max_abs_d2v_dt2",
}


def _is_range(value) -> bool:
    return isinstance(value, tuple) and len(value) == 2


def _append_condition(clauses, arguments, expression, value):
    if _is_range(value):
        lower, upper = value
        if lower is not None:
            clauses.append(f"{expression} >= ?")
            arguments.append(lower)
        if upper is not None:
            clauses.append(f"{expression} <= ?")
            arguments.append(upper)
    elif isinstance(value, (list, set, frozenset)):
        values = list(value)
        if not values:
            clauses.append("0")
        else:
            clauses.append(f"{expression} IN ({','.join('?' for _ in values)})")
            arguments.extend(values)
    else:
        clauses.append(f"{expression} = ?")
        arguments.append(value)


def _eav_condition(table: str, run_expression: str, name: str, value):
    condition = [f"value.run_id={run_expression}", "value.name=?"]
    arguments = [name]
    if _is_range(value):
        lower, upper = value
        if lower is not None:
            condition.append("value.numeric_value >= ?")
            arguments.append(lower)
        if upper is not None:
            condition.append("value.numeric_value <= ?")
            arguments.append(upper)
    elif isinstance(value, (list, set, frozenset)):
        encoded = [
            json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value
        ]
        if not encoded:
            return "0", []
        condition.append(
            f"value.json_value IN ({','.join('?' for _ in encoded)})"
        )
        arguments.extend(encoded)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        condition.append("value.numeric_value=?")
        arguments.append(float(value))
    else:
        condition.append("value.json_value=?")
        arguments.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return f"EXISTS(SELECT 1 FROM {table} value WHERE {' AND '.join(condition)})", arguments


def _chunks(values, size=900):
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start : start + size]


class Results:
    """Search scalar metadata efficiently and load arrays only on request."""

    def __init__(self, database: str | Path = "results/results.sqlite3"):
        self.database = Path(database).expanduser().resolve()
        self.parameter_database = parameter_database_path(self.database)

    def connect(self):
        connection = sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute(
            "ATTACH DATABASE ? AS methodology",
            (f"file:{self.parameter_database}?mode=ro",),
        )
        return connection

    def connect_parameters(self):
        connection = sqlite3.connect(
            f"file:{self.parameter_database}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _physical_name(name: str) -> str:
        return PHYSICAL_ALIASES.get(name, name)

    def search(
        self,
        *,
        limit: int | None = None,
        order_by="run_id",
        descending: bool = False,
        **filters,
    ):
        """Return runs matching physical values or arbitrary config parameters."""

        if not isinstance(descending, bool):
            raise ValueError("descending must be a boolean.")
        clauses, arguments = [], []
        for requested_name, value in filters.items():
            if requested_name in RUN_COLUMNS:
                _append_condition(
                    clauses, arguments, RUN_COLUMNS[requested_name], value
                )
                continue
            name = self._physical_name(requested_name)
            if name in PHYSICAL_VALUES:
                clause, values = _eav_condition(
                    "physical_values", "r.run_id", name, value
                )
            else:
                clause, values = _eav_condition(
                    "methodology.run_parameter_values", "r.run_id", name, value
                )
            clauses.append(clause)
            arguments.extend(values)

        order_name = self._physical_name(str(order_by))
        if order_by in RUN_COLUMNS:
            order_expression = RUN_COLUMNS[order_by]
        elif order_name in PHYSICAL_VALUES:
            order_expression = (
                "(SELECT COALESCE(value.numeric_value, value.text_value) "
                "FROM physical_values value WHERE value.run_id=r.run_id "
                "AND value.name=?)"
            )
            arguments.append(order_name)
        else:
            order_expression = (
                "(SELECT COALESCE(value.numeric_value, value.text_value) "
                "FROM methodology.run_parameter_values value "
                "WHERE value.run_id=r.run_id AND value.name=?)"
            )
            arguments.append(str(order_by))

        direction = "DESC" if descending else "ASC"
        sql = f"""
            SELECT r.run_id, rp.status, rp.parameters_json
            FROM runs r
            JOIN methodology.run_parameters rp ON rp.run_id=r.run_id
            {'WHERE ' + ' AND '.join(clauses) if clauses else ''}
            ORDER BY {order_expression} {direction}, r.run_id {direction}
        """
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError("limit must be a positive integer.")
            sql += " LIMIT ?"
            arguments.append(limit)

        with self.connect() as connection:
            rows = connection.execute(sql, arguments).fetchall()
            ids = [int(row["run_id"]) for row in rows]
            physical = {run_id: {} for run_id in ids}
            for chunk in _chunks(ids):
                placeholders = ",".join("?" for _ in chunk)
                values = connection.execute(
                    f"""SELECT run_id, name, json_value FROM physical_values
                        WHERE run_id IN ({placeholders})""",
                    chunk,
                ).fetchall()
                for value in values:
                    physical[int(value["run_id"])][value["name"]] = json.loads(
                        value["json_value"]
                    )

        output = []
        for row in rows:
            run_id = int(row["run_id"])
            item = json.loads(row["parameters_json"])
            item.update(physical[run_id])
            item.update({"run_id": run_id, "status": row["status"]})
            output.append(item)
        return output

    def controls(self, run_id: int, kind: str = "best") -> dict[str, np.ndarray]:
        if kind not in {"initial", "best", "final"}:
            raise ValueError("kind must be initial, best, or final.")
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT name, values_blob FROM control_arrays WHERE run_id=? AND kind=?",
                (run_id, kind),
            ).fetchall()
        if not rows:
            raise KeyError(f"No {kind} controls found for run_id={run_id}.")
        return {row["name"]: decode_array(row["values_blob"]) for row in rows}

    def history(self, run_id: int) -> dict[str, np.ndarray]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT chunk_index, start_step, end_step, name, values_blob
                   FROM history_arrays WHERE run_id=?
                   ORDER BY name, chunk_index""",
                (run_id,),
            ).fetchall()
        if not rows:
            raise KeyError(f"No history found for run_id={run_id}.")

        names = sorted({row["name"] for row in rows})
        output = {}
        for name in names:
            pieces = [
                decode_array(row["values_blob"])
                for row in rows
                if row["name"] == name
            ]
            output[name] = np.concatenate(
                [pieces[0], *(piece[1:] for piece in pieces[1:])]
            )

        with self.connect_parameters() as connection:
            stages = connection.execute(
                """SELECT stage_index, start_step, end_step, parameters_json
                   FROM optimizer_stages WHERE run_id=? ORDER BY stage_index""",
                (run_id,),
            ).fetchall()
        sizes = []
        for index, stage in enumerate(stages):
            values = json.loads(stage["parameters_json"])
            length = stage["end_step"] - stage["start_step"] + (
                1 if index == 0 else 0
            )
            sizes.append(
                np.full(length, values.get("optimizer_step_size", np.nan), dtype=float)
            )
        if sizes:
            output["optimizer_step_size"] = np.concatenate(sizes)
            output["optimizer_step_size_change_steps"] = np.asarray(
                [stage["start_step"] for stage in stages], dtype=int
            )
            output["stage_optimizer_step_sizes"] = np.asarray(
                [
                    json.loads(stage["parameters_json"]).get(
                        "optimizer_step_size", np.nan
                    )
                    for stage in stages
                ],
                dtype=float,
            )
            # Compatibility aliases for existing convergence plots.
            output["learning_rate"] = output["optimizer_step_size"]
            output["learning_rate_change_steps"] = output[
                "optimizer_step_size_change_steps"
            ]
            output["stage_learning_rates"] = output[
                "stage_optimizer_step_sizes"
            ]
        output["step"] = np.arange(next(iter(output.values())).size)
        return output

    def tolerances(self, run_id: int) -> dict[str, np.ndarray]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT step, values_json FROM tolerance_history
                   WHERE run_id=? ORDER BY step""",
                (run_id,),
            ).fetchall()
        records = [json.loads(row["values_json"]) for row in rows]
        names = sorted({name for record in records for name in record})
        output = {
            "step": np.asarray([row["step"] for row in rows], dtype=int),
            **{
                name: np.asarray([record.get(name, np.nan) for record in records])
                for name in names
            },
        }
        if "passed" in output:
            output["passed"] = output["passed"].astype(np.uint8)
        return output

    def run_parameters(self, run_id: int) -> dict[str, Any]:
        with self.connect_parameters() as connection:
            row = connection.execute(
                "SELECT parameters_json FROM run_parameters WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"No run parameters found for run_id={run_id}.")
        return json.loads(row["parameters_json"])

    def config_document(self, run_id: int) -> dict[str, Any]:
        with self.connect_parameters() as connection:
            row = connection.execute(
                """SELECT c.config_json FROM config_documents c
                   JOIN run_parameters r
                     ON r.config_document_id=c.config_document_id
                   WHERE r.run_id=?""",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"No config document found for run_id={run_id}.")
        return json.loads(row["config_json"])

    def get(self, run_id: int, *, arrays: bool = True) -> Mapping[str, Any]:
        matches = self.search(run_id=run_id, limit=1)
        if not matches:
            raise KeyError(f"Unknown run_id={run_id}.")
        result = dict(matches[0])
        result["config"] = self.config_document(run_id)
        if arrays:
            result["history"] = self.history(run_id)
            result["tolerances"] = self.tolerances(run_id)
            result["controls"] = {
                kind: self.controls(run_id, kind)
                for kind in ("initial", "best", "final")
            }
        return result
