"""Transactional persistence split into physical results and run methodology.

The physical database contains only calculated quantities keyed by ``run_id``.
The adjacent parameter database contains exact config documents, resolved
per-run settings, and optimizer-stage provenance. Both schemas use generic
name/value or array records so adding config fields or calculated diagnostics
does not require a database migration.

Isolation boundary: this module accepts plain records and arrays. It does not
load configs, perform calculations, invoke the runner, or plot.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
import json
from numbers import Integral, Real
from pathlib import Path
import sqlite3
import zlib

import numpy as np


PHYSICAL_DATABASE_VERSION = 1
PARAMETER_DATABASE_VERSION = 1
SQLITE_TIMEOUT_SECONDS = 3600.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parameter_database_path(physical_database: str | Path) -> Path:
    """Return the adjacent methodology database associated with a result file."""

    path = Path(physical_database).expanduser().resolve()
    suffix = path.suffix or ".sqlite3"
    return path.with_name(f"{path.stem}.parameters{suffix}")


def encode_array(values) -> bytes:
    buffer = BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    return zlib.compress(buffer.getvalue(), level=3)


def decode_array(blob: bytes) -> np.ndarray:
    return np.load(BytesIO(zlib.decompress(blob)), allow_pickle=False)


PHYSICAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS physical_values (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    numeric_value REAL,
    text_value TEXT,
    json_value TEXT NOT NULL,
    PRIMARY KEY(run_id, name)
);
CREATE INDEX IF NOT EXISTS idx_physical_number
    ON physical_values(name, numeric_value, run_id);
CREATE INDEX IF NOT EXISTS idx_physical_text
    ON physical_values(name, text_value, run_id);

CREATE TABLE IF NOT EXISTS control_arrays (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    values_blob BLOB NOT NULL,
    PRIMARY KEY(run_id, kind, name)
);
CREATE TABLE IF NOT EXISTS history_arrays (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    start_step INTEGER NOT NULL,
    end_step INTEGER NOT NULL,
    name TEXT NOT NULL,
    values_blob BLOB NOT NULL,
    PRIMARY KEY(run_id, chunk_index, name)
);
CREATE TABLE IF NOT EXISTS tolerance_history (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    values_json TEXT NOT NULL,
    PRIMARY KEY(run_id, step)
);
"""


PARAMETER_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS config_documents (
    config_document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    registered_utc TEXT NOT NULL,
    UNIQUE(config_id, config_json)
);
CREATE INDEX IF NOT EXISTS idx_config_documents_id
    ON config_documents(config_id, config_document_id);

CREATE TABLE IF NOT EXISTS execution_batches (
    execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_document_id INTEGER NOT NULL
        REFERENCES config_documents(config_document_id),
    batch_id INTEGER NOT NULL,
    queue_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    execution_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    started_utc TEXT NOT NULL,
    completed_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_execution_batch
    ON execution_batches(batch_id, execution_id);
CREATE INDEX IF NOT EXISTS idx_execution_queue
    ON execution_batches(queue_id, execution_id);

CREATE TABLE IF NOT EXISTS run_parameters (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id INTEGER NOT NULL
        REFERENCES execution_batches(execution_id),
    config_document_id INTEGER NOT NULL
        REFERENCES config_documents(config_document_id),
    status TEXT NOT NULL,
    error TEXT,
    started_utc TEXT NOT NULL,
    completed_utc TEXT,
    parameters_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_parameters_execution
    ON run_parameters(execution_id, run_id);
CREATE INDEX IF NOT EXISTS idx_run_parameters_status
    ON run_parameters(status, run_id);

CREATE TABLE IF NOT EXISTS run_parameter_values (
    run_id INTEGER NOT NULL REFERENCES run_parameters(run_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    numeric_value REAL,
    text_value TEXT,
    json_value TEXT NOT NULL,
    PRIMARY KEY(run_id, name)
);
CREATE INDEX IF NOT EXISTS idx_run_parameter_number
    ON run_parameter_values(name, numeric_value, run_id);
CREATE INDEX IF NOT EXISTS idx_run_parameter_text
    ON run_parameter_values(name, text_value, run_id);

CREATE TABLE IF NOT EXISTS optimizer_stages (
    run_id INTEGER NOT NULL REFERENCES run_parameters(run_id) ON DELETE CASCADE,
    stage_index INTEGER NOT NULL,
    start_step INTEGER NOT NULL,
    end_step INTEGER NOT NULL,
    parameters_json TEXT NOT NULL,
    saved_utc TEXT NOT NULL,
    PRIMARY KEY(run_id, stage_index)
);
"""


class ResultStore:
    """Write physical outputs and flexible methodology through short connections."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.parameter_path = parameter_database_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.parameter_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialise()

    @staticmethod
    def _connect(path: Path):
        connection = sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            f"PRAGMA busy_timeout = {int(SQLITE_TIMEOUT_SECONDS * 1000)}"
        )
        return connection

    def connect(self):
        return self._connect(self.path)

    def connect_parameters(self):
        return self._connect(self.parameter_path)

    @staticmethod
    def _initialise_database(connection, schema: str, *, version: int, role: str):
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.executescript(schema)
        current = connection.execute(
            "SELECT value FROM metadata WHERE key='database_version'"
        ).fetchone()
        if current is not None and int(current[0]) != version:
            raise RuntimeError(
                f"{role} database version {current[0]} is not supported by version {version}."
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('database_version', ?)",
            (str(version),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('database_role', ?)",
            (role,),
        )

    def initialise(self) -> None:
        with self.connect() as connection:
            self._initialise_database(
                connection,
                PHYSICAL_SCHEMA,
                version=PHYSICAL_DATABASE_VERSION,
                role="physical_results",
            )
        with self.connect_parameters() as connection:
            self._initialise_database(
                connection,
                PARAMETER_SCHEMA,
                version=PARAMETER_DATABASE_VERSION,
                role="run_parameters",
            )

    @contextmanager
    def transaction(self, *, parameters: bool = False):
        connection = self.connect_parameters() if parameters else self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _value_columns(value):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if isinstance(value, bool):
            return None, "true" if value else "false", encoded
        if isinstance(value, (Integral, Real)) and value is not None:
            return float(value), None, encoded
        if value is None:
            return None, "null", encoded
        if isinstance(value, str):
            return None, value, encoded
        return None, encoded, encoded

    @classmethod
    def _write_values(cls, connection, table: str, run_id: int, values: dict):
        for name, value in values.items():
            if value is None:
                continue
            numeric, text, encoded = cls._value_columns(value)
            connection.execute(
                f"""INSERT INTO {table}(
                           run_id, name, numeric_value, text_value, json_value
                       ) VALUES(?, ?, ?, ?, ?)
                       ON CONFLICT(run_id, name) DO UPDATE SET
                           numeric_value=excluded.numeric_value,
                           text_value=excluded.text_value,
                           json_value=excluded.json_value""",
                (run_id, name, numeric, text, encoded),
            )

    def register_config(self, record: dict) -> int:
        """Store an exact config document and return its internal document ID.

        A reused ``config_id`` with changed content creates another immutable
        document instead of invalidating or overwriting earlier provenance.
        """

        payload = json.dumps(record["config"], sort_keys=True, separators=(",", ":"))
        with self.transaction(parameters=True) as connection:
            connection.execute(
                """INSERT OR IGNORE INTO config_documents(
                       config_id, config_json, registered_utc
                   ) VALUES(?, ?, ?)""",
                (record["config_id"], payload, utc_now()),
            )
            row = connection.execute(
                """SELECT config_document_id FROM config_documents
                   WHERE config_id=? AND config_json=?""",
                (record["config_id"], payload),
            ).fetchone()
        return int(row["config_document_id"])

    def prepare_batch(
        self,
        record: dict,
        cases: list[dict],
        initialisations: int,
        *,
        config_document_id: int,
        initialization_metadata: list[dict] | None = None,
    ):
        """Allocate fresh run IDs and store exact settings before computation."""

        mapping = {}
        run_records = []
        if initialization_metadata is None:
            initialization_metadata = [{} for _ in range(initialisations)]
        if len(initialization_metadata) != initialisations:
            raise ValueError(
                "initialization_metadata must contain one record per initialization."
            )
        execution_payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self.transaction(parameters=True) as connection:
            cursor = connection.execute(
                """INSERT INTO execution_batches(
                       config_document_id, batch_id, queue_id, batch_index,
                       execution_json, status, error, started_utc, completed_utc
                   ) VALUES(?, ?, ?, ?, ?, 'running', NULL, ?, NULL)""",
                (
                    config_document_id,
                    record["batch_id"],
                    record["queue_id"],
                    record["batch_index"],
                    execution_payload,
                    utc_now(),
                ),
            )
            execution_id = int(cursor.lastrowid)
            for case_index, parameters in enumerate(cases):
                for initialization_index in range(initialisations):
                    resolved = {
                        **parameters,
                        "config_id": record["config_id"],
                        "config_document_id": config_document_id,
                        "config_name": record["config_name"],
                        "config_file": record["config_file"],
                        "description": record.get("description", ""),
                        "created_utc": record.get("created_utc", ""),
                        "batch_id": record["batch_id"],
                        "execution_id": execution_id,
                        "queue_id": record["queue_id"],
                        "batch_index": record["batch_index"],
                        "batch_key": record["batch_key"],
                        "case_index": case_index,
                        "initialization_index": initialization_index,
                        "initialisation_index": initialization_index,
                        "seed": record["seed"],
                        "initialization_count_total": initialisations,
                        **initialization_metadata[initialization_index],
                    }
                    cursor = connection.execute(
                        """INSERT INTO run_parameters(
                               execution_id, config_document_id, status, error,
                               started_utc, completed_utc, parameters_json
                           ) VALUES(?, ?, 'running', NULL, ?, NULL, ?)""",
                        (
                            execution_id,
                            config_document_id,
                            utc_now(),
                            json.dumps(resolved, sort_keys=True, separators=(",", ":")),
                        ),
                    )
                    run_id = int(cursor.lastrowid)
                    self._write_values(
                        connection, "run_parameter_values", run_id, resolved
                    )
                    mapping[(case_index, initialization_index)] = run_id
                    run_records.append((run_id, parameters))

        try:
            with self.transaction() as connection:
                for run_id, parameters in run_records:
                    connection.execute("INSERT INTO runs(run_id) VALUES(?)", (run_id,))
                    self._write_values(
                        connection,
                        "physical_values",
                        run_id,
                        {
                            name: parameters[name]
                            for name in ("N", "t_interval", "r_bg", "u_max", "v_max")
                        },
                    )
        except Exception:
            with self.transaction(parameters=True) as connection:
                connection.execute(
                    "DELETE FROM run_parameters WHERE execution_id=?",
                    (execution_id,),
                )
                connection.execute(
                    "DELETE FROM execution_batches WHERE execution_id=?",
                    (execution_id,),
                )
            raise
        return execution_id, mapping

    def save_stage(
        self,
        *,
        execution_id: int,
        stage_index: int,
        start_step: int,
        end_step: int,
        members: list[dict],
        tolerances: list[dict],
    ) -> None:
        """Commit physical histories and separate optimizer-stage provenance."""

        encoded_members = []
        for member in members:
            history_blobs = {
                name: encode_array(values) for name, values in member["history"].items()
            }
            control_blobs = {
                (kind, name): encode_array(values)
                for kind in ("initial", "best", "final")
                if kind in member
                for name, values in member[kind].items()
            }
            encoded_members.append((member, history_blobs, control_blobs))

        with self.transaction(parameters=True) as connection:
            for member in members:
                connection.execute(
                    """INSERT OR REPLACE INTO optimizer_stages(
                           run_id, stage_index, start_step, end_step,
                           parameters_json, saved_utc
                       ) VALUES(?, ?, ?, ?, ?, ?)""",
                    (
                        member["run_id"],
                        stage_index,
                        start_step,
                        end_step,
                        json.dumps(
                            member.get("optimizer_stage", {}),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        utc_now(),
                    ),
                )

        with self.transaction() as connection:
            for member, history_blobs, control_blobs in encoded_members:
                run_id = member["run_id"]
                self._write_values(
                    connection,
                    "physical_values",
                    run_id,
                    {
                        "completed_steps": end_step,
                        "best_score": member["best_score"],
                        "best_objective": member["best_objective"],
                        "best_penalty": member["best_penalty"],
                        "best_step": member["best_step"],
                        "final_score": member["final_score"],
                        "final_objective": member["final_objective"],
                        "final_penalty": member["final_penalty"],
                        **member.get("best_derivatives", {}),
                    },
                )
                for name, blob in history_blobs.items():
                    connection.execute(
                        """INSERT OR REPLACE INTO history_arrays(
                               run_id, chunk_index, start_step, end_step,
                               name, values_blob
                           ) VALUES(?, ?, ?, ?, ?, ?)""",
                        (run_id, stage_index, start_step, end_step, name, blob),
                    )
                for (kind, name), blob in control_blobs.items():
                    connection.execute(
                        """INSERT OR REPLACE INTO control_arrays(
                               run_id, kind, name, values_blob
                           ) VALUES(?, ?, ?, ?)""",
                        (run_id, kind, name, blob),
                    )
            for tolerance in tolerances:
                values = {
                    name: value
                    for name, value in tolerance.items()
                    if name not in {"run_id", "member", "step"}
                }
                connection.execute(
                    """INSERT OR REPLACE INTO tolerance_history(
                           run_id, step, values_json
                       ) VALUES(?, ?, ?)""",
                    (
                        tolerance["run_id"],
                        tolerance["step"],
                        json.dumps(values, sort_keys=True, separators=(",", ":")),
                    ),
                )

    def complete_batch(self, execution_id: int, run_ids) -> None:
        completed = utc_now()
        with self.transaction(parameters=True) as connection:
            connection.executemany(
                """UPDATE run_parameters
                   SET status='complete', completed_utc=?, error=NULL
                   WHERE run_id=?""",
                [(completed, int(run_id)) for run_id in run_ids],
            )
            connection.execute(
                """UPDATE execution_batches
                   SET status='complete', completed_utc=?, error=NULL
                   WHERE execution_id=?""",
                (completed, execution_id),
            )

    def fail_batch(self, execution_id: int, run_ids, error: str) -> None:
        with self.transaction(parameters=True) as connection:
            connection.executemany(
                """UPDATE run_parameters SET status='failed', error=?
                   WHERE run_id=?""",
                [(str(error), int(run_id)) for run_id in run_ids],
            )
            connection.execute(
                """UPDATE execution_batches SET status='failed', error=?
                   WHERE execution_id=?""",
                (str(error), execution_id),
            )
