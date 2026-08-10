#!/usr/bin/env python3
"""Resume the dedicated RTX 4060 manifest and package its result databases."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ofc.config import load_config, random_id
from ofc.storage import parameter_database_path


MANIFEST = ROOT / "scripts" / "local" / "rtx4060.manifest"
DATABASE = ROOT / "results" / "local_rtx4060_underexplored_v1.sqlite3"
STATE = ROOT / "logs" / "local_rtx4060_state.json"
TRANSFER = ROOT / "results" / "local_rtx4060_underexplored_v1-transfer.zip"
AUTO_PRIOR_DATABASE = ROOT / "results" / "auto_fourier_intensity_priors.sqlite3"
AUTO_PRIOR_TRANSFER = (
    ROOT / "results" / "auto_fourier_intensity_priors-transfer.zip"
)


def _ensure_auto_prior_database() -> None:
    if AUTO_PRIOR_DATABASE.is_file():
        return
    if not AUTO_PRIOR_TRANSFER.is_file():
        raise FileNotFoundError(
            "The auto-center prior bundle is required. Download "
            f"{AUTO_PRIOR_TRANSFER.name} from the server into results/, then run "
            "this same script again."
        )
    with zipfile.ZipFile(AUTO_PRIOR_TRANSFER) as archive:
        matching = [
            info
            for info in archive.infolist()
            if Path(info.filename).name == AUTO_PRIOR_DATABASE.name
            and not info.is_dir()
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"{AUTO_PRIOR_TRANSFER} does not contain exactly one "
                f"{AUTO_PRIOR_DATABASE.name}."
            )
        AUTO_PRIOR_DATABASE.parent.mkdir(parents=True, exist_ok=True)
        temporary = AUTO_PRIOR_DATABASE.with_suffix(".sqlite3.tmp")
        with archive.open(matching[0]) as source, temporary.open("wb") as destination:
            while block := source.read(1024 * 1024):
                destination.write(block)
        temporary.replace(AUTO_PRIOR_DATABASE)
    print(f"EXTRACTED AUTO-CENTER PRIORS | {AUTO_PRIOR_DATABASE}", flush=True)


def _manifest_paths() -> tuple[Path, ...]:
    paths = tuple(
        (ROOT / line.strip()).resolve()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not paths:
        raise RuntimeError(f"The local manifest is empty: {MANIFEST}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local configs: " + ", ".join(missing))
    return paths


def _fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_state(fingerprint: str) -> dict:
    if not STATE.is_file():
        return {
            "schema_version": 1,
            "queue_id": random_id(),
            "manifest_fingerprint": fingerprint,
            "completed_config_ids": [],
            "recovered_config_ids": [],
        }
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("manifest_fingerprint") != fingerprint:
        raise RuntimeError(
            "The committed RTX 4060 manifest changed after this local queue "
            f"started. Preserve {DATABASE.name} and {STATE.name}, then ask an "
            "agent to prepare a compatible continuation."
        )
    return state


def _write_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def _attempt_status(config_id: int) -> str:
    if not DATABASE.is_file():
        return "absent"
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        total, complete = connection.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN status='complete' THEN 1 ELSE 0 END)
                 FROM runs WHERE config_id=?""",
            (config_id,),
        ).fetchone()
    if not total:
        return "absent"
    return "complete" if int(complete or 0) == int(total) else "incomplete"


def _purge_incomplete_config(config_id: int) -> int:
    """Remove only an incomplete local config attempt before retrying it.

    This database is dedicated to the laptop manifest. No downstream config is
    launched until its predecessor returns successfully, so removing a partial
    current attempt cannot invalidate completed descendants.
    """

    removed = 0
    if DATABASE.is_file():
        with sqlite3.connect(DATABASE) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            removed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE config_id=?", (config_id,)
                ).fetchone()[0]
            )
            connection.execute("DELETE FROM runs WHERE config_id=?", (config_id,))

    parameter_database = parameter_database_path(DATABASE)
    if parameter_database.is_file():
        with sqlite3.connect(parameter_database) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            document_ids = [
                int(row[0])
                for row in connection.execute(
                    "SELECT config_document_id FROM config_documents WHERE config_id=?",
                    (config_id,),
                )
            ]
            for document_id in document_ids:
                connection.execute(
                    "DELETE FROM run_parameters WHERE config_document_id=?",
                    (document_id,),
                )
                connection.execute(
                    "DELETE FROM execution_batches WHERE config_document_id=?",
                    (document_id,),
                )
                connection.execute(
                    "DELETE FROM config_documents WHERE config_document_id=?",
                    (document_id,),
                )
    return removed


def _checkpoint(path: Path) -> None:
    if path.is_file():
        with sqlite3.connect(path) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        check=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _package(state: dict, configs: tuple[Path, ...]) -> None:
    parameter_database = parameter_database_path(DATABASE)
    _checkpoint(DATABASE)
    _checkpoint(parameter_database)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_fingerprint": state["manifest_fingerprint"],
        "queue_id": state["queue_id"],
        "completed_configs": [str(path.relative_to(ROOT)) for path in configs],
        "database_files": [DATABASE.name, parameter_database.name],
    }
    TRANSFER.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRANSFER.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(DATABASE, DATABASE.name)
        archive.write(parameter_database, parameter_database.name)
        archive.writestr("local_run_metadata.json", json.dumps(metadata, indent=2) + "\n")
    temporary.replace(TRANSFER)


def main() -> int:
    _ensure_auto_prior_database()
    configs = _manifest_paths()
    documents = tuple(load_config(path) for path in configs)
    expected_database = str(DATABASE.relative_to(ROOT))
    wrong_database = [
        document.name
        for document in documents
        if document.runtime.database != expected_database
    ]
    if wrong_database:
        raise RuntimeError(
            "Local configs do not use the dedicated transfer database: "
            + ", ".join(wrong_database)
        )
    wrong_prior_database = [
        document.name
        for document in documents
        if document.runtime.fourier_intensity_fraction == "auto"
        and document.runtime.fourier_intensity_auto_database
        != str(AUTO_PRIOR_DATABASE.relative_to(ROOT))
    ]
    if wrong_prior_database:
        raise RuntimeError(
            "Local auto-center configs do not use the portable prior database: "
            + ", ".join(wrong_prior_database)
        )

    fingerprint = _fingerprint(configs)
    state = _read_state(fingerprint)
    completed = {int(value) for value in state["completed_config_ids"]}
    _write_state(state)

    for index, (path, document) in enumerate(zip(configs, documents), start=1):
        if document.config_id in completed:
            print(f"LOCAL {index}/{len(configs)} SKIP | {document.name}", flush=True)
            continue

        attempt_status = _attempt_status(document.config_id)
        if attempt_status == "complete":
            print(
                f"LOCAL {index}/{len(configs)} RECOVERED COMPLETE | {document.name}",
                flush=True,
            )
        else:
            if attempt_status == "incomplete":
                removed = _purge_incomplete_config(document.config_id)
                state["recovered_config_ids"].append(document.config_id)
                _write_state(state)
                print(
                    f"LOCAL {index}/{len(configs)} RETRY | {document.name} | "
                    f"removed_partial_runs={removed}",
                    flush=True,
                )
            print(
                f"LOCAL {index}/{len(configs)} START | {document.name}", flush=True
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "--queue-id",
                    str(state["queue_id"]),
                    str(path),
                ],
                cwd=ROOT,
                check=False,
            )
            if result.returncode:
                print(
                    f"LOCAL {index}/{len(configs)} FAILED | {document.name} | "
                    "run the same shell script again to retry this config safely",
                    flush=True,
                )
                return result.returncode

        completed.add(document.config_id)
        state["completed_config_ids"] = sorted(completed)
        _write_state(state)
        print(f"LOCAL {index}/{len(configs)} COMPLETE | {document.name}", flush=True)

    _package(state, configs)
    print(f"LOCAL QUEUE COMPLETE | transfer bundle: {TRANSFER}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
