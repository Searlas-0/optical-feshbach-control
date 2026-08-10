from pathlib import Path
import sqlite3
import zipfile

from ofc.config import load_config
from ofc.storage import ResultStore, parameter_database_path
from scripts.local import run_local_queue


def test_prior_bundle_is_extracted_by_the_one_command_worker(tmp_path, monkeypatch):
    database = tmp_path / "auto_fourier_intensity_priors.sqlite3"
    transfer = tmp_path / "auto_fourier_intensity_priors-transfer.zip"
    with zipfile.ZipFile(transfer, "w") as archive:
        archive.writestr(database.name, b"portable-priors")

    monkeypatch.setattr(run_local_queue, "AUTO_PRIOR_DATABASE", database)
    monkeypatch.setattr(run_local_queue, "AUTO_PRIOR_TRANSFER", transfer)
    run_local_queue._ensure_auto_prior_database()

    assert database.read_bytes() == b"portable-priors"


def test_committed_local_manifest_is_self_contained_and_uses_transfer_database():
    paths = run_local_queue._manifest_paths()
    documents = tuple(load_config(path) for path in paths)
    config_ids = {document.config_id for document in documents}

    assert len(paths) == 42
    assert all(
        document.runtime.database
        == "results/local_rtx4060_underexplored_v1.sqlite3"
        for document in documents
    )
    assert all(document.scalar_cases()[0].u_max in {10.0, 20.0, 80.0} for document in documents)
    assert all(
        document.query is None
        or document.query.where["config_id"] in config_ids
        for document in documents
    )
    assert all(
        document.query is None or document.query.database is None
        for document in documents
    )
    scouts = [document for document in documents if "scout500" in document.name]
    assert all(document.runtime.fourier_intensity_fraction == "auto" for document in scouts)
    assert all(
        document.runtime.fourier_intensity_auto_database
        == "results/auto_fourier_intensity_priors.sqlite3"
        for document in scouts
    )


def test_incomplete_retry_purge_removes_both_database_halves(tmp_path, monkeypatch):
    database = tmp_path / "local.sqlite3"
    store = ResultStore(database)
    config_id = 123
    with store.connect_parameters() as connection:
        document_id = connection.execute(
            """INSERT INTO config_documents(config_id, config_json, registered_utc)
               VALUES(?, '{}', 'now')""",
            (config_id,),
        ).lastrowid
        execution_id = connection.execute(
            """INSERT INTO execution_batches(
                   config_document_id, batch_id, queue_id, batch_index,
                   execution_json, status, started_utc
               ) VALUES(?, 1, 2, 0, '{}', 'running', 'now')""",
            (document_id,),
        ).lastrowid
        run_id = connection.execute(
            """INSERT INTO run_parameters(
                   execution_id, config_document_id, status, started_utc,
                   parameters_json
               ) VALUES(?, ?, 'running', 'now', '{}')""",
            (execution_id, document_id),
        ).lastrowid
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO runs(
                   run_id, config_id, config_document_id, batch_id, execution_id,
                   queue_id, batch_index, status, started_utc
               ) VALUES(?, ?, ?, 1, ?, 2, 0, 'running', 'now')""",
            (run_id, config_id, document_id, execution_id),
        )

    monkeypatch.setattr(run_local_queue, "DATABASE", database)
    assert run_local_queue._purge_incomplete_config(config_id) == 1

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    with sqlite3.connect(parameter_database_path(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM run_parameters").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM execution_batches").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM config_documents").fetchone()[0] == 0
