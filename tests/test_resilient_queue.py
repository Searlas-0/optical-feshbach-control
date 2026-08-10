from pathlib import Path
from types import SimpleNamespace

import subprocess

from ofc.resilient_queue import (
    mark_interrupted_runs_failed,
    run_queue,
    run_timed_queue,
)
from ofc.results import Results
from ofc.storage import ResultStore


def test_interrupted_queue_rows_are_closed_as_failed(tmp_path):
    database = tmp_path / "results.sqlite3"
    store = ResultStore(database)
    config_document_id = store.register_config(
        {"config_id": 11, "config_name": "interrupted", "config": {}}
    )
    execution_id, run_ids = store.prepare_batch(
        {
            "batch_id": 22,
            "config_id": 11,
            "queue_id": 33,
            "batch_index": 0,
            "batch_key": "test",
            "seed": 44,
            "config_name": "interrupted",
            "config_file": "interrupted.yaml",
            "description": "",
            "created_utc": "",
        },
        [{"N": 4, "t_interval": 1.0, "r_bg": 1.0, "u_max": 2.0, "v_max": 3.0}],
        2,
        config_document_id=config_document_id,
    )
    assert execution_id > 0 and len(run_ids) == 2

    repaired = mark_interrupted_runs_failed(
        database, queue_id=33, message="test interruption"
    )

    assert repaired == 2
    assert {row["status"] for row in Results(database).search(queue_id=33)} == {
        "failed"
    }


def test_resilient_queue_launches_the_next_config_after_failure(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "run.py").touch()
    database = tmp_path / "results.sqlite3"
    ResultStore(database)
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.touch()
    second.touch()
    calls = []
    return_codes = iter((7, 0))

    def fake_run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=next(return_codes))

    monkeypatch.setattr("ofc.resilient_queue.subprocess.run", fake_run)
    monkeypatch.setattr(
        "ofc.resilient_queue.mark_interrupted_runs_failed", lambda *args, **kwargs: 0
    )

    status = run_queue(
        [(101, first), (102, second)],
        project_root=root,
        python_executable=Path("/usr/bin/python3"),
        database=database,
    )

    assert status == 1
    assert len(calls) == 2
    assert calls[0][0][-1] == str(first)
    assert calls[1][0][-1] == str(second)


def test_timed_queue_repeats_then_closes_the_deadline_child(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "run.py").touch()
    database = tmp_path / "results.sqlite3"
    ResultStore(database)
    config = tmp_path / "refine.yaml"
    config.touch()
    clock = [0.0]
    processes = []
    repaired = []

    class FakeProcess:
        def __init__(self, command, **options):
            self.command = command
            self.options = options
            self.terminated = False
            processes.append(self)

        def wait(self, timeout=None):
            if self.terminated:
                return -15
            if len(processes) == 1:
                clock[0] += 6.0
                return 0
            clock[0] += float(timeout)
            raise subprocess.TimeoutExpired(self.command, timeout)

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    monkeypatch.setattr("ofc.resilient_queue.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("ofc.resilient_queue.subprocess.Popen", FakeProcess)
    monkeypatch.setattr(
        "ofc.resilient_queue.mark_interrupted_runs_failed",
        lambda *args, **kwargs: repaired.append((args, kwargs)) or 3,
    )

    status = run_timed_queue(
        [(101, config)],
        project_root=root,
        python_executable=Path("/usr/bin/python3"),
        database=database,
        max_elapsed_seconds=10.0,
        repeat_until_deadline=True,
    )

    assert status == 0
    assert len(processes) == 2
    assert processes[1].terminated is True
    assert repaired[0][1]["queue_id"] == 101
    assert "deadline" in repaired[0][1]["message"]
