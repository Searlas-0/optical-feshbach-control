from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ofc.notebook_workflow import RunNotebook, find_project_root


def _project(tmp_path):
    (tmp_path / "src" / "ofc").mkdir(parents=True)
    (tmp_path / "run_config").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "run.py").touch()
    return tmp_path


def test_notebook_workflow_discovers_root_and_guards_every_mutating_action(tmp_path):
    root = _project(tmp_path)
    workflow = RunNotebook("guarded_run", project_root=root / "src")

    assert find_project_root(root / "src" / "ofc") == root
    assert workflow.create_config(
        activated=False,
        description="guard test",
        parameters={"N": 4},
        runtime={},
    ) is None
    assert not workflow.config_path.exists()
    assert workflow.run_local(activated=False) is None
    assert workflow.submit_slurm(activated=False) is None

    document = workflow.create_config(
        activated=True,
        description="guard test",
        parameters={"N": 4},
        runtime={},
    )
    assert document.name == "guarded_run"
    assert workflow.config_path.is_file()


def test_grouped_local_execution_can_detach_with_one_queue_and_log(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    first = RunNotebook("first_gpu", project_root=root)
    second = RunNotebook("second_gpu", project_root=root)
    for workflow in (first, second):
        workflow.create_config(
            activated=True,
            description="grouped GPU test",
            parameters={"N": 4},
            runtime={"device": "gpu"},
        )

    calls = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command, **options):
        calls["command"] = command
        calls["options"] = options
        return FakeProcess()

    monkeypatch.setattr("ofc.notebook_workflow.subprocess.Popen", fake_popen)
    log_path = root / "logs" / "gpu-group.log"
    queue_id = first.run_local_group(
        activated=True,
        additional_workflows=[second],
        queue_id=1234,
        detached=True,
        log_path=log_path,
    )

    assert queue_id == 1234
    assert calls["command"][-2:] == [str(first.config_path), str(second.config_path)]
    assert calls["options"]["start_new_session"] is True
    assert calls["options"]["stderr"] == subprocess.STDOUT
    assert first.active_queue_id == second.active_queue_id == 1234
    assert first.local_process_id == second.local_process_id == 4321
    assert first.local_log_path == second.local_log_path == log_path
    assert log_path.is_file()


def test_grouped_local_execution_can_wait_for_an_existing_process(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    first = RunNotebook("queued_first", project_root=root)
    second = RunNotebook("queued_second", project_root=root)
    for workflow in (first, second):
        workflow.create_config(
            activated=True,
            description="queued GPU test",
            parameters={"N": 4},
            runtime={"device": "gpu"},
        )

    calls = {}

    class FakeProcess:
        pid = 4322

    def fake_popen(command, **options):
        calls["command"] = command
        calls["options"] = options
        return FakeProcess()

    monkeypatch.setattr("ofc.notebook_workflow.subprocess.Popen", fake_popen)
    first.run_local_group(
        activated=True,
        additional_workflows=[second],
        queue_id=1235,
        detached=True,
        wait_for_process_id=999,
    )

    assert calls["command"][1:6] == [
        "-m",
        "ofc.process_queue",
        "--wait-pid",
        "999",
        "--",
    ]
    assert calls["command"][-2:] == [str(first.config_path), str(second.config_path)]
    assert calls["options"]["start_new_session"] is True


def test_bar_gpu_execution_verifies_host_and_detaches_single_config(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    workflow = RunNotebook("bar_gpu", project_root=root)
    workflow.create_config(
        activated=True,
        description="bar GPU test",
        parameters={"N": 4},
        runtime={"device": "gpu"},
    )
    calls = {"run": [], "popen": []}

    class Probe:
        stdout = "CudaDevice(id=0)\n"

    class FakeProcess:
        pid = 9876

    def fake_run(command, **options):
        calls["run"].append((command, options))
        return Probe()

    def fake_popen(command, **options):
        calls["popen"].append((command, options))
        return FakeProcess()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("ofc.notebook_workflow.socket.gethostname", lambda: "bar")
    monkeypatch.setattr("ofc.notebook_workflow.subprocess.run", fake_run)
    monkeypatch.setattr("ofc.notebook_workflow.subprocess.Popen", fake_popen)
    queue_id = workflow.run_on_bar_gpu(
        activated=True,
        queue_id=5678,
        detached=True,
    )

    assert queue_id == 5678
    assert calls["run"][0][0][-1].endswith("print(devices[0])")
    assert calls["popen"][0][0][-1] == str(workflow.config_path)
    assert calls["popen"][0][1]["start_new_session"] is True
    assert calls["run"][0][1]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert calls["popen"][0][1]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert workflow.local_process_id == 9876
    assert workflow.local_log_path.is_file()


def test_grouped_bar_gpu_execution_requires_explicit_gpu_and_one_process(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    first = RunNotebook("first_explicit_gpu", project_root=root)
    second = RunNotebook("second_explicit_gpu", project_root=root)
    for workflow in (first, second):
        workflow.create_config(
            activated=True,
            description="grouped explicit GPU test",
            parameters={"N": 4},
            runtime={"device": "gpu"},
        )
    calls = {"run": [], "popen": []}

    class Probe:
        stdout = "CudaDevice(id=0)\n"

    class FakeProcess:
        pid = 2468

    def fake_run(command, **options):
        calls["run"].append((command, options))
        return Probe()

    def fake_popen(command, **options):
        calls["popen"].append((command, options))
        return FakeProcess()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("ofc.notebook_workflow.socket.gethostname", lambda: "bar")
    monkeypatch.setattr("ofc.notebook_workflow.subprocess.run", fake_run)
    monkeypatch.setattr("ofc.notebook_workflow.subprocess.Popen", fake_popen)

    queue_id = first.run_on_bar_gpu_group(
        activated=True,
        additional_workflows=[second],
        queue_id=9876,
    )

    assert queue_id == 9876
    assert calls["run"][0][0][-1].endswith("print(devices[0])")
    assert calls["popen"][0][0][-2:] == [
        str(first.config_path),
        str(second.config_path),
    ]
    assert calls["run"][0][1]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert calls["popen"][0][1]["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert first.local_process_id == second.local_process_id == 2468


def test_grouped_query_combines_same_queue_configs_for_one_cap_plot(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    first = RunNotebook("query_u40", project_root=root)
    second = RunNotebook("query_u160", project_root=root)
    database = root / "results" / "results.sqlite3"
    database.touch()
    fake_results = SimpleNamespace(database=database)

    def fake_query(workflow, **options):
        cap = 40.0 if workflow is first else 160.0
        return SimpleNamespace(
            results=fake_results,
            execution={"config_name": workflow.run_name, "queue_id": 123},
            rows=[
                {
                    "run_id": int(cap),
                    "config_name": workflow.run_name,
                    "batch_index": 0,
                    "case_index": 0,
                    "initialization_index": 0,
                    "u_max": cap,
                    "best_score": cap,
                }
            ],
            registered_rows=[],
        )

    monkeypatch.setattr(RunNotebook, "query", fake_query)

    combined = first.query_group(
        additional_workflows=[second],
        queue_id=123,
        sweep_parameters=["u_max"],
    )

    assert len(combined.rows) == 2
    assert combined.sweep_parameters == ("u_max",)
    assert combined.sweep("u_max").display_values == (40.0, 160.0)


def test_grouped_query_skips_later_configs_that_have_not_started(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    first = RunNotebook("query_u40", project_root=root)
    second = RunNotebook("query_u160", project_root=root)
    database = root / "results" / "results.sqlite3"
    database.touch()
    fake_results = SimpleNamespace(database=database)

    def fake_query(workflow, **options):
        if workflow is second:
            raise RuntimeError(
                "Only 0 execution(s) match the inherited config 'query_u160'."
            )
        return SimpleNamespace(
            results=fake_results,
            execution={"config_name": workflow.run_name, "queue_id": 123},
            rows=[
                {
                    "run_id": 40,
                    "config_name": workflow.run_name,
                    "batch_index": 0,
                    "case_index": 0,
                    "initialization_index": 0,
                    "u_max": 40.0,
                    "best_score": 1.0,
                }
            ],
            registered_rows=[],
        )

    monkeypatch.setattr(RunNotebook, "query", fake_query)

    combined = first.query_group(
        additional_workflows=[second],
        sweep_parameters=["u_max"],
    )

    assert [row["u_max"] for row in combined.rows] == [40.0]
    assert combined.execution["queue_id"] == 123
    assert combined.sweep("u_max").display_values == (40.0,)


def test_figure_preview_and_saved_png_dpi_are_independently_adjustable(
    tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    root = _project(tmp_path)
    workflow = RunNotebook("figure_dpi", project_root=root)
    figure, _ = plt.subplots()
    calls = []
    original_savefig = figure.savefig

    def record_savefig(*arguments, **options):
        calls.append((arguments, options.copy()))
        return original_savefig(*arguments, **options)

    monkeypatch.setattr(figure, "savefig", record_savefig)
    workflow.present_figure(
        figure,
        "summary",
        save_figure="dpi-test",
        preview_dpi=225,
        save_dpi=725,
    )

    assert calls[0][1]["dpi"] == 725.0
    assert calls[1][1]["dpi"] == 225.0
    assert (root / "figures" / "dpi-test" / "summary.png").is_file()

    with pytest.raises(ValueError, match="preview_dpi"):
        workflow.present_figure(figure, "invalid", preview_dpi=0)


def test_notebook_query_accepts_any_number_of_sweep_dimensions(tmp_path, monkeypatch):
    root = _project(tmp_path)
    workflow = RunNotebook("multi_sweep", project_root=root)
    document = workflow.create_config(
        activated=True,
        description="multi-dimensional query",
        parameters={"N": [4, 8], "u_max": [10.0, 100.0]},
        runtime={"database": "results/results.sqlite3"},
    )
    (root / "results" / "results.sqlite3").touch()

    rows = [
        {
            "run_id": index,
            "config_document_id": 1,
            "config_id": document.config_id,
            "config_name": document.name,
            "queue_id": 42,
            "batch_index": 0,
            "case_index": index,
            "status": "running",
            "N": N,
            "u_max": u_max,
            "best_score": float(index),
        }
        for index, (N, u_max) in enumerate(
            ((4, 10.0), (4, 100.0), (8, 10.0), (8, 100.0)), start=1
        )
    ]

    class FakeResults:
        def __init__(self, database):
            self.database = Path(database)

        def config_runs(self, **filters):
            return [
                {
                    "config_document_id": 1,
                    "config_id": document.config_id,
                    "config_name": document.name,
                    "queue_id": 42,
                    "status": "running",
                }
            ]

        def search(self, **filters):
            return list(rows)

    monkeypatch.setattr("ofc.notebook_workflow.Results", FakeResults)
    result = workflow.query(sweep_parameters=["N", "u_max"])

    assert result.sweep_parameters == ("N", "u_max")
    assert result.sweep("N").display_values == (4, 8)
    assert result.sweep("u_max").display_values == (10.0, 100.0)
    assert len(result.rows) == 4


def test_notebook_query_can_ignore_local_config_and_select_historical_data(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    database = root / "results" / "results.sqlite3"
    database.touch()
    workflow = RunNotebook("plot_only", project_root=root)
    calls = {}
    rows = [
        {
            "run_id": 11,
            "config_document_id": 7,
            "config_id": 8,
            "config_name": "archived",
            "queue_id": 9,
            "batch_index": 0,
            "case_index": 0,
            "status": "complete",
            "u_max": 40.0,
            "best_score": 3.0,
        },
        {
            "run_id": 12,
            "config_document_id": 7,
            "config_id": 8,
            "config_name": "archived",
            "queue_id": 9,
            "batch_index": 0,
            "case_index": 1,
            "status": "running",
            "u_max": 160.0,
            "best_score": 4.0,
        },
    ]

    class FakeResults:
        def __init__(self, selected_database):
            calls["database"] = Path(selected_database)

        def config_runs(self, **filters):
            calls["config_runs"] = filters
            return [
                {
                    "config_document_id": 7,
                    "config_id": 8,
                    "config_name": "archived",
                    "queue_id": 9,
                    "status": "mixed",
                }
            ]

        def search(self, **filters):
            calls["search"] = filters
            return list(rows)

        def config_document(self, run_id):
            assert run_id == 11
            return {"parameters": {"u_max": [40.0, 160.0]}}

    monkeypatch.setattr("ofc.notebook_workflow.Results", FakeResults)
    result = workflow.query(
        inherit_config=False,
        filters={"config_name": "archived"},
        sweep_parameters=["u_max"],
    )

    assert not workflow.config_path.exists()
    assert calls["database"] == database
    assert calls["config_runs"] == {"config_name": "archived"}
    assert calls["search"]["config_document_id"] == 7
    assert calls["search"]["queue_id"] == 9
    assert calls["search"]["config_name"] == "archived"
    assert result.config_parameters == {"u_max": [40.0, 160.0]}
    assert [row["status"] for row in result.rows] == ["complete", "running"]
