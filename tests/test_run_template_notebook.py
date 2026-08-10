import ast
import importlib.util
import json
from pathlib import Path

import pytest

from ofc.config import make_document


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "scripts" / "templates" / "boilerplate_run.ipynb"
THREE_CAP_LONG_NOTEBOOK = (
    ROOT / "scripts" / "n100" / "multi_cap" / "three_cap_adam_long_gpu.ipynb"
)
FINAL_LBFGS_NOTEBOOK = (
    ROOT / "scripts" / "n100" / "multi_cap" / "lbfgs_saved_parameter_refinement.ipynb"
)
SCRIPTS_AGENT_RULES = ROOT / "scripts" / "AGENTS.md"
GPU_CONTINUATION_NOTEBOOKS = {
    ROOT / "scripts" / "n100" / f"u{cap}" / "adam_gpu_continuation_41k.ipynb": {
        "cap": cap,
        "source_queue_id": source_queue_id,
        "source_run_id": source_run_id,
        "smoothness": smoothness,
        "sharpness": sharpness,
    }
    for cap, source_queue_id, source_run_id, smoothness, sharpness in (
        (40, 702007, 80651, 1.25e-7, 5e-8),
        (160, 702013, 80906, 1.25e-7, 1.25e-8),
        (1280, 702019, 81268, 2.5e-7, 2.5e-8),
    )
}
U_MAX_SMOOTH_SHARP_NOTEBOOKS = {
    ROOT / "scripts" / "n100" / f"u{cap}" / "smoothness_sharpness_sweep_10k.ipynb": cap
    for cap in (40, 160, 1280)
}
LONG_SMOOTH_SHARP_NOTEBOOKS = {
    ROOT / "scripts" / "n100" / f"u{cap}" / "smoothness_sharpness_long_adam_gpu.ipynb": {
        "cap": cap,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "smoothness": smoothness,
        "sharpness": sharpness,
    }
    for cap, learning_rate, beta1, beta2, smoothness, sharpness in (
        (40, 0.1, 0.9, 0.99, 1.25e-7, 5e-8),
        (160, 0.02, 0.95, 0.99, 1.25e-7, 1.25e-8),
        (1280, 0.1, 0.95, 0.999, 2.5e-7, 2.5e-8),
    )
}
MAKER_PATH = ROOT / "run_config" / "make_config.py"
REQUIRED_WORKFLOW_CELLS = [
    "imports-and-project-root",
    "create-config",
    "run-locally",
    "submit-slurm",
    "query-config-run",
    "figure-output-controls",
    "plot-sweep-summary",
    "plot-double-sweep-summary",
    "plot-triple-sweep-summary",
]


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells_by_id():
    return {cell["id"]: cell for cell in _notebook()["cells"]}


def _source(cell):
    return "".join(cell["source"])


def _literal_assignment(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"No literal assignment for {name!r}")


def _maker_module():
    spec = importlib.util.spec_from_file_location("template_make_config", MAKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_notebooks_select_the_project_kernel():
    for notebook_path in (ROOT / "scripts").rglob("*.ipynb"):
        if ".ipynb_checkpoints" in notebook_path.parts:
            continue
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
        assert kernelspec.get("name") == "optical-feshbach-control"
        assert kernelspec.get("display_name") == "Python (optical-feshbach-control)"


def test_template_has_ordered_end_to_end_workflow_cells():
    cells = _notebook()["cells"]
    ids = [cell["id"] for cell in cells]
    assert [ids.index(cell_id) for cell_id in REQUIRED_WORKFLOW_CELLS] == sorted(
        ids.index(cell_id) for cell_id in REQUIRED_WORKFLOW_CELLS
    )
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            source = _source(cell)
            compile(source, f"{NOTEBOOK.name}:{cell['id']}", "exec")
            assert "def " not in source


def test_template_declares_the_canonical_argument_only_contract():
    overview = _source(_cells_by_id()["title-and-workflow"])
    setup = _source(_cells_by_id()["imports-and-project-root"])

    assert "Every run notebook must use this boilerplate design wherever possible" in overview
    assert "source of truth" in overview
    assert "explicit permission" in overview
    assert "A request to create a run is not permission to change this style" in overview
    assert "Do not add notebook-local functions or classes" in overview
    assert "Reusable behavior belongs in `ofc.notebook_workflow`" in overview
    assert setup.startswith("from ofc.notebook_workflow import RunNotebook")
    assert "RunNotebook(run_name)" in setup
    assert "show_context" not in setup


def test_scripts_agent_rules_require_permission_before_notebook_format_changes():
    rules = SCRIPTS_AGENT_RULES.read_text(encoding="utf-8")

    assert "Use `scripts/templates/boilerplate_run.ipynb` as the source of truth" in rules
    assert "user's explicit permission in the current conversation" in rules
    assert "request for a new run or analysis is not permission" in rules
    assert "Never define functions or classes in a run notebook" in rules
    assert "`resume_optimizer`" in rules
    assert "false" in rules and "true" in rules
    assert "stop and ask" in rules


def test_template_exposes_exact_current_make_config_defaults():
    setup_source = _source(_cells_by_id()["imports-and-project-root"])
    source = _source(_cells_by_id()["create-config"])
    maker = _maker_module()

    assert _literal_assignment(setup_source, "run_name") == "my_optimization_run"
    assert "run_name =" not in source
    assert _literal_assignment(source, "parameters") == maker.default_parameters()
    assert _literal_assignment(source, "runtime") == maker.default_runtime()
    assert source.lstrip().startswith("Activated = False")
    assert "workflow.create_config(" in source
    assert "reuse_existing" in source
    assert '"resume_optimizer": False (reset) or True (restore)' in source


def test_template_guards_mutating_cells_and_delegates_all_mechanics():
    cells = _cells_by_id()
    setup = _source(cells["imports-and-project-root"])
    local = _source(cells["run-locally"])
    slurm = _source(cells["submit-slurm"])
    query = _source(cells["query-config-run"])

    config = _source(cells["create-config"])
    for source in (config, local, slurm):
        assert source.lstrip().startswith("Activated = False")
    assert "workflow.run_on_bar_gpu(" in local
    assert "detached = True" in local
    assert "log_path = None" in local
    assert "workflow.submit_slurm(" in slurm
    assert "workflow.query(" in query
    assert "inherit_config = True" in query
    assert "inherit_config=inherit_config" in query
    assert "database=database" in query
    assert "historical database only" in query
    assert "sweep_parameters = None" in query
    assert "running" in query and "complete" in query and "failed" in query
    assert "list:" in query and "range:" in query
    combined = setup + config + local + slurm + query
    for forbidden in ("subprocess", "Results(", "load_config(", "_detect_sweep"):
        assert forbidden not in combined


def test_template_exposes_standard_and_multi_sweep_plot_arguments():
    cells = _cells_by_id()
    summary = _source(cells["plot-sweep-summary"])

    assert "_plot_sweep_" not in summary
    assert "query_result.plot_summary(" in summary
    assert "sweep_parameter = None" in summary
    assert "sweep_parameter=sweep_parameter" in summary
    assert "history_points = 1200" in summary
    double = _source(cells["plot-double-sweep-summary"])
    triple = _source(cells["plot-triple-sweep-summary"])
    output_controls = _source(cells["figure-output-controls"])
    assert "separate_sweep_parameter" in double
    assert "colour_sweep_parameter" in double
    assert "row_sweep_parameter" in triple
    assert "column_sweep_parameter" in triple
    assert "colour_sweep_parameter" in triple
    assert "preview_dpi =" in output_controls
    assert "save_dpi =" in output_controls
    for plot_source in (summary, double, triple):
        assert "preview_dpi=preview_dpi" in plot_source
        assert "save_dpi=save_dpi" in plot_source


def test_u_max_smoothness_sharpness_notebook_is_balanced_and_canonical():
    for notebook_path, cap in U_MAX_SMOOTH_SHARP_NOTEBOOKS.items():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = _source(cell)
            compile(source, f"{notebook_path.name}:{cell['id']}", "exec")
            assert "def " not in source

        setup = _source(cells["imports-and-project-root"])
        config = _source(cells["create-config"])
        slurm = _source(cells["submit-slurm"])
        query_cell = _source(cells["query-config-run"])
        parameters = _literal_assignment(config, "parameters")
        runtime = _literal_assignment(config, "runtime")
        initialization_query = _literal_assignment(config, "initialization_query")
        document = make_document(
            name=_literal_assignment(setup, "run_name"),
            parameters=parameters,
            runtime=runtime,
            query=initialization_query,
        )

        assert config.startswith("Activated = False")
        assert slurm.startswith("Activated = False")
        assert parameters["u_max"] == float(cap)
        assert parameters["adam_learning_rate"] == 0.05
        assert parameters["adam_beta1"] == 0.9
        assert parameters["adam_beta2"] == 0.999
        assert len(parameters["smoothness"]) == 6
        assert len(parameters["sharpness"]) == 5
        assert len(document.scalar_cases()) == 30
        assert [len(batch.cases) for batch in document.batches()] == [5] * 6
        assert all(
            {case.u_max for case in batch.cases} == {float(cap)}
            for batch in document.batches()
        )
        assert all(
            len({case.smoothness for case in batch.cases}) == 1
            for batch in document.batches()
        )
        assert runtime["initialisations"] == 10
        assert runtime["max_cases_per_batch"] == 5
        assert initialization_query["where"]["u_max"] == float(cap)
        assert initialization_query["limit"] == 1
        assert initialization_query["match_parameters"] == ["u_max"]
        assert len(initialization_query["perturbation_levels"]) == 5
        assert "cpus = 2" in slurm
        assert 'memory = "1G"' in slurm
        assert 'time = "00:10:00"' in slurm
        assert "array = True" in slurm
        assert "array_max_concurrent = None" in slurm
        assert 'sweep_parameters = ["smoothness", "sharpness"]' in query_cell


def test_generic_slurm_launcher_maps_array_tasks_to_config_batches():
    launcher = (ROOT / "slurm" / "run_config.slurm").read_text(encoding="utf-8")

    assert 'QUEUE_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"' in launcher
    assert '--batch-index "${SLURM_ARRAY_TASK_ID}"' in launcher
    assert '"${RUN_ARGUMENTS[@]}"' in launcher


def test_cap_adam_continuations_are_split_into_canonical_declarative_notebooks():
    assert not (ROOT / "scripts" / "n100" / "multi_cap" / "cap_adam_gpu_continuations.ipynb").exists()

    forbidden_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Lambda,
    )
    for notebook_path, expected in GPU_CONTINUATION_NOTEBOOKS.items():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        ids = [cell["id"] for cell in notebook["cells"]]
        assert [ids.index(cell_id) for cell_id in REQUIRED_WORKFLOW_CELLS] == sorted(
            ids.index(cell_id) for cell_id in REQUIRED_WORKFLOW_CELLS
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = _source(cell)
            tree = ast.parse(source, filename=f"{notebook_path.name}:{cell['id']}")
            assert not any(isinstance(node, forbidden_nodes) for node in ast.walk(tree))

        setup = _source(cells["imports-and-project-root"])
        config = _source(cells["create-config"])
        run = _source(cells["run-locally"])
        query = _source(cells["query-config-run"])
        summary = _source(cells["plot-sweep-summary"])
        parameters = _literal_assignment(config, "parameters")
        runtime = _literal_assignment(config, "runtime")
        initialization_query = _literal_assignment(config, "initialization_query")
        document = make_document(
            name=_literal_assignment(setup, "run_name"),
            parameters=parameters,
            runtime=runtime,
            query=initialization_query,
        )

        assert _literal_assignment(setup, "run_name") == (
            f"N100_u{expected['cap']}_adam_gpu_continuation_41k"
        )
        assert parameters["u_max"] == float(expected["cap"])
        assert parameters["smoothness"] == expected["smoothness"]
        assert parameters["sharpness"] == expected["sharpness"]
        assert parameters["schedule"] == [
            (1000, 1.0),
            (5000, 0.1),
            (5000, 0.1),
            (30000, 0.1),
        ]
        assert parameters["adam_learning_rate"] == [0.02, 0.05, 0.1]
        assert parameters["adam_beta1"] == [0.8, 0.9, 0.95]
        assert parameters["adam_beta2"] == [0.99, 0.999]
        assert runtime["device"] == "gpu"
        assert runtime["initialisations"] == 10
        assert runtime["max_cases_per_batch"] == 18
        assert len(document.scalar_cases()) == 18
        assert len(document.batches()) == 1
        assert initialization_query["where"]["run_id"] == expected["source_run_id"]
        assert initialization_query["where"]["queue_id"] == expected["source_queue_id"]
        assert len(initialization_query["perturbation_levels"]) == 5
        assert run.startswith("Activated = False")
        assert "workflow.run_on_bar_gpu(" in run
        assert "detached = True" in run
        assert 'sweep_parameters = ["adam_learning_rate", "adam_beta1", "adam_beta2"]' in query
        assert "query_result.plot_summary(" in summary


def test_three_cap_long_gpu_notebook_is_grouped_declarative_and_exact():
    notebook = json.loads(THREE_CAP_LONG_NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    ids = [cell["id"] for cell in notebook["cells"]]
    assert [ids.index(cell_id) for cell_id in REQUIRED_WORKFLOW_CELLS] == sorted(
        ids.index(cell_id) for cell_id in REQUIRED_WORKFLOW_CELLS
    )
    forbidden = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Lambda,
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        tree = ast.parse(
            _source(cell), filename=f"{THREE_CAP_LONG_NOTEBOOK.name}:{cell['id']}"
        )
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree))

    namespace = {}
    exec(_source(cells["imports-and-project-root"]), namespace)
    exec(_source(cells["create-config"]), namespace)
    expected = {
        "u40": (40.0, 0.1, 0.9, 0.99, 1.25e-7, 5e-8),
        "u160": (160.0, 0.02, 0.95, 0.99, 1.25e-7, 1.25e-8),
        "u1280": (1280.0, 0.1, 0.95, 0.999, 2.5e-7, 2.5e-8),
    }
    for suffix, values in expected.items():
        parameters = namespace[f"parameters_{suffix}"]
        query = namespace[f"query_{suffix}"]
        assert (
            parameters["u_max"],
            parameters["adam_learning_rate"],
            parameters["adam_beta1"],
            parameters["adam_beta2"],
            parameters["smoothness"],
            parameters["sharpness"],
        ) == values
        expected_schedule = (
            [(1_000, 1.0), (50_000, 0.1), (100_000, 0.1), (1_000_000, 0.1)]
            if suffix == "u40"
            else [(1_000, 1.0), (30_000, 0.1), (60_000, 0.5)]
        )
        assert parameters["schedule"] == expected_schedule
        assert parameters["block_size"] == 1_000
        assert query["limit"] == 5
        assert query["resume_optimizer"] is False
        assert query["perturbation_levels"] == [
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.01,
        ]
    assert namespace["runtime"]["initialisations"] == 50
    assert namespace["runtime"]["device"] == "gpu"
    assert namespace["runtime"]["auto_halt"] is True
    assert namespace["runtime"]["concurrent_workers"] == 1
    assert namespace["runtime_u160"]["max_batch_elapsed_seconds"] == 5 * 60 * 60
    assert namespace["runtime_u1280"]["max_batch_elapsed_seconds"] == 5 * 60 * 60
    assert 50 + 5 * 5 == 75
    assert "run_on_bar_gpu_group" in _source(cells["run-locally"])
    assert "query_group" in _source(cells["query-config-run"])


def test_long_smoothness_sharpness_gpu_notebooks_are_exact_and_sharded():
    forbidden = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Lambda,
    )
    for notebook_path, expected in LONG_SMOOTH_SHARP_NOTEBOOKS.items():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: cell for cell in notebook["cells"]}
        ids = [cell["id"] for cell in notebook["cells"]]
        assert [ids.index(cell_id) for cell_id in REQUIRED_WORKFLOW_CELLS] == sorted(
            ids.index(cell_id) for cell_id in REQUIRED_WORKFLOW_CELLS
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            tree = ast.parse(
                _source(cell), filename=f"{notebook_path.name}:{cell['id']}"
            )
            assert not any(isinstance(node, forbidden) for node in ast.walk(tree))

        config = _source(cells["create-config"])
        parameters = _literal_assignment(config, "parameters")
        runtime = _literal_assignment(config, "runtime")
        query = _literal_assignment(config, "initialization_query")
        document = make_document(
            name=_literal_assignment(_source(cells["imports-and-project-root"]), "run_name"),
            parameters=parameters,
            runtime=runtime,
            query=query,
        )

        assert parameters["u_max"] == float(expected["cap"])
        assert parameters["adam_learning_rate"] == expected["learning_rate"]
        assert parameters["adam_beta1"] == expected["beta1"]
        assert parameters["adam_beta2"] == expected["beta2"]
        assert parameters["schedule"] == [
            (1_000, 1.0),
            (30_000, 0.1),
            (60_000, 0.5),
        ]
        assert len(parameters["smoothness"]) == 15
        assert len(parameters["sharpness"]) == 15
        assert expected["smoothness"] == parameters["smoothness"][7]
        assert expected["sharpness"] == parameters["sharpness"][7]
        assert parameters["smoothness"][-1] / parameters["smoothness"][0] == pytest.approx(1_000)
        assert parameters["sharpness"][-1] / parameters["sharpness"][0] == pytest.approx(1_000)
        assert len(document.scalar_cases()) == 225
        assert len(document.batches()) == 225
        assert all(len(batch.cases) == 1 for batch in document.batches())
        assert runtime["initialisations"] == 50
        assert runtime["max_cases_per_batch"] == 1
        assert runtime["max_initialisations_per_batch"] == 25
        assert runtime["max_batch_elapsed_seconds"] == 5 * 60 * 60
        assert runtime["device"] == "gpu"
        assert runtime["auto_halt"] is True
        assert query["where"]["smoothness"] == expected["smoothness"]
        assert query["where"]["sharpness"] == expected["sharpness"]
        assert query["limit"] == 5
        assert query["resume_optimizer"] is False
        assert query["perturbation_levels"] == [
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.01,
        ]
        assert "run_on_bar_gpu" in _source(cells["run-locally"])
        assert 'sweep_parameters = ["smoothness", "sharpness"]' in _source(
            cells["query-config-run"]
        )


def test_final_saved_parameter_lbfgs_notebook_is_canonical_and_exact():
    notebook = json.loads(FINAL_LBFGS_NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    assert [cell["id"] for cell in notebook["cells"]] == [
        cell["id"] for cell in _notebook()["cells"]
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(
                _source(cell),
                f"{FINAL_LBFGS_NOTEBOOK.name}:{cell['id']}",
                "exec",
            )
            assert "def " not in _source(cell)

    namespace = {}
    exec(_source(cells["imports-and-project-root"]), namespace)
    exec(_source(cells["create-config"]), namespace)
    parameters = namespace["parameters"]
    runtime = namespace["runtime"]
    query = namespace["initialization_query"]
    assert namespace["run_name"] == "N100_three_cap_saved_parameter_lbfgs_refinement"
    assert parameters["optimizer"] == "lbfgs"
    assert parameters["u_max"] == [1280.0, 160.0, 40.0]
    assert parameters["schedule"] == [(50, 1.0)]
    assert parameters["projected_gradient_tol"] == 1e-5
    assert parameters["block_size"] == 10
    assert runtime["initialisations"] == 0
    assert runtime["device"] == "gpu"
    assert runtime["max_elapsed_seconds"] == 12 * 60 * 60
    assert runtime["distribute_max_elapsed_across_batches"] is True
    assert runtime["repeat_schedule_until_stable"] is True
    assert query["limit"] == 5
    assert query["resume_optimizer"] is False
    assert query["perturbed"] is False
    assert query["match_parameters"] == ["u_max", "smoothness", "sharpness"]
    assert query["discover_parameters"] == ["u_max", "smoothness", "sharpness"]
    assert query["discover_group_parameters"] == ["u_max"]
