import sqlite3

import numpy as np
import pytest

from ofc.config import make_document
from ofc.plot_cli import main as plot_main, select_rows
from ofc.results import Results
from ofc.runner import _validate_slurm_cpu_allocation, run_config
from ofc.storage import ResultStore, parameter_database_path


def tiny_document(database, *, use_jit=False):
    return make_document(
        name="tiny_matrix",
        parameters={
            "N": 4,
            "r_bg": [-1.5, 0.75],
            "u_max": [10.0, 20.0],
            "schedule": [(2, 1.0), (2, 0.5)],
            "block_size": 2,
            "smoothness": 1e-3,
        },
        runtime={
            "initialisations": 2,
            "use_jit": use_jit,
            "use_x64": True,
            "device": "cpu",
            "concurrent_workers": 1,
            "database": str(database),
        },
    )


def test_slurm_cpu_allocation_must_cover_config_workers(tmp_path, monkeypatch):
    document = make_document(
        name="allocation",
        runtime={
            "concurrent_workers": 2,
            "database": str(tmp_path / "allocation.sqlite3"),
        },
    )

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    with pytest.raises(RuntimeError, match="allocation is too small"):
        _validate_slurm_cpu_allocation([document])

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    _validate_slurm_cpu_allocation([document])


def test_generic_methodology_storage_accepts_future_config_fields_without_schema_changes(
    tmp_path,
):
    database = tmp_path / "future.sqlite3"
    store = ResultStore(database)
    first_document_id = store.register_config(
        {"config_id": 123, "config": {"future_top_level": {"version": 1}}}
    )
    second_document_id = store.register_config(
        {"config_id": 123, "config": {"future_top_level": {"version": 2}}}
    )
    record = {
        "config_id": 123,
        "config_name": "future",
        "config_file": "future.yaml",
        "description": "",
        "created_utc": "",
        "batch_id": 456,
        "queue_id": 789,
        "batch_index": 0,
        "batch_key": "future-batch",
        "seed": 42,
    }
    _, run_ids = store.prepare_batch(
        record,
        [
            {
                "N": 4,
                "t_interval": 1.0,
                "r_bg": 0.5,
                "u_max": 10.0,
                "v_max": 20.0,
                "future_optimizer_setting": {"nested": [1, 2, 3]},
            }
        ],
        1,
        config_document_id=second_document_id,
    )

    assert first_document_id != second_document_id
    row = Results(database).search(
        future_optimizer_setting={"nested": [1, 2, 3]}
    )[0]
    assert row["run_id"] == run_ids[(0, 0)]
    assert row["future_optimizer_setting"] == {"nested": [1, 2, 3]}
    assert Results(database).config_document(row["run_id"])["future_top_level"] == {
        "version": 2
    }


def test_stage_commits_search_and_array_retrieval(tmp_path):
    database = tmp_path / "results.sqlite3"
    document = tiny_document(database)
    summary = run_config(document, queue_id=777)

    assert summary[0]["cases"] == 4
    assert summary[0]["runs"] == 8
    results = Results(database)
    low_cap = results.search(config_name="tiny_matrix", queue_id=777, u_max=(9.0, 11.0))
    assert len(low_cap) == 4
    assert all(row["status"] == "complete" for row in low_cap)
    assert all(row["best_max_abs_du_dt"] is not None for row in low_cap)
    assert all(row["best_max_abs_dv_dt"] is not None for row in low_cap)
    assert all("best_max_abs_d2u_dt2" not in row for row in low_cap)
    assert all("best_max_abs_d2v_dt2" not in row for row in low_cap)
    assert all(row["fourier_num_modes"] == 5 for row in low_cap)
    assert all(row["device"] == "cpu" for row in low_cap)
    assert all(row["execution_device"] == "cpu" for row in low_cap)
    assert len(results.search(fourier_rms_amplitude=(0.29, 0.31))) == 8
    assert len(results.search(r_bg=-1.5)) == 4

    run = results.get(low_cap[0]["run_id"])
    assert "dt" not in run
    assert run["history"]["step"].tolist() == [0, 1, 2, 3, 4]
    assert run["tolerances"]["step"].tolist() == [2, 4]
    assert run["tolerances"]["passed"].dtype.kind in {"i", "u"}
    assert set(run["controls"]) == {"initial", "best", "final"}
    assert all(values.shape == (5,) for values in run["controls"]["best"].values())
    assert run["config"]["config_id"] == document.config_id
    assert run["config"]["parameters"]["r_bg"] == [-1.5, 0.75]

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert tables == {
            "metadata",
            "runs",
            "physical_values",
            "control_arrays",
            "history_arrays",
            "tolerance_history",
        }
        assert [
            row[1] for row in connection.execute("PRAGMA table_info(runs)")
        ] == ["run_id"]
        assert connection.execute("SELECT count(*) FROM history_arrays").fetchone()[0] == 48
    with sqlite3.connect(parameter_database_path(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "config_documents" in tables
        assert "run_parameters" in tables
        assert "run_parameter_values" in tables
        assert "optimizer_stages" in tables
        assert connection.execute("SELECT count(*) FROM optimizer_stages").fetchone()[0] == 16

    figure_dir = tmp_path / "figures"
    assert plot_main(
        [
            "--database",
            str(database),
            "--where",
            "config_name=tiny_matrix",
            "--sweep-parameter",
            "u_max",
            "--output-dir",
            str(figure_dir),
        ]
    ) == 0
    assert {path.name for path in figure_dir.glob("*.png")} == {
        "01_convergence.png",
        "02_distribution.png",
        "03_controls.png",
    }


def test_batch_selection_runs_only_requested_time_shard(tmp_path):
    database = tmp_path / "selected.sqlite3"
    document = make_document(
        name="selected_time",
        parameters={
            "N": 4,
            "t_interval": [1.0, 2.0],
            "r_bg": 0.5,
            "u_max": 10.0,
            "schedule": [(1, 1.0)],
            "block_size": 1,
        },
        runtime={
            "initialisations": 1,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "database": str(database),
        },
    )

    summary = run_config(document, batch_indices=[1], queue_id=888)
    rows = Results(database).search(config_name="selected_time")

    assert len(summary) == 1
    assert summary[0]["batch_id"] == document.batches()[1].batch_id
    assert len(rows) == 1
    assert rows[0]["t_interval"] == 2.0
    with pytest.raises(ValueError, match="valid indices are 0..1"):
        run_config(document, batch_indices=[2], queue_id=888)


def test_config_query_appends_stored_controls_to_random_initializations(tmp_path):
    database = tmp_path / "queried.sqlite3"
    source = make_document(
        name="query_source",
        parameters={
            "N": 4,
            "r_bg": 0.75,
            "u_max": 10.0,
            "v_max": 20.0,
            "schedule": [(1, 1.0)],
            "block_size": 1,
        },
        runtime={
            "initialisations": 2,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "database": str(database),
        },
    )
    run_config(source, queue_id=1001)
    results = Results(database)
    source_row = max(
        results.search(queue_id=1001), key=lambda row: row["best_score"]
    )
    source_best = results.controls(source_row["run_id"], "best")

    target = make_document(
        name="query_target",
        parameters={
            "N": 4,
            "r_bg": 0.75,
            "u_max": 10.0,
            "v_max": 20.0,
            "schedule": [(1, 1.0)],
            "block_size": 1,
        },
        runtime={
            "initialisations": 2,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "database": str(database),
        },
        query={
            "where": {"queue_id": 1001},
            "limit": 1,
            "order_by": "best_score",
            "descending": True,
            "control_kind": "best",
        },
    )
    summary = run_config(target, queue_id=1002)
    target_rows = results.search(queue_id=1002)

    assert summary[0]["random_initialisations"] == 2
    assert summary[0]["queried_initialisations"] == 1
    assert summary[0]["runs"] == 3
    assert [row["initialization_source"] for row in target_rows].count("fourier") == 2
    queried = [
        row for row in target_rows if row["initialization_source"] == "query"
    ][0]
    assert queried["source_run_id"] == source_row["run_id"]
    assert queried["source_control_kind"] == "best"
    assert queried["initialization_count_total"] == 3
    queried_initial = results.controls(queried["run_id"], "initial")
    np.testing.assert_allclose(queried_initial["u"], source_best["u"], rtol=1e-12)
    np.testing.assert_allclose(queried_initial["v"], source_best["v"], rtol=1e-12)
    assert results.config_document(queried["run_id"])["query"]["where"] == {
        "queue_id": 1001
    }


def test_config_query_fails_clearly_when_it_matches_no_runs(tmp_path):
    database = tmp_path / "empty_query.sqlite3"
    document = make_document(
        name="empty_query",
        parameters={"N": 4, "schedule": [(1, 1.0)], "block_size": 1},
        runtime={
            "initialisations": 1,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "database": str(database),
        },
        query={"where": {"config_name": "does_not_exist"}},
    )

    with pytest.raises(ValueError, match="matched no runs"):
        run_config(document, queue_id=1003)


def test_lbfgs_config_runs_and_persists_optimizer_provenance(tmp_path):
    database = tmp_path / "lbfgs.sqlite3"
    document = make_document(
        name="tiny_lbfgs",
        parameters={
            "N": 4,
            "r_bg": 0.75,
            "u_max": 10.0,
            "optimizer": "lbfgs",
            "schedule": [(2, 1.0), (1, 1.0)],
            "lbfgs_history_size": 3,
            "lbfgs_max_linesearch_steps": 8,
            "lbfgs_tolerance": 1e-8,
            "block_size": 1,
            "smoothness": 1e-3,
            "sharpness": 1e-4,
        },
        runtime={
            "initialisations": 1,
            "use_jit": True,
            "use_x64": True,
            "device": "cpu",
            "concurrent_workers": 1,
            "database": str(database),
        },
    )

    summary = run_config(document, queue_id=999)
    results = Results(database)
    rows = results.search(config_name="tiny_lbfgs")
    run = results.get(rows[0]["run_id"])

    assert summary[0]["runs"] == 1
    assert len(rows) == 1
    assert run["optimizer"] == "lbfgs"
    assert run["lbfgs_history_size"] == 3
    assert run["lbfgs_max_linesearch_steps"] == 8
    assert run["sharpness"] == 1e-4
    assert run["best_max_abs_d2u_dt2"] >= 0.0
    assert run["best_max_abs_d2v_dt2"] >= 0.0
    assert run["history"]["step"].tolist() == [0, 1, 2, 3]
    assert np.isfinite(run["history"]["score"]).all()
    assert all(np.isfinite(values).all() for values in run["controls"]["best"].values())


def test_rerun_preserves_old_results_and_allocates_fresh_run_ids(tmp_path):
    database = tmp_path / "rerun.sqlite3"
    document = tiny_document(database)
    first_summary = run_config(document, queue_id=1)
    results = Results(database)
    first_rows = results.search(queue_id=1)
    first_rows.sort(
        key=lambda row: (row["r_bg"], row["u_max"], row["initialization_index"])
    )
    first_histories = [results.history(row["run_id"])["score"] for row in first_rows]

    second_summary = run_config(document, queue_id=2)
    second_rows = results.search(queue_id=2)
    second_rows.sort(
        key=lambda row: (row["r_bg"], row["u_max"], row["initialization_index"])
    )
    second_histories = [results.history(row["run_id"])["score"] for row in second_rows]

    assert first_summary[0]["batch_id"] == second_summary[0]["batch_id"]
    assert len(first_rows) == len(second_rows) == 8
    assert len(results.search()) == 16
    assert {row["run_id"] for row in first_rows}.isdisjoint(
        {row["run_id"] for row in second_rows}
    )
    assert all(row["queue_id"] == 2 for row in second_rows)
    selected = select_rows(results, run_ids=[], filters={}, limit=None)
    assert {row["run_id"] for row in selected} == {
        row["run_id"] for row in second_rows
    }
    for first, second in zip(first_histories, second_histories):
        np.testing.assert_array_equal(first, second)
