import json
import sqlite3

import numpy as np
import pytest

from ofc.config import make_document
from ofc.plot_cli import main as plot_main, select_rows
from ofc.query_cli import main as query_main
from ofc.results import Results
from ofc.runner import _validate_slurm_cpu_allocation, run_config
from ofc.storage import PARAMETER_SCHEMA, ResultStore, parameter_database_path


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
    assert all(np.isfinite(row["best_projected_gradient_rms"]) for row in low_cap)
    assert all("best_max_abs_d2u_dt2" not in row for row in low_cap)
    assert all("best_max_abs_d2v_dt2" not in row for row in low_cap)
    assert all(row["fourier_num_modes"] == 5 for row in low_cap)
    assert all(len(row["fourier_u_sin_coefficients"]) == 5 for row in low_cap)
    assert all(len(row["fourier_u_cos_coefficients"]) == 5 for row in low_cap)
    assert all(len(row["fourier_v_sin_coefficients"]) == 5 for row in low_cap)
    assert all(len(row["fourier_v_cos_coefficients"]) == 5 for row in low_cap)
    assert all(row["device"] == "cpu" for row in low_cap)
    assert all(row["execution_device"] == "cpu" for row in low_cap)
    assert len(results.search(fourier_rms_amplitude=(0.29, 0.31))) == 8
    assert len(results.search(r_bg=-1.5)) == 4

    run = results.get(low_cap[0]["run_id"])
    assert "dt" not in run
    assert run["history"]["step"].tolist() == [0, 1, 2, 3, 4]
    assert run["history"]["learning_rate_change_steps"].tolist() == [0, 2]
    # Only the terminal stability values are persisted; checkpoint history is
    # kept on-device solely for the in-flight consecutive-block decision.
    assert run["tolerances"]["step"].tolist() == [4]
    assert run["tolerances"]["passed"].dtype.kind in {"i", "u"}
    assert set(run["controls"]) == {
        "initial",
        "best",
        "final",
        "initial_raw",
        "best_raw",
        "final_raw",
    }
    assert all(values.shape == (5,) for values in run["controls"]["best"].values())
    grid = np.linspace(0.0, 1.0, run["N"] + 1)
    modes = np.arange(1, run["fourier_num_modes"] + 1)
    phase = 2.0 * np.pi * modes[:, None] * grid[None, :]
    raw_u = run["fourier_u_offset"] + np.sum(
        np.asarray(run["fourier_u_sin_coefficients"])[:, None] * np.sin(phase)
        + np.asarray(run["fourier_u_cos_coefficients"])[:, None] * np.cos(phase),
        axis=0,
    )
    raw_v = run["fourier_v_offset"] + np.sum(
        np.asarray(run["fourier_v_sin_coefficients"])[:, None] * np.sin(phase)
        + np.asarray(run["fourier_v_cos_coefficients"])[:, None] * np.cos(phase),
        axis=0,
    )
    np.testing.assert_allclose(run["controls"]["initial_raw"]["u"], raw_u)
    np.testing.assert_allclose(run["controls"]["initial_raw"]["v"], raw_v)
    np.testing.assert_allclose(
        run["controls"]["initial"]["u"],
        run["u_max"] / (1.0 + np.exp(-raw_u)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        run["controls"]["initial"]["v"],
        run["v_max"] * np.tanh(raw_v),
        rtol=1e-12,
        atol=1e-12,
    )
    for kind in ("best", "final"):
        raw_controls = run["controls"][f"{kind}_raw"]
        np.testing.assert_allclose(
            run["controls"][kind]["u"],
            run["u_max"] / (1.0 + np.exp(-raw_controls["u"])),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            run["controls"][kind]["v"],
            run["v_max"] * np.tanh(raw_controls["v"]),
            rtol=1e-12,
            atol=1e-12,
        )


    assert set(run["adam_states"]) == {"best", "final"}
    for kind, state in run["adam_states"].items():
        assert state["count"] == (
            run["best_step"] if kind == "best" else run["completed_steps"]
        )
        assert set(state) == {"raw", "count", "first_moment", "second_moment"}
        for name in ("u", "v"):
            np.testing.assert_allclose(
                state["raw"][name], run["controls"][f"{kind}_raw"][name]
            )
            assert state["first_moment"][name].shape == (5,)
            assert state["second_moment"][name].shape == (5,)
            assert np.isfinite(state["first_moment"][name]).all()
            assert np.isfinite(state["second_moment"][name]).all()
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
            "optimizer_state_arrays",
            "optimizer_states",
            "history_arrays",
            "tolerance_history",
        }
        assert [
            row[1] for row in connection.execute("PRAGMA table_info(runs)")
        ] == [
            "run_id",
            "config_id",
            "config_document_id",
            "batch_id",
            "execution_id",
            "queue_id",
            "batch_index",
            "status",
            "started_utc",
            "completed_utc",
        ]
        indexed = connection.execute(
            """SELECT config_id, config_document_id, batch_id, execution_id,
                      queue_id, status, started_utc, completed_utc
               FROM runs WHERE run_id=?""",
            (run["run_id"],),
        ).fetchone()
        assert indexed[0] == document.config_id
        assert all(value is not None for value in indexed[1:5])
        assert indexed[5] == "complete"
        assert indexed[6] is not None
        assert indexed[7] is not None
        assert connection.execute("SELECT count(*) FROM history_arrays").fetchone()[0] == 48
        assert connection.execute(
            "SELECT count(*) FROM optimizer_state_arrays"
        ).fetchone()[0] == 64
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
        "01_sweep_summary.png",
    }


def test_per_batch_elapsed_limit_stops_and_records_partial_run(tmp_path):
    database = tmp_path / "batch-time-limit.sqlite3"
    document = make_document(
        name="batch_time_limit",
        parameters={
            "N": 4,
            "schedule": [(2, 1.0), (2, 0.5)],
            "block_size": 1,
            "J_tol": None,
            "u_tol": None,
            "v_tol": None,
            "projected_gradient_tol": None,
        },
        runtime={
            "initialisations": 1,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "max_batch_elapsed_seconds": 1e-9,
            "auto_halt": False,
            "database": str(database),
        },
    )

    summary = run_config(document, queue_id=778)
    row = Results(database).search(queue_id=778)[0]
    assert summary[0]["time_limited"] is True
    assert summary[0]["termination_reason"] == "time_limit"
    assert row["termination_reason"] == "time_limit"
    assert row["completed_steps"] == 2
    assert row["status"] == "complete"


def test_walltime_chunks_do_not_create_false_schedule_change_markers(
    tmp_path, monkeypatch
):
    database = tmp_path / "walltime-chunks.sqlite3"
    monkeypatch.setattr("ofc.runner.WALLTIME_CHECK_MAX_STEPS", 2)
    document = make_document(
        name="walltime_chunks",
        parameters={
            "N": 4,
            "schedule": [(5, 1.0)],
            "block_size": 1,
            "J_tol": None,
            "u_tol": None,
            "v_tol": None,
            "projected_gradient_tol": None,
        },
        runtime={
            "initialisations": 1,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "auto_halt": False,
            "database": str(database),
        },
    )

    run_config(document, queue_id=779)
    results = Results(database)
    row = results.search(queue_id=779)[0]
    history = results.history(row["run_id"])
    assert history["step"].tolist() == list(range(6))
    assert history["optimizer_step_size_change_steps"].tolist() == [0]
    with sqlite3.connect(parameter_database_path(database)) as connection:
        connection.row_factory = sqlite3.Row
        stages = connection.execute(
            "SELECT parameters_json FROM optimizer_stages WHERE run_id=?",
            (row["run_id"],),
        ).fetchall()
    assert len(stages) == 3
    assert not any(
        json.loads(stage["parameters_json"])["learning_rate_update"]
        for stage in stages
    )


def test_history_chunks_tighten_after_one_million_equivalent_steps(
    tmp_path, monkeypatch
):
    database = tmp_path / "long-history-chunks.sqlite3"
    monkeypatch.setattr("ofc.runner.WALLTIME_CHECK_MAX_STEPS", 2)
    monkeypatch.setattr("ofc.runner.LONG_RUN_HISTORY_THRESHOLD_STEPS", 2)
    monkeypatch.setattr("ofc.runner.LONG_RUN_HISTORY_CHUNK_STEPS", 1)
    document = make_document(
        name="long_history_chunks",
        parameters={
            "N": 4,
            "schedule": [(5, 1.0)],
            "block_size": 1,
            "J_tol": None,
            "u_tol": None,
            "v_tol": None,
            "projected_gradient_tol": None,
        },
        runtime={
            "initialisations": 1,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "auto_halt": False,
            "database": str(database),
        },
    )

    run_config(document, queue_id=780)
    results = Results(database)
    row = results.search(queue_id=780)[0]
    np.testing.assert_array_equal(results.history(row["run_id"])["step"], range(6))
    with sqlite3.connect(parameter_database_path(database)) as connection:
        stages = connection.execute(
            """SELECT start_step, end_step FROM optimizer_stages
               WHERE run_id=? ORDER BY stage_index""",
            (row["run_id"],),
        ).fetchall()
    assert stages == [(0, 2), (2, 3), (3, 4), (4, 5)]


def test_runner_auto_halts_whole_batch_after_three_consecutive_blocks(tmp_path):
    database = tmp_path / "auto_halt.sqlite3"
    document = make_document(
        name="auto_halt",
        parameters={
            "N": 4,
            "schedule": [(20, 1.0)],
            "block_size": 2,
            "J_tol": 1e30,
            "u_tol": 1e30,
            "v_tol": 1e30,
            "projected_gradient_tol": None,
        },
        runtime={
            "initialisations": 2,
            "use_jit": True,
            "device": "cpu",
            "concurrent_workers": 1,
            "auto_halt": True,
            "database": str(database),
        },
    )

    summary = run_config(document, queue_id=778)

    assert summary[0]["steps"] == 6
    assert summary[0]["auto_halted"] is True
    results = Results(database)
    for row in results.search(queue_id=778):
        run = results.get(row["run_id"])
        assert run["history"]["step"].tolist() == list(range(7))
        assert run["tolerances"]["step"].tolist() == [6]
        assert run["tolerances"]["consecutive_blocks"].tolist() == [3]
        assert run["tolerances"]["passed"].tolist() == [1]
        assert "projected_gradient_tolerance" not in run["tolerances"]


def test_learning_rate_boundary_prunes_stable_members_and_logs_count(
    tmp_path, capsys
):
    database = tmp_path / "boundary_pruning.sqlite3"
    document = make_document(
        name="boundary_pruning",
        parameters={
            "N": 4,
            "schedule": [(3, 1.0), (3, 0.1)],
            "block_size": 1,
            "J_tol": [1e30, 1e-100],
            "u_tol": None,
            "v_tol": None,
            "projected_gradient_tol": None,
        },
        runtime={
            "initialisations": 1,
            "use_jit": True,
            "device": "cpu",
            "concurrent_workers": 1,
            "auto_halt": True,
            "database": str(database),
        },
    )

    summary = run_config(document, queue_id=779)

    output = capsys.readouterr().out
    assert "halted 1 stable run(s); 1 remain in the next batch" in output
    assert summary[0]["halted_runs"] == 1
    results = Results(database)
    easy = results.get(results.search(queue_id=779, J_tol=1e30)[0]["run_id"])
    strict = results.get(results.search(queue_id=779, J_tol=1e-100)[0]["run_id"])
    assert easy["history"]["step"].tolist() == list(range(4))
    assert strict["history"]["step"].tolist() == list(range(7))
    assert easy["tolerances"]["auto_halted"].tolist() == [True]
    assert easy["tolerances"]["passed"].tolist() == [1]


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
            "max_initialisations_per_batch": 3,
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

    assert [item["random_initialisations"] for item in summary] == [2, 0, 0]
    assert [item["queried_initialisations"] for item in summary] == [1, 3, 1]
    assert [item["runs"] for item in summary] == [3, 3, 1]
    assert [item["initialization_batch_index"] for item in summary] == [0, 1, 2]
    assert all(item["initialization_batch_count"] == 3 for item in summary)
    assert [row["initialization_source"] for row in target_rows].count("fourier") == 2
    queried = sorted(
        [
        row for row in target_rows if row["initialization_source"] == "query"
        ],
        key=lambda row: row["query_perturbation_index"],
    )
    assert len(queried) == 5
    assert all(row["source_run_id"] == source_row["run_id"] for row in queried)
    assert all(row["source_control_kind"] == "best" for row in queried)
    assert all(row["initialization_count_total"] == 7 for row in queried)
    assert sorted(row["initialization_index"] for row in target_rows) == list(range(7))
    assert all(row["query_perturbed"] is True for row in queried)
    assert [row["query_perturbation_level"] for row in queried] == [
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
    ]
    assert len({row["query_perturbation_seed"] for row in queried}) == 5
    for row in queried:
        queried_initial = results.controls(row["run_id"], "initial")
        assert not np.allclose(queried_initial["u"], source_best["u"])
        assert not np.allclose(queried_initial["v"], source_best["v"])
        assert np.asarray(queried_initial["u"]).min() >= 0.0
        assert np.asarray(queried_initial["u"]).max() <= 10.0
        assert np.asarray(queried_initial["v"]).min() >= -20.0
        assert np.asarray(queried_initial["v"]).max() <= 20.0
    assert results.config_document(queried[0]["run_id"])["query"]["where"] == {
        "queue_id": 1001
    }


def test_config_query_can_read_controls_from_an_isolated_source_database(tmp_path):
    source_database = tmp_path / "source.sqlite3"
    target_database = tmp_path / "target.sqlite3"
    source = make_document(
        name="cross_database_source",
        parameters={
            "N": 4,
            "r_bg": 0.75,
            "u_max": 10.0,
            "v_max": 20.0,
            "schedule": [(1, 1.0)],
            "block_size": 1,
        },
        runtime={
            "initialisations": 1,
            "use_jit": False,
            "device": "cpu",
            "database": str(source_database),
        },
    )
    run_config(source, queue_id=1101)
    source_results = Results(source_database)
    source_row = source_results.search(queue_id=1101)[0]
    source_best = source_results.controls(source_row["run_id"], "best")

    target = make_document(
        name="cross_database_target",
        parameters={
            "N": 4,
            "r_bg": 0.75,
            "u_max": 10.0,
            "v_max": 20.0,
            "schedule": [(1, 1.0)],
            "block_size": 1,
        },
        runtime={
            "initialisations": 0,
            "use_jit": False,
            "device": "cpu",
            "database": str(target_database),
        },
        query={
            "database": str(source_database),
            "where": {"run_id": source_row["run_id"]},
            "limit": 1,
            "control_kind": "best",
            "resume_optimizer": False,
            "perturbed": False,
        },
    )
    run_config(target, queue_id=1102)
    target_results = Results(target_database)
    target_row = target_results.search(queue_id=1102)[0]

    assert target_row["source_run_id"] == source_row["run_id"]
    target_initial = target_results.controls(target_row["run_id"], "initial")
    np.testing.assert_allclose(target_initial["u"], source_best["u"])
    np.testing.assert_allclose(target_initial["v"], source_best["v"])


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


def test_case_matched_query_uses_partial_controls_and_falls_back_per_case(tmp_path):
    database = tmp_path / "matched_query.sqlite3"
    common_parameters = {
        "N": 4,
        "r_bg": 0.75,
        "u_max": 10.0,
        "v_max": 20.0,
        "schedule": [(1, 1.0)],
        "block_size": 1,
    }
    common_runtime = {
        "initialisations": 1,
        "use_jit": False,
        "device": "cpu",
        "concurrent_workers": 1,
        "database": str(database),
    }
    fallback = make_document(
        name="fallback",
        parameters={**common_parameters, "adam_learning_rate": 0.5},
        runtime=common_runtime,
    )
    partial = make_document(
        name="partial",
        parameters={
            **common_parameters,
            "adam_learning_rate": [0.01, 0.02],
        },
        runtime=common_runtime,
    )
    run_config(fallback, queue_id=2000)
    run_config(partial, queue_id=2001)

    target = make_document(
        name="matched_target",
        parameters={
            **common_parameters,
            "adam_learning_rate": [0.01, 0.02, 0.03],
        },
        runtime={
            **common_runtime,
            "initialisations": 0,
            "max_cases_per_batch": 2,
        },
        query={
            "where": {"queue_id": 2001},
            "limit": 1,
            "order_by": "best_score",
            "descending": True,
            "control_kind": "best",
            "perturbed": False,
            "match_parameters": ["adam_learning_rate"],
            "fallback_where": {"queue_id": 2000},
        },
    )
    summary = run_config(target, queue_id=2002)
    results = Results(database)
    target_rows = results.search(queue_id=2002)
    partial_rows = {
        row["adam_learning_rate"]: row for row in results.search(queue_id=2001)
    }
    fallback_row = results.search(queue_id=2000)[0]

    assert [item["cases"] for item in summary] == [2, 1]
    assert all(item["runs"] == item["cases"] for item in summary)
    assert all(item["random_initialisations"] == 0 for item in summary)
    assert all(item["queried_initialisations"] == 1 for item in summary)
    assert len(target_rows) == 3
    for row in target_rows:
        source = partial_rows.get(row["adam_learning_rate"], fallback_row)
        assert row["initialization_source"] == "query"
        assert row["source_run_id"] == source["run_id"]
        initial = results.controls(row["run_id"], "initial")
        source_best = results.controls(source["run_id"], "best")
        np.testing.assert_allclose(initial["u"], source_best["u"], atol=1e-10)
        np.testing.assert_allclose(initial["v"], source_best["v"], atol=1e-10)


def test_query_can_resume_the_exact_final_adam_state(tmp_path):
    database = tmp_path / "adam_resume.sqlite3"
    parameters = {
        "N": 4,
        "r_bg": 0.75,
        "u_max": 10.0,
        "v_max": 20.0,
        "schedule": [(2, 1.0)],
        "block_size": 1,
        "adam_learning_rate": 0.01,
    }
    runtime = {
        "initialisations": 1,
        "use_jit": False,
        "device": "cpu",
        "concurrent_workers": 1,
        "database": str(database),
    }
    source = make_document(
        name="adam_resume_source",
        parameters=parameters,
        runtime=runtime,
    )
    run_config(source, queue_id=3000)
    results = Results(database)
    source_row = results.search(queue_id=3000)[0]
    source_state = results.adam_state(source_row["run_id"], "final")

    target = make_document(
        name="adam_resume_target",
        parameters={**parameters, "schedule": [(1, 1.0)]},
        runtime={**runtime, "initialisations": 0},
        query={
            "where": {"run_id": source_row["run_id"]},
            "limit": 1,
            "control_kind": "final",
            "perturbed": False,
            "resume_optimizer": True,
        },
    )
    run_config(target, queue_id=3001)
    target_row = results.search(queue_id=3001)[0]
    target_run = results.get(target_row["run_id"])

    assert target_row["source_optimizer_resumed"] is True
    for name in ("u", "v"):
        np.testing.assert_allclose(
            target_run["controls"]["initial_raw"][name],
            source_state["raw"][name],
        )
    assert target_run["adam_states"]["final"]["count"] == source_state["count"] + 1


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
    assert run["history"]["learning_rate_change_steps"].tolist() == [0]
    assert np.isfinite(run["history"]["score"]).all()
    assert all(np.isfinite(values).all() for values in run["controls"]["best"].values())

    # Legacy L-BFGS stages had only the broader schedule-change flag. They
    # must still not appear as learning-rate updates on a step plot.
    with sqlite3.connect(parameter_database_path(database)) as connection:
        connection.row_factory = sqlite3.Row
        stages = connection.execute(
            """SELECT stage_index, parameters_json FROM optimizer_stages
               WHERE run_id=? ORDER BY stage_index""",
            (run["run_id"],),
        ).fetchall()
        for stage in stages:
            parameters = json.loads(stage["parameters_json"])
            parameters.pop("learning_rate_update")
            connection.execute(
                """UPDATE optimizer_stages SET parameters_json=?
                   WHERE run_id=? AND stage_index=?""",
                (json.dumps(parameters), run["run_id"], stage["stage_index"]),
            )
    assert Results(database).history(run["run_id"])[
        "learning_rate_change_steps"
    ].tolist() == [0]


def test_peak_refinement_runs_monotonically_until_stability(tmp_path):
    database = tmp_path / "peak-refinement.sqlite3"
    document = make_document(
        name="tiny_peak_refinement",
        parameters={
            "N": 4,
            "r_bg": 0.75,
            "u_max": 10.0,
            "optimizer": "peak_refinement",
            "schedule": [(2, 1.0)],
            "peak_initial_step_size": 0.05,
            "peak_max_linesearch_steps": 12,
            "block_size": 1,
            "J_tol": 1e30,
            "u_tol": 1e30,
            "v_tol": 1e30,
            "projected_gradient_tol": 1e30,
        },
        runtime={
            "initialisations": 1,
            "use_jit": True,
            "use_x64": True,
            "device": "cpu",
            "concurrent_workers": 1,
            "max_batch_elapsed_seconds": 60.0,
            "max_steps_per_chunk": 2,
            "repeat_schedule_until_stable": True,
            "auto_halt": True,
            "database": str(database),
        },
    )

    summary = run_config(document, queue_id=1001)
    run = Results(database).get(
        Results(database).search(config_name="tiny_peak_refinement")[0]["run_id"]
    )

    assert summary[0]["termination_reason"] == "stability"
    assert run["optimizer"] == "peak_refinement"
    assert run["termination_reason"] == "stability"
    assert run["history"]["learning_rate_change_steps"].tolist() == [0]
    assert np.all(np.diff(run["history"]["score"]) >= -1e-12)
    assert run["tolerances"]["consecutive_blocks"].tolist() == [3]


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
    assert [summary["queue_id"] for summary in results.config_runs()] == [2, 1]
    assert {row["queue_id"] for row in results.search_config_run(rank=1)} == {2}
    assert {row["queue_id"] for row in results.search_config_run(rank=2)} == {1}
    assert results.search_config_run(rank=3) == []
    latest_best = results.search_config_run(
        rank=1,
        u_max=(None, 10.0),
        order_by="best_score",
        descending=True,
        limit=1,
    )
    assert len(latest_best) == 1
    assert latest_best[0]["queue_id"] == 2
    selected = select_rows(results, run_ids=[], filters={}, limit=None)
    assert {row["run_id"] for row in selected} == {
        row["run_id"] for row in second_rows
    }
    for first, second in zip(first_histories, second_histories):
        np.testing.assert_array_equal(first, second)


def test_query_cli_selects_recent_config_run_and_best_score(tmp_path, capsys):
    database = tmp_path / "query.sqlite3"
    document = tiny_document(database)
    run_config(document, queue_id=1)
    run_config(document, queue_id=2)
    capsys.readouterr()

    assert query_main(
        [
            "--database",
            str(database),
            "--config-run",
            "2",
            "--where",
            "u_max=:10",
            "--best",
        ]
    ) == 0
    matches = json.loads(capsys.readouterr().out)
    assert len(matches) == 1
    assert matches[0]["queue_id"] == 1
    assert matches[0]["u_max"] == 10.0


def test_physical_v1_database_is_migrated_and_backfilled(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    parameters = parameter_database_path(database)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
               INSERT INTO metadata VALUES('database_version', '1');
               INSERT INTO metadata VALUES('database_role', 'physical_results');
               CREATE TABLE runs(run_id INTEGER PRIMARY KEY);
               INSERT INTO runs VALUES(1);"""
        )
    with sqlite3.connect(parameters) as connection:
        connection.executescript(PARAMETER_SCHEMA)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES(?, ?)",
            [("database_version", "1"), ("database_role", "run_parameters")],
        )
        connection.execute(
            """INSERT INTO config_documents(
                   config_document_id, config_id, config_json, registered_utc
               ) VALUES(1, 123, '{}', '2026-01-01T00:00:00+00:00')"""
        )
        connection.execute(
            """INSERT INTO execution_batches(
                   execution_id, config_document_id, batch_id, queue_id,
                   batch_index, execution_json, status, error,
                   started_utc, completed_utc
               ) VALUES(1, 1, 456, 789, 0, '{}', 'complete', NULL,
                        '2026-01-02T00:00:00+00:00',
                        '2026-01-02T01:00:00+00:00')"""
        )
        connection.execute(
            """INSERT INTO run_parameters(
                   run_id, execution_id, config_document_id, status, error,
                   started_utc, completed_utc, parameters_json
               ) VALUES(1, 1, 1, 'complete', NULL,
                        '2026-01-02T00:00:00+00:00',
                        '2026-01-02T01:00:00+00:00',
                        '{\"config_id\":123,\"config_name\":\"legacy\",\"queue_id\":789}')"""
        )

    ResultStore(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='database_version'"
        ).fetchone()[0] == "2"
        row = connection.execute(
            """SELECT config_id, config_document_id, batch_id, execution_id,
                      queue_id, status, started_utc, completed_utc
               FROM runs WHERE run_id=1"""
        ).fetchone()
    assert row == (
        123,
        1,
        456,
        1,
        789,
        "complete",
        "2026-01-02T00:00:00+00:00",
        "2026-01-02T01:00:00+00:00",
    )
