import sqlite3

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ofc.config import ResolvedConfig, make_document
from ofc.grid_refinement import (
    grid_refinement_diagnostics,
    interpolate_controls_2n_minus_1,
    refined_grid_size,
)
from ofc.grid_refinement_backfill import backfill_database
from ofc.physics import Physics
from ofc.results import Results
from ofc.runner import run_config


jax.config.update("jax_enable_x64", True)


def test_interpolation_uses_the_requested_2n_minus_1_physics_grid():
    N = 3
    source_times = np.linspace(0.0, 1.0, N + 1)
    controls = {
        "u": 2.0 + 3.0 * source_times,
        "v": -4.0 + 0.5 * source_times,
    }

    refined = interpolate_controls_2n_minus_1(controls, N)
    comparison_times = np.linspace(0.0, 1.0, 2 * N)

    assert refined_grid_size(N) == 5
    assert refined["u"].shape == refined["v"].shape == (2 * N,)
    np.testing.assert_allclose(refined["u"], 2.0 + 3.0 * comparison_times)
    np.testing.assert_allclose(refined["v"], -4.0 + 0.5 * comparison_times)


def test_grid_metric_matches_the_requested_formula_and_tolerance():
    N = 4
    controls = {
        "u": jnp.asarray([[0.2, 0.5, 1.0, 0.7, 0.1]]),
        "v": jnp.asarray([[0.0, -0.3, 0.4, 0.2, -0.1]]),
    }
    physics = Physics(N, u_isbound=False, v_isbound=False)
    parameters = {
        "r_bg": jnp.asarray(-0.008716),
        "dt": jnp.asarray(1.0 / N),
        "u_smooth": jnp.asarray(0.0),
        "v_smooth": jnp.asarray(0.0),
        "u_sharp": jnp.asarray(0.0),
        "v_sharp": jnp.asarray(0.0),
    }
    _, base_objective, _ = physics.metrics_from_controls(
        {name: values[0] for name, values in controls.items()}, parameters
    )
    diagnostic = jax.device_get(
        grid_refinement_diagnostics(
            controls,
            jnp.asarray([base_objective]),
            N=N,
            r_bg=-0.008716,
            t_interval=1.0,
            tolerance=1e-2,
            y_floor=1e-12,
            use_jit=False,
        )
    )
    refined = float(diagnostic["best_objective_refined_grid"][0])
    expected = abs(refined - float(base_objective)) / max(abs(refined), 1e-12)

    assert diagnostic["best_grid_refinement_refined_N"][0] == 2 * N - 1
    assert diagnostic["best_grid_refinement_relative_error"][0] == pytest.approx(
        expected
    )
    assert bool(diagnostic["best_grid_refinement_passed"][0]) == (
        expected <= 1e-2
    )


def test_config_validates_and_defaults_to_the_loose_grid_tolerance():
    assert ResolvedConfig().grid_refinement_tol == 1e-2
    assert ResolvedConfig().grid_refinement_y_floor == 1e-12
    with pytest.raises(ValueError, match="grid_refinement_tol"):
        ResolvedConfig(grid_refinement_tol=0.0)
    with pytest.raises(ValueError, match="grid_refinement_y_floor"):
        ResolvedConfig(grid_refinement_y_floor=-1.0)


def test_runner_persists_queryable_grid_diagnostics(tmp_path):
    database = tmp_path / "future.sqlite3"
    document = make_document(
        name="grid_metric",
        parameters={
            "N": 4,
            "r_bg": -0.008716,
            "u_max": 10.0,
            "v_max": 20.0,
            "schedule": ((2, 1.0),),
            "block_size": 1,
            "grid_refinement_tol": 1e-3,
            "grid_refinement_y_floor": 1e-10,
        },
        runtime={
            "initialisations": 1,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "database": str(database),
        },
    )

    run_config(document, queue_id=111)
    row = Results(database).search(
        queue_id=111, best_grid_refinement_status="computed"
    )[0]

    assert row["best_grid_refinement_refined_N"] == 7
    assert row["best_grid_refinement_tolerance"] == 1e-3
    assert row["best_grid_refinement_y_floor"] == 1e-10
    expected = abs(
        row["best_objective_refined_grid"] - row["best_objective"]
    ) / max(abs(row["best_objective_refined_grid"]), 1e-10)
    assert row["best_grid_refinement_relative_error"] == pytest.approx(expected)
    assert row["best_grid_refinement_passed"] == (expected <= 1e-3)


def test_backfill_recreates_deleted_metrics_for_existing_runs(tmp_path):
    database = tmp_path / "past.sqlite3"
    document = make_document(
        name="past_grid_metric",
        parameters={"N": 4, "schedule": ((1, 1.0),), "block_size": 1},
        runtime={
            "initialisations": 2,
            "use_jit": False,
            "device": "cpu",
            "concurrent_workers": 1,
            "database": str(database),
        },
    )
    run_config(document, queue_id=222)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM physical_values WHERE name LIKE 'best_grid_refinement_%'"
        )
        connection.execute(
            "DELETE FROM physical_values WHERE name='best_objective_refined_grid'"
        )

    summary = backfill_database(database, batch_size=2)
    rows = Results(database).search(best_grid_refinement_status="computed")

    assert summary == {"pending": 2, "computed": 2, "unavailable": 0, "failed": 0}
    assert len(rows) == 2
    assert all(
        np.isfinite(row["best_grid_refinement_relative_error"]) for row in rows
    )
