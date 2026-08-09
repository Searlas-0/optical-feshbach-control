import jax.numpy as jnp
import numpy as np

from ofc.optimization import (
    BatchedAdamOptimizer,
    BatchedLBFGSOptimizer,
    BatchedPeakRefinementOptimizer,
    normalised_control_rms,
    projected_gradient_rms,
)


def test_peak_refinement_is_monotone_feasible_and_adapts_step_size():
    physics = Physics(3, dtype=jnp.float64)
    optimizer = BatchedPeakRefinementOptimizer(
        physics,
        block_size=2,
        max_linesearch_steps=20,
        use_jit=True,
    )
    raw = {
        "u": jnp.zeros((2, 4), dtype=jnp.float64),
        "v": jnp.asarray(
            [[0.0, 0.1, -0.1, 0.0], [0.2, -0.2, 0.1, -0.1]],
            dtype=jnp.float64,
        ),
    }
    parameters = {
        "r_bg": jnp.asarray([1.0, 1.0]),
        "u_max": jnp.asarray([10.0, 10.0]),
        "v_max": jnp.asarray([10.0, 10.0]),
        "dt": jnp.asarray([1.0 / 3.0, 1.0 / 3.0]),
        "t_interval": jnp.asarray([1.0, 1.0]),
        "u_smooth": jnp.asarray([1e-3, 1e-3]),
        "v_smooth": jnp.asarray([1e-3, 1e-3]),
        "u_sharp": jnp.asarray([1e-4, 1e-4]),
        "v_sharp": jnp.asarray([1e-4, 1e-4]),
        "peak_initial_step_size": jnp.asarray([0.1, 0.1]),
        "peak_min_step_size": jnp.asarray([1e-12, 1e-12]),
        "peak_max_step_size": jnp.asarray([0.2, 0.2]),
        "peak_backtracking_factor": jnp.asarray([0.5, 0.5]),
        "peak_step_growth": jnp.asarray([1.5, 1.5]),
        "peak_armijo": jnp.asarray([1e-4, 1e-4]),
    }
    state = optimizer.initialise(raw, parameters)
    initial_scores = np.asarray(state.best_score)
    state, output = optimizer.run_stage(
        state,
        parameters,
        steps=8,
        start_step=0,
    )

    history = np.asarray(output["score_history"])
    assert int(output["actual_steps"]) == 8
    assert np.isfinite(history).all()
    assert np.all(np.diff(history, axis=1) >= -1e-12)
    assert np.all(np.asarray(state.best_score) >= initial_scores)
    assert np.all(np.asarray(state.step_size) >= 1e-12)
    assert np.all(np.asarray(state.step_size) <= 0.2)
    for name, values in state.normalised.items():
        values = np.asarray(values)
        if name == "u":
            assert np.all((0.0 <= values) & (values <= 1.0))
        else:
            assert np.all((-1.0 <= values) & (values <= 1.0))
from ofc.physics import Physics


def test_jitted_adam_stages_keep_state_and_capture_checkpoints():
    physics = Physics(3, dtype=jnp.float64)
    optimizer = BatchedAdamOptimizer(physics, block_size=1, use_jit=True)
    raw = {"u": jnp.zeros((1, 4)), "v": jnp.zeros((1, 4))}
    parameters = {
        "r_bg": jnp.asarray([1.0]),
        "u_max": jnp.asarray([10.0]),
        "v_max": jnp.asarray([10.0]),
        "dt": jnp.asarray([1.0 / 3.0]),
        "t_interval": jnp.asarray([1.0]),
        "u_smooth": jnp.asarray([1e-3]),
        "v_smooth": jnp.asarray([1e-3]),
        "u_sharp": jnp.asarray([1e-4]),
        "v_sharp": jnp.asarray([1e-4]),
        "adam_beta1": jnp.asarray([0.9]),
        "adam_beta2": jnp.asarray([0.999]),
        "adam_eps": jnp.asarray([1e-8]),
        "adam_learning_rate": jnp.asarray([1e-2]),
    }
    state = optimizer.initialise(raw, parameters)
    state, first = optimizer.run_stage(
        state,
        parameters,
        steps=1,
        start_step=0,
        learning_rate=jnp.asarray([1e-2]),
    )
    checkpoint_state = state
    state, second = optimizer.run_stage(
        state,
        parameters,
        steps=2,
        start_step=1,
        learning_rate=jnp.asarray([5e-3]),
    )
    resumed_state = optimizer.initialise(
        checkpoint_state.raw,
        parameters,
        count=checkpoint_state.count,
        first_moment=checkpoint_state.first_moment,
        second_moment=checkpoint_state.second_moment,
    )
    resumed_state, _ = optimizer.run_stage(
        resumed_state,
        parameters,
        steps=2,
        start_step=1,
        learning_rate=jnp.asarray([5e-3]),
    )

    assert int(state.count[0]) == 3
    assert first["score_history"].shape == (1, 2)
    assert second["score_history"].shape == (1, 3)
    assert int(first["actual_steps"]) == 1
    assert int(second["actual_steps"]) == 2
    assert first["stability_values"] == {}
    assert first["best_stability_values"] == {}
    assert np.isfinite(np.asarray(state.best_score)).all()
    assert int(state.best_step[0]) <= 3
    for moment in (state.best_first_moment, state.best_second_moment):
        assert set(moment) == {"u", "v"}
        assert all(values.shape == (1, 4) for values in moment.values())
        assert all(np.isfinite(np.asarray(values)).all() for values in moment.values())
    if int(state.best_step[0]) == int(state.count[0]):
        for name in ("u", "v"):
            np.testing.assert_allclose(
                state.best_first_moment[name], state.first_moment[name]
            )
            np.testing.assert_allclose(
                state.best_second_moment[name], state.second_moment[name]
            )
    for name in ("u", "v"):
        np.testing.assert_allclose(resumed_state.raw[name], state.raw[name])
        np.testing.assert_allclose(
            resumed_state.first_moment[name], state.first_moment[name]
        )
        np.testing.assert_allclose(
            resumed_state.second_moment[name], state.second_moment[name]
        )
    np.testing.assert_array_equal(resumed_state.count, state.count)

    checkpoint_score, checkpoint_objective, checkpoint_penalty = physics.metrics(
        {name: values[0] for name, values in state.best_raw.items()},
        {name: values[0] for name, values in parameters.items()},
    )
    np.testing.assert_allclose(state.best_score[0], checkpoint_score)
    np.testing.assert_allclose(state.best_objective[0], checkpoint_objective)
    np.testing.assert_allclose(state.best_penalty[0], checkpoint_penalty)


def test_jitted_lbfgs_stages_keep_state_and_capture_checkpoints():
    physics = Physics(3, dtype=jnp.float64)
    optimizer = BatchedLBFGSOptimizer(
        physics,
        block_size=1,
        history_size=3,
        max_linesearch_steps=8,
        use_jit=True,
    )
    raw = {"u": jnp.zeros((1, 4)), "v": jnp.zeros((1, 4))}
    parameters = {
        "r_bg": jnp.asarray([1.0]),
        "u_max": jnp.asarray([10.0]),
        "v_max": jnp.asarray([10.0]),
        "dt": jnp.asarray([1.0 / 3.0]),
        "t_interval": jnp.asarray([1.0]),
        "u_smooth": jnp.asarray([1e-3]),
        "v_smooth": jnp.asarray([1e-3]),
        "u_sharp": jnp.asarray([1e-4]),
        "v_sharp": jnp.asarray([1e-4]),
        "lbfgs_tolerance": jnp.asarray([1e-8]),
    }
    state = optimizer.initialise(raw, parameters)
    initial_raw = {
        name: np.asarray(values).copy() for name, values in state.raw.items()
    }
    initial_score = np.asarray(state.best_score).copy()
    state, first = optimizer.run_stage(state, parameters, steps=1, start_step=0)
    state, second = optimizer.run_stage(state, parameters, steps=2, start_step=1)

    assert int(state.count) == 3
    assert first["score_history"].shape == (1, 2)
    assert second["score_history"].shape == (1, 3)
    assert int(first["actual_steps"]) == 1
    assert int(second["actual_steps"]) == 2
    assert second["stability_values"] == {}
    assert second["best_stability_values"] == {}
    assert np.isfinite(np.asarray(state.best_score)).all()
    assert np.isfinite(np.asarray(state.solver_state.error)).all()
    assert any(
        not np.allclose(np.asarray(state.raw[name]), initial_raw[name])
        for name in state.raw
    )
    assert np.all(np.asarray(state.best_score) > initial_score)
    assert int(state.best_step[0]) <= 3
    checkpoint_score, checkpoint_objective, checkpoint_penalty = physics.metrics(
        {name: values[0] for name, values in state.best_raw.items()},
        {name: values[0] for name, values in parameters.items()},
    )
    np.testing.assert_allclose(state.best_score[0], checkpoint_score)
    np.testing.assert_allclose(state.best_objective[0], checkpoint_objective)
    np.testing.assert_allclose(state.best_penalty[0], checkpoint_penalty)


def test_projected_gradient_and_control_rms_use_normalised_feasible_coordinates():
    physics = Physics(1, dtype=jnp.float64)
    controls = {
        "u": jnp.asarray([0.5, 0.2]),
        "v": jnp.asarray([0.0, 0.5]),
    }
    gradients = {
        "u": jnp.asarray([1.0, -1.0]),
        "v": jnp.asarray([2.0, -2.0]),
    }

    value = projected_gradient_rms(
        physics, controls, gradients, test_step=1.0
    )
    np.testing.assert_allclose(value, np.sqrt(2.14 / 4.0))
    np.testing.assert_allclose(
        normalised_control_rms(
            controls,
            {"u": controls["u"] + 0.1, "v": controls["v"] - 0.1},
        ),
        0.1,
    )


def test_adam_auto_halts_inside_compiled_stage_after_three_stable_blocks():
    physics = Physics(3, dtype=jnp.float64)
    optimizer = BatchedAdamOptimizer(
        physics,
        block_size=2,
        score_tolerance=True,
        u_tolerance=True,
        v_tolerance=True,
        projected_gradient_tolerance=True,
        auto_halt=True,
        use_jit=True,
    )
    raw = {
        "u": jnp.zeros((2, 4), dtype=jnp.float64),
        "v": jnp.zeros((2, 4), dtype=jnp.float64),
    }
    parameters = {
        "r_bg": jnp.asarray([1.0, 1.0]),
        "u_max": jnp.asarray([10.0, 10.0]),
        "v_max": jnp.asarray([10.0, 10.0]),
        "dt": jnp.asarray([1.0 / 3.0, 1.0 / 3.0]),
        "t_interval": jnp.asarray([1.0, 1.0]),
        "u_smooth": jnp.asarray([1e-3, 1e-3]),
        "v_smooth": jnp.asarray([1e-3, 1e-3]),
        "u_sharp": jnp.asarray([0.0, 0.0]),
        "v_sharp": jnp.asarray([0.0, 0.0]),
        "adam_beta1": jnp.asarray([0.9, 0.9]),
        "adam_beta2": jnp.asarray([0.999, 0.999]),
        "adam_eps": jnp.asarray([1e-8, 1e-8]),
        "adam_learning_rate": jnp.asarray([1e-2, 1e-2]),
        "J_tol": jnp.asarray([1e30, 1e30]),
        "u_tol": jnp.asarray([1e30, 1e30]),
        "v_tol": jnp.asarray([1e30, 1e30]),
        "projected_gradient_tol": jnp.asarray([1e30, 1e30]),
        "projected_gradient_alpha": jnp.asarray([1.0, 1.0]),
    }
    state = optimizer.initialise(raw, parameters)
    state, output = optimizer.run_stage(
        state,
        parameters,
        steps=20,
        start_step=0,
        learning_rate=parameters["adam_learning_rate"],
    )

    assert int(output["actual_steps"]) == 6
    assert bool(output["halted"]) is True
    np.testing.assert_array_equal(
        output["stability_consecutive_blocks"], np.asarray([3, 3])
    )
    assert int(state.count[0]) == 6
    assert np.isnan(np.asarray(output["score_history"])[:, 7:]).all()
    assert set(output["stability_values"]) == {
        "score_tolerance",
        "u_tolerance",
        "v_tolerance",
        "control_tolerance",
        "projected_gradient_tolerance",
    }
    assert set(output["best_stability_values"]) == {
        "best_projected_gradient_rms"
    }
    assert np.isfinite(
        np.asarray(output["best_stability_values"]["best_projected_gradient_rms"])
    ).all()
