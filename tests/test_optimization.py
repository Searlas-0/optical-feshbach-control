import jax.numpy as jnp
import numpy as np

from ofc.optimization import BatchedAdamOptimizer, BatchedLBFGSOptimizer
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
    state, second = optimizer.run_stage(
        state,
        parameters,
        steps=2,
        start_step=1,
        learning_rate=jnp.asarray([5e-3]),
    )

    assert int(state.count) == 3
    assert first["score_history"].shape == (1, 2)
    assert second["score_history"].shape == (1, 3)
    assert first["checkpoint_raw"]["u"].shape == (1, 1, 4)
    assert second["checkpoint_raw"]["u"].shape == (2, 1, 4)
    assert np.isfinite(np.asarray(state.best_score)).all()
    assert int(state.best_step[0]) <= 3

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
    assert second["checkpoint_raw"]["u"].shape == (2, 1, 4)
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
