import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ofc.physics import Physics


def test_inelastic_physics_matches_signed_background_golden_values():
    """Golden values lock the fixed-time-scale scattering convention."""

    physics = Physics(4, dtype=jnp.float64)
    controls = {
        "u": jnp.asarray([2.0, 3.0, 5.0, 7.0, 11.0]),
        "v": jnp.asarray([-3.0, -1.0, 0.0, 1.0, 4.0]),
    }
    r_bg = -0.75
    scattering = physics.dimensionless_scattering_length(controls, r_bg)
    eta = physics.pair_amplitude(scattering, 0.25)
    objective = physics.molecular_objective(scattering, eta, 0.25, r_bg)
    penalty = physics.smoothness_penalty(
        controls,
        {
            "dt": 0.25,
            "t_interval": 1.0,
            "u_max": 11.0,
            "v_max": 4.0,
            "u_smooth": 10**-1.5,
            "v_smooth": 10**-1.5,
            "u_sharp": 0.0,
            "v_sharp": 0.0,
        },
    )

    np.testing.assert_allclose(
        np.asarray(scattering.real),
        [
            -0.45297029702970293,
            -0.19615384615384612,
            -0.007425742574257432,
            0.11896551724137922,
            0.42258883248730966,
        ],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(eta.real),
        [
            5.692192629771605,
            3.4213796475045877,
            0.6639507014955301,
            -1.3788139301155655,
            -2.8217028198922027,
        ],
        rtol=1e-13,
    )
    assert float(objective) == pytest.approx(3.4020230916796876, rel=1e-13)
    assert float(penalty) == pytest.approx(5.059644256269406, rel=1e-13)


def test_smoothness_uses_absolute_controls_and_dimensionless_timestep():
    physics = Physics(2, dtype=jnp.float64)
    controls = {
        "u": jnp.asarray([0.0, 2.0, 4.0]),
        "v": jnp.asarray([-3.0, 0.0, 3.0]),
    }
    parameters = {
        "dt": 2.0,
        "t_interval": 4.0,
        "u_max": 4.0,
        "v_max": 6.0,
        "u_smooth": 0.1,
        "v_smooth": 0.2,
        "u_sharp": 0.0,
        "v_sharp": 0.0,
    }
    scaled_controls = {"u": controls["u"] * 3.0, "v": controls["v"] * 5.0}
    scaled_parameters = {
        **parameters,
        "u_max": 12.0,
        "v_max": 30.0,
    }
    longer_timestep_parameters = {
        **scaled_parameters,
        "dt": 7.0,
        "t_interval": 14.0,
    }

    penalty = physics.smoothness_penalty(controls, parameters)
    scaled_penalty = physics.smoothness_penalty(
        scaled_controls, scaled_parameters
    )
    longer_timestep_penalty = physics.smoothness_penalty(
        scaled_controls, longer_timestep_parameters
    )

    assert float(penalty) == pytest.approx(2.2)
    assert float(scaled_penalty) == pytest.approx(48.6)
    assert float(longer_timestep_penalty) == pytest.approx(
        float(scaled_penalty) * 2.0 / 7.0
    )


def test_sharpness_penalizes_second_derivatives_with_dt_cubed():
    physics = Physics(2, dtype=jnp.float64)
    controls = {
        "u": jnp.asarray([0.0, 1.0, 4.0]),
        "v": jnp.asarray([0.0, 2.0, 1.0]),
    }
    penalty = physics.smoothness_penalty(
        controls,
        {
            "dt": 2.0,
            "u_smooth": 0.0,
            "v_smooth": 0.0,
            "u_sharp": 0.3,
            "v_sharp": 0.4,
        },
    )

    assert float(penalty) == pytest.approx(0.6)


def test_zero_sharpness_omits_second_differences_from_the_compiled_program():
    controls = {
        "u": jnp.arange(5.0),
        "v": jnp.arange(5.0),
    }
    parameters = {
        "dt": 0.25,
        "u_smooth": 1e-3,
        "v_smooth": 1e-3,
        "u_sharp": 0.0,
        "v_sharp": 0.0,
    }
    inactive = Physics(
        4,
        dtype=jnp.float64,
        u_sharp_active=False,
        v_sharp_active=False,
    )
    active = Physics(
        4,
        dtype=jnp.float64,
        u_sharp_active=True,
        v_sharp_active=True,
    )

    inactive_jaxpr = jax.make_jaxpr(inactive.smoothness_penalty)(
        controls, parameters
    )
    active_jaxpr = jax.make_jaxpr(active.smoothness_penalty)(controls, parameters)
    inactive_differences = sum(
        equation.primitive.name == "pjit" for equation in inactive_jaxpr.jaxpr.eqns
    )
    active_differences = sum(
        equation.primitive.name == "pjit" for equation in active_jaxpr.jaxpr.eqns
    )

    assert inactive_differences == 2
    assert active_differences == 4


def test_metrics_and_gradient_are_finite_under_jit():
    physics = Physics(4, dtype=jnp.float64)
    raw = {"u": jnp.zeros(5), "v": jnp.zeros(5)}
    parameters = {
        "u_max": jnp.asarray(10.0),
        "v_max": jnp.asarray(10.0),
        "dt": jnp.asarray(0.25),
        "t_interval": jnp.asarray(1.0),
        "u_smooth": jnp.asarray(1e-3),
        "v_smooth": jnp.asarray(1e-3),
        "u_sharp": jnp.asarray(1e-4),
        "v_sharp": jnp.asarray(1e-4),
        "r_bg": jnp.asarray(1.0),
    }
    metrics = physics.metrics(raw, parameters)
    value, gradient = jax.jit(
        jax.value_and_grad(lambda value: physics.metrics(value, parameters)[0])
    )(raw)
    assert float(metrics[0]) == pytest.approx(float(metrics[1] - metrics[2]))
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(values)).all() for values in gradient.values())


def test_r_bg_sets_background_magnitude_sign_and_inelastic_convention():
    physics = Physics(4, dtype=jnp.float64)
    raw = {
        "u": jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
        "v": jnp.asarray([1.0, -0.5, 0.0, 0.5, -1.0]),
    }
    common = {
        "u_max": jnp.asarray(10.0),
        "v_max": jnp.asarray(10.0),
        "dt": jnp.asarray(0.25),
        "t_interval": jnp.asarray(1.0),
        "u_smooth": jnp.asarray(1e-3),
        "v_smooth": jnp.asarray(1e-3),
        "u_sharp": jnp.asarray(1e-4),
        "v_sharp": jnp.asarray(1e-4),
    }
    controls = physics.bounded_controls(raw, common)
    positive = physics.dimensionless_scattering_length(controls, 0.75)
    negative = physics.dimensionless_scattering_length(controls, -1.5)

    for r_bg, actual in ((0.75, positive), (-1.5, negative)):
        sign = np.sign(r_bg)
        expected = r_bg * (
            1.0
            + sign
            * np.asarray(controls["u"])
            / (
                -sign * np.asarray(controls["u"])
                - np.asarray(controls["v"])
                + 0.5j
            )
        )
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-13)

    # At zero optical width the scattering length is exactly r_bg. The signed
    # optical width keeps the absorptive imaginary part negative for either
    # background sign under the +i/2 convention.
    zero_controls = {"u": jnp.zeros(2), "v": jnp.asarray([-3.0, 4.0])}
    np.testing.assert_allclose(
        physics.dimensionless_scattering_length(zero_controls, 0.75), 0.75
    )
    np.testing.assert_allclose(
        physics.dimensionless_scattering_length(zero_controls, -1.5), -1.5
    )
    assert np.all(np.asarray(positive.imag) < 0.0)
    assert np.all(np.asarray(negative.imag) < 0.0)
    assert np.all(np.asarray(jnp.imag(1.0 / positive)) > 0.0)
    assert np.all(np.asarray(jnp.imag(1.0 / negative)) > 0.0)

    positive_metrics = physics.metrics(raw, {**common, "r_bg": 0.75})
    negative_metrics = physics.metrics(raw, {**common, "r_bg": -1.5})
    assert float(positive_metrics[1]) > 0.0
    assert float(negative_metrics[1]) > 0.0
    assert not np.allclose(
        np.asarray(negative_metrics, dtype=float),
        np.asarray(positive_metrics, dtype=float),
    )


def test_unit_positive_background_matches_working_plus_i_half_convention():
    physics = Physics(2, dtype=jnp.float64)
    controls = {
        "u": jnp.asarray([1.0, 3.0, 7.0]),
        "v": jnp.asarray([-2.0, 0.5, 4.0]),
    }

    actual = physics.dimensionless_scattering_length(controls, 1.0)
    expected = 1.0 + controls["u"] / (
        -controls["u"] - controls["v"] + 0.5j
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-13)


@pytest.mark.parametrize("r_bg", [-1.5, 0.75])
def test_r_bg_metrics_and_gradients_are_finite_under_jit(r_bg):
    physics = Physics(4, dtype=jnp.float64)
    raw = {"u": jnp.zeros(5), "v": jnp.zeros(5)}
    parameters = {
        "u_max": jnp.asarray(10.0),
        "v_max": jnp.asarray(10.0),
        "dt": jnp.asarray(0.25),
        "t_interval": jnp.asarray(1.0),
        "u_smooth": jnp.asarray(1e-3),
        "v_smooth": jnp.asarray(1e-3),
        "u_sharp": jnp.asarray(1e-4),
        "v_sharp": jnp.asarray(1e-4),
        "r_bg": r_bg,
    }
    value, gradient = jax.jit(
        jax.value_and_grad(lambda value: physics.metrics(value, parameters)[0])
    )(raw)

    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(values)).all() for values in gradient.values())
