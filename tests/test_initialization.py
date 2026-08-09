import math

import jax.numpy as jnp
import numpy as np
import pytest

from ofc.initialization import random_fourier_controls, stored_controls_to_raw
from ofc.physics import Physics


def test_random_fourier_curves_are_reproducible_and_grid_independent():
    coarse = random_fourier_controls(1234, 3, 100, dtype=jnp.float64)
    fine = random_fourier_controls(1234, 3, 400, dtype=jnp.float64)
    short = random_fourier_controls(1234, 1, 100, dtype=jnp.float64)
    offsets = {"u": math.log(0.3 / 0.7), "v": 0.0}

    for name in ("u", "v"):
        np.testing.assert_array_equal(np.asarray(short[name]), np.asarray(coarse[name][:1]))
        np.testing.assert_array_equal(np.asarray(coarse[name]), np.asarray(fine[name][:, ::4]))
        for curve in np.asarray(coarse[name]):
            oscillation = curve - offsets[name]
            assert np.trapezoid(oscillation, dx=0.01) == pytest.approx(0.0, abs=2e-7)
            rms = math.sqrt(np.trapezoid(oscillation**2, dx=0.01))
            assert rms == pytest.approx(0.3, abs=2e-6)


def test_random_fourier_controls_can_return_the_exact_sampled_parameters():
    controls, parameters = random_fourier_controls(
        1234,
        2,
        100,
        dtype=jnp.float64,
        return_parameters=True,
    )
    grid = jnp.linspace(0.0, 1.0, 101, dtype=jnp.float64)
    modes = jnp.arange(1, 6, dtype=jnp.float64)
    phase = 2.0 * jnp.pi * modes[:, None] * grid[None, :]

    assert len(parameters) == 2
    for index, record in enumerate(parameters):
        for name in ("u", "v"):
            sine = jnp.asarray(record[f"fourier_{name}_sin_coefficients"])
            cosine = jnp.asarray(record[f"fourier_{name}_cos_coefficients"])
            reconstructed = (
                record[f"fourier_{name}_offset"]
                + jnp.sum(
                    sine[:, None] * jnp.sin(phase)
                    + cosine[:, None] * jnp.cos(phase),
                    axis=0,
                )
            )
            np.testing.assert_allclose(
                reconstructed,
                controls[name][index],
                rtol=1e-14,
                atol=1e-14,
            )


@pytest.mark.parametrize("modes", [2, 7])
def test_mode_count_is_restricted(modes):
    with pytest.raises(ValueError, match="between 3 and 6"):
        random_fourier_controls(1, 1, 4, num_modes=modes)


def test_stored_controls_are_resampled_and_inverted_for_current_bounds():
    stored = {
        "u": np.asarray([1.0, 3.0, 5.0]),
        "v": np.asarray([-2.0, 0.0, 2.0]),
    }
    raw = stored_controls_to_raw(
        stored,
        4,
        u_max=10.0,
        v_max=5.0,
        u_isbound=True,
        v_isbound=True,
        dtype=jnp.float64,
    )
    restored = Physics(4, dtype=jnp.float64).bounded_controls(
        raw,
        {"u_max": 10.0, "v_max": 5.0},
    )

    np.testing.assert_allclose(restored["u"], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(restored["v"], [-2.0, -1.0, 0.0, 1.0, 2.0])


def test_stored_controls_are_perturbed_in_bounded_space_and_clipped():
    stored = {
        "u": np.full(5, 9.8),
        "v": np.asarray([-4.9, -2.0, 0.0, 2.0, 4.9]),
    }
    arguments = {
        "N": 12,
        "u_max": 10.0,
        "v_max": 5.0,
        "u_isbound": True,
        "v_isbound": True,
        "perturbation_level": 0.2,
        "perturbation_seed": 1234,
        "dtype": jnp.float64,
    }
    first = stored_controls_to_raw(stored, **arguments)
    second = stored_controls_to_raw(stored, **arguments)
    different = stored_controls_to_raw(
        stored,
        **{**arguments, "perturbation_seed": 5678},
    )
    physics = Physics(12, dtype=jnp.float64)
    bounded = physics.bounded_controls(first, {"u_max": 10.0, "v_max": 5.0})
    bounded_again = physics.bounded_controls(
        second, {"u_max": 10.0, "v_max": 5.0}
    )
    bounded_different = physics.bounded_controls(
        different, {"u_max": 10.0, "v_max": 5.0}
    )

    np.testing.assert_allclose(bounded["u"], bounded_again["u"])
    np.testing.assert_allclose(bounded["v"], bounded_again["v"])
    assert not np.allclose(bounded["u"], bounded_different["u"])
    assert not np.allclose(bounded["v"], bounded_different["v"])
    assert np.asarray(bounded["u"]).min() >= 0.0
    assert np.asarray(bounded["u"]).max() <= 10.0
    assert np.asarray(bounded["v"]).min() >= -5.0
    assert np.asarray(bounded["v"]).max() <= 5.0


def test_perturbed_stored_controls_require_a_valid_seed():
    stored = {"u": np.ones(3), "v": np.zeros(3)}

    with pytest.raises(ValueError, match="perturbation_seed"):
        stored_controls_to_raw(
            stored,
            2,
            u_max=10.0,
            v_max=5.0,
            u_isbound=True,
            v_isbound=True,
            perturbation_level=0.1,
        )
