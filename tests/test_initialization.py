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
