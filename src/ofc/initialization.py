"""Random control initialization.

All initialization choices live here so they are easy to audit.  The defaults
match the reference code: five Fourier modes, a ``1/m²`` coefficient envelope,
continuous raw-space RMS 0.3, an intensity baseline at 30% of ``u_max``, and a
zero detuning baseline.  Structured starting curves are intentionally absent.
"""

from __future__ import annotations

from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np


DEFAULT_NUM_MODES = 5
DEFAULT_RMS_AMPLITUDE = 0.3
DEFAULT_INTENSITY_FRACTION = 0.3
FOURIER_NAMESPACE = 0x46555249
CONTROL_IDS = {"u": 0, "v": 1}
QUERY_PERTURBATION_NAMESPACE = 0x51525950


def _coefficients(key, num_modes: int, rms_amplitude: float, dtype):
    modes = jnp.arange(1, num_modes + 1, dtype=dtype)
    sin_key, cos_key = jax.random.split(key)
    sin_values = jax.random.normal(sin_key, (num_modes,), dtype=dtype) / modes**2
    cos_values = jax.random.normal(cos_key, (num_modes,), dtype=dtype) / modes**2
    continuous_rms = jnp.sqrt(0.5 * jnp.sum(sin_values**2 + cos_values**2))
    scale = jnp.asarray(rms_amplitude, dtype=dtype) / jnp.maximum(
        continuous_rms, jnp.finfo(dtype).eps
    )
    return sin_values * scale, cos_values * scale


def random_fourier_controls(
    seed: int,
    count: int,
    N: int,
    *,
    num_modes: int = DEFAULT_NUM_MODES,
    rms_amplitude: float = DEFAULT_RMS_AMPLITUDE,
    intensity_fraction: float = DEFAULT_INTENSITY_FRACTION,
    dtype=jnp.float64,
    return_parameters: bool = False,
) -> dict[str, jax.Array] | tuple[dict[str, jax.Array], tuple[dict, ...]]:
    """Return reproducible unconstrained ``u``/``v`` optimizer controls.

    Coefficients are continuous functions of normalized time, so initialization
    index ``i`` describes the same curve for every compatible grid resolution.
    With ``return_parameters=True``, also return one JSON-compatible record per
    initialization containing the exact offsets and sampled sine/cosine
    coefficients used to construct both controls.
    """

    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("count must be a non-negative integer.")
    if isinstance(N, bool) or not isinstance(N, Integral) or N < 1:
        raise ValueError("N must be a positive integer.")
    if isinstance(num_modes, bool) or not isinstance(num_modes, Integral):
        raise ValueError("num_modes must be an integer between 3 and 6.")
    if not 3 <= num_modes <= 6:
        raise ValueError("num_modes must be between 3 and 6.")
    if not 0.0 < intensity_fraction < 1.0:
        raise ValueError("intensity_fraction must be in (0, 1).")
    if not isinstance(return_parameters, bool):
        raise ValueError("return_parameters must be a boolean.")

    grid = jnp.linspace(0.0, 1.0, int(N) + 1, dtype=dtype)
    modes = jnp.arange(1, num_modes + 1, dtype=dtype)
    phase = 2.0 * jnp.pi * modes[:, None] * grid[None, :]
    sin_phase, cos_phase = jnp.sin(phase), jnp.cos(phase)
    indices = jnp.arange(count, dtype=jnp.uint32)
    base_key = jax.random.fold_in(jax.random.PRNGKey(seed), FOURIER_NAMESPACE)
    offsets = {
        "u": jnp.log(intensity_fraction) - jnp.log1p(-intensity_fraction),
        "v": 0.0,
    }
    result = {}
    sampled = {}
    for name in ("u", "v"):
        if count == 0:
            result[name] = jnp.empty((0, int(N) + 1), dtype=dtype)
            sampled[name] = (
                jnp.empty((0, num_modes), dtype=dtype),
                jnp.empty((0, num_modes), dtype=dtype),
            )
            continue
        control_key = jax.random.fold_in(base_key, CONTROL_IDS[name])
        keys = jax.vmap(lambda index: jax.random.fold_in(control_key, index))(indices)
        sin_coefficients, cos_coefficients = jax.vmap(
            lambda key: _coefficients(key, num_modes, rms_amplitude, dtype)
        )(keys)
        sampled[name] = (sin_coefficients, cos_coefficients)
        # Keep the reduction local to each curve. A batch-sized einsum can
        # select a different floating-point reduction and make index zero vary
        # by a few ulps when the requested initialization count changes.
        result[name] = jax.vmap(
            lambda sin_values, cos_values: (
                jnp.asarray(offsets[name], dtype=dtype)
                + jnp.sum(
                    sin_values[:, None] * sin_phase
                    + cos_values[:, None] * cos_phase,
                    axis=0,
                )
            )
        )(sin_coefficients, cos_coefficients)
    if not return_parameters:
        return result

    host_coefficients = {
        name: tuple(np.asarray(values) for values in sampled[name])
        for name in ("u", "v")
    }
    host_offsets = {
        name: float(np.asarray(jnp.asarray(offsets[name], dtype=dtype)))
        for name in ("u", "v")
    }
    parameter_records = tuple(
        {
            "fourier_u_offset": host_offsets["u"],
            "fourier_u_sin_coefficients": host_coefficients["u"][0][index].tolist(),
            "fourier_u_cos_coefficients": host_coefficients["u"][1][index].tolist(),
            "fourier_v_offset": host_offsets["v"],
            "fourier_v_sin_coefficients": host_coefficients["v"][0][index].tolist(),
            "fourier_v_cos_coefficients": host_coefficients["v"][1][index].tolist(),
        }
        for index in range(count)
    )
    return result, parameter_records


def stored_controls_to_raw(
    controls,
    N: int,
    *,
    u_max: float,
    v_max: float,
    u_isbound: bool,
    v_isbound: bool,
    perturbation_level: float = 0.0,
    perturbation_seed: int | None = None,
    dtype=jnp.float64,
) -> dict[str, jax.Array]:
    """Resample stored bounded controls and map them into current raw space.

    Resampling uses normalized time, so stored controls may come from a
    different ``N`` or time interval. A positive perturbation level adds a
    reproducible continuous five-mode Fourier curve to the resampled bounded
    controls before clipping. The level is the continuous RMS fraction of
    ``u_max`` and ``v_max`` respectively. Values outside the configured bounds
    are clipped before applying the inverse sigmoid/tanh maps.
    """

    if isinstance(N, bool) or not isinstance(N, Integral) or N < 1:
        raise ValueError("N must be a positive integer.")
    if (
        isinstance(perturbation_level, bool)
        or not isinstance(perturbation_level, Real)
        or not np.isfinite(perturbation_level)
        or not 0.0 <= float(perturbation_level) <= 1.0
    ):
        raise ValueError("perturbation_level must be finite and in [0, 1].")
    perturbation_level = float(perturbation_level)
    if perturbation_level > 0.0 and (
        isinstance(perturbation_seed, bool)
        or not isinstance(perturbation_seed, Integral)
        or perturbation_seed < 0
    ):
        raise ValueError(
            "perturbation_seed must be a non-negative integer for a perturbed start."
        )
    arrays = {}
    for name in ("u", "v"):
        values = np.asarray(controls[name], dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError(f"Stored {name} control must be a one-dimensional array.")
        if not np.isfinite(values).all():
            raise ValueError(f"Stored {name} control must contain only finite values.")
        old_grid = jnp.linspace(0.0, 1.0, values.size, dtype=dtype)
        new_grid = jnp.linspace(0.0, 1.0, int(N) + 1, dtype=dtype)
        arrays[name] = jnp.interp(
            new_grid,
            old_grid,
            jnp.asarray(values, dtype=dtype),
        )

    if perturbation_level > 0.0:
        grid = jnp.linspace(0.0, 1.0, int(N) + 1, dtype=dtype)
        modes = jnp.arange(1, DEFAULT_NUM_MODES + 1, dtype=dtype)
        phase = 2.0 * jnp.pi * modes[:, None] * grid[None, :]
        base_key = jax.random.fold_in(
            jax.random.PRNGKey(int(perturbation_seed) & 0xFFFFFFFF),
            QUERY_PERTURBATION_NAMESPACE,
        )
        scales = {"u": float(u_max), "v": float(v_max)}
        for name in ("u", "v"):
            key = jax.random.fold_in(base_key, CONTROL_IDS[name])
            sin_values, cos_values = _coefficients(
                key,
                DEFAULT_NUM_MODES,
                perturbation_level,
                dtype,
            )
            normalized_curve = jnp.sum(
                sin_values[:, None] * jnp.sin(phase)
                + cos_values[:, None] * jnp.cos(phase),
                axis=0,
            )
            arrays[name] = arrays[name] + scales[name] * normalized_curve
        arrays["u"] = jnp.clip(arrays["u"], 0.0, float(u_max))
        arrays["v"] = jnp.clip(arrays["v"], -float(v_max), float(v_max))

    # A modest interior clip avoids infinite raw values while changing an
    # exactly saturated stored control by only one part in 10^7 of its cap.
    interior = jnp.asarray(1e-7, dtype=dtype)
    if u_isbound:
        fraction = jnp.clip(arrays["u"] / u_max, interior, 1.0 - interior)
        arrays["u"] = jnp.log(fraction) - jnp.log1p(-fraction)
    if v_isbound:
        fraction = jnp.clip(
            arrays["v"] / v_max,
            -1.0 + interior,
            1.0 - interior,
        )
        arrays["v"] = jnp.arctanh(fraction)
    return arrays
