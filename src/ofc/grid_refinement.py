"""Nested-grid objective diagnostics for stored physical controls."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import jax
import jax.numpy as jnp

from .physics import Physics


def refined_grid_size(N: int) -> int:
    """Return the requested comparison grid size ``2*N - 1``."""

    if isinstance(N, bool) or not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer of at least two.")
    return 2 * N - 1


def interpolate_controls_2n_minus_1(
    controls: Mapping[str, jax.Array], N: int, *, dtype=jnp.float64
) -> dict[str, jax.Array]:
    """Linearly interpolate N+1 samples onto a ``2*N-1`` run grid.

    ``Physics.N`` counts time intervals, so a base control has N+1 samples and
    the requested comparison run has ``(2*N-1)+1 = 2*N`` samples. Interpolation
    uses normalized time in [0, 1].
    """

    comparison_N = refined_grid_size(N)
    source_times = jnp.linspace(0.0, 1.0, N + 1, dtype=dtype)
    comparison_times = jnp.linspace(0.0, 1.0, comparison_N + 1, dtype=dtype)
    output = {}
    for name in ("u", "v"):
        values = jnp.asarray(controls[name], dtype=dtype)
        squeezed = values.ndim == 1
        if squeezed:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != N + 1:
            raise ValueError(
                f"controls[{name!r}] must have trailing shape ({N + 1},)."
            )
        interpolated = jax.vmap(
            lambda member: jnp.interp(comparison_times, source_times, member)
        )(values)
        output[name] = interpolated[0] if squeezed else interpolated
    return output


@lru_cache(maxsize=None)
def _objective_evaluator(comparison_N: int, dtype_name: str, use_jit: bool):
    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    physics = Physics(
        comparison_N,
        dtype=dtype,
        u_isbound=False,
        v_isbound=False,
        u_sharp_active=False,
        v_sharp_active=False,
    )

    def objective_one(u, v, background, duration):
        member_controls = {"u": u, "v": v}
        scattering_length = physics.dimensionless_scattering_length(
            member_controls, background
        )
        dt = duration / comparison_N
        pair_amplitude = physics.pair_amplitude(scattering_length, dt)
        return physics.molecular_objective(
            scattering_length, pair_amplitude, dt, background
        )

    evaluate = jax.vmap(objective_one)
    return jax.jit(evaluate) if use_jit else evaluate


def grid_refinement_diagnostics(
    controls: Mapping[str, jax.Array],
    base_objectives,
    *,
    N: int,
    r_bg,
    t_interval,
    tolerance,
    y_floor,
    dtype=jnp.float64,
    use_jit: bool = True,
) -> dict[str, jax.Array]:
    """Evaluate Y on ``2*N-1`` and return the configured relative error.

    The error is ``abs(Y_2N-1 - Y_N) / max(abs(Y_2N-1), Y_floor)``. Controls
    are the physical best controls, not raw optimizer coordinates.
    """

    comparison_N = refined_grid_size(N)
    refined_controls = interpolate_controls_2n_minus_1(controls, N, dtype=dtype)
    base_objectives = jnp.atleast_1d(jnp.asarray(base_objectives, dtype=dtype))
    member_count = base_objectives.shape[0]
    if any(
        values.ndim != 2 or values.shape[0] != member_count
        for values in refined_controls.values()
    ):
        raise ValueError("controls and base_objectives must have the same member axis.")
    r_bg = jnp.broadcast_to(jnp.asarray(r_bg, dtype=dtype), (member_count,))
    t_interval = jnp.broadcast_to(
        jnp.asarray(t_interval, dtype=dtype), (member_count,)
    )
    tolerance = jnp.broadcast_to(
        jnp.asarray(tolerance, dtype=dtype), (member_count,)
    )
    y_floor = jnp.broadcast_to(jnp.asarray(y_floor, dtype=dtype), (member_count,))
    dtype_name = "float64" if dtype == jnp.float64 else "float32"
    evaluate = _objective_evaluator(comparison_N, dtype_name, use_jit)
    refined_objectives = evaluate(
        refined_controls["u"], refined_controls["v"], r_bg, t_interval
    )
    relative_error = jnp.abs(refined_objectives - base_objectives) / jnp.maximum(
        jnp.abs(refined_objectives), y_floor
    )
    return {
        "best_grid_refinement_base_objective": base_objectives,
        "best_objective_refined_grid": refined_objectives,
        "best_grid_refinement_relative_error": relative_error,
        "best_grid_refinement_refined_N": jnp.full(
            (member_count,), comparison_N, dtype=jnp.int32
        ),
        "best_grid_refinement_tolerance": tolerance,
        "best_grid_refinement_y_floor": y_floor,
        "best_grid_refinement_passed": relative_error <= tolerance,
    }
