"""Dimensionless inelastic optical Feshbach physics.

``u`` is the non-negative optical width and ``v`` the signed detuning, both in
linewidth units.  Their complex scattering length is measured in the fixed
length ``l* = sqrt(t* hbar / m)`` and drives a discretized Volterra equation for
the pair amplitude.  The optimized objective is the integrated inelastic
molecular density.  No elastic interaction branch is retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class Physics:
    N: int
    dtype: object = jnp.float64
    u_isbound: bool = True
    v_isbound: bool = True
    u_sharp_active: bool = True
    v_sharp_active: bool = True

    @property
    def complex_dtype(self):
        return jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64

    def bounded_controls(
        self, raw: Mapping[str, jax.Array], parameters: Mapping[str, jax.Array]
    ) -> dict[str, jax.Array]:
        u = jnp.asarray(raw["u"], dtype=self.dtype)
        v = jnp.asarray(raw["v"], dtype=self.dtype)
        if self.u_isbound:
            u = parameters["u_max"] * jax.nn.sigmoid(u)
        if self.v_isbound:
            v = parameters["v_max"] * jnp.tanh(v)
        return {"u": u, "v": v}

    def dimensionless_scattering_length(
        self,
        controls: Mapping[str, jax.Array],
        r_bg,
    ):
        """Return ``a_s/l*`` for the signed background ratio ``r_bg=a_bg/l*``.

        Writing ``s=sign(r_bg)``, the optical map is
        ``r_bg * (1 + s*u/(-s*u-v+i/2))``.  Equivalently, it is
        ``|r_bg| * (s + u/(-s*u-v+i/2))``.  The sign multiplying ``u`` is the
        sign of the optical width Gamma; at zero optical width the result is
        exactly ``r_bg``.
        """

        u = jnp.asarray(controls["u"], dtype=self.dtype)
        v = jnp.asarray(controls["v"], dtype=self.dtype)
        r_bg = jnp.asarray(r_bg, dtype=self.dtype)
        sign = jnp.sign(r_bg)
        return r_bg * (1.0 + sign * u / (-sign * u - v + 0.5j))

    def pair_amplitude(self, scattering_ratio, dt):
        """Solve the discretized Volterra equation for the pair amplitude η."""

        scattering_ratio = jnp.asarray(scattering_ratio, dtype=self.complex_dtype)
        dt = jnp.asarray(dt, dtype=self.dtype)
        eta = jnp.zeros_like(scattering_ratio).at[0].set(
            -4.0 * jnp.pi * scattering_ratio[0]
        )
        kernel_prefactor = -1.0 / (
            4.0 * jnp.pi**(3.0 / 2.0) * jnp.sqrt(jnp.asarray(1j, self.complex_dtype))
        )
        l1_prefactor = 2.0 * kernel_prefactor / jnp.sqrt(dt)
        j = jnp.arange(self.N)

        def time_step(history, k):
            differences = jnp.diff(history)
            valid = j < (k - 1)
            m = k - j
            safe_m = jnp.maximum(m, 1)
            weights = jnp.where(
                valid, jnp.sqrt(safe_m) - jnp.sqrt(safe_m - 1), 0.0
            )
            known_history = jnp.sum(weights * differences)
            numerator = -1.0 + l1_prefactor * (
                history[k - 1] - known_history
            )
            denominator = 1.0 / (4.0 * jnp.pi * scattering_ratio[k]) + l1_prefactor
            eta_k = numerator / denominator
            return history.at[k].set(eta_k), eta_k

        eta, _ = lax.scan(time_step, eta, jnp.arange(1, self.N + 1))
        return eta

    @staticmethod
    def trapezoid(values, dx):
        return jnp.sum(0.5 * dx * (values[:-1] + values[1:]))

    def molecular_objective(self, scattering_length, pair_amplitude, dt, r_bg):
        """Integrated dimensionless molecule density from inelastic contact."""

        # The signed optical width is already included in the scattering map.
        # With the +i/2 convention, Im(a_s) <= 0 and therefore Im(1/a_s) >= 0
        # for either sign of the non-zero background length.
        inelastic_factor = jnp.imag(1.0 / scattering_length)
        contact = inelastic_factor * jnp.abs(pair_amplitude) ** 2
        return self.trapezoid(contact, dt) / (2.0 * jnp.pi)

    def smoothness_penalty(self, controls, parameters):
        """Penalize first and second control derivatives over dimensionless time."""

        dt = parameters["dt"]
        penalty = (
            parameters["u_smooth"] * jnp.sum(jnp.diff(controls["u"]) ** 2)
            / dt
            + parameters["v_smooth"] * jnp.sum(jnp.diff(controls["v"]) ** 2)
            / dt
        )
        # These are Python/static branches, so an inactive sharpness term is
        # absent from the compiled JAX program rather than multiplied by zero.
        if self.u_sharp_active:
            penalty = (
                penalty
                + parameters["u_sharp"]
                * jnp.sum(jnp.diff(controls["u"], n=2) ** 2)
                / dt**3
            )
        if self.v_sharp_active:
            penalty = (
                penalty
                + parameters["v_sharp"]
                * jnp.sum(jnp.diff(controls["v"], n=2) ** 2)
                / dt**3
            )
        return penalty

    def metrics(self, raw, parameters):
        """Return maximized score, molecular objective, and regularization."""

        controls = self.bounded_controls(raw, parameters)
        scattering_length = self.dimensionless_scattering_length(
            controls,
            parameters["r_bg"],
        )
        eta = self.pair_amplitude(scattering_length, parameters["dt"])
        objective = self.molecular_objective(
            scattering_length, eta, parameters["dt"], parameters["r_bg"]
        )
        penalty = self.smoothness_penalty(controls, parameters)
        return objective - penalty, objective, penalty

    def minimization_target(self, raw, parameters):
        """Negate the maximized score for gradient-based minimization."""

        score, objective, penalty = self.metrics(raw, parameters)
        return -score, (objective, penalty)
