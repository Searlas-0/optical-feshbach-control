from dataclasses import dataclass, replace
import math

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    a_bg: float = 1.0
    gamma: float = 1.0
    Gamma_max: float | None = None
    detuning_max: float | None = None

    t_interval: float = 1.0
    N: int | None = None
    dt: float | None = None

    loss: bool = True

    u_isbound: bool = True
    v_isbound: bool = True
    u_max: float = 50.0
    v_max: float = 50.0

    a_isbound: bool = True
    a_max: float = 50.0
    a_min: float = 1e-5

    seed: int = 0
    rng_sim_num: int = 13
    struct_curves: bool = True

    num_steps: int = 1000
    learning_rate: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    u_smooth: float = 1e-4
    v_smooth: float = 1e-4
    a_smooth: float = 1e-4

    use_jit: bool = True
    use_x64: bool = True

    def __post_init__(self):
        t_interval = float(self.t_interval)
        if t_interval <= 0.0:
            raise ValueError("t_interval must be positive.")

        if self.N is None:
            if self.dt is None:
                N = 100
            else:
                supplied_dt = float(self.dt)
                if supplied_dt <= 0.0:
                    raise ValueError("dt must be positive.")
                N = round(t_interval / supplied_dt)
                if N < 1 or not math.isclose(
                    supplied_dt * N,
                    t_interval,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise ValueError("dt must divide t_interval into an integer number of steps.")
        else:
            N = int(self.N)
            if N < 1:
                raise ValueError("N must be at least 1.")
        dt = t_interval / N

        gamma = float(self.gamma)
        if gamma <= 0.0:
            raise ValueError("gamma must be positive.")

        if self.Gamma_max is None:
            if self.u_max is None:
                Gamma_max = 10.0
            else:
                u_max = float(self.u_max)
                if u_max <= 0.0:
                    raise ValueError("u_max must be positive.")
                Gamma_max = gamma * u_max
        else:
            Gamma_max = float(self.Gamma_max)
        if Gamma_max <= 0.0:
            raise ValueError("Gamma_max must be positive.")
        u_max = Gamma_max / gamma

        if self.detuning_max is None:
            if self.v_max is None:
                detuning_max = 40.0
            else:
                v_max = float(self.v_max)
                if v_max <= 0.0:
                    raise ValueError("v_max must be positive.")
                detuning_max = gamma * v_max
        else:
            detuning_max = float(self.detuning_max)
        if detuning_max <= 0.0:
            raise ValueError("detuning_max must be positive.")
        v_max = detuning_max / gamma

        a_min = float(self.a_min)
        a_max = float(self.a_max)
        if a_max <= a_min:
            raise ValueError("a_max must be greater than a_min.")

        rng_sim_num = int(self.rng_sim_num)
        num_steps = int(self.num_steps)
        if rng_sim_num < 0:
            raise ValueError("rng_sim_num cannot be negative.")
        if rng_sim_num == 0 and not self.struct_curves:
            raise ValueError(
                "At least one initial curve is required; enable struct_curves "
                "or set rng_sim_num above zero."
            )
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if self.u_smooth < 0.0 or self.v_smooth < 0.0 or self.a_smooth < 0.0:
            raise ValueError("Smoothness penalties cannot be negative.")

        object.__setattr__(self, "a_bg", float(self.a_bg))
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "Gamma_max", Gamma_max)
        object.__setattr__(self, "detuning_max", detuning_max)
        object.__setattr__(self, "t_interval", t_interval)
        object.__setattr__(self, "N", N)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "loss", bool(self.loss))
        object.__setattr__(self, "u_isbound", bool(self.u_isbound))
        object.__setattr__(self, "v_isbound", bool(self.v_isbound))
        object.__setattr__(self, "u_max", u_max)
        object.__setattr__(self, "v_max", v_max)
        object.__setattr__(self, "a_isbound", bool(self.a_isbound))
        object.__setattr__(self, "a_max", a_max)
        object.__setattr__(self, "a_min", a_min)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "rng_sim_num", rng_sim_num)
        object.__setattr__(self, "struct_curves", bool(self.struct_curves))
        object.__setattr__(self, "num_steps", num_steps)
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "beta1", float(self.beta1))
        object.__setattr__(self, "beta2", float(self.beta2))
        object.__setattr__(self, "eps", float(self.eps))
        object.__setattr__(self, "u_smooth", float(self.u_smooth))
        object.__setattr__(self, "v_smooth", float(self.v_smooth))
        object.__setattr__(self, "a_smooth", float(self.a_smooth))
        object.__setattr__(self, "use_jit", bool(self.use_jit))
        object.__setattr__(self, "use_x64", bool(self.use_x64))

        jax.config.update("jax_enable_x64", self.use_x64)

    def update(self, **changes):
        """Return a validated copy with the specified fields changed."""
        if "dt" in changes and "N" not in changes:
            changes["N"] = None
        if "u_max" in changes and "Gamma_max" not in changes:
            changes["Gamma_max"] = None
        if "v_max" in changes and "detuning_max" not in changes:
            changes["detuning_max"] = None
        return replace(self, **changes)

    @property
    def dtype(self):
        return jnp.float64 if self.use_x64 else jnp.float32

    @property
    def complex_dtype(self):
        return jnp.complex128 if self.use_x64 else jnp.complex64

    @property
    def time_grid(self):
        return jnp.linspace(0.0, self.t_interval, self.N + 1, dtype=self.dtype)

    @property
    def key(self):
        return jax.random.PRNGKey(self.seed)
