from typing import NamedTuple
import math

import jax
import jax.numpy as jnp


class Config(NamedTuple):
    a_bg: float = 1.0
    gamma: float = 1.0
    Gamma_max: float = 10.0

    t_interval: float = 1.0
    N: int = 100
    dt: float = 0.01

    bound_u: bool = True
    bound_v: bool = True
    u_max: float = 40.0
    v_max: float = 20.0

    seed: int = 0
    sim_num: int = 13

    learning_smooth: float = 1e-2
    num_steps: int = 1000
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    loss: bool = True

    smooth: bool = False
    u_smooth: float = 0.0
    v_smooth: float = 0.0

    use_jit: bool = True
    use_x64: bool = True

    @classmethod
    def init(
        cls,
        *,
        a_bg: float = 1.0,
        gamma: float = 1.0,
        Gamma_max: float = 10.0,
        t_interval: float = 1.0,
        N: int = 100,
        dt: float | None = None,
        u_max: float = 9999.0,
        v_max: float = 9999.0,
        a_max: float = 9999.0,
        seed: int = 0,
        sim_num: int = 13,
        learning_smooth: float = 1e-2,
        num_steps: int = 1000,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        loss: bool = True,
        u_smooth: float = 0.0,
        v_smooth: float = 0.0,
        a_smooth:float = 0.0,
        use_jit: bool = True,
        use_x64: bool = True,
    ):
        N = int(N)
        t_interval = float(t_interval)
        if N < 1:
            raise ValueError("N must be at least 1.")
        if t_interval <= 0.0:
            raise ValueError("t_interval must be positive.")

        resolved_dt = t_interval / N if dt is None else float(dt)
        if resolved_dt <= 0.0:
            raise ValueError("dt must be positive.")
        if not math.isclose(resolved_dt * N, t_interval, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("dt * N must equal t_interval.")

        sim_num = int(sim_num)
        num_steps = int(num_steps)
        if sim_num < 1:
            raise ValueError("sim_num must be at least 1.")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if learning_smooth <= 0.0:
            raise ValueError("learning_smooth must be positive.")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if gamma == 0.0:
            raise ValueError("gamma must be non-zero.")
        if u_max <= 0.0 or v_max <= 0.0:
            raise ValueError("u_max and v_max must be positive.")
        if u_smooth <= 0.0 or v_smooth <= 0.0:
            raise ValueError("u_smooth and v_smooth cannot be negative.")

        jax.config.update("jax_enable_x64", bool(use_x64))
        return cls(
            a_bg=float(a_bg),
            gamma=float(gamma),
            Gamma_max=float(Gamma_max),
            t_interval=t_interval,
            N=N,
            dt=resolved_dt,
            u_max=float(u_max),
            v_max=float(v_max),
            a_max=float(a_max),
            seed=int(seed),
            sim_num=sim_num,
            learning_smooth=float(learning_smooth),
            num_steps=num_steps,
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(eps),
            loss=bool(loss),
            u_smooth=float(u_smooth),
            v_smooth=float(v_smooth),
            a_smooth=float(a_smooth),
            use_jit=bool(use_jit),
            use_x64=bool(use_x64),
        )

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
