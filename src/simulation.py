import jax
import jax.numpy as jnp
from jax import lax
import optax
from config import Config


class Simulation:
    def __init__(self, config: Config):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config configuration.")
        self.config = config
        self.initial_raw_data = None
        self.initial_data = None
        self.raw_data = None
        self.data = None
        self.history = None

    @property
    def required_controls(self):
        return ("u", "v") if self.config.loss else ("a_s",)

    def _normalise_raw_data(self, raw_data):
        if not isinstance(raw_data, dict):
            raise TypeError("raw_data must be a dictionary of control arrays.")

        missing = set(self.required_controls) - set(raw_data)
        extra = set(raw_data) - set(self.required_controls)
        if missing or extra:
            raise ValueError(
                f"Expected controls {self.required_controls}; missing={sorted(missing)}, extra={sorted(extra)}."
            )

        arrays = {name: jnp.asarray(raw_data[name], dtype=self.config.dtype) for name in self.required_controls}
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("All control arrays must have the same shape.")

        shape = next(iter(shapes))
        if len(shape) not in (1, 2):
            raise ValueError("Control arrays must have shape (N + 1,) or (batch, N + 1).")
        if shape[-1] != self.config.N + 1:
            raise ValueError(f"Control arrays must contain N + 1 = {self.config.N + 1} points.")
        return arrays

    def bounded(self, raw_data):
        if not self.config.loss:
            return {"a_s": raw_data["a_s"]}

        u = raw_data["u"]
        v = raw_data["v"]
        if self.config.bound_u:
            u = self.config.u_max * jax.nn.sigmoid(u)
        if self.config.bound_v:
            v = self.config.v_max * jnp.tanh(v)
        return {"u": u, "v": v}

    def scattering_length(self, data):
        u = jnp.asarray(data["u"], dtype=self.config.dtype)
        v = jnp.asarray(data["v"], dtype=self.config.dtype)
        return self.config.a_bg * (1.0 + u / (-v - u + 0.5j))

    def solve_eta(self, a_s):
        a_s = jnp.asarray(a_s, dtype=self.config.complex_dtype)
        num_points = a_s.shape[0]
        num_steps = num_points - 1
        eta_0 = -4.0 * jnp.pi * a_s[0]
        eta = jnp.zeros_like(a_s).at[0].set(eta_0)

        kernel_prefactor = -1.0 / (4.0 * (jnp.pi**3 / 2.0) * jnp.sqrt(1j))
        l1_prefactor = 2.0 * kernel_prefactor / jnp.sqrt(self.config.dt)
        j = jnp.arange(num_steps)

        def time_step(history, k):
            diffs = history[1:] - history[:-1]
            valid = j < (k - 1)
            m = k - j
            safe_m = jnp.maximum(m, 1)
            weights = jnp.where(
                valid,
                jnp.sqrt(safe_m) - jnp.sqrt(safe_m - 1),
                0.0,
            )
            known_history = jnp.sum(weights * diffs)
            numerator = -1.0 + l1_prefactor * (history[k - 1] - known_history)
            denominator = 1.0 / (4.0 * jnp.pi * a_s[k]) + l1_prefactor
            eta_k = numerator / denominator
            return history.at[k].set(eta_k), eta_k

        indices = jnp.arange(1, num_points)
        eta, _ = lax.scan(time_step, eta, indices)
        return eta

    @staticmethod
    def trapezoid(integrand, dx):
        return jnp.sum(0.5 * dx * (integrand[:-1] + integrand[1:]))

    def molecule_density(self, a_s, eta):
        contact = jnp.imag(1.0 / a_s) * jnp.abs(eta) ** 2
        return self.trapezoid(contact, self.config.dt) / (2.0 * jnp.pi)

    def energy_density(self, a_s, eta):
        integrand = jnp.abs(eta) ** 2 / a_s**2
        return self.trapezoid(integrand, jnp.diff(a_s)) / (8.0 * jnp.pi)

    def _smoothness_penalty(self, data):
        if not self.config.smooth:
            return jnp.asarray(0.0, dtype=self.config.dtype)

        dt = self.config.dt

        if self.config.loss:
            u_rate = jnp.diff(data["u"]) 
            v_rate = jnp.diff(data["v"]) / self.config.dt
            return (
                self.config.u_smooth * jnp.mean(u_rate**2)
                + self.config.v_smooth * jnp.mean(v_rate**2)
            )

        a_rate = jnp.diff(data["a_s"]) / self.config.dt
        return self.config.a_smooth * jnp.mean(a_rate**2)

    def _single_objective(self, raw_data):
        data = self.bounded(raw_data)
        if self.config.loss:
            a_s = self.scattering_length(data)
            eta = self.solve_eta(a_s)
            objective = self.molecule_density(a_s, eta)
        else:
            a_s = data["a_s"]
            eta = self.solve_eta(a_s)
            objective = self.energy_density(a_s, eta)
        return jnp.real(objective) - self._smoothness_penalty(data)

    def _objective_values(self, raw_data):
        sample = next(iter(raw_data.values()))
        if sample.ndim == 1:
            return self._single_objective(raw_data)[None]
        return jax.vmap(self._single_objective)(raw_data)

    def objective(self, raw_data):
        raw_data = self._normalise_raw_data(raw_data)
        values = self._objective_values(raw_data)
        return values[0] if next(iter(raw_data.values())).ndim == 1 else values

    def optimise(self, raw_data):
        raw_data = self._normalise_raw_data(raw_data)
        self.initial_raw_data = raw_data
        self.initial_data = self.bounded(raw_data)

        optimiser = optax.adam(
            learning_rate=self.config.learning_rate,
            b1=self.config.beta1,
            b2=self.config.beta2,
            eps=self.config.eps,
        )
        opt_state = optimiser.init(raw_data)

        def train_step(current, current_opt_state):
            def loss_fn(candidate):
                return -jnp.mean(self._objective_values(candidate))

            _, grads = jax.value_and_grad(loss_fn)(current)
            updates, next_opt_state = optimiser.update(grads, current_opt_state, current)
            updated = optax.apply_updates(current, updates)
            return updated, next_opt_state, self._objective_values(updated)

        if self.config.use_jit:
            train_step = jax.jit(train_step)

        history = []
        for _ in range(self.config.num_steps):
            raw_data, opt_state, objectives = train_step(raw_data, opt_state)
            history.append(objectives)

        self.raw_data = raw_data
        self.data = self.bounded(raw_data)
        self.history = jnp.stack(history)
        if next(iter(raw_data.values())).ndim == 1:
            self.history = self.history[:, 0]
        return self.data