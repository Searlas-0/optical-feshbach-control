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

    @property
    def required_controls(self):
        return ("u", "v") if self.config.loss else ("a",)

    def _normalise_raw_data(self, raw_data):
        if not isinstance(raw_data, dict):
            raise TypeError("raw_data must be a dictionary of control arrays.")

        missing = set(self.required_controls) - set(raw_data)
        extra = set(raw_data) - set(self.required_controls)
        if missing or extra:
            raise ValueError(
                f"Expected controls {self.required_controls}; missing={sorted(missing)}, extra={sorted(extra)}."
            )

        normalised = {
            name: jnp.asarray(values, dtype=self.config.dtype)
            for name, values in raw_data.items()
        }
        shapes = {array.shape for array in normalised.values()}
        if len(shapes) != 1:
            raise ValueError("All control arrays must have the same shape.")
        shape = next(iter(shapes))
        if len(shape) != 2 or shape[1] != self.config.N + 1:
            raise ValueError(
                f"Control arrays must have shape (batch_size, {self.config.N + 1})."
            )
        if shape[0] < 1:
            raise ValueError("Control arrays must contain at least one batch member.")
        return normalised


    def bounded(self, raw_data):
        c = self.config
        if c.loss:
            u = raw_data["u"]
            v = raw_data["v"]
            u_bound = c.u_max * jax.nn.sigmoid(u) if c.u_isbound else u
            v_bound = c.v_max * jnp.tanh(v) if c.v_isbound else v
            return {"u": u_bound, "v": v_bound}
        a = raw_data['a']
        a_bound = c.a_min + (c.a_max - c.a_min) * jax.nn.sigmoid(a) if c.a_isbound else a
        return {'a': a_bound}

    def scattering_length(self, data):
        u = jnp.asarray(data["u"], dtype=self.config.dtype)
        v = jnp.asarray(data["v"], dtype=self.config.dtype)
        return self.config.a_bg * (1.0 + u / (-v - u + 0.5j))

    def solve_eta(self, a_s):
        c = self.config
        a_s = jnp.asarray(a_s, dtype=c.complex_dtype)
        num_steps = c.N 
        num_points = num_steps + 1
        eta_0 = -4.0 * jnp.pi * a_s[0]
        eta = jnp.zeros_like(a_s).at[0].set(eta_0)

        kernel_prefactor = -1.0 / (4.0 * (jnp.pi**3 / 2.0) * jnp.sqrt(1j))
        l1_prefactor = 2.0 * kernel_prefactor / jnp.sqrt(self.config.dt)
        j = jnp.arange(num_steps)

        def time_step(history, k):
            diffs = jnp.diff(history)
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
        c = self.config
        dt = c.dt

        if c.loss:
            u_rate = jnp.diff(data["u"]) 
            v_rate = jnp.diff(data["v"]) 
            return (
                c.u_smooth * jnp.sum(u_rate**2) / dt
                + c.v_smooth * jnp.sum(v_rate**2) / dt
            )
        a_rate = jnp.diff(data["a"])
        return c.a_smooth * jnp.mean(a_rate**2) / dt

    def loss_fn(self, raw_data):
        data = self.bounded(raw_data)
        if self.config.loss:
            a_s = self.scattering_length(data)
            eta = self.solve_eta(a_s)
            objective = self.molecule_density(a_s, eta)
        else:
            a_s = data["a"]
            eta = self.solve_eta(a_s)
            objective = self.energy_density(a_s, eta)
        loss = -objective + self._smoothness_penalty(data)
        return loss, objective
        
    def optimise(self, raw_data):
        c = self.config
        raw_data = self._normalise_raw_data(raw_data)

        optimiser = optax.adam(
            learning_rate=c.learning_rate,
            b1=c.beta1,
            b2=c.beta2,
            eps=c.eps,
        )
        opt_state = optimiser.init(raw_data)
        value_and_grad = jax.vmap(
            jax.value_and_grad(self.loss_fn, has_aux=True)
        )

        def train_step(current_data, current_opt_state):
            (_, objectives), grads = value_and_grad(current_data)
            updates, next_opt_state = optimiser.update(grads, current_opt_state, current_data)
            updated = optax.apply_updates(current_data, updates)
            return updated, next_opt_state, objectives

        if c.use_jit:
            train_step = jax.jit(train_step)

        history = []
        for _ in range(c.num_steps):
            raw_data, opt_state, objectives = train_step(raw_data, opt_state)
            history.append(objectives)
        history = jnp.stack(history, axis=1)
        objective_fn = jax.vmap(lambda member: self.loss_fn(member)[1])
        if c.use_jit:
            objective_fn = jax.jit(objective_fn)
        final_objectives = objective_fn(raw_data)
        history = jnp.concatenate([history, final_objectives[:, None]], axis=1)

        return {
            "raw": raw_data,
            "bound": self.bounded(raw_data),
            "history": history,
            "objective": final_objectives,
        }
