from dataclasses import asdict, dataclass
import math
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from simulation import Simulation


@dataclass(frozen=True)
class TestResult:
    config: Config
    initial_arrays: dict
    initial_data: dict
    optimised_data: dict
    objective_history: object
    batch_labels: tuple
    best_index: int
    best_objective: float
    new_max: bool
    optimisation_time: float
    fig: object = None
    axes: object = None
    best_fig: object = None
    best_axes: object = None


class Curves:
    """Build reproducible, batched raw control curves from a ``Config``."""

    structured_names = (
        "Constant midpoint",
        "Constant low intensity",
        "Linear ramp",
        "Gaussian pulse",
    )

    @staticmethod
    def _validate_config(config):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config configuration.")

    @staticmethod
    def _control_names(config):
        return ("u", "v") if config.loss else ("a",)

    @staticmethod
    def _fraction_to_physical(config, name, fraction):
        fraction = jnp.asarray(fraction, dtype=config.dtype)
        if name == "u":
            return config.u_max * fraction
        if name == "v":
            return config.v_max * (2.0 * fraction - 1.0)
        if name == "a":
            return config.a_min + (config.a_max - config.a_min) * fraction
        raise KeyError(f"Unknown control {name!r}.")

    def _physical_profile(self, config, fraction):
        return {
            name: self._fraction_to_physical(config, name, fraction)
            for name in self._control_names(config)
        }

    def constant_midpoint(self, config):
        """Return midpoint-valued physical controls on the configured grid."""

        self._validate_config(config)
        fraction = jnp.full(config.N + 1, 0.5, dtype=config.dtype)
        return self._physical_profile(config, fraction)

    def constant_low_intensity(self, config):
        """Return physical controls at ten percent of their allowed range."""

        self._validate_config(config)
        fraction = jnp.full(config.N + 1, 0.1, dtype=config.dtype)
        return self._physical_profile(config, fraction)

    def linear_ramp(self, config):
        """Return physical controls ramping from ten to ninety percent."""

        self._validate_config(config)
        fraction = jnp.linspace(0.1, 0.9, config.N + 1, dtype=config.dtype)
        return self._physical_profile(config, fraction)

    def gaussian_pulse(self, config):
        """Return low-background physical controls with a Gaussian peak."""

        self._validate_config(config)
        x = jnp.linspace(0.0, 1.0, config.N + 1, dtype=config.dtype)
        gaussian = jnp.exp(-0.5 * ((x - 0.5) / 0.12) ** 2)
        fraction = 0.1 + 0.8 * gaussian
        return self._physical_profile(config, fraction)

    @staticmethod
    def random_smooth_curve(key, time_grid, num_modes=5, scale=0.5):
        """Return a low-frequency random Fourier series in raw space."""

        time_grid = jnp.asarray(time_grid)
        keys = jax.random.split(key, 2)
        sin_coeffs = scale * jax.random.normal(
            keys[0], (num_modes,), dtype=time_grid.dtype
        )
        cos_coeffs = scale * jax.random.normal(
            keys[1], (num_modes,), dtype=time_grid.dtype
        )
        modes = jnp.arange(1, num_modes + 1, dtype=time_grid.dtype)
        phase = (
            2.0
            * jnp.pi
            * modes[:, None]
            * time_grid[None, :]
            / time_grid[-1]
        )
        return jnp.sum(
            sin_coeffs[:, None] * jnp.sin(phase)
            + cos_coeffs[:, None] * jnp.cos(phase),
            axis=0,
        )

    @staticmethod
    def _logit(values):
        return jnp.log(values) - jnp.log1p(-values)

    def _to_raw(self, config, name, physical):
        """Invert a bound only when that control is configured as bounded."""

        if name == "u":
            if not config.u_isbound:
                return physical
            return self._logit(physical / config.u_max)
        if name == "v":
            if not config.v_isbound:
                return physical
            return jnp.arctanh(physical / config.v_max)
        if name == "a":
            if not config.a_isbound:
                return physical
            fraction = (physical - config.a_min) / (
                config.a_max - config.a_min
            )
            return self._logit(fraction)
        raise KeyError(f"Unknown control {name!r}.")

    def _structured_raw(self, config):
        profile_functions = (
            self.constant_midpoint,
            self.constant_low_intensity,
            self.linear_ramp,
            self.gaussian_pulse,
        )
        physical_profiles = [function(config) for function in profile_functions]
        raw_by_control = {
            name: jnp.stack(
                [self._to_raw(config, name, profile[name]) for profile in physical_profiles]
            )
            for name in self._control_names(config)
        }

        if config.loss:
            return {
                "u": jnp.repeat(raw_by_control["u"], 4, axis=0),
                "v": jnp.tile(raw_by_control["v"], (4, 1)),
            }
        return {"a": raw_by_control["a"]}

    def _random_raw(self, config):
        control_names = self._control_names(config)
        key_count = config.rng_sim_num * len(control_names)
        keys = jax.random.split(config.key, key_count) if key_count else ()
        key_index = 0
        random_raw = {}
        for name in control_names:
            control_curves = []
            for _ in range(config.rng_sim_num):
                control_curves.append(
                    self.random_smooth_curve(keys[key_index], config.time_grid)
                )
                key_index += 1
            if control_curves:
                random_raw[name] = jnp.stack(control_curves)
        return random_raw

    def make_init(self, config):
        """Return all requested initial curves as one batched control dictionary."""

        self._validate_config(config)
        batches = []
        if config.struct_curves:
            batches.append(self._structured_raw(config))
        if config.rng_sim_num:
            batches.append(self._random_raw(config))

        return {
            name: jnp.concatenate([batch[name] for batch in batches], axis=0)
            for name in self._control_names(config)
        }

    def labels(self, config):
        """Return labels in the same order as ``make_init`` batch members."""

        self._validate_config(config)
        labels = []
        if config.struct_curves:
            if config.loss:
                labels.extend(
                    f"u: {u_name}; v: {v_name}"
                    for u_name in self.structured_names
                    for v_name in self.structured_names
                )
            else:
                labels.extend(f"a: {name}" for name in self.structured_names)
        labels.extend(
            f"Random Fourier {index + 1}" for index in range(config.rng_sim_num)
        )
        return tuple(labels)

    def add_perturbed_optimal(self, config, initial_arrays, optimal_raw, scale=0.05):
        """Append one smoothly perturbed saved optimum to an existing batch."""

        self._validate_config(config)
        if optimal_raw is None:
            return initial_arrays
        if not isinstance(optimal_raw, dict):
            raise TypeError("optimal_raw must be a dictionary of control arrays.")

        expected = set(self._control_names(config))
        supplied = set(optimal_raw)
        if supplied != expected:
            raise ValueError(
                "Saved optimal_raw controls do not match this configuration; "
                f"expected={sorted(expected)}, supplied={sorted(supplied)}."
            )

        perturbation_key = jax.random.fold_in(config.key, 104729)
        keys = jax.random.split(perturbation_key, len(expected))
        augmented = {}
        for key, name in zip(keys, self._control_names(config)):
            values = jnp.asarray(optimal_raw[name], dtype=config.dtype)
            if values.ndim == 2 and values.shape[0] == 1:
                values = values[0]
            if values.shape != (config.N + 1,):
                raise ValueError(
                    f"optimal_raw[{name!r}] must have shape ({config.N + 1},), "
                    f"got {values.shape}."
                )
            perturbation = self.random_smooth_curve(
                key, config.time_grid, scale=scale
            )
            candidate = (values + perturbation)[None, :]
            augmented[name] = jnp.concatenate(
                [jnp.asarray(initial_arrays[name]), candidate], axis=0
            )
        return augmented


curves = Curves()


class Test:
    def __init__(
        self,
        config: Config,
        plot_columns=2,
        optimal_raw=None,
        max_obj=0.0,
        autorun=True,
        make_plots=True,
    ):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config configuration.")
        if int(plot_columns) < 1:
            raise ValueError("plot_columns must be at least 1.")
        if not np.isfinite(max_obj):
            raise ValueError("max_obj must be finite.")

        self.config = config
        self.plot_columns = int(plot_columns)
        self.max_obj = float(max_obj)
        self._optimal_raw = None
        self._optimal_time_grid = None
        self.optimal_raw = optimal_raw

        self.initial_arrays = None
        self.simulation = None
        self.result = None
        if autorun:
            self.run(make_plots=make_plots)

    @property
    def optimal_raw(self):
        return self._optimal_raw

    @optimal_raw.setter
    def optimal_raw(self, value):
        if value is not None and not isinstance(value, dict):
            raise TypeError("optimal_raw must be a dictionary or None.")
        self._optimal_raw = value
        self._optimal_time_grid = (
            None
            if value is None
            else np.asarray(self.config.time_grid, dtype=float).copy()
        )

    @property
    def results(self):
        """Compatibility alias for code using ``test.results``."""

        return self.result

    def _validate_new_config(self, config):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config configuration.")
        if self.optimal_raw is None:
            return

        new_grid = np.asarray(config.time_grid, dtype=float)
        if not np.array_equal(self._optimal_time_grid, new_grid):
            raise ValueError(
                "The new configuration time grid differs from the grid associated "
                "with optimal_raw. Clear optimal_raw or use the same N and t_interval."
            )

        expected = {"u", "v"} if config.loss else {"a"}
        supplied = set(self.optimal_raw)
        if supplied != expected:
            raise ValueError(
                "The saved optimal_raw controls are incompatible with the new "
                f"configuration; expected={sorted(expected)}, "
                f"supplied={sorted(supplied)}."
            )
        for name, values in self.optimal_raw.items():
            shape = np.shape(values)
            if shape not in ((config.N + 1,), (1, config.N + 1)):
                raise ValueError(
                    f"optimal_raw[{name!r}] must have shape ({config.N + 1},) "
                    f"or (1, {config.N + 1}), got {shape}."
                )

    def _print_parameters(self, batch_size):
        print("Parameters used:")
        for name, value in asdict(self.config).items():
            print(f"  {name}: {value}")
        print(f"  resolved_device: {self.simulation.device}")
        print(f"  batch_size: {batch_size}")
        print(f"  plot_columns: {self.plot_columns}")

    @staticmethod
    def _best_finite_index(objectives):
        objectives = np.asarray(objectives, dtype=float)
        finite = np.isfinite(objectives)
        if not finite.any():
            raise ValueError("Every final objective in the batch is non-finite.")
        return int(np.argmax(np.where(finite, objectives, -np.inf)))

    def run(self, config=None, make_plots=True):
        """Run one batched optimisation, optionally using an updated configuration."""

        next_config = self.config if config is None else config
        self._validate_new_config(next_config)
        self.config = next_config

        initial_arrays = curves.make_init(self.config)
        labels = curves.labels(self.config)
        if self.optimal_raw is not None:
            initial_arrays = curves.add_perturbed_optimal(
                self.config, initial_arrays, self.optimal_raw
            )
            labels += ("Perturbed saved optimum",)

        self.initial_arrays = initial_arrays
        self.simulation = Simulation(self.config)
        initial_data = self.simulation.bounded(initial_arrays)
        batch_size = next(iter(initial_arrays.values())).shape[0]
        self._print_parameters(batch_size)

        start_time = time.perf_counter()
        optimised_data = self.simulation.optimise(initial_arrays)
        jax.block_until_ready(optimised_data)
        optimisation_time = time.perf_counter() - start_time

        final_objectives = np.asarray(optimised_data["objective"])
        best_index = self._best_finite_index(final_objectives)
        best_objective = float(final_objectives[best_index])
        new_max = best_objective > self.max_obj
        if new_max:
            self.max_obj = best_objective
            self.optimal_raw = {
                name: jnp.asarray(values[best_index]).copy()
                for name, values in optimised_data["raw"].items()
            }

        fig = axes = best_fig = best_axes = None
        if make_plots:
            fig, axes = self._plot_batch(initial_data, optimised_data, labels)
            if new_max:
                best_fig, best_axes = self._plot_best(
                    optimised_data,
                    labels,
                    best_index,
                )

        print(f"Total batch optimisation time: {optimisation_time:.3f} s")
        print(
            f"Best final objective: {best_objective:.8g} "
            f"(batch member {best_index + 1})"
        )
        if new_max:
            print(f"New maximum objective saved: {self.max_obj:.8g}")

        self.result = TestResult(
            config=self.config,
            initial_arrays=initial_arrays,
            initial_data=initial_data,
            optimised_data=optimised_data,
            objective_history=optimised_data["history"],
            batch_labels=labels,
            best_index=best_index,
            best_objective=best_objective,
            new_max=new_max,
            optimisation_time=optimisation_time,
            fig=fig,
            axes=axes,
            best_fig=best_fig,
            best_axes=best_axes,
        )
        return self.result

    def _plot_member(
        self,
        ax,
        initial_data,
        optimised_data,
        index,
        label,
    ):
        time_grid = np.asarray(self.config.time_grid)
        colours = {
            "u": ("#78add2", "#2474b5"),
            "v": ("#df9295", "#c44e52"),
            "a": ("#78add2", "#2474b5"),
        }
        display_names = {"u": "u", "v": "v", "a": "a_s"}

        for name in self.simulation.required_controls:
            initial_colour, optimised_colour = colours[name]
            display_name = display_names[name]
            ax.plot(
                time_grid,
                np.asarray(initial_data[name][index]),
                linestyle="--",
                color=initial_colour,
                alpha=0.55,
                linewidth=1.0,
                label=f"initial {display_name}",
            )
            ax.plot(
                time_grid,
                np.asarray(optimised_data["bound"][name][index]),
                linestyle="-",
                color=optimised_colour,
                linewidth=1.5,
                label=f"optimised {display_name}",
            )

        history = np.asarray(optimised_data["history"][index], dtype=float)
        history_axis = ax.twinx()
        history_x = np.linspace(time_grid[0], time_grid[-1], history.size)
        history_axis.plot(
            history_x,
            history,
            color="lightgray",
            linewidth=1.0,
            label=f"objective final = {history[-1]:.6g}",
            zorder=0,
        )
        history_axis.set_ylabel("Objective", color="gray")
        history_axis.tick_params(axis="y", colors="gray", labelsize="small")

        lines, line_labels = ax.get_legend_handles_labels()
        history_lines, history_labels = history_axis.get_legend_handles_labels()
        ax.legend(
            lines + history_lines,
            line_labels + history_labels,
            fontsize="x-small",
            loc="best",
        )
        ax.set_title(label, fontsize="small")
        ax.set_xlabel("Dimensionless time")
        ax.set_ylabel("Physical control")
        return history_axis

    def _plot_batch(self, initial_data, optimised_data, labels):
        batch_size = len(labels)
        ncols = min(self.plot_columns, batch_size)
        nrows = math.ceil(batch_size / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.6 * ncols, 3.4 * nrows),
            squeeze=False,
            sharex=True,
        )
        axes = axes.ravel()
        for index, (ax, label) in enumerate(zip(axes, labels)):
            self._plot_member(ax, initial_data, optimised_data, index, label)
        for ax in axes[batch_size:]:
            ax.set_visible(False)

        interaction_type = "Inelastic" if self.config.loss else "Elastic"
        fig.suptitle(f"{interaction_type} batched optimisation")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return fig, axes

    def _plot_best(self, optimised_data, labels, best_index):
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        best_final_data = optimised_data["bound"]
        history_axis = self._plot_member(
            ax,
            best_final_data,
            optimised_data,
            best_index,
            f"New maximum — {labels[best_index]}",
        )
        fig.tight_layout()
        return fig, (ax, history_axis)

    def plot_simulations(self):
        """Recreate the batch plot from the most recent result."""

        if self.result is None:
            raise ValueError("Run the test before plotting its outcome.")
        return self._plot_batch(
            self.result.initial_data,
            self.result.optimised_data,
            self.result.batch_labels,
        )
