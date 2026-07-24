from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation
from config import Config


@dataclass(frozen=True)
class TestResult:
    config: Config
    initial_arrays: dict
    initial_data: dict
    optimised_data: dict
    objective_history: object
    fig: object = None
    axes: object = None
    history_fig: object = None
    history_ax: object = None


class Test:
    def __init__(
        self,
        config: Config,
        init_scale=0.1,
        plot_columns=3,
        autorun=True,
        make_plots=True,
    ):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config configuration.")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")
        if int(plot_columns) < 1:
            raise ValueError("plot_columns must be at least 1.")

        self.config = config
        self.init_scale = float(init_scale)
        self.plot_columns = int(plot_columns)
        self.initial_arrays = self._make_initial_arrays()
        self.simulation = Simulation(config)
        self.result = None
        if autorun:
            self.run(make_plots=make_plots)

    def _make_initial_arrays(self):
        shape = (self.config.sim_num, self.config.N + 1)
        if self.config.loss:
            key_u, key_v = jax.random.split(self.config.key)
            return {
                "u": self.init_scale * jax.random.normal(key_u, shape, dtype=self.config.dtype),
                "v": self.init_scale * jax.random.normal(key_v, shape, dtype=self.config.dtype),
            }

        offsets = self.init_scale * jax.random.normal(self.config.key, shape, dtype=self.config.dtype)
        a_s = self.config.a_bg + offsets
        epsilon = jnp.asarray(1e-6, dtype=self.config.dtype)
        a_s = jnp.where(jnp.abs(a_s) < epsilon, epsilon, a_s)
        return {"a_s": a_s}

    def run(self, make_plots=True):
        optimised_data = self.simulation.optimise(self.initial_arrays)

        fig = axes = history_fig = history_ax = None
        if make_plots:
            fig, axes = self.plot_controls(
                self.simulation.initial_data,
                optimised_data,
            )
            history_fig, history_ax = self.plot_history(self.simulation.history)

        self.result = TestResult(
            config=self.config,
            initial_arrays=self.initial_arrays,
            initial_data=self.simulation.initial_data,
            optimised_data=optimised_data,
            objective_history=self.simulation.history,
            fig=fig,
            axes=axes,
            history_fig=history_fig,
            history_ax=history_ax,
        )
        return self.result

    def plot_controls(self, initial_data=None, optimised_data=None):
        if initial_data is None or optimised_data is None:
            if self.result is None:
                raise ValueError("Run the test before plotting its outcome.")
            initial_data = self.result.initial_data
            optimised_data = self.result.optimised_data

        ncols = min(self.plot_columns, self.config.sim_num)
        nrows = math.ceil(self.config.sim_num / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.0 * nrows),
            squeeze=False,
            sharex=True,
        )
        axes = axes.ravel()
        time = np.asarray(self.config.time_grid)

        for index, ax in enumerate(axes[: self.config.sim_num]):
            if self.config.loss:
                ax.plot(time, np.asarray(initial_data["u"][index]), "--", color="#2474b5", alpha=0.65, label="initial u")
                ax.plot(time, np.asarray(optimised_data["u"][index]), color="#2474b5", label="optimised u")
                ax.plot(time, np.asarray(initial_data["v"][index]), "--", color="#c44e52", alpha=0.65, label="initial v")
                ax.plot(time, np.asarray(optimised_data["v"][index]), color="#c44e52", label="optimised v")
                ax.set_ylabel("control")
            else:
                ax.plot(time, np.asarray(initial_data["a_s"][index]), "--", color="#2474b5", alpha=0.65, label="initial a_s")
                ax.plot(time, np.asarray(optimised_data["a_s"][index]), color="#c44e52", label="optimised a_s")
                ax.set_ylabel("scattering length")

            ax.set_title(f"Simulation {index + 1}")
            ax.set_xlabel("dimensionless time")
            ax.legend(fontsize="small")

        for ax in axes[self.config.sim_num :]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig, axes

    def plot_history(self, history=None):
        if history is None:
            if self.result is None:
                raise ValueError("Run the test before plotting objective history.")
            history = self.result.objective_history

        values = np.asarray(history)
        if values.ndim == 1:
            values = values[:, None]

        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        steps = np.arange(1, values.shape[0] + 1)
        for index in range(values.shape[1]):
            ax.plot(steps, values[:, index], linewidth=1.0, alpha=0.45)
        ax.plot(steps, values.mean(axis=1), color="#111111", linewidth=2.0, label="mean")
        ax.set_xlabel("optimisation step")
        ax.set_ylabel("objective")
        ax.legend()
        fig.tight_layout()
        return fig, ax