from dataclasses import asdict, dataclass
import math
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation
from config import Config


@dataclass(frozen=True)
class InitialArraySet:
    """One set of input arrays and the curve names used to create it."""

    arrays: dict
    curve_types: dict


@dataclass(frozen=True)
class SimulationResult:
    initial_arrays: dict
    initial_curve_types: dict
    optimised_data: dict
    objective_history: object


@dataclass(frozen=True)
class TestResult:
    config: Config
    simulations: tuple
    fig: object = None
    axes: object = None

    @property
    def initial_arrays(self):
        return tuple(simulation.initial_arrays for simulation in self.simulations)


# Keep initialization in this section so that the available curves or the random
# selection strategy can be changed without touching Test or the plotting code.
def _curve_templates(length, dtype):
    x = jnp.linspace(0.0, 1.0, length, dtype=dtype)
    centre = 0.5
    width = 0.12
    delta = jnp.zeros(length, dtype=dtype).at[length // 2].set(1.0)

    return {
        "constant": jnp.ones_like(x),
        "zero": jnp.zeros_like(x),
        "identity function": x,
        "linear": 2.0 * x - 1.0,
        "quadratic": x**2,
        "power law": jnp.sqrt(x),
        "gaussian": jnp.exp(-0.5 * ((x - centre) / width) ** 2),
        "delta": delta,
        "square pulse": jnp.where((x >= 0.35) & (x <= 0.65), 1.0, 0.0),
        "smooth step": 0.5 * (1.0 + jnp.tanh((x - centre) / 0.08)),
        "sinusoidal": 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x)),
        "lorentzian": 1.0 / (1.0 + ((x - centre) / width) ** 2),
        "exponential decay": jnp.exp(-4.0 * x),
    }


def _random_curve_names(names, count, rng):
    """Return exactly count names, using each curve once before repeating."""

    selections = []
    while len(selections) < count:
        selections.extend(rng.permutation(names).tolist())
    return selections[:count]


def make_initial_array_sets(config, init_scale=0.1):
    """Create ``sim_num`` independently selected initial control sets."""

    templates = _curve_templates(config.N + 1, config.dtype)
    curve_names = tuple(templates)
    control_names = ("u", "v") if config.loss else ("a",)
    rng = np.random.default_rng(config.seed)
    choices = {
        name: _random_curve_names(curve_names, config.sim_num, rng)
        for name in control_names
    }

    initial_sets = []
    for index in range(config.sim_num):
        selected_types = {name: choices[name][index] for name in control_names}
        arrays = {
            name: init_scale * templates[curve_type]
            for name, curve_type in selected_types.items()
        }
        initial_sets.append(
            InitialArraySet(arrays=arrays, curve_types=selected_types)
        )
    return tuple(initial_sets)


class Test:
    def __init__(
        self,
        config: Config,
        init_scale=0.1,
        plot_columns=3,
        initial_array_factory=make_initial_array_sets,
        autorun=True,
        make_plots=True,
    ):
        if not isinstance(config, Config):
            raise TypeError("config must be a Config configuration.")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive.")
        if int(plot_columns) < 1:
            raise ValueError("plot_columns must be at least 1.")
        if not callable(initial_array_factory):
            raise TypeError("initial_array_factory must be callable.")

        self.config = config
        self.init_scale = float(init_scale)
        self.plot_columns = int(plot_columns)
        self.initial_array_factory = initial_array_factory
        self.initial_array_sets = initial_array_factory(config, self.init_scale)
        if len(self.initial_array_sets) != config.sim_num:
            raise ValueError("initial_array_factory must return config.sim_num sets.")

        self.result = None
        if autorun:
            self.run(make_plots=make_plots)

    @property
    def initial_arrays(self):
        return tuple(initial_set.arrays for initial_set in self.initial_array_sets)

    @property
    def results(self):
        """Compatibility alias for code that previously used ``test.results``."""

        return self.result

    def _print_parameters(self):
        print("Parameters used:")
        for name, value in asdict(self.config).items():
            print(f"  {name}: {value}")
        print(f"  init_scale: {self.init_scale}")
        print(f"  plot_columns: {self.plot_columns}")

    def run(self, make_plots=True):
        self._print_parameters()

        simulations = []
        optimisation_runtimes = []
        for index, initial_set in enumerate(self.initial_array_sets, start=1):
            start_time = time.perf_counter()
            optimised_data = Simulation(self.config).optimise(initial_set.arrays)
            jax.block_until_ready(optimised_data)
            runtime = time.perf_counter() - start_time
            optimisation_runtimes.append(runtime)
            print(f"{index}/{self.config.sim_num} : {runtime:.3f} s")

            simulations.append(
                SimulationResult(
                    initial_arrays=initial_set.arrays,
                    initial_curve_types=initial_set.curve_types,
                    optimised_data=optimised_data,
                    objective_history=optimised_data["history"],
                )
            )

        simulations = tuple(simulations)
        fig = axes = None
        if make_plots:
            fig, axes = self.plot_simulations(simulations)

        average_runtime = sum(optimisation_runtimes) / len(optimisation_runtimes)
        print(f"Average runtime per optimisation: {average_runtime:.3f} s")

        self.result = TestResult(
            config=self.config,
            simulations=simulations,
            fig=fig,
            axes=axes,
        )
        return self.result

    def plot_simulations(self, simulations=None):
        if simulations is None:
            if self.result is None:
                raise ValueError("Run the test before plotting its outcome.")
            simulations = self.result.simulations

        ncols = min(self.plot_columns, len(simulations))
        nrows = math.ceil(len(simulations) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.0 * nrows),
            squeeze=False,
            sharex=True,
        )
        axes = axes.ravel()
        time = np.asarray(self.config.time_grid)

        for ax, simulation in zip(axes, simulations):
            bounded = simulation.optimised_data["bound"]
            if self.config.loss:
                ax.plot(time, np.asarray(bounded["u"]), label="u")
                ax.plot(time, np.asarray(bounded["v"]), label="v")
                ax.legend(fontsize="small")
            else:
                ax.plot(time, np.asarray(bounded["a"]))
                ax.set_ylabel("Scattering length")

            title = ", ".join(
                f"{name}: {curve_type}"
                for name, curve_type in simulation.initial_curve_types.items()
            )
            ax.set_title(title)
            ax.set_xlabel("Dimensionless time")

        for ax in axes[len(simulations):]:
            ax.set_visible(False)

        interaction_type = "Inelastic" if self.config.loss else "Elastic"
        fig.suptitle(f"{interaction_type} interaction")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return fig, axes
