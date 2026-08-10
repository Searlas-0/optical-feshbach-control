#!/usr/bin/env python3
"""Create immutable YAML configs from stable defaults plus optional overrides.

Running this file without arguments writes a uniquely named config containing
the canonical defaults.  For an experiment, call ``make_config(...)`` with only
the values that differ; the default mappings themselves are never edited.

ISOLATION RULE: this process only validates settings and emits YAML.  It must
never import the runner, access results, perform calculations, or plot.  Random
config/batch IDs are generated once and embedded so reruns reproduce their
initial controls exactly.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofc.config import DEFAULT_RESULTS_DATABASE, make_document, write_config


def default_parameters() -> dict[str, Any]:
    """Return a fresh copy of the canonical optimization defaults."""

    return {
        "N": 100,
        "t_interval": 4.0,
        "r_bg": 0.01,
        "u_isbound": True,
        "v_isbound": True,
        "u_max": 70.0,
        "v_max": 100.0,
        "slew_limit": 0.05,
        "optimizer": "adam",
        "schedule": [
            (5_000, 1.0),
            (5_000, 0.5),
            (7_500, 0.5),
        ],
        "adam_learning_rate": 1e-2,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "lbfgs_history_size": 10,
        "lbfgs_max_linesearch_steps": 20,
        "lbfgs_tolerance": 1e-6,
        "peak_initial_step_size": 1e-2,
        "peak_min_step_size": 1e-12,
        "peak_max_step_size": 0.1,
        "peak_backtracking_factor": 0.5,
        "peak_step_growth": 1.5,
        "peak_armijo": 1e-4,
        "peak_max_linesearch_steps": 24,
        "smoothness": 1e-2,
        "u_smooth": None,
        "v_smooth": None,
        "sharpness": 0.0,
        "u_sharp": None,
        "v_sharp": None,
        "block_size": 500,
        "J_tol": 1e-5,
        "u_tol": 1e-4,
        "v_tol": 1e-4,
        "projected_gradient_tol": 1e-4,
        "projected_gradient_alpha": 1.0,
        "grid_refinement_tol": 1e-2,
        "grid_refinement_y_floor": 1e-12,
    }


def default_runtime() -> dict[str, Any]:
    """Return execution defaults targeting the one canonical database pair."""

    # These match the previous Fourier starts. Set concurrent_workers through
    # a runtime override when a particular allocation requires a different value.
    return {
        "initialisations": 10,
        "fourier_num_modes": 5,
        "fourier_rms_amplitude": 0.3,
        "fourier_intensity_fraction": 0.3,
        "use_jit": True,
        "use_x64": True,
        "device": "auto",
        "concurrent_workers": 4,
        "max_cases_per_batch": None,
        "max_initialisations_per_batch": None,
        "max_steps_per_chunk": None,
        "max_batch_elapsed_seconds": None,
        "max_elapsed_seconds": None,
        "distribute_max_elapsed_across_batches": False,
        "repeat_schedule_until_stable": False,
        "auto_halt": True,
        "database": DEFAULT_RESULTS_DATABASE,
    }


def _default_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"default_{timestamp}"


def make_config(
    name: str | None = None,
    description: str | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    query: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Write one config using defaults updated by the supplied overrides.

    With no arguments this writes a complete default config under
    ``run_config/``. Lists in ``parameters`` retain their sweep meaning. An
    optional query selects stored controls that the runner appends to the
    random Fourier starts.
    """

    config_name = _default_name() if name is None else str(name)
    if not config_name or Path(config_name).name != config_name:
        raise ValueError("name must be a non-empty filename-safe config name.")

    parameter_values = default_parameters()
    parameter_values.update(dict(parameters or {}))
    runtime_values = default_runtime()
    runtime_values.update(dict(runtime or {}))

    document = make_document(
        name=config_name,
        description=(
            "Canonical default inelastic optical Feshbach optimization"
            if description is None
            else str(description)
        ),
        parameters=parameter_values,
        runtime=runtime_values,
        query=query,
    )
    directory = Path(__file__).resolve().parent if output_dir is None else Path(output_dir)
    return write_config(document, directory / f"{config_name}.yaml")


def main() -> Path:
    # Default behavior needs no editing. For a named experiment, replace the
    # next line with, for example:
    # path = make_config(
    #     name="u_max_sweep",
    #     description="Determine a useful intensity cap",
    #     parameters={"u_max": [25.0, 50.0, 100.0]},
    # )
    path = make_config()
    print(path)
    return path


if __name__ == "__main__":
    main()
