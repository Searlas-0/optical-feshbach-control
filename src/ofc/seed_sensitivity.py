"""Configuration composition for the fixed-parameter seed-sensitivity study.

The notebook exposes only editable experiment arguments.  This module owns the
repeated four-stage dependency wiring so every stored-control handoff is exact,
auditable, and tied to one immutable source config ID.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from .config import ConfigDocument, make_document


STAGES = ("exploration", "all_polish", "top20_loose", "top1_strict")


def loose_stage_name(top_count: int) -> str:
    """Return the config-stage label for a loose-refinement population."""

    return f"top{_positive_integer(top_count, 'top_count')}_loose"


def fixed_seed_stage_name(
    N: int,
    cap: int,
    stage: str,
    exploration_initialisations: int = 1_000,
    *,
    parameter_label: str | None = None,
) -> str:
    """Return the stable config name for one seed-sensitivity stage."""

    dynamic_loose_stage = (
        stage.startswith("top")
        and stage.endswith("_loose")
        and stage[3:-6].isdigit()
        and int(stage[3:-6]) > 0
    )
    if stage not in STAGES and not dynamic_loose_stage:
        raise ValueError(
            f"stage must be one of {STAGES} or have the form 'top<count>_loose'."
        )
    initialization_count = _positive_integer(
        exploration_initialisations, "exploration_initialisations"
    )
    if parameter_label is not None:
        if not parameter_label or any(
            not (character.isalnum() or character == "_")
            for character in parameter_label
        ):
            raise ValueError(
                "parameter_label must contain only letters, numbers, and underscores."
            )
        label = f"_{parameter_label}"
    else:
        label = ""
    return (
        f"N{int(N)}_u{int(cap)}_fixed_parameter{label}_"
        f"seed{initialization_count}_"
        f"{stage}_gpu"
    )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a positive number.")
    return float(value)


def _cap_settings(settings: Mapping[str, Any], cap: int) -> dict[str, float]:
    required = (
        "adam_learning_rate",
        "adam_beta1",
        "adam_beta2",
        "smoothness",
        "sharpness",
    )
    missing = [name for name in required if name not in settings]
    if missing:
        raise ValueError(f"u_max={cap} is missing settings: {', '.join(missing)}.")
    output = {name: float(settings[name]) for name in required}
    if output["adam_learning_rate"] <= 0.0:
        raise ValueError(f"u_max={cap} adam_learning_rate must be positive.")
    for name in ("adam_beta1", "adam_beta2"):
        if not 0.0 <= output[name] < 1.0:
            raise ValueError(f"u_max={cap} {name} must be in [0, 1).")
    for name in ("smoothness", "sharpness"):
        if output[name] < 0.0:
            raise ValueError(f"u_max={cap} {name} must be non-negative.")
    return output


def _parameters(
    N: int,
    cap: int,
    settings: Mapping[str, float],
    *,
    optimizer: str,
    schedule: Sequence[Sequence[Real]],
    block_size: int,
    score_tolerance: float | None,
    control_tolerance: float | None,
    gradient_tolerance: float | None,
) -> dict[str, Any]:
    return {
        "N": N,
        "t_interval": 4.0,
        "r_bg": -0.008716,
        "u_isbound": True,
        "v_isbound": True,
        "u_max": float(cap),
        "v_max": 1000.0,
        "slew_limit": 0.05,
        "optimizer": optimizer,
        "schedule": [tuple(stage) for stage in schedule],
        "adam_learning_rate": settings["adam_learning_rate"],
        "adam_beta1": settings["adam_beta1"],
        "adam_beta2": settings["adam_beta2"],
        "adam_eps": 1e-8,
        "lbfgs_history_size": 100,
        "lbfgs_max_linesearch_steps": 100,
        "lbfgs_tolerance": 1e-12,
        "peak_initial_step_size": 1e-2,
        "peak_min_step_size": 1e-12,
        "peak_max_step_size": 0.1,
        "peak_backtracking_factor": 0.5,
        "peak_step_growth": 1.5,
        "peak_armijo": 1e-4,
        "peak_max_linesearch_steps": 24,
        "smoothness": settings["smoothness"],
        "u_smooth": None,
        "v_smooth": None,
        "sharpness": settings["sharpness"],
        "u_sharp": None,
        "v_sharp": None,
        "block_size": block_size,
        "J_tol": score_tolerance,
        "u_tol": control_tolerance,
        "v_tol": control_tolerance,
        "projected_gradient_tol": gradient_tolerance,
        "projected_gradient_alpha": 1.0,
        "grid_refinement_tol": (
            1e-3 if score_tolerance is not None and score_tolerance <= 1e-6 else 1e-2
        ),
        "grid_refinement_y_floor": 1e-12,
    }


def _runtime(
    *,
    database: str,
    initialisations: int,
    batch_size: int,
    max_steps_per_chunk: int,
    repeat_until_stable: bool,
    auto_halt: bool,
    max_batch_elapsed_seconds: float | None,
    max_elapsed_seconds: float | None,
    fourier_num_modes: int,
    fourier_rms_amplitude: float,
    fourier_intensity_fraction: float,
    device: str = "gpu",
) -> dict[str, Any]:
    return {
        "initialisations": initialisations,
        "fourier_num_modes": fourier_num_modes,
        "fourier_rms_amplitude": fourier_rms_amplitude,
        "fourier_intensity_fraction": fourier_intensity_fraction,
        "use_jit": True,
        "use_x64": True,
        "device": device,
        "concurrent_workers": 1,
        "max_cases_per_batch": None,
        "max_initialisations_per_batch": batch_size,
        "max_steps_per_chunk": max_steps_per_chunk,
        "max_batch_elapsed_seconds": max_batch_elapsed_seconds,
        "max_elapsed_seconds": max_elapsed_seconds,
        "distribute_max_elapsed_across_batches": False,
        "repeat_schedule_until_stable": repeat_until_stable,
        "auto_halt": auto_halt,
        "database": database,
    }


def _source_query(
    source: ConfigDocument,
    *,
    N: int,
    cap: int,
    settings: Mapping[str, float],
    limit: int | None,
    terminal_only: bool = False,
    resume_optimizer: bool,
) -> dict[str, Any]:
    where: dict[str, Any] = {
        "status": "complete",
        "config_id": source.config_id,
        "config_name": source.name,
        "N": N,
        "u_max": float(cap),
        "smoothness": settings["smoothness"],
        "sharpness": settings["sharpness"],
    }
    if terminal_only:
        where["termination_reason"] = ["stability", "time_limit"]
    return {
        "where": where,
        "limit": limit,
        "order_by": "best_objective",
        "descending": True,
        "control_kind": "best",
        "resume_optimizer": resume_optimizer,
        "perturbed": False,
        "match_parameters": [],
    }


def fixed_cap_seed_sensitivity_documents(
    *,
    cap_settings: Mapping[int, Mapping[str, Any]],
    resolutions: Sequence[int],
    database: str,
    exploration_initialisations: int = 1_000,
    exploration_schedule: Sequence[Sequence[Real]] = (
        (5_000, 1.5),
        (15_000, 0.75),
    ),
    all_polish_steps: int = 2_000,
    batch_sizes: Mapping[int, int] | None = None,
    loose_batch_sizes: Mapping[int, int] | None = None,
    top_count: int = 20,
    loose_max_elapsed_seconds: Real = 2 * 60 * 60,
    strict_max_elapsed_seconds: Real = 4 * 60 * 60,
    fourier_num_modes: int = 6,
    fourier_rms_amplitude: Real = 0.8,
    fourier_intensity_fraction: Real = 0.5,
    resume_optimizer: bool = False,
    parameter_label: str | None = None,
) -> tuple[ConfigDocument, ...]:
    """Build four dependent GPU configs for every resolution and cap.

    Ordering is resolution-major and preserves the insertion order of
    ``cap_settings``.  A notebook can therefore put 1280, 160, and 40 in that
    mapping to retain the requested descending-cap order.
    """

    if not cap_settings:
        raise ValueError("cap_settings cannot be empty.")
    if not resolutions:
        raise ValueError("resolutions cannot be empty.")
    if not isinstance(database, str) or not database:
        raise ValueError("database must be a non-empty path string.")
    if not isinstance(resume_optimizer, bool):
        raise ValueError("resume_optimizer must be a boolean.")
    if resume_optimizer:
        raise ValueError(
            "This mixed-optimizer pipeline requires resume_optimizer=False; "
            "every handoff restores controls and starts a fresh optimizer state."
        )

    resolution_values = tuple(_positive_integer(value, "resolution") for value in resolutions)
    if len(set(resolution_values)) != len(resolution_values):
        raise ValueError("resolutions cannot contain duplicates.")
    initialization_count = _positive_integer(
        exploration_initialisations, "exploration_initialisations"
    )
    polish_steps = _positive_integer(all_polish_steps, "all_polish_steps")
    selected_top_count = _positive_integer(top_count, "top_count")
    if selected_top_count > initialization_count:
        raise ValueError("top_count cannot exceed exploration_initialisations.")

    schedule = tuple(tuple(stage) for stage in exploration_schedule)
    if not schedule:
        raise ValueError("exploration_schedule cannot be empty.")
    exploration_steps = sum(_positive_integer(stage[0], "schedule steps") for stage in schedule)
    if exploration_steps > 20_000:
        raise ValueError("The exploratory schedule must not exceed 20,000 steps.")

    configured_batch_sizes = dict(batch_sizes or {})
    missing_batch_sizes = [N for N in resolution_values if N not in configured_batch_sizes]
    if missing_batch_sizes:
        raise ValueError(
            "batch_sizes is missing resolutions: "
            + ", ".join(map(str, missing_batch_sizes))
        )
    for N in resolution_values:
        configured_batch_sizes[N] = _positive_integer(
            configured_batch_sizes[N], f"batch_sizes[{N}]"
        )

    configured_loose_batch_sizes = (
        dict(loose_batch_sizes) if loose_batch_sizes is not None else None
    )
    if configured_loose_batch_sizes is not None:
        missing_loose_batch_sizes = [
            N for N in resolution_values if N not in configured_loose_batch_sizes
        ]
        if missing_loose_batch_sizes:
            raise ValueError(
                "loose_batch_sizes is missing resolutions: "
                + ", ".join(map(str, missing_loose_batch_sizes))
            )
        for N in resolution_values:
            configured_loose_batch_sizes[N] = _positive_integer(
                configured_loose_batch_sizes[N], f"loose_batch_sizes[{N}]"
            )

    loose_seconds = _positive_float(
        loose_max_elapsed_seconds, "loose_max_elapsed_seconds"
    )
    strict_seconds = _positive_float(
        strict_max_elapsed_seconds, "strict_max_elapsed_seconds"
    )
    modes = _positive_integer(fourier_num_modes, "fourier_num_modes")
    amplitude = _positive_float(fourier_rms_amplitude, "fourier_rms_amplitude")
    intensity_fraction = _positive_float(
        fourier_intensity_fraction, "fourier_intensity_fraction"
    )

    settings_by_cap = {
        _positive_integer(cap, "u_max"): _cap_settings(settings, int(cap))
        for cap, settings in cap_settings.items()
    }
    documents: list[ConfigDocument] = []
    for N in resolution_values:
        for cap, settings in settings_by_cap.items():
            exploration = make_document(
                name=fixed_seed_stage_name(
                    N,
                    cap,
                    "exploration",
                    initialization_count,
                    parameter_label=parameter_label,
                ),
                description=(
                    f"N={N} u_max={cap} fixed-parameter seed-sensitivity search "
                    f"with {initialization_count} broad random Fourier starts and "
                    f"{exploration_steps} high-learning-rate Adam steps."
                ),
                parameters=_parameters(
                    N,
                    cap,
                    settings,
                    optimizer="adam",
                    schedule=schedule,
                    block_size=1_000,
                    score_tolerance=None,
                    control_tolerance=None,
                    gradient_tolerance=None,
                ),
                runtime=_runtime(
                    database=database,
                    initialisations=initialization_count,
                    batch_size=configured_batch_sizes[N],
                    max_steps_per_chunk=2_000,
                    repeat_until_stable=False,
                    auto_halt=False,
                    max_batch_elapsed_seconds=None,
                    max_elapsed_seconds=None,
                    fourier_num_modes=modes,
                    fourier_rms_amplitude=amplitude,
                    fourier_intensity_fraction=intensity_fraction,
                ),
                query=None,
            )
            all_polish = make_document(
                name=fixed_seed_stage_name(
                    N,
                    cap,
                    "all_polish",
                    initialization_count,
                    parameter_label=parameter_label,
                ),
                description=(
                    f"N={N} u_max={cap} monotone peak polish of every saved "
                    f"exploratory seed for {polish_steps} steps."
                ),
                parameters=_parameters(
                    N,
                    cap,
                    settings,
                    optimizer="peak_refinement",
                    schedule=((polish_steps, 1.0),),
                    block_size=250,
                    score_tolerance=None,
                    control_tolerance=None,
                    gradient_tolerance=None,
                ),
                runtime=_runtime(
                    database=database,
                    initialisations=0,
                    batch_size=configured_batch_sizes[N],
                    max_steps_per_chunk=250,
                    repeat_until_stable=False,
                    auto_halt=False,
                    max_batch_elapsed_seconds=None,
                    max_elapsed_seconds=None,
                    fourier_num_modes=modes,
                    fourier_rms_amplitude=amplitude,
                    fourier_intensity_fraction=intensity_fraction,
                ),
                query=_source_query(
                    exploration,
                    N=N,
                    cap=cap,
                    settings=settings,
                    limit=None,
                    resume_optimizer=resume_optimizer,
                ),
            )
            top_loose = make_document(
                name=fixed_seed_stage_name(
                    N,
                    cap,
                    loose_stage_name(selected_top_count),
                    initialization_count,
                    parameter_label=parameter_label,
                ),
                description=(
                    f"N={N} u_max={cap} loose L-BFGS-B refinement of the top "
                    f"{selected_top_count} polished random seeds."
                ),
                parameters=_parameters(
                    N,
                    cap,
                    settings,
                    optimizer="lbfgs",
                    schedule=((50, 1.0),),
                    block_size=10,
                    score_tolerance=1e-5,
                    control_tolerance=1e-4,
                    gradient_tolerance=1e-4,
                ),
                runtime=_runtime(
                    database=database,
                    initialisations=0,
                    batch_size=(
                        selected_top_count
                        if configured_loose_batch_sizes is None
                        else configured_loose_batch_sizes[N]
                    ),
                    max_steps_per_chunk=50,
                    repeat_until_stable=True,
                    auto_halt=True,
                    max_batch_elapsed_seconds=loose_seconds,
                    max_elapsed_seconds=(
                        loose_seconds if configured_loose_batch_sizes is None else None
                    ),
                    fourier_num_modes=modes,
                    fourier_rms_amplitude=amplitude,
                    fourier_intensity_fraction=intensity_fraction,
                ),
                query=_source_query(
                    all_polish,
                    N=N,
                    cap=cap,
                    settings=settings,
                    limit=selected_top_count,
                    resume_optimizer=resume_optimizer,
                ),
            )
            top_strict = make_document(
                name=fixed_seed_stage_name(
                    N,
                    cap,
                    "top1_strict",
                    initialization_count,
                    parameter_label=parameter_label,
                ),
                description=(
                    f"N={N} u_max={cap} strict monotone peak refinement of the "
                    "best loose L-BFGS-B solution."
                ),
                parameters=_parameters(
                    N,
                    cap,
                    settings,
                    optimizer="peak_refinement",
                    schedule=((250, 1.0),),
                    block_size=25,
                    score_tolerance=1e-6,
                    control_tolerance=1e-5,
                    gradient_tolerance=1e-5,
                ),
                runtime=_runtime(
                    database=database,
                    initialisations=0,
                    batch_size=1,
                    max_steps_per_chunk=250,
                    repeat_until_stable=True,
                    auto_halt=True,
                    max_batch_elapsed_seconds=strict_seconds,
                    max_elapsed_seconds=strict_seconds,
                    fourier_num_modes=modes,
                    fourier_rms_amplitude=amplitude,
                    fourier_intensity_fraction=intensity_fraction,
                ),
                query=_source_query(
                    top_loose,
                    N=N,
                    cap=cap,
                    settings=settings,
                    limit=1,
                    terminal_only=True,
                    resume_optimizer=resume_optimizer,
                ),
            )
            documents.extend((exploration, all_polish, top_loose, top_strict))
    return tuple(documents)


def fixed_endpoint_seed_sensitivity_documents(
    *,
    cap_endpoint_settings: Mapping[int, Mapping[str, Mapping[str, Any]]],
    resolutions: Sequence[int],
    database: str,
    exploration_initialisations: int = 1_000,
    exploration_schedule: Sequence[Sequence[Real]] = (
        (5_000, 1.5),
        (15_000, 0.75),
    ),
    all_polish_steps: int = 2_000,
    batch_sizes: Mapping[int, int] | None = None,
    loose_batch_sizes: Mapping[int, int] | None = None,
    top_count: int = 20,
    loose_max_elapsed_seconds: Real = 2 * 60 * 60,
    strict_max_elapsed_seconds: Real = 4 * 60 * 60,
    fourier_num_modes: int = 6,
    fourier_rms_amplitude: Real = 0.8,
    fourier_intensity_fraction: Real = 0.5,
    resume_optimizer: bool = False,
    include_strict: bool = True,
    parameter_label_suffix: str = "",
) -> tuple[ConfigDocument, ...]:
    """Build matched low/high regularization pipelines in resolution-major order."""

    if not cap_endpoint_settings:
        raise ValueError("cap_endpoint_settings cannot be empty.")
    if not isinstance(include_strict, bool):
        raise ValueError("include_strict must be a boolean.")
    if any(
        not (character.isalnum() or character == "_")
        for character in parameter_label_suffix
    ):
        raise ValueError(
            "parameter_label_suffix must contain only letters, numbers, and underscores."
        )
    endpoint_order = ("low", "high")
    for cap, endpoints in cap_endpoint_settings.items():
        if set(endpoints) != set(endpoint_order):
            raise ValueError(
                f"u_max={cap} must provide exactly the 'low' and 'high' endpoints."
            )

    documents: list[ConfigDocument] = []
    for N in resolutions:
        for cap, endpoints in cap_endpoint_settings.items():
            for endpoint in endpoint_order:
                endpoint_documents = fixed_cap_seed_sensitivity_documents(
                    cap_settings={cap: endpoints[endpoint]},
                    resolutions=(N,),
                    database=database,
                    exploration_initialisations=exploration_initialisations,
                    exploration_schedule=exploration_schedule,
                    all_polish_steps=all_polish_steps,
                    batch_sizes=batch_sizes,
                    loose_batch_sizes=loose_batch_sizes,
                    top_count=top_count,
                    loose_max_elapsed_seconds=loose_max_elapsed_seconds,
                    strict_max_elapsed_seconds=strict_max_elapsed_seconds,
                    fourier_num_modes=fourier_num_modes,
                    fourier_rms_amplitude=fourier_rms_amplitude,
                    fourier_intensity_fraction=fourier_intensity_fraction,
                    resume_optimizer=resume_optimizer,
                    parameter_label=(
                        f"{endpoint}_regularization{parameter_label_suffix}"
                    ),
                )
                documents.extend(
                    endpoint_documents if include_strict else endpoint_documents[:3]
                )
    return tuple(documents)


def strict_refinement_document(
    *,
    name: str,
    N: int,
    cap: int,
    settings: Mapping[str, Any],
    source_database: str,
    target_database: str,
    source_run_id: int,
    max_elapsed_seconds: Real = 4 * 60 * 60,
) -> ConfigDocument:
    """Build one strict CPU refinement from an exact cross-database challenger."""

    resolution = _positive_integer(N, "N")
    u_max = _positive_integer(cap, "u_max")
    run_id = _positive_integer(source_run_id, "source_run_id")
    strict_seconds = _positive_float(max_elapsed_seconds, "max_elapsed_seconds")
    normalized_settings = _cap_settings(settings, u_max)
    if not isinstance(source_database, str) or not source_database:
        raise ValueError("source_database must be a non-empty path string.")
    if not isinstance(target_database, str) or not target_database:
        raise ValueError("target_database must be a non-empty path string.")
    return make_document(
        name=name,
        description=(
            f"Strict Slurm CPU peak refinement for N={resolution}, u_max={u_max}, "
            f"starting from bar challenger run_id={run_id}."
        ),
        parameters=_parameters(
            resolution,
            u_max,
            normalized_settings,
            optimizer="peak_refinement",
            schedule=((250, 1.0),),
            block_size=25,
            score_tolerance=1e-6,
            control_tolerance=1e-5,
            gradient_tolerance=1e-5,
        ),
        runtime=_runtime(
            database=target_database,
            initialisations=0,
            batch_size=1,
            max_steps_per_chunk=250,
            repeat_until_stable=True,
            auto_halt=True,
            max_batch_elapsed_seconds=strict_seconds,
            max_elapsed_seconds=strict_seconds,
            fourier_num_modes=6,
            fourier_rms_amplitude=0.8,
            fourier_intensity_fraction=0.5,
            device="cpu",
        ),
        query={
            "database": source_database,
            "where": {
                "run_id": run_id,
                "status": ["running", "complete", "failed"],
            },
            "limit": 1,
            "order_by": "best_objective",
            "descending": True,
            "control_kind": "best",
            "resume_optimizer": False,
            "perturbed": False,
            "match_parameters": [],
        },
    )
