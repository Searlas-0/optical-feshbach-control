"""Configuration schema and validation.

Sweep values are ordinary YAML lists.  Every combination is expanded, while
``N``, ``schedule``, and ``t_interval`` define separate persisted batches.
The time split permits independent Slurm array tasks while preserving one
immutable sweep configuration. All other numerical sweeps share a JAX batch.

Isolation boundary: config code only validates/serializes configuration data.
It never imports the runner, reads results, performs calculations, or plots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
import json
import math
from numbers import Integral, Real
from pathlib import Path
import secrets
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = {2, SCHEMA_VERSION}
ID_MAX = 2**63 - 1
DEFAULT_RESULTS_DATABASE = "results/results.sqlite3"
NON_SWEEP_FIELDS = {
    "u_isbound",
    "v_isbound",
    "optimizer",
    "block_size",
    "lbfgs_history_size",
    "lbfgs_max_linesearch_steps",
    "peak_max_linesearch_steps",
}
DEFAULT_SCHEDULE = (
    (5_000, 1.0),
    (5_000, 0.5),
    (7_500, 0.5),
    (7_500, 0.5),
)
DEFAULT_QUERY_PERTURBATION_LEVELS = (0.0005, 0.001, 0.0025, 0.005, 0.01)


def random_id() -> int:
    """Return a non-zero cryptographically random signed-64-bit identifier."""

    return secrets.randbelow(ID_MAX - 1) + 1


@dataclass(frozen=True)
class ResolvedConfig:
    """One scalar point in a possibly multidimensional configuration sweep."""

    N: int = 100
    t_interval: float = 1.0
    r_bg: float = 1.0
    u_isbound: bool = True
    v_isbound: bool = True
    u_max: float = 50.0
    v_max: float = 200.0
    slew_limit: float = 0.05
    optimizer: str = "adam"

    schedule: tuple[tuple[int, float], ...] = field(
        default_factory=lambda: DEFAULT_SCHEDULE
    )
    adam_learning_rate: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    lbfgs_history_size: int = 10
    lbfgs_max_linesearch_steps: int = 20
    lbfgs_tolerance: float = 1e-6
    peak_initial_step_size: float = 1e-2
    peak_min_step_size: float = 1e-12
    peak_max_step_size: float = 0.1
    peak_backtracking_factor: float = 0.5
    peak_step_growth: float = 1.5
    peak_armijo: float = 1e-4
    peak_max_linesearch_steps: int = 24

    smoothness: float = 1e-5
    u_smooth: float | None = None
    v_smooth: float | None = None
    sharpness: float = 0.0
    u_sharp: float | None = None
    v_sharp: float | None = None

    block_size: int = 500
    J_tol: float | None = 1e-5
    u_tol: float | None = 1e-4
    v_tol: float | None = 1e-4
    projected_gradient_tol: float | None = 1e-4
    projected_gradient_alpha: float = 1.0
    grid_refinement_tol: float = 1e-2
    grid_refinement_y_floor: float = 1e-12

    def __post_init__(self) -> None:
        integer_fields = {
            "N": self.N,
            "block_size": self.block_size,
            "lbfgs_history_size": self.lbfgs_history_size,
            "lbfgs_max_linesearch_steps": self.lbfgs_max_linesearch_steps,
            "peak_max_linesearch_steps": self.peak_max_linesearch_steps,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))

        if isinstance(self.r_bg, bool) or not isinstance(self.r_bg, Real):
            raise ValueError("r_bg must be a finite non-zero number.")
        r_bg = float(self.r_bg)
        if not math.isfinite(r_bg) or r_bg == 0.0:
            raise ValueError("r_bg must be a finite non-zero number.")
        object.__setattr__(self, "r_bg", r_bg)

        positive = {
            "t_interval": self.t_interval,
            "u_max": self.u_max,
            "v_max": self.v_max,
            "slew_limit": self.slew_limit,
            "adam_learning_rate": self.adam_learning_rate,
            "adam_eps": self.adam_eps,
            "lbfgs_tolerance": self.lbfgs_tolerance,
            "peak_initial_step_size": self.peak_initial_step_size,
            "peak_min_step_size": self.peak_min_step_size,
            "peak_max_step_size": self.peak_max_step_size,
            "projected_gradient_alpha": self.projected_gradient_alpha,
            "grid_refinement_tol": self.grid_refinement_tol,
            "grid_refinement_y_floor": self.grid_refinement_y_floor,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite positive number.")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number.")
            object.__setattr__(self, name, value)
        if self.slew_limit > 1.0:
            raise ValueError("slew_limit must not exceed 1.")
        if self.peak_min_step_size > self.peak_initial_step_size:
            raise ValueError(
                "peak_min_step_size must not exceed peak_initial_step_size."
            )
        if self.peak_initial_step_size > self.peak_max_step_size:
            raise ValueError(
                "peak_initial_step_size must not exceed peak_max_step_size."
            )

        for name in ("peak_backtracking_factor", "peak_armijo"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite number in (0, 1).")
            value = float(value)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be a finite number in (0, 1).")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.peak_step_growth, bool)
            or not isinstance(self.peak_step_growth, Real)
            or not math.isfinite(float(self.peak_step_growth))
            or float(self.peak_step_growth) < 1.0
        ):
            raise ValueError("peak_step_growth must be finite and at least 1.")
        object.__setattr__(self, "peak_step_growth", float(self.peak_step_growth))

        for name in (
            "smoothness",
            "u_smooth",
            "v_smooth",
            "sharpness",
            "u_sharp",
            "v_sharp",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be finite and non-negative or null.")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative or null.")
            object.__setattr__(self, name, value)

        for name in ("J_tol", "u_tol", "v_tol", "projected_gradient_tol"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be finite and positive or null.")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive or null.")
            object.__setattr__(self, name, value)

        for name in ("adam_beta1", "adam_beta2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
            object.__setattr__(self, name, value)

        optimizer = str(self.optimizer).lower()
        if optimizer not in {"adam", "lbfgs", "peak_refinement"}:
            raise ValueError(
                "optimizer must be 'adam', 'lbfgs', or 'peak_refinement'."
            )
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "u_isbound", bool(self.u_isbound))
        object.__setattr__(self, "v_isbound", bool(self.v_isbound))
        object.__setattr__(self, "schedule", normalise_schedule(self.schedule))
        if optimizer != "adam" and any(
            multiplier != 1.0 for _, multiplier in self.schedule
        ):
            raise ValueError(
                "Non-Adam schedule multipliers must all equal 1.0."
            )

    @property
    def dt(self) -> float:
        return self.t_interval / self.N

    @property
    def effective_u_smooth(self) -> float:
        return self.smoothness if self.u_smooth is None else self.u_smooth

    @property
    def effective_v_smooth(self) -> float:
        return self.smoothness if self.v_smooth is None else self.v_smooth

    @property
    def effective_u_sharp(self) -> float:
        return self.sharpness if self.u_sharp is None else self.u_sharp

    @property
    def effective_v_sharp(self) -> float:
        return self.sharpness if self.v_sharp is None else self.v_sharp

    def parameters(self) -> dict[str, Any]:
        values = asdict(self)
        values["schedule"] = [list(stage) for stage in self.schedule]
        values["dt"] = self.dt
        return values


@dataclass(frozen=True)
class RuntimeConfig:
    initialisations: int = 24
    fourier_num_modes: int = 5
    fourier_rms_amplitude: float = 0.3
    fourier_intensity_fraction: float | str = 0.3
    fourier_intensity_auto_database: str | None = None
    use_jit: bool = True
    use_x64: bool = True
    device: str = "auto"
    concurrent_workers: int = 2
    max_cases_per_batch: int | None = None
    max_initialisations_per_batch: int | None = None
    max_steps_per_chunk: int | None = None
    max_batch_elapsed_seconds: float | None = None
    max_elapsed_seconds: float | None = None
    distribute_max_elapsed_across_batches: bool = False
    repeat_schedule_until_stable: bool = False
    auto_halt: bool = True
    # All production configs share one canonical physical database. The
    # adjacent results.parameters.sqlite3 file is derived automatically.
    database: str = DEFAULT_RESULTS_DATABASE

    def __post_init__(self) -> None:
        for name in ("initialisations", "fourier_num_modes", "concurrent_workers"):
            value = getattr(self, name)
            minimum = 0 if name == "initialisations" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
            object.__setattr__(self, name, int(value))
        if self.max_cases_per_batch is not None:
            value = self.max_cases_per_batch
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("max_cases_per_batch must be a positive integer or null.")
            object.__setattr__(self, "max_cases_per_batch", int(value))
        if self.max_initialisations_per_batch is not None:
            value = self.max_initialisations_per_batch
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(
                    "max_initialisations_per_batch must be a positive integer or null."
                )
            object.__setattr__(self, "max_initialisations_per_batch", int(value))
        if self.max_steps_per_chunk is not None:
            value = self.max_steps_per_chunk
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(
                    "max_steps_per_chunk must be a positive integer or null."
                )
            object.__setattr__(self, "max_steps_per_chunk", int(value))
        for name in ("max_batch_elapsed_seconds", "max_elapsed_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be finite and positive or null.")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive or null.")
            object.__setattr__(self, name, value)
        if not isinstance(self.repeat_schedule_until_stable, bool):
            raise ValueError("repeat_schedule_until_stable must be a boolean.")
        if not isinstance(self.distribute_max_elapsed_across_batches, bool):
            raise ValueError(
                "distribute_max_elapsed_across_batches must be a boolean."
            )
        if (
            self.distribute_max_elapsed_across_batches
            and self.max_elapsed_seconds is None
        ):
            raise ValueError(
                "distribute_max_elapsed_across_batches requires max_elapsed_seconds."
            )
        if (
            self.repeat_schedule_until_stable
            and self.max_elapsed_seconds is None
            and self.max_batch_elapsed_seconds is None
        ):
            raise ValueError(
                "repeat_schedule_until_stable requires max_elapsed_seconds or "
                "max_batch_elapsed_seconds."
            )
        if self.max_elapsed_seconds is not None and self.concurrent_workers != 1:
            raise ValueError(
                "max_elapsed_seconds requires concurrent_workers: 1 for one global deadline."
            )
        if not 3 <= self.fourier_num_modes <= 6:
            raise ValueError("fourier_num_modes must be between 3 and 6.")
        value = self.fourier_intensity_fraction
        if isinstance(value, str):
            if value.strip().lower() != "auto":
                raise ValueError(
                    "fourier_intensity_fraction must be a fraction in (0, 1) "
                    "or 'auto'."
                )
            object.__setattr__(self, "fourier_intensity_fraction", "auto")
        else:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    "fourier_intensity_fraction must be a fraction in (0, 1) "
                    "or 'auto'."
                )
            value = float(value)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(
                    "fourier_intensity_fraction must be a fraction in (0, 1) "
                    "or 'auto'."
                )
            object.__setattr__(self, "fourier_intensity_fraction", value)

        auto_database = self.fourier_intensity_auto_database
        if auto_database is not None:
            if not isinstance(auto_database, str) or not auto_database.strip():
                raise ValueError(
                    "fourier_intensity_auto_database must be a non-empty path or null."
                )
            object.__setattr__(
                self, "fourier_intensity_auto_database", auto_database.strip()
            )
        if auto_database is not None and self.fourier_intensity_fraction != "auto":
            raise ValueError(
                "fourier_intensity_auto_database requires "
                "fourier_intensity_fraction: auto."
            )

        for name in ("fourier_rms_amplitude",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if not isinstance(self.auto_halt, bool):
            raise ValueError("auto_halt must be a boolean.")
        device = str(self.device).lower()
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be 'auto', 'cpu', or 'gpu'.")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "use_jit", bool(self.use_jit))
        object.__setattr__(self, "use_x64", bool(self.use_x64))


@dataclass(frozen=True)
class InitializationQuery:
    """Select stored controls to append to the random Fourier starts."""

    where: Mapping[str, Any]
    database: str | None = None
    limit: int | None = None
    order_by: str = "best_score"
    descending: bool = True
    control_kind: str = "best"
    resume_optimizer: bool = False
    perturbed: bool = True
    perturbation_levels: tuple[float, ...] = DEFAULT_QUERY_PERTURBATION_LEVELS
    match_parameters: tuple[str, ...] = ()
    discover_parameters: tuple[str, ...] = ()
    discover_group_parameters: tuple[str, ...] = ()
    fallback_where: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.where, Mapping) or not self.where:
            raise ValueError("query.where must be a non-empty mapping of result filters.")
        where = dict(_plain_data(self.where))
        if any(not isinstance(name, str) or not name for name in where):
            raise ValueError("query.where filter names must be non-empty strings.")
        object.__setattr__(self, "where", where)

        if self.database is not None:
            if not isinstance(self.database, str) or not self.database.strip():
                raise ValueError("query.database must be a non-empty path or null.")
            object.__setattr__(self, "database", self.database.strip())

        if self.limit is not None:
            if (
                isinstance(self.limit, bool)
                or not isinstance(self.limit, Integral)
                or self.limit < 1
            ):
                raise ValueError("query.limit must be a positive integer or null.")
            object.__setattr__(self, "limit", int(self.limit))
        if not isinstance(self.order_by, str) or not self.order_by:
            raise ValueError("query.order_by must be a non-empty result field name.")
        if not isinstance(self.descending, bool):
            raise ValueError("query.descending must be a boolean.")
        control_kind = str(self.control_kind).lower()
        if control_kind not in {"initial", "best", "final"}:
            raise ValueError("query.control_kind must be initial, best, or final.")
        object.__setattr__(self, "control_kind", control_kind)
        if not isinstance(self.resume_optimizer, bool):
            raise ValueError("query.resume_optimizer must be a boolean.")
        if not isinstance(self.perturbed, bool):
            raise ValueError("query.perturbed must be a boolean.")
        if self.resume_optimizer and control_kind == "initial":
            raise ValueError(
                "query.resume_optimizer requires best or final controls."
            )
        if self.resume_optimizer and self.perturbed:
            raise ValueError(
                "query.resume_optimizer requires perturbed: false because Adam "
                "moments only match the exact stored raw controls."
            )

        levels = self.perturbation_levels
        if isinstance(levels, (str, bytes)):
            raise ValueError("query.perturbation_levels must be a non-empty sequence.")
        try:
            levels = tuple(levels)
        except TypeError as exc:
            raise ValueError(
                "query.perturbation_levels must be a non-empty sequence."
            ) from exc
        if not levels:
            raise ValueError("query.perturbation_levels must be a non-empty sequence.")
        normalized_levels = []
        for level in levels:
            if isinstance(level, bool) or not isinstance(level, Real):
                raise ValueError(
                    "query.perturbation_levels must contain finite values in (0, 1]."
                )
            level = float(level)
            if not math.isfinite(level) or not 0.0 < level <= 1.0:
                raise ValueError(
                    "query.perturbation_levels must contain finite values in (0, 1]."
                )
            normalized_levels.append(level)
        if len(set(normalized_levels)) != len(normalized_levels):
            raise ValueError("query.perturbation_levels must not contain duplicates.")
        object.__setattr__(self, "perturbation_levels", tuple(normalized_levels))

        match_parameters = self.match_parameters
        if isinstance(match_parameters, (str, bytes)):
            raise ValueError("query.match_parameters must be a sequence of parameter names.")
        try:
            match_parameters = tuple(match_parameters)
        except TypeError as exc:
            raise ValueError(
                "query.match_parameters must be a sequence of parameter names."
            ) from exc
        if any(not isinstance(name, str) or not name for name in match_parameters):
            raise ValueError("query.match_parameters must contain non-empty strings.")
        if len(set(match_parameters)) != len(match_parameters):
            raise ValueError("query.match_parameters must not contain duplicates.")
        unknown = sorted(set(match_parameters) - set(ResolvedConfig.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"query.match_parameters contains unknown parameters: {unknown}"
            )
        if match_parameters and self.limit is None:
            raise ValueError("query.limit is required when match_parameters is set.")
        object.__setattr__(self, "match_parameters", match_parameters)

        discover_parameters = self.discover_parameters
        if isinstance(discover_parameters, (str, bytes)):
            raise ValueError(
                "query.discover_parameters must be a sequence of parameter names."
            )
        try:
            discover_parameters = tuple(discover_parameters)
        except TypeError as exc:
            raise ValueError(
                "query.discover_parameters must be a sequence of parameter names."
            ) from exc
        if any(not isinstance(name, str) or not name for name in discover_parameters):
            raise ValueError(
                "query.discover_parameters must contain non-empty strings."
            )
        if len(set(discover_parameters)) != len(discover_parameters):
            raise ValueError("query.discover_parameters must not contain duplicates.")
        unknown = sorted(
            set(discover_parameters) - set(ResolvedConfig.__dataclass_fields__)
        )
        if unknown:
            raise ValueError(
                f"query.discover_parameters contains unknown parameters: {unknown}"
            )
        if discover_parameters and set(discover_parameters) != set(match_parameters):
            raise ValueError(
                "query.discover_parameters must contain exactly match_parameters."
            )
        object.__setattr__(self, "discover_parameters", discover_parameters)

        group_parameters = self.discover_group_parameters
        if isinstance(group_parameters, (str, bytes)):
            raise ValueError(
                "query.discover_group_parameters must be a sequence of parameter names."
            )
        try:
            group_parameters = tuple(group_parameters)
        except TypeError as exc:
            raise ValueError(
                "query.discover_group_parameters must be a sequence of parameter names."
            ) from exc
        if any(not isinstance(name, str) or not name for name in group_parameters):
            raise ValueError(
                "query.discover_group_parameters must contain non-empty strings."
            )
        if len(set(group_parameters)) != len(group_parameters):
            raise ValueError(
                "query.discover_group_parameters must not contain duplicates."
            )
        if set(group_parameters) - set(discover_parameters):
            raise ValueError(
                "query.discover_group_parameters must be a subset of "
                "query.discover_parameters."
            )
        object.__setattr__(self, "discover_group_parameters", group_parameters)

        fallback_where = self.fallback_where
        if fallback_where is not None:
            if not isinstance(fallback_where, Mapping) or not fallback_where:
                raise ValueError("query.fallback_where must be a non-empty mapping or null.")
            fallback_where = dict(_plain_data(fallback_where))
            if any(not isinstance(name, str) or not name for name in fallback_where):
                raise ValueError(
                    "query.fallback_where filter names must be non-empty strings."
                )
            if not match_parameters:
                raise ValueError(
                    "query.fallback_where requires query.match_parameters."
                )
        object.__setattr__(self, "fallback_where", fallback_where)


@dataclass(frozen=True)
class BatchSpec:
    batch_id: int
    batch_index: int
    seed: int
    N: int
    t_interval: float
    schedule: tuple[tuple[int, float], ...]
    cases: tuple[ResolvedConfig, ...]

    @property
    def key(self) -> str:
        return batch_key(self.N, self.schedule, self.t_interval)


@dataclass(frozen=True)
class ConfigDocument:
    schema_version: int
    config_id: int
    name: str
    description: str
    created_utc: str
    parameters: Mapping[str, Any]
    runtime: RuntimeConfig
    batch_ids: Mapping[str, int]
    query: InitializationQuery | None = None
    source_path: Path | None = None

    @property
    def config_file(self) -> str:
        return self.source_path.name if self.source_path else "<memory>"

    def scalar_cases(self) -> tuple[ResolvedConfig, ...]:
        return expand_parameters(self.parameters)

    def batches(self) -> tuple[BatchSpec, ...]:
        cases = self.scalar_cases()
        groups: dict[tuple[str, bool, ...], list[ResolvedConfig]] = {}
        for case in cases:
            groups.setdefault(_case_compilation_key(case), []).append(case)

        # N is the outer loop, followed by schedule and time. This gives stable
        # Slurm array indices for a fixed (N, T) production shard.
        ordered_keys: list[tuple[str, bool, ...]] = []
        n_order = _sweep_values("N", self.parameters.get("N", 100))
        schedule_order = schedule_options(self.parameters.get("schedule", DEFAULT_SCHEDULE))
        time_order = _sweep_values(
            "t_interval", self.parameters.get("t_interval", 1.0)
        )
        for n_value in n_order:
            for schedule in schedule_order:
                for time_value in time_order:
                    for case in cases:
                        if (
                            case.N == int(n_value)
                            and case.schedule == schedule
                            and case.t_interval == float(time_value)
                        ):
                            key = _case_compilation_key(case)
                            if key not in ordered_keys:
                                ordered_keys.append(key)

        batches = []
        for key in ordered_keys:
            persisted_key = key[0]
            if persisted_key not in self.batch_ids:
                raise ValueError(
                    f"Config has no batch_id for {persisted_key}. Regenerate it with make_config.py."
                )
            members = groups[key]
            batch_id = int(self.batch_ids[persisted_key])
            shard_size = self.runtime.max_cases_per_batch or len(members)
            for start in range(0, len(members), shard_size):
                shard = members[start : start + shard_size]
                batches.append(
                    BatchSpec(
                        batch_id=batch_id,
                        batch_index=len(batches),
                        seed=(self.config_id + batch_id) % (2**32 - 1),
                        N=shard[0].N,
                        t_interval=shard[0].t_interval,
                        schedule=shard[0].schedule,
                        cases=tuple(shard),
                    )
                )
        return tuple(batches)


def normalise_schedule(value: Any) -> tuple[tuple[int, float], ...]:
    if isinstance(value, Mapping):
        value = list(value.items())
    if isinstance(value, (str, bytes)):
        raise ValueError("schedule must be a non-empty sequence of [steps, multiplier].")
    try:
        stages = tuple(value)
    except TypeError as exc:
        raise ValueError("schedule must be a non-empty sequence of stages.") from exc
    if not stages:
        raise ValueError("schedule cannot be empty.")
    result = []
    for stage in stages:
        try:
            steps, multiplier = stage
        except (TypeError, ValueError) as exc:
            raise ValueError("each schedule stage must contain steps and multiplier.") from exc
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("schedule steps must be positive integers.")
        if isinstance(multiplier, bool) or not isinstance(multiplier, Real):
            raise ValueError("schedule multipliers must be positive numbers.")
        multiplier = float(multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("schedule multipliers must be finite and positive.")
        result.append((int(steps), multiplier))
    return tuple(result)


def _looks_like_schedule(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return False
    try:
        return all(
            len(stage) == 2
            and isinstance(stage[0], Integral)
            and not isinstance(stage[0], bool)
            and isinstance(stage[1], Real)
            and not isinstance(stage[1], bool)
            for stage in value
        )
    except TypeError:
        return False


def schedule_options(value: Any) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Accept one schedule or a list of schedules for a schedule sweep."""

    if _looks_like_schedule(value) or isinstance(value, Mapping):
        return (normalise_schedule(value),)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("schedule must contain one schedule or a list of schedules.")
    return tuple(normalise_schedule(option) for option in value)


def _sweep_values(name: str, value: Any) -> tuple[Any, ...]:
    if name == "schedule":
        return schedule_options(value)
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Sweep parameter {name!r} cannot be empty.")
        return tuple(value)
    return (value,)


def _plain_data(value: Any) -> Any:
    """Convert array-library containers/scalars to YAML-safe Python values.

    Config creation deliberately does not depend on NumPy, but accepts objects
    such as ``numpy.ndarray`` and NumPy scalars through their standard
    ``tolist`` conversion hook.
    """

    if isinstance(value, Mapping):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_data(item) for item in value)
    if not isinstance(value, (str, bytes)):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            converted = tolist()
            if converted is not value:
                return _plain_data(converted)
    return value


def expand_parameters(parameters: Mapping[str, Any]) -> tuple[ResolvedConfig, ...]:
    unknown = set(parameters) - set(ResolvedConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown optimisation parameters: {sorted(unknown)}")
    invalid_static_sweeps = sorted(
        name
        for name in NON_SWEEP_FIELDS
        if name in parameters and isinstance(parameters[name], list)
    )
    if invalid_static_sweeps:
        raise ValueError(
            "These compile/diagnostic-shape fields cannot be swept: "
            f"{invalid_static_sweeps}"
        )
    field_names = list(parameters)
    choices = [_sweep_values(name, parameters[name]) for name in field_names]
    cases = []
    for combination in product(*choices):
        values = dict(zip(field_names, combination))
        cases.append(ResolvedConfig(**values))
    return tuple(cases) if cases else (ResolvedConfig(),)


def batch_key(N: int, schedule: Any, t_interval: float = 1.0) -> str:
    normalised = normalise_schedule(schedule)
    return json.dumps(
        {
            "N": int(N),
            "schedule": [list(stage) for stage in normalised],
            "t_interval": float(t_interval),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _case_compilation_key(case: ResolvedConfig) -> tuple[str, bool, ...]:
    """Separate static compute features without changing persisted batch IDs."""

    return (
        batch_key(case.N, case.schedule, case.t_interval),
        case.effective_u_sharp != 0.0,
        case.effective_v_sharp != 0.0,
        case.J_tol is not None,
        case.u_tol is not None,
        case.v_tol is not None,
        case.projected_gradient_tol is not None,
    )


def make_document(
    *,
    name: str,
    description: str = "",
    parameters: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    query: Mapping[str, Any] | InitializationQuery | None = None,
) -> ConfigDocument:
    """Create a validated document and assign all random IDs once."""

    parameters = dict(_plain_data(parameters or {}))
    runtime = dict(_plain_data(runtime or {}))
    if query is not None and not isinstance(query, InitializationQuery):
        if not isinstance(query, Mapping):
            raise ValueError("query must be a mapping or null.")
        query = InitializationQuery(**dict(_plain_data(query)))
    cases = expand_parameters(parameters)
    keys = list(
        dict.fromkeys(
            batch_key(case.N, case.schedule, case.t_interval) for case in cases
        )
    )
    return ConfigDocument(
        schema_version=SCHEMA_VERSION,
        config_id=random_id(),
        name=str(name),
        description=str(description),
        created_utc=datetime.now(timezone.utc).isoformat(),
        parameters=parameters,
        runtime=RuntimeConfig(**runtime),
        batch_ids={key: random_id() for key in keys},
        query=query,
    )


def document_to_dict(document: ConfigDocument) -> dict[str, Any]:
    return {
        "schema_version": document.schema_version,
        "config_id": document.config_id,
        "name": document.name,
        "description": document.description,
        "created_utc": document.created_utc,
        "parameters": dict(document.parameters),
        "runtime": asdict(document.runtime),
        "query": None if document.query is None else asdict(document.query),
        "batch_ids": dict(document.batch_ids),
    }


def write_config(
    document: ConfigDocument, path: str | Path, *, overwrite: bool = False
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Choose a new config name so existing run provenance is preserved."
        )
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document_to_dict(document), stream, sort_keys=False)
    return path


def load_config(path: str | Path) -> ConfigDocument:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping.")
    version = int(raw.get("schema_version", 0))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version {version}; expected one of "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    config_id = int(raw["config_id"])
    if not 0 < config_id <= ID_MAX:
        raise ValueError("config_id must be a non-zero signed-64-bit integer.")
    parameters = dict(raw.get("parameters") or {})
    parameters.setdefault("grid_refinement_y_floor", 1e-12)
    if "grid_refinement_tol" not in parameters:
        j_tolerance = parameters.get("J_tol")
        strict = "strict" in str(raw.get("name", "")).lower() or (
            isinstance(j_tolerance, Real)
            and not isinstance(j_tolerance, bool)
            and float(j_tolerance) <= 1e-6
        )
        parameters["grid_refinement_tol"] = 1e-3 if strict else 1e-2
    runtime_values = dict(raw.get("runtime") or {})
    if version == 2:
        # Preserve exact rerun behavior for immutable pre-auto-halt configs.
        parameters.setdefault("u_tol", 1e-3)
        parameters.setdefault("v_tol", 1e-3)
        parameters.setdefault("projected_gradient_tol", None)
        runtime_values.setdefault("auto_halt", False)
    runtime = RuntimeConfig(**runtime_values)
    query_values = raw.get("query")
    query = (
        None
        if query_values is None
        else InitializationQuery(**dict(query_values))
    )
    document = ConfigDocument(
        schema_version=version,
        config_id=config_id,
        name=str(raw.get("name") or path.stem),
        description=str(raw.get("description") or ""),
        created_utc=str(raw.get("created_utc") or ""),
        parameters=parameters,
        runtime=runtime,
        batch_ids={str(key): int(value) for key, value in dict(raw.get("batch_ids") or {}).items()},
        query=query,
        source_path=path,
    )
    document.batches()  # eagerly validate every scalar case and generated ID
    return document
