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


SCHEMA_VERSION = 2
ID_MAX = 2**63 - 1
NON_SWEEP_FIELDS = {
    "u_isbound",
    "v_isbound",
    "optimizer",
    "block_size",
    "lbfgs_history_size",
    "lbfgs_max_linesearch_steps",
}
DEFAULT_SCHEDULE = (
    (5_000, 1.0),
    (5_000, 0.5),
    (7_500, 0.5),
    (7_500, 0.5),
)


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

    smoothness: float = 1e-5
    u_smooth: float | None = None
    v_smooth: float | None = None
    sharpness: float = 0.0
    u_sharp: float | None = None
    v_sharp: float | None = None

    block_size: int = 500
    J_tol: float = 1e-5
    u_tol: float | None = 1e-3
    v_tol: float | None = 1e-3

    def __post_init__(self) -> None:
        integer_fields = {
            "N": self.N,
            "block_size": self.block_size,
            "lbfgs_history_size": self.lbfgs_history_size,
            "lbfgs_max_linesearch_steps": self.lbfgs_max_linesearch_steps,
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
            "J_tol": self.J_tol,
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

        for name in ("u_tol", "v_tol"):
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
        if optimizer not in {"adam", "lbfgs"}:
            raise ValueError("optimizer must be 'adam' or 'lbfgs'.")
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "u_isbound", bool(self.u_isbound))
        object.__setattr__(self, "v_isbound", bool(self.v_isbound))
        object.__setattr__(self, "schedule", normalise_schedule(self.schedule))
        if optimizer == "lbfgs" and any(multiplier != 1.0 for _, multiplier in self.schedule):
            raise ValueError("L-BFGS schedule multipliers must all equal 1.0.")

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
    fourier_intensity_fraction: float = 0.3
    use_jit: bool = True
    use_x64: bool = True
    device: str = "auto"
    concurrent_workers: int = 2
    database: str = "results/results.sqlite3"

    def __post_init__(self) -> None:
        for name in ("initialisations", "fourier_num_modes", "concurrent_workers"):
            value = getattr(self, name)
            minimum = 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
            object.__setattr__(self, name, int(value))
        if not 3 <= self.fourier_num_modes <= 6:
            raise ValueError("fourier_num_modes must be between 3 and 6.")
        for name in ("fourier_rms_amplitude", "fourier_intensity_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        if not 0.0 < self.fourier_intensity_fraction < 1.0:
            raise ValueError("fourier_intensity_fraction must be in (0, 1).")
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
    limit: int | None = None
    order_by: str = "best_score"
    descending: bool = True
    control_kind: str = "best"

    def __post_init__(self) -> None:
        if not isinstance(self.where, Mapping) or not self.where:
            raise ValueError("query.where must be a non-empty mapping of result filters.")
        where = dict(_plain_data(self.where))
        if any(not isinstance(name, str) or not name for name in where):
            raise ValueError("query.where filter names must be non-empty strings.")
        object.__setattr__(self, "where", where)

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
        groups: dict[tuple[str, bool, bool], list[ResolvedConfig]] = {}
        for case in cases:
            groups.setdefault(_case_compilation_key(case), []).append(case)

        # N is the outer loop, followed by schedule and time. This gives stable
        # Slurm array indices for a fixed (N, T) production shard.
        ordered_keys: list[tuple[str, bool, bool]] = []
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
        for index, key in enumerate(ordered_keys):
            persisted_key = key[0]
            if persisted_key not in self.batch_ids:
                raise ValueError(
                    f"Config has no batch_id for {persisted_key}. Regenerate it with make_config.py."
                )
            members = groups[key]
            batch_id = int(self.batch_ids[persisted_key])
            batches.append(
                BatchSpec(
                    batch_id=batch_id,
                    batch_index=index,
                    seed=(self.config_id + batch_id) % (2**32 - 1),
                    N=members[0].N,
                    t_interval=members[0].t_interval,
                    schedule=members[0].schedule,
                    cases=tuple(members),
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


def _case_compilation_key(case: ResolvedConfig) -> tuple[str, bool, bool]:
    """Separate static sharpness profiles without changing persisted batch IDs."""

    return (
        batch_key(case.N, case.schedule, case.t_interval),
        case.effective_u_sharp != 0.0,
        case.effective_v_sharp != 0.0,
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
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version {version}; expected {SCHEMA_VERSION}.")
    config_id = int(raw["config_id"])
    if not 0 < config_id <= ID_MAX:
        raise ValueError("config_id must be a non-zero signed-64-bit integer.")
    parameters = dict(raw.get("parameters") or {})
    runtime = RuntimeConfig(**dict(raw.get("runtime") or {}))
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
