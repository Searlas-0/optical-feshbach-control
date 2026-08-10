"""Thin orchestration between isolated configuration, compute, and storage.

Architectural rule: this is the only package module allowed to coordinate
configuration documents, numerical modules, and result writes.  Physics,
optimization, analysis, config generation, result querying, and plotting stay
independent and exchange only explicit arguments/return values.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import gc
import os
from pathlib import Path
import time
from typing import Iterable

from .device import configure_jax_environment, effective_device

configure_jax_environment()

import jax
import jax.numpy as jnp
import numpy as np

from .analysis import best_control_derivatives
from .auto_initialization import AutoIntensityCenter, resolve_auto_intensity_center
from .config import (
    BatchSpec,
    ConfigDocument,
    document_to_dict,
    load_config,
    random_id,
)
from .filtering import parse_value
from .grid_refinement import grid_refinement_diagnostics
from .initialization import random_fourier_controls, stored_controls_to_raw
from .optimization import (
    BatchedAdamOptimizer,
    BatchedLBFGSOptimizer,
    BatchedPeakRefinementOptimizer,
)
from .physics import Physics
from .results import Results
from .storage import ResultStore


QUERY_PERTURBATION_SEED_MULTIPLIER = 0x9E3779B1
QUERY_PERTURBATION_INDEX_MULTIPLIER = 0x85EBCA6B
WALLTIME_CHECK_MAX_STEPS = 10_000
LONG_RUN_HISTORY_THRESHOLD_STEPS = 1_000_000
LONG_RUN_HISTORY_CHUNK_STEPS = 1_000


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _history_chunk_step_limit(runtime, start_step: int) -> int:
    """Bound disposable in-memory trace arrays, tightening after one million steps."""

    configured = runtime.max_steps_per_chunk or WALLTIME_CHECK_MAX_STEPS
    if start_step >= LONG_RUN_HISTORY_THRESHOLD_STEPS:
        return min(configured, LONG_RUN_HISTORY_CHUNK_STEPS)
    steps_to_threshold = LONG_RUN_HISTORY_THRESHOLD_STEPS - start_step
    return min(configured, steps_to_threshold)


def _validate_slurm_cpu_allocation(documents: Iterable[ConfigDocument]) -> None:
    """Prevent a SLURM job from oversubscribing its allocated CPU task slots."""

    supplied = os.environ.get("SLURM_CPUS_PER_TASK")
    if supplied is None:
        return
    try:
        allocated = int(supplied)
    except ValueError as error:
        raise RuntimeError(
            f"SLURM_CPUS_PER_TASK={supplied!r} is not a valid integer."
        ) from error
    required = max(document.runtime.concurrent_workers for document in documents)
    if allocated < required:
        raise RuntimeError(
            "SLURM allocation is too small: "
            f"--cpus-per-task={allocated}, but the queued configs require "
            f"concurrent_workers={required}. Increase --cpus-per-task or lower "
            "the config value."
        )


def _database_path(document: ConfigDocument) -> Path:
    path = Path(document.runtime.database).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _fourier_intensity_centers(document, cases, output_database):
    """Snapshot one resolved center per case before initialization sharding."""

    configured = document.runtime.fourier_intensity_fraction
    if configured != "auto":
        return {
            case: AutoIntensityCenter(
                bounded_center=float(configured) * case.u_max,
                fraction=float(configured),
                source_count=0,
                source_keys=(),
                global_source_count=0,
                exact_cap_source_count=0,
            )
            for case in cases
        }
    prior_path = document.runtime.fourier_intensity_auto_database
    prior_database = None if prior_path is None else _project_path(prior_path)
    return {
        case: resolve_auto_intensity_center(
            output_database=output_database,
            prior_database=prior_database,
            t_interval=case.t_interval,
            r_bg=case.r_bg,
            u_max=case.u_max,
        )
        for case in cases
    }


def _device(name: str):
    name = effective_device(name)
    if name == "auto":
        devices = jax.devices()
    else:
        devices = jax.devices(name)
    if not devices:
        raise RuntimeError(f"No JAX {name} device is available.")
    return devices[0]


def _member_parameters(batch: BatchSpec, initialisations: int, dtype):
    fields = [
        "r_bg",
        "u_max",
        "v_max",
        "dt",
        "t_interval",
        "u_smooth",
        "v_smooth",
        "u_sharp",
        "v_sharp",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "adam_learning_rate",
        "lbfgs_tolerance",
        "peak_initial_step_size",
        "peak_min_step_size",
        "peak_max_step_size",
        "peak_backtracking_factor",
        "peak_step_growth",
        "peak_armijo",
    ]
    first_case = batch.cases[0]
    optional_tolerances = (
        ("J_tol", first_case.J_tol),
        ("u_tol", first_case.u_tol),
        ("v_tol", first_case.v_tol),
        ("projected_gradient_tol", first_case.projected_gradient_tol),
    )
    fields.extend(name for name, value in optional_tolerances if value is not None)
    if first_case.projected_gradient_tol is not None:
        fields.append("projected_gradient_alpha")
    values = {name: [] for name in fields}
    for case in batch.cases:
        resolved = {
            "r_bg": case.r_bg,
            "u_max": case.u_max,
            "v_max": case.v_max,
            "dt": case.dt,
            "t_interval": case.t_interval,
            "u_smooth": case.effective_u_smooth,
            "v_smooth": case.effective_v_smooth,
            "u_sharp": case.effective_u_sharp,
            "v_sharp": case.effective_v_sharp,
            "adam_beta1": case.adam_beta1,
            "adam_beta2": case.adam_beta2,
            "adam_eps": case.adam_eps,
            "adam_learning_rate": case.adam_learning_rate,
            "lbfgs_tolerance": case.lbfgs_tolerance,
            "peak_initial_step_size": case.peak_initial_step_size,
            "peak_min_step_size": case.peak_min_step_size,
            "peak_max_step_size": case.peak_max_step_size,
            "peak_backtracking_factor": case.peak_backtracking_factor,
            "peak_step_growth": case.peak_step_growth,
            "peak_armijo": case.peak_armijo,
        }
        resolved.update(
            {
                name: getattr(case, name)
                for name, enabled_value in optional_tolerances
                if enabled_value is not None
            }
        )
        if first_case.projected_gradient_tol is not None:
            resolved["projected_gradient_alpha"] = case.projected_gradient_alpha
        for name in fields:
            values[name].extend([resolved[name]] * initialisations)
    return {name: jnp.asarray(value, dtype=dtype) for name, value in values.items()}


def _bounded_batch(physics, raw, parameters, use_jit):
    function = jax.vmap(physics.bounded_controls)
    if use_jit:
        function = jax.jit(function)
    return function(raw, parameters)


def _flat_member(flat_index: int, initialisations: int):
    return divmod(flat_index, initialisations)


def _slice_member_tree(tree, keep_indices, member_count: int):
    """Keep selected leading member axes while preserving scalar loop state."""

    def select(values):
        if (
            hasattr(values, "ndim")
            and values.ndim > 0
            and values.shape[0] == member_count
        ):
            return values[keep_indices]
        return values

    return jax.tree.map(select, tree)


def _slice_optimizer_members(state, keep_indices, member_count: int):
    """Shrink an optimizer state between schedule stages without resetting it."""

    state_values = {
        name: _slice_member_tree(getattr(state, name), keep_indices, member_count)
        for name in state.__dataclass_fields__
    }
    return replace(state, **state_values)


def _stored_case_parameters(case, runtime, execution_device: str):
    """Flatten numerical and execution provenance for per-run searchability."""

    # Persist only current config fields. ``dt`` is a derived compute input,
    # not a config parameter, and can always be recovered as T/N.
    values = asdict(case)
    values["schedule"] = [list(stage) for stage in case.schedule]
    return {
        **values,
        **asdict(runtime),
        "execution_device": execution_device,
    }


def _load_query_starts(
    document: ConfigDocument,
    store: ResultStore,
    *,
    cases=None,
):
    query = document.query
    if query is None:
        return ()
    results = Results(query.database or store.path)
    controls_by_run = {}
    optimizer_states_by_run = {}

    def search(where, *, limit):
        filters = {
            name: parse_value(value) if isinstance(value, str) else value
            for name, value in where.items()
        }
        filters.setdefault("status", "complete")
        return results.search(
            limit=limit,
            order_by=query.order_by,
            descending=query.descending,
            **filters,
        )

    def starts_from_rows(rows):
        starts = []
        for row in rows:
            run_id = int(row["run_id"])
            if query.resume_optimizer:
                if run_id not in optimizer_states_by_run:
                    try:
                        optimizer_states_by_run[run_id] = results.adam_state(
                            run_id, query.control_kind
                        )
                    except KeyError as error:
                        raise ValueError(
                            f"Initialization query selected run_id={run_id}, which "
                            f"has no resumable {query.control_kind} Adam state."
                        ) from error
                controls = None
                optimizer_state = optimizer_states_by_run[run_id]
            else:
                if run_id not in controls_by_run:
                    try:
                        controls_by_run[run_id] = results.controls(
                            run_id, query.control_kind
                        )
                    except KeyError as error:
                        raise ValueError(
                            f"Initialization query selected run_id={run_id}, which has no "
                            f"{query.control_kind} controls."
                        ) from error
                controls = controls_by_run[run_id]
                optimizer_state = None
            levels = query.perturbation_levels if query.perturbed else (0.0,)
            for perturbation_index, perturbation_level in enumerate(levels):
                starts.append(
                    {
                        "run_id": run_id,
                        "controls": controls,
                        "optimizer_state": optimizer_state,
                        "query_perturbed": query.perturbed,
                        "query_perturbation_index": perturbation_index,
                        "query_perturbation_level": perturbation_level,
                    }
                )
        return tuple(starts)

    if not query.match_parameters:
        rows = search(query.where, limit=query.limit)
        if not rows:
            raise ValueError(
                f"Initialization query matched no runs: {dict(query.where)!r}"
            )
        return starts_from_rows(rows)

    def frozen(value):
        if isinstance(value, dict):
            return tuple(sorted((name, frozen(item)) for name, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(frozen(item) for item in value)
        return value

    primary_rows = search(query.where, limit=None)
    rows_by_parameters = {}
    for row in primary_rows:
        try:
            key = tuple(frozen(row[name]) for name in query.match_parameters)
        except KeyError as error:
            raise ValueError(
                f"Initialization match parameter {error.args[0]!r} is absent from "
                "a selected source run."
            ) from error
        rows_by_parameters.setdefault(key, []).append(row)

    fallback_rows = (
        []
        if query.fallback_where is None
        else search(query.fallback_where, limit=query.limit)
    )
    starts_by_case = {}
    selected_cases = document.scalar_cases() if cases is None else tuple(cases)
    for case in selected_cases:
        key = tuple(
            frozen(getattr(case, name)) for name in query.match_parameters
        )
        rows = rows_by_parameters.get(key, ())[: query.limit]
        if not rows:
            rows = fallback_rows
        if len(rows) < query.limit and not query.discover_parameters:
            raise ValueError(
                "Initialization query found neither enough case-matched runs nor "
                f"fallback runs for {dict(zip(query.match_parameters, key))!r}."
            )
        starts_by_case[case] = starts_from_rows(rows)
    return starts_by_case


def _discover_query_cases(document: ConfigDocument, store: ResultStore):
    """Snapshot exact stored parameter combinations requested by the query."""

    query = document.query
    if query is None or not query.discover_parameters:
        return document.scalar_cases()
    filters = {
        name: parse_value(value) if isinstance(value, str) else value
        for name, value in query.where.items()
    }
    filters.setdefault("status", "complete")
    rows = Results(query.database or store.path).search(
        order_by=query.order_by,
        descending=query.descending,
        **filters,
    )
    combinations = {
        tuple(row[name] for name in query.discover_parameters)
        for row in rows
        if "best_score" in row
        and all(name in row for name in query.discover_parameters)
    }
    if not combinations:
        raise ValueError(
            "Initialization query found no saved parameter combinations to refine."
        )
    template = document.scalar_cases()[0]
    cases = tuple(
        replace(
            template,
            **dict(zip(query.discover_parameters, combination)),
        )
        for combination in sorted(combinations, key=lambda values: tuple(map(str, values)))
    )
    return cases


def _discovered_case_batches(document: ConfigDocument, cases):
    """Group exact discovered cases into deterministic persisted batches."""

    template_batch = document.batches()[0]
    group_names = document.query.discover_group_parameters
    grouped = {}
    for case in cases:
        key = tuple(getattr(case, name) for name in group_names)
        grouped.setdefault(key, []).append(case)
    batches = []
    for group_cases in grouped.values():
        shard_size = document.runtime.max_cases_per_batch or len(group_cases)
        for start in range(0, len(group_cases), shard_size):
            shard = tuple(group_cases[start : start + shard_size])
            index = len(batches)
            batches.append(BatchSpec(
                batch_id=template_batch.batch_id,
                batch_index=index,
                seed=(template_batch.seed + index) % (2**32 - 1),
                N=shard[0].N,
                t_interval=shard[0].t_interval,
                schedule=shard[0].schedule,
                cases=shard,
            ))
    return tuple(batches)


def _seed_query_starts(batch, query_starts):
    seeded = []
    for start in query_starts:
        seed = None
        if start["query_perturbed"]:
            seed = (
                int(batch.seed)
                ^ (
                    int(start["run_id"])
                    * QUERY_PERTURBATION_SEED_MULTIPLIER
                )
                ^ (
                    (int(start["query_perturbation_index"]) + 1)
                    * QUERY_PERTURBATION_INDEX_MULTIPLIER
                )
            ) & 0xFFFFFFFF
        seeded.append({**start, "query_perturbation_seed": seed})
    return tuple(seeded)


def _batch_query_starts(batch, query_starts):
    """Return uniformly sized, deterministically seeded starts for each case."""

    if isinstance(query_starts, dict):
        return tuple(
            _seed_query_starts(batch, query_starts[case]) for case in batch.cases
        )
    seeded = _seed_query_starts(batch, query_starts)
    return tuple(seeded for _ in batch.cases)


def _initial_raw(batch, case_base_raw, case_query_starts, dtype):
    blocks = {"u": [], "v": []}
    for case, base_raw, query_starts in zip(
        batch.cases, case_base_raw, case_query_starts
    ):
        queried = []
        for start in query_starts:
            if start["optimizer_state"] is not None:
                raw = {
                    name: jnp.asarray(values, dtype=dtype)
                    for name, values in start["optimizer_state"]["raw"].items()
                }
                expected_shape = (case.N + 1,)
                if set(raw) != {"u", "v"} or any(
                    values.shape != expected_shape for values in raw.values()
                ):
                    raise ValueError(
                        "Resumed Adam controls must contain u and v arrays with "
                        f"shape {expected_shape}; source run_id={start['run_id']}."
                    )
                queried.append(raw)
            else:
                queried.append(
                    stored_controls_to_raw(
                        start["controls"],
                        case.N,
                        u_max=case.u_max,
                        v_max=case.v_max,
                        u_isbound=case.u_isbound,
                        v_isbound=case.v_isbound,
                        perturbation_level=start["query_perturbation_level"],
                        perturbation_seed=start["query_perturbation_seed"],
                        dtype=dtype,
                    )
                )
        for name in blocks:
            pieces = [base_raw[name]]
            if queried:
                pieces.append(jnp.stack([start[name] for start in queried]))
            blocks[name].append(jnp.concatenate(pieces, axis=0))
    return {name: jnp.concatenate(values, axis=0) for name, values in blocks.items()}


def _initial_adam_moments(batch, case_query_starts, random_initialisations, dtype):
    """Assemble per-member Adam counters and moments for exact query resumes."""

    counts = []
    blocks = {
        moment: {"u": [], "v": []}
        for moment in ("first_moment", "second_moment")
    }
    control_shape = (batch.N + 1,)
    for query_starts in case_query_starts:
        counts.extend([0] * random_initialisations)
        for moment in blocks:
            for name in blocks[moment]:
                blocks[moment][name].append(
                    jnp.zeros((random_initialisations, *control_shape), dtype=dtype)
                )
        for start in query_starts:
            state = start["optimizer_state"]
            if state is None:
                counts.append(0)
            else:
                counts.append(int(state["count"]))
            for moment in blocks:
                for name in blocks[moment]:
                    values = (
                        jnp.zeros(control_shape, dtype=dtype)
                        if state is None
                        else jnp.asarray(state[moment][name], dtype=dtype)
                    )
                    if values.shape != control_shape:
                        raise ValueError(
                            f"Resumed Adam {moment}[{name!r}] must have shape "
                            f"{control_shape}; source run_id={start['run_id']}."
                        )
                    blocks[moment][name].append(values[None, :])
    return {
        "count": jnp.asarray(counts, dtype=jnp.int32),
        **{
            moment: {
                name: jnp.concatenate(values, axis=0)
                for name, values in controls.items()
            }
            for moment, controls in blocks.items()
        },
    }


def _selected_batches(document: ConfigDocument, batch_indices=None, *, batches=None):
    batches = document.batches() if batches is None else tuple(batches)
    if batch_indices is None:
        return batches
    requested = set()
    for value in batch_indices:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("batch indices must be non-negative integers.")
        requested.add(value)
    unknown = sorted(requested - {batch.batch_index for batch in batches})
    if unknown:
        raise ValueError(
            f"Unknown batch indices {unknown}; valid indices are 0..{len(batches) - 1}."
        )
    selected = tuple(batch for batch in batches if batch.batch_index in requested)
    if not selected:
        raise ValueError("At least one batch index must be selected.")
    return selected


def _initialization_work_items(document, batches, query_starts):
    """Shard each case batch's complete initialization population in order."""

    work_items = []
    for batch in batches:
        case_query_starts = _batch_query_starts(batch, query_starts)
        query_counts = {len(starts) for starts in case_query_starts}
        if len(query_counts) != 1:
            raise ValueError(
                "Case-matched initialization queries must select the same number "
                "of starts for every case in a batch."
            )
        total = document.runtime.initialisations + query_counts.pop()
        if total < 1:
            raise ValueError(
                "Each batch needs at least one random or queried initialization."
            )
        shard_size = document.runtime.max_initialisations_per_batch or total
        ranges = tuple(
            (start, min(start + shard_size, total))
            for start in range(0, total, shard_size)
        )
        work_items.extend(
            (batch, initialization_range, shard_index, len(ranges))
            for shard_index, initialization_range in enumerate(ranges)
        )
    return tuple(work_items)


def _run_batch(
    document: ConfigDocument,
    batch: BatchSpec,
    *,
    queue_id: int,
    store: ResultStore,
    config_document_id: int,
    query_starts,
    intensity_centers,
    device,
    initialization_range=None,
    initialization_batch_index: int = 0,
    initialization_batch_count: int = 1,
    deadline_monotonic: float | None = None,
    batch_count: int | None = None,
) -> dict:
    batch_started_monotonic = time.monotonic()
    runtime = document.runtime
    if runtime.max_batch_elapsed_seconds is not None:
        batch_deadline = (
            batch_started_monotonic + runtime.max_batch_elapsed_seconds
        )
        deadline_monotonic = (
            batch_deadline
            if deadline_monotonic is None
            else min(deadline_monotonic, batch_deadline)
        )
    displayed_batch_count = len(document.batches()) if batch_count is None else batch_count
    all_case_query_starts = _batch_query_starts(batch, query_starts)
    total_random_initialisations = runtime.initialisations
    query_counts = {len(starts) for starts in all_case_query_starts}
    if len(query_counts) != 1:
        raise ValueError(
            "Case-matched initialization queries must select the same number "
            "of starts for every case in a batch."
        )
    total_queried_initialisations = query_counts.pop()
    total_initialisations = (
        total_random_initialisations + total_queried_initialisations
    )
    if total_initialisations < 1:
        raise ValueError(
            "Each batch needs at least one random or queried initialization."
        )
    if initialization_range is None:
        initialization_range = (0, total_initialisations)
    initialization_start, initialization_end = initialization_range
    if not 0 <= initialization_start < initialization_end <= total_initialisations:
        raise ValueError("Initialization batch range is outside the population.")
    random_start = min(initialization_start, total_random_initialisations)
    random_end = min(initialization_end, total_random_initialisations)
    query_start = max(0, initialization_start - total_random_initialisations)
    query_end = max(0, initialization_end - total_random_initialisations)
    random_initialisations = random_end - random_start
    queried_initialisations = query_end - query_start
    initialisations = random_initialisations + queried_initialisations
    case_query_starts = tuple(
        starts[query_start:query_end] for starts in all_case_query_starts
    )
    dtype = jnp.float64 if runtime.use_x64 else jnp.float32
    first_case = batch.cases[0]
    resume_optimizer = bool(
        document.query is not None and document.query.resume_optimizer
    )
    if resume_optimizer and first_case.optimizer != "adam":
        raise ValueError(
            "query.resume_optimizer is only supported for optimizer: adam."
        )
    batch_record = {
        "batch_id": batch.batch_id,
        "config_id": document.config_id,
        "queue_id": queue_id,
        "batch_index": batch.batch_index,
        "batch_key": batch.key,
        "seed": batch.seed,
        "config_name": document.name,
        "config_file": document.config_file,
        "description": document.description,
        "created_utc": document.created_utc,
    }
    case_base_raw = []
    case_fourier_parameters = []
    with jax.default_device(device):
        for case in batch.cases:
            center = intensity_centers[case]
            all_base_raw, all_parameters = random_fourier_controls(
                batch.seed,
                total_random_initialisations,
                batch.N,
                num_modes=runtime.fourier_num_modes,
                rms_amplitude=runtime.fourier_rms_amplitude,
                intensity_fraction=center.fraction,
                dtype=dtype,
                return_parameters=True,
            )
            case_base_raw.append(
                {
                    name: values[random_start:random_end]
                    for name, values in all_base_raw.items()
                }
            )
            center_metadata = (
                center.metadata()
                if runtime.fourier_intensity_fraction == "auto"
                else {
                    "fourier_u_center_mode": "fixed",
                    "fourier_u_center": center.bounded_center,
                    "fourier_u_center_fraction": center.fraction,
                }
            )
            case_fourier_parameters.append(
                tuple(
                    {**parameters, **center_metadata}
                    for parameters in all_parameters[random_start:random_end]
                )
            )
    execution_id, run_ids = store.prepare_batch(
        batch_record,
        [
            _stored_case_parameters(case, runtime, device.platform)
            for case in batch.cases
        ],
        initialisations,
        config_document_id=config_document_id,
        initialization_index_offset=initialization_start,
        initialization_count_total=total_initialisations,
        initialization_metadata=[
            (
                [
                    {
                        "initialization_source": "fourier",
                        **parameters,
                    }
                    for parameters in fourier_parameters
                ]
                + [
                    {
                        "initialization_source": "query",
                        "source_run_id": start["run_id"],
                        "source_control_kind": document.query.control_kind,
                        "source_optimizer_resumed": resume_optimizer,
                        "query_perturbed": start["query_perturbed"],
                        "query_perturbation_index": start[
                            "query_perturbation_index"
                        ],
                        "query_perturbation_level": start[
                            "query_perturbation_level"
                        ],
                        "query_perturbation_seed": start[
                            "query_perturbation_seed"
                        ],
                    }
                    for start in starts
                ]
            )
            for fourier_parameters, starts in zip(
                case_fourier_parameters, case_query_starts
            )
        ],
    )
    try:
        with jax.default_device(device):
            physics = Physics(
                batch.N,
                dtype=dtype,
                u_isbound=first_case.u_isbound,
                v_isbound=first_case.v_isbound,
                u_sharp_active=any(
                    case.effective_u_sharp != 0.0 for case in batch.cases
                ),
                v_sharp_active=any(
                    case.effective_v_sharp != 0.0 for case in batch.cases
                ),
            )
            raw = _initial_raw(batch, case_base_raw, case_query_starts, dtype)
            parameters = _member_parameters(batch, initialisations, dtype)
            if first_case.optimizer == "adam":
                optimizer = BatchedAdamOptimizer(
                    physics,
                    block_size=first_case.block_size,
                    score_tolerance=first_case.J_tol is not None,
                    u_tolerance=first_case.u_tol is not None,
                    v_tolerance=first_case.v_tol is not None,
                    projected_gradient_tolerance=(
                        first_case.projected_gradient_tol is not None
                    ),
                    auto_halt=runtime.auto_halt,
                    use_jit=runtime.use_jit,
                )
            elif first_case.optimizer == "lbfgs":
                optimizer = BatchedLBFGSOptimizer(
                    physics,
                    block_size=first_case.block_size,
                    history_size=first_case.lbfgs_history_size,
                    max_linesearch_steps=first_case.lbfgs_max_linesearch_steps,
                    score_tolerance=first_case.J_tol is not None,
                    u_tolerance=first_case.u_tol is not None,
                    v_tolerance=first_case.v_tol is not None,
                    projected_gradient_tolerance=(
                        first_case.projected_gradient_tol is not None
                    ),
                    auto_halt=runtime.auto_halt,
                    use_jit=runtime.use_jit,
                )
            else:
                optimizer = BatchedPeakRefinementOptimizer(
                    physics,
                    block_size=first_case.block_size,
                    max_linesearch_steps=first_case.peak_max_linesearch_steps,
                    score_tolerance=first_case.J_tol is not None,
                    u_tolerance=first_case.u_tol is not None,
                    v_tolerance=first_case.v_tol is not None,
                    projected_gradient_tolerance=(
                        first_case.projected_gradient_tol is not None
                    ),
                    auto_halt=runtime.auto_halt,
                    use_jit=runtime.use_jit,
                )
            if resume_optimizer:
                restored_adam = _initial_adam_moments(
                    batch,
                    case_query_starts,
                    random_initialisations,
                    dtype,
                )
                state = optimizer.initialise(raw, parameters, **restored_adam)
            else:
                state = optimizer.initialise(raw, parameters)
            initial_raw_host = jax.device_get(raw)
            initial_controls = jax.device_get(
                _bounded_batch(physics, raw, parameters, runtime.use_jit)
            )
            optimizer_step_sizes = (
                parameters["adam_learning_rate"]
                if first_case.optimizer == "adam"
                else jnp.ones_like(parameters["lbfgs_tolerance"])
            )
            start_step = 0
            batch_halted = False
            batch_time_limited = False
            active_members = np.arange(
                len(batch.cases) * initialisations, dtype=int
            )
            halted_run_count = 0

            schedule_stage_number = 0
            stored_stage_index = 0
            stage_steps_remaining = 0
            while True:
                schedule_stage_index = schedule_stage_number % len(batch.schedule)
                configured_steps, multiplier = batch.schedule[schedule_stage_index]
                schedule_change = stage_steps_remaining == 0
                learning_rate_update = bool(
                    first_case.optimizer == "adam"
                    and schedule_change
                    and schedule_stage_number > 0
                    and float(multiplier) != 1.0
                )
                if schedule_change:
                    stage_steps_remaining = configured_steps
                member_count = len(active_members)
                if first_case.optimizer == "adam" and schedule_change:
                    optimizer_step_sizes = optimizer_step_sizes * multiplier
                max_steps_per_chunk = _history_chunk_step_limit(
                    runtime, start_step
                )
                steps = min(
                    stage_steps_remaining,
                    max(first_case.block_size, max_steps_per_chunk),
                )
                state, device_output = optimizer.run_stage(
                    state,
                    parameters,
                    steps=steps,
                    start_step=start_step,
                    learning_rate=optimizer_step_sizes,
                )
                jax.block_until_ready(device_output["actual_steps"])
                actual_steps = int(jax.device_get(device_output["actual_steps"]))
                end_step = start_step + actual_steps
                stage_steps_remaining -= actual_steps
                schedule_stage_complete = stage_steps_remaining == 0
                deadline_reached = (
                    deadline_monotonic is not None
                    and time.monotonic() >= deadline_monotonic
                )
                output = jax.device_get(
                    {
                        name: device_output[name]
                        for name in (
                            "score_history",
                            "objective_history",
                            "penalty_history",
                            "best_score",
                            "best_objective",
                            "best_penalty",
                            "best_step",
                            "stability_values",
                            "best_stability_values",
                            "stability_step",
                            "stability_consecutive_blocks",
                            "halted",
                        )
                    }
                )
                for name in (
                    "score_history",
                    "objective_history",
                    "penalty_history",
                ):
                    output[name] = output[name][:, : actual_steps + 1]
                halted = bool(output["halted"])
                batch_halted = halted
                consecutive = np.asarray(
                    output["stability_consecutive_blocks"]
                )
                stable_members = consecutive >= 3
                removable_members = (
                    stable_members
                    if runtime.auto_halt and (schedule_stage_complete or halted)
                    else np.zeros(member_count, dtype=bool)
                )
                final_schedule_stage = (
                    not runtime.repeat_schedule_until_stable
                    and schedule_stage_index == len(batch.schedule) - 1
                    and schedule_stage_complete
                )
                terminal_members = (
                    np.ones(member_count, dtype=bool)
                    if halted or final_schedule_stage or deadline_reached
                    else removable_members
                )
                best_controls = jax.device_get(
                    _bounded_batch(physics, state.best_raw, parameters, runtime.use_jit)
                )
                active_cases = [
                    batch.cases[_flat_member(int(index), initialisations)[0]]
                    for index in active_members
                ]
                grid_diagnostics = jax.device_get(
                    grid_refinement_diagnostics(
                        best_controls,
                        output["best_objective"],
                        N=batch.N,
                        r_bg=parameters["r_bg"],
                        t_interval=parameters["t_interval"],
                        tolerance=np.asarray(
                            [case.grid_refinement_tol for case in active_cases]
                        ),
                        y_floor=np.asarray(
                            [case.grid_refinement_y_floor for case in active_cases]
                        ),
                        dtype=dtype,
                        use_jit=runtime.use_jit,
                    )
                )
                final_controls = jax.device_get(
                    _bounded_batch(physics, state.raw, parameters, runtime.use_jit)
                )
                raw_controls = jax.device_get(
                    {"best": state.best_raw, "final": state.raw}
                )
                adam_states = None
                if first_case.optimizer == "adam":
                    adam_states = jax.device_get(
                        {
                            "best": {
                                "count": state.best_count,
                                "first_moment": state.best_first_moment,
                                "second_moment": state.best_second_moment,
                            },
                            "final": {
                                "count": state.count,
                                "first_moment": state.first_moment,
                                "second_moment": state.second_moment,
                            },
                        }
                    )
                member_records = []
                if first_case.optimizer == "adam":
                    optimizer_step_sizes_host = np.asarray(
                        jax.device_get(optimizer_step_sizes)
                    )
                else:
                    optimizer_step_sizes_host = np.asarray(
                        jax.device_get(device_output["optimizer_step_size"])
                    )
                for member_index in range(member_count):
                    original_member_index = int(active_members[member_index])
                    case_index, initialization_index = _flat_member(
                        original_member_index, initialisations
                    )
                    case = batch.cases[case_index]
                    record = {
                        "run_id": run_ids[(case_index, initialization_index)],
                        "best_score": float(output["best_score"][member_index]),
                        "best_objective": float(output["best_objective"][member_index]),
                        "best_penalty": float(output["best_penalty"][member_index]),
                        "best_step": int(output["best_step"][member_index]),
                        "final_score": float(output["score_history"][member_index, -1]),
                        "final_objective": float(output["objective_history"][member_index, -1]),
                        "final_penalty": float(output["penalty_history"][member_index, -1]),
                        "learning_rate": float(optimizer_step_sizes_host[member_index]),
                        "optimizer_stage": {
                            "optimizer": case.optimizer,
                            "optimizer_step_size": float(
                                optimizer_step_sizes_host[member_index]
                            ),
                            "schedule_stage_index": schedule_stage_index,
                            "schedule_change": schedule_change,
                            "learning_rate_update": learning_rate_update,
                        },
                        "history": {
                            "score": output["score_history"][member_index],
                            "objective": output["objective_history"][member_index],
                            "penalty": output["penalty_history"][member_index],
                        },
                        "best": {
                            name: values[member_index]
                            for name, values in best_controls.items()
                        },
                        "final": {
                            name: values[member_index]
                            for name, values in final_controls.items()
                        },
                        "best_raw": {
                            name: values[member_index]
                            for name, values in raw_controls["best"].items()
                        },
                        "final_raw": {
                            name: values[member_index]
                            for name, values in raw_controls["final"].items()
                        },
                        "best_derivatives": best_control_derivatives(
                            {
                                name: values[member_index]
                                for name, values in best_controls.items()
                            },
                            dt=case.dt,
                            u_sharp_active=case.effective_u_sharp != 0.0,
                            v_sharp_active=case.effective_v_sharp != 0.0,
                        ),
                        "best_diagnostics": {
                            name: float(values[member_index])
                            for name, values in output[
                                "best_stability_values"
                            ].items()
                        }
                        | {
                            name: np.asarray(values)[member_index].item()
                            for name, values in grid_diagnostics.items()
                        }
                        | {"best_grid_refinement_status": "computed"},
                    }
                    if stored_stage_index == 0:
                        record["initial"] = {
                            name: values[original_member_index]
                            for name, values in initial_controls.items()
                        }
                        record["initial_raw"] = {
                            name: values[original_member_index]
                            for name, values in initial_raw_host.items()
                        }
                    if adam_states is not None:
                        record["adam_states"] = {
                            kind: {
                                "count": int(state_values["count"][member_index]),
                                **{
                                    moment: {
                                        name: values[member_index]
                                        for name, values in controls.items()
                                    }
                                    for moment, controls in state_values.items()
                                    if moment != "count"
                                },
                            }
                            for kind, state_values in adam_states.items()
                        }
                    if terminal_members[member_index]:
                        if deadline_reached:
                            record["termination_reason"] = "time_limit"
                        elif consecutive[member_index] >= 3:
                            record["termination_reason"] = "stability"
                        else:
                            record["termination_reason"] = "schedule_end"
                    member_records.append(record)
                stored_tolerances = []
                if np.any(terminal_members):
                    stability_step = int(output["stability_step"])
                    stability_values = {
                        name: np.asarray(values)
                        for name, values in output["stability_values"].items()
                    }
                    for member_index in range(member_count):
                        if not terminal_members[member_index]:
                            continue
                        original_member_index = int(active_members[member_index])
                        case_index, initialization_index = _flat_member(
                            original_member_index, initialisations
                        )
                        stored_tolerances.append(
                            {
                                "member": member_index,
                                "run_id": run_ids[
                                    (case_index, initialization_index)
                                ],
                                "step": stability_step,
                                **{
                                    name: float(values[member_index])
                                    for name, values in stability_values.items()
                                },
                                "consecutive_blocks": int(consecutive[member_index]),
                                "required_consecutive_blocks": 3,
                                "passed": bool(consecutive[member_index] >= 3),
                                "auto_halted": bool(
                                    removable_members[member_index]
                                ),
                            }
                        )
                store.save_stage(
                    execution_id=execution_id,
                    stage_index=stored_stage_index,
                    start_step=start_step,
                    end_step=end_step,
                    members=member_records,
                    tolerances=stored_tolerances,
                )
                start_step = end_step
                newly_halted = int(np.count_nonzero(removable_members))
                elapsed_seconds = time.monotonic() - batch_started_monotonic
                print(
                    f"{document.config_file} | batch {batch.batch_index + 1}/"
                    f"{displayed_batch_count} | initialization batch "
                    f"{initialization_batch_index + 1}/{initialization_batch_count} | "
                    f"stage {schedule_stage_index + 1}/"
                    f"{len(batch.schedule)} chunk saved at step {start_step}"
                    f"{' (cycle ' + str(schedule_stage_number // len(batch.schedule) + 1) + ')' if runtime.repeat_schedule_until_stable else ''}"
                    f"{' (stable: auto-halted)' if halted else ''}",
                    f"halted {halted_run_count + newly_halted}/"
                    f"{len(batch.cases) * initialisations} | "
                    f"elapsed {elapsed_seconds / 3600:.2f} h",
                    flush=True,
                )
                # Histories are calculation outputs, not optimizer state. They
                # have now been encoded and committed as one on-disk chunk, so
                # discard every host/device reference before the next chunk.
                # Only current optimizer/stability state and best checkpoints
                # remain resident, irrespective of total run length.
                del (
                    output,
                    device_output,
                    member_records,
                    best_controls,
                    final_controls,
                    raw_controls,
                    grid_diagnostics,
                    active_cases,
                    adam_states,
                    stored_tolerances,
                    optimizer_step_sizes_host,
                )
                if stored_stage_index == 0:
                    initial_raw_host = None
                    initial_controls = None
                gc.collect()
                if deadline_reached:
                    batch_time_limited = True
                    break
                if halted:
                    halted_run_count += newly_halted
                    batch_halted = True
                    break
                if not schedule_stage_complete:
                    stored_stage_index += 1
                    continue
                if final_schedule_stage:
                    break
                remaining_count = member_count - newly_halted
                halted_run_count += newly_halted
                next_stage_index = (schedule_stage_number + 1) % len(batch.schedule)
                next_multiplier = batch.schedule[next_stage_index][1]
                boundary_name = (
                    "learning-rate change"
                    if first_case.optimizer == "adam" and next_multiplier != 1.0
                    else "refinement-cycle boundary"
                    if first_case.optimizer == "peak_refinement"
                    else "schedule boundary"
                )
                print(
                    f"{document.config_file} | initialization batch "
                    f"{initialization_batch_index + 1}/{initialization_batch_count} | "
                    f"{boundary_name} after "
                    f"step {start_step}: halted {newly_halted} stable run(s); "
                    f"{remaining_count} remain in the next batch",
                    flush=True,
                )
                if remaining_count == 0:
                    batch_halted = True
                    break
                if newly_halted:
                    keep_positions = np.flatnonzero(~removable_members)
                    keep_indices = jnp.asarray(
                        keep_positions, dtype=jnp.int32
                    )
                    state = _slice_optimizer_members(
                        state, keep_indices, member_count
                    )
                    parameters = _slice_member_tree(
                        parameters, keep_indices, member_count
                    )
                    optimizer_step_sizes = _slice_member_tree(
                        optimizer_step_sizes, keep_indices, member_count
                    )
                    active_members = active_members[keep_positions]
                schedule_stage_number += 1
                stored_stage_index += 1
                stage_steps_remaining = 0

            store.complete_batch(execution_id, run_ids.values())
        return {
            "batch_id": batch.batch_id,
            "batch_index": batch.batch_index,
            "initialization_batch_index": initialization_batch_index,
            "initialization_batch_count": initialization_batch_count,
            "cases": len(batch.cases),
            "runs": len(batch.cases) * initialisations,
            "random_initialisations": random_initialisations,
            "queried_initialisations": queried_initialisations,
            "steps": start_step,
            "auto_halted": batch_halted,
            "time_limited": batch_time_limited,
            "termination_reason": (
                "time_limit"
                if batch_time_limited
                else "stability" if batch_halted else "schedule_end"
            ),
            "halted_runs": halted_run_count,
        }
    except Exception as error:
        store.fail_batch(
            execution_id,
            run_ids.values(),
            f"{type(error).__name__}: {error}",
        )
        raise


def run_config(
    config: ConfigDocument | str | Path,
    *,
    queue_id: int | None = None,
    batch_indices=None,
    deadline_monotonic: float | None = None,
) -> list[dict]:
    """Run one config; compatible points batch while shape-changing points queue."""

    document = load_config(config) if isinstance(config, (str, Path)) else config
    queue_id = random_id() if queue_id is None else int(queue_id)
    jax.config.update("jax_enable_x64", document.runtime.use_x64)
    device = _device(document.runtime.device)
    output_database = _database_path(document)
    store = ResultStore(output_database)
    config_document_id = store.register_config(
        {
            "config_id": document.config_id,
            "config_name": document.name,
            "config_file": document.config_file,
            "description": document.description,
            "created_utc": document.created_utc,
            "config": document_to_dict(document),
        }
    )
    discovered_cases = _discover_query_cases(document, store)
    available_batches = (
        _discovered_case_batches(document, discovered_cases)
        if document.query is not None and document.query.discover_parameters
        else document.batches()
    )
    query_starts = _load_query_starts(
        document,
        store,
        cases=discovered_cases,
    )
    intensity_centers = _fourier_intensity_centers(
        document, discovered_cases, output_database
    )
    batches = _selected_batches(
        document,
        batch_indices,
        batches=available_batches,
    )
    if document.runtime.repeat_schedule_until_stable and any(
        case.optimizer not in {"lbfgs", "peak_refinement"}
        for case in discovered_cases
    ):
        raise ValueError(
            "repeat_schedule_until_stable is supported only for optimizer: "
            "lbfgs or peak_refinement."
        )
    configured_deadline = (
        None
        if document.runtime.max_elapsed_seconds is None
        else time.monotonic() + document.runtime.max_elapsed_seconds
    )
    if deadline_monotonic is None:
        deadline_monotonic = configured_deadline
    elif configured_deadline is not None:
        deadline_monotonic = min(deadline_monotonic, configured_deadline)
    work_items = _initialization_work_items(document, batches, query_starts)
    per_batch_elapsed_seconds = (
        document.runtime.max_elapsed_seconds / len(work_items)
        if document.runtime.distribute_max_elapsed_across_batches
        else None
    )
    workers = min(document.runtime.concurrent_workers, len(work_items))
    if workers == 1:
        completed = []
        for (
            batch,
            initialization_range,
            initialization_batch_index,
            initialization_batch_count,
        ) in work_items:
            if (
                deadline_monotonic is not None
                and time.monotonic() >= deadline_monotonic
            ):
                break
            batch_deadline = deadline_monotonic
            if per_batch_elapsed_seconds is not None:
                fair_deadline = time.monotonic() + per_batch_elapsed_seconds
                batch_deadline = (
                    fair_deadline
                    if batch_deadline is None
                    else min(batch_deadline, fair_deadline)
                )
            completed.append(_run_batch(
                document,
                batch,
                queue_id=queue_id,
                store=store,
                config_document_id=config_document_id,
                query_starts=query_starts,
                intensity_centers=intensity_centers,
                device=device,
                initialization_range=initialization_range,
                initialization_batch_index=initialization_batch_index,
                initialization_batch_count=initialization_batch_count,
                deadline_monotonic=batch_deadline,
                batch_count=len(batches),
            ))
        return completed
    completed = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ofc-batch") as pool:
        futures = {
            pool.submit(
                _run_batch,
                document,
                batch,
                queue_id=queue_id,
                store=store,
                config_document_id=config_document_id,
                query_starts=query_starts,
                intensity_centers=intensity_centers,
                device=device,
                initialization_range=initialization_range,
                initialization_batch_index=initialization_batch_index,
                initialization_batch_count=initialization_batch_count,
                deadline_monotonic=deadline_monotonic,
                batch_count=len(batches),
            ): work_index
            for work_index, (
                batch,
                initialization_range,
                initialization_batch_index,
                initialization_batch_count,
            ) in enumerate(work_items)
        }
        for future in as_completed(futures):
            completed.append((futures[future], future.result()))
    return [result for _, result in sorted(completed)]


def run_configs(
    configs: Iterable[str | Path],
    *,
    batch_indices=None,
    queue_id: int | None = None,
    continue_on_error: bool = False,
) -> dict:
    """Queue config files in argument order under one random queue identifier."""

    paths = list(configs)
    if not paths:
        raise ValueError("At least one config path is required.")
    documents = [load_config(path) for path in paths]
    if batch_indices is not None and len(documents) != 1:
        raise ValueError("Batch selection requires exactly one config file.")
    _validate_slurm_cpu_allocation(documents)
    x64_values = {document.runtime.use_x64 for document in documents}
    if len(x64_values) != 1:
        raise ValueError("Queued configs must agree on use_x64 (a process-global JAX setting).")
    queue_id = random_id() if queue_id is None else int(queue_id)
    if not 0 < queue_id <= 2**63 - 1:
        raise ValueError("queue_id must be a non-zero signed-64-bit integer.")
    if not isinstance(continue_on_error, bool):
        raise ValueError("continue_on_error must be a boolean.")
    output = []
    for document in documents:
        try:
            output.append(
                {
                    "config_id": document.config_id,
                    "config_file": document.config_file,
                    "status": "complete",
                    "batches": run_config(
                        document, queue_id=queue_id, batch_indices=batch_indices
                    ),
                }
            )
        except Exception as error:
            if not continue_on_error:
                raise
            failure = {
                "config_id": document.config_id,
                "config_file": document.config_file,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "batches": [],
            }
            output.append(failure)
            print(
                f"{document.config_file} failed; continuing to the next queued "
                f"config: {failure['error']}",
                flush=True,
            )
    return {"queue_id": queue_id, "configs": output}
