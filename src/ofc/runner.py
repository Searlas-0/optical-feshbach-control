"""Thin orchestration between isolated configuration, compute, and storage.

Architectural rule: this is the only package module allowed to coordinate
configuration documents, numerical modules, and result writes.  Physics,
optimization, analysis, config generation, result querying, and plotting stay
independent and exchange only explicit arguments/return values.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import gc
import os
from pathlib import Path
from typing import Iterable

from .device import configure_jax_environment, effective_device

configure_jax_environment()

import jax
import jax.numpy as jnp
import numpy as np

from .analysis import ToleranceTracker, best_control_derivatives
from .config import (
    BatchSpec,
    ConfigDocument,
    document_to_dict,
    load_config,
    random_id,
)
from .filtering import parse_value
from .initialization import random_fourier_controls, stored_controls_to_raw
from .optimization import BatchedAdamOptimizer, BatchedLBFGSOptimizer
from .physics import Physics
from .results import Results
from .storage import ResultStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    fields = (
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
    )
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
        }
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


def _load_query_starts(document: ConfigDocument, store: ResultStore):
    query = document.query
    if query is None:
        return ()
    filters = {
        name: parse_value(value) if isinstance(value, str) else value
        for name, value in query.where.items()
    }
    filters.setdefault("status", "complete")
    results = Results(store.path)
    rows = results.search(
        limit=query.limit,
        order_by=query.order_by,
        descending=query.descending,
        **filters,
    )
    if not rows:
        raise ValueError(f"Initialization query matched no runs: {dict(query.where)!r}")
    starts = []
    for row in rows:
        run_id = int(row["run_id"])
        try:
            controls = results.controls(run_id, query.control_kind)
        except KeyError as error:
            raise ValueError(
                f"Initialization query selected run_id={run_id}, which has no "
                f"{query.control_kind} controls."
            ) from error
        starts.append({"run_id": run_id, "controls": controls})
    return tuple(starts)


def _initial_raw(batch, base_raw, query_starts, dtype):
    if not query_starts:
        return {
            name: jnp.tile(values, (len(batch.cases), 1))
            for name, values in base_raw.items()
        }
    blocks = {"u": [], "v": []}
    for case in batch.cases:
        queried = [
            stored_controls_to_raw(
                start["controls"],
                case.N,
                u_max=case.u_max,
                v_max=case.v_max,
                u_isbound=case.u_isbound,
                v_isbound=case.v_isbound,
                dtype=dtype,
            )
            for start in query_starts
        ]
        for name in blocks:
            blocks[name].append(
                jnp.concatenate(
                    [
                        base_raw[name],
                        jnp.stack([start[name] for start in queried]),
                    ],
                    axis=0,
                )
            )
    return {name: jnp.concatenate(values, axis=0) for name, values in blocks.items()}


def _selected_batches(document: ConfigDocument, batch_indices=None):
    batches = document.batches()
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


def _run_batch(
    document: ConfigDocument,
    batch: BatchSpec,
    *,
    queue_id: int,
    store: ResultStore,
    config_document_id: int,
    query_starts,
    device,
) -> dict:
    runtime = document.runtime
    random_initialisations = runtime.initialisations
    queried_initialisations = len(query_starts)
    initialisations = random_initialisations + queried_initialisations
    dtype = jnp.float64 if runtime.use_x64 else jnp.float32
    first_case = batch.cases[0]
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
    execution_id, run_ids = store.prepare_batch(
        batch_record,
        [
            _stored_case_parameters(case, runtime, device.platform)
            for case in batch.cases
        ],
        initialisations,
        config_document_id=config_document_id,
        initialization_metadata=(
            [{"initialization_source": "fourier"}] * random_initialisations
            + [
                {
                    "initialization_source": "query",
                    "source_run_id": start["run_id"],
                    "source_control_kind": document.query.control_kind,
                }
                for start in query_starts
            ]
        ),
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
            base_raw = random_fourier_controls(
                batch.seed,
                random_initialisations,
                batch.N,
                num_modes=runtime.fourier_num_modes,
                rms_amplitude=runtime.fourier_rms_amplitude,
                intensity_fraction=runtime.fourier_intensity_fraction,
                dtype=dtype,
            )
            raw = _initial_raw(batch, base_raw, query_starts, dtype)
            parameters = _member_parameters(batch, initialisations, dtype)
            if first_case.optimizer == "adam":
                optimizer = BatchedAdamOptimizer(
                    physics,
                    block_size=first_case.block_size,
                    use_jit=runtime.use_jit,
                )
            else:
                optimizer = BatchedLBFGSOptimizer(
                    physics,
                    block_size=first_case.block_size,
                    history_size=first_case.lbfgs_history_size,
                    max_linesearch_steps=first_case.lbfgs_max_linesearch_steps,
                    use_jit=runtime.use_jit,
                )
            state = optimizer.initialise(raw, parameters)
            initial_raw_host = jax.device_get(raw)
            initial_controls = jax.device_get(
                _bounded_batch(physics, raw, parameters, runtime.use_jit)
            )
            tracker = ToleranceTracker(first_case.block_size, initial_raw_host)
            optimizer_step_sizes = (
                parameters["adam_learning_rate"]
                if first_case.optimizer == "adam"
                else jnp.ones_like(parameters["lbfgs_tolerance"])
            )
            start_step = 0

            for stage_index, (steps, multiplier) in enumerate(batch.schedule):
                if first_case.optimizer == "adam":
                    optimizer_step_sizes = optimizer_step_sizes * multiplier
                state, device_output = optimizer.run_stage(
                    state,
                    parameters,
                    steps=steps,
                    start_step=start_step,
                    learning_rate=optimizer_step_sizes,
                )
                jax.block_until_ready(device_output["score_history"])
                # Do not copy Adam moments or live controls to host. Only the
                # stage artifacts needed for persistence leave the device.
                output = jax.device_get(
                    {
                        name: device_output[name]
                        for name in (
                            "score_history",
                            "objective_history",
                            "penalty_history",
                            "checkpoint_raw",
                            "best_score",
                            "best_objective",
                            "best_penalty",
                            "best_step",
                        )
                    }
                )
                tolerance_rows = tracker.consume_stage(
                    start_step=start_step,
                    score_history=output["score_history"],
                    checkpoint_raw=output["checkpoint_raw"],
                )
                best_controls = jax.device_get(
                    _bounded_batch(physics, state.best_raw, parameters, runtime.use_jit)
                )
                final_controls = jax.device_get(
                    _bounded_batch(physics, state.raw, parameters, runtime.use_jit)
                )
                member_records = []
                member_count = len(batch.cases) * initialisations
                if first_case.optimizer == "adam":
                    optimizer_step_sizes_host = np.asarray(
                        jax.device_get(optimizer_step_sizes)
                    )
                else:
                    optimizer_step_sizes_host = np.asarray(
                        jax.device_get(device_output["optimizer_step_size"])
                    )
                for member_index in range(member_count):
                    case_index, initialization_index = _flat_member(
                        member_index, initialisations
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
                        "best_derivatives": best_control_derivatives(
                            {
                                name: values[member_index]
                                for name, values in best_controls.items()
                            },
                            dt=case.dt,
                            u_sharp_active=case.effective_u_sharp != 0.0,
                            v_sharp_active=case.effective_v_sharp != 0.0,
                        ),
                    }
                    if stage_index == 0:
                        record["initial"] = {
                            name: values[member_index]
                            for name, values in initial_controls.items()
                        }
                    member_records.append(record)
                stored_tolerances = []
                for item in tolerance_rows:
                    case_index, initialization_index = _flat_member(
                        item["member"], initialisations
                    )
                    case = batch.cases[case_index]
                    passed = (
                        item["score_tolerance"] < case.J_tol
                        and (case.u_tol is None or item["u_tolerance"] < case.u_tol)
                        and (case.v_tol is None or item["v_tolerance"] < case.v_tol)
                    )
                    stored_tolerances.append(
                        {
                            **item,
                            "run_id": run_ids[(case_index, initialization_index)],
                            "passed": passed,
                        }
                    )
                store.save_stage(
                    execution_id=execution_id,
                    stage_index=stage_index,
                    start_step=start_step,
                    end_step=start_step + steps,
                    members=member_records,
                    tolerances=stored_tolerances,
                )
                start_step += steps
                print(
                    f"{document.config_file} | batch {batch.batch_index + 1}/"
                    f"{len(document.batches())} | stage {stage_index + 1}/"
                    f"{len(batch.schedule)} saved at step {start_step}",
                    flush=True,
                )
                # Only optimizer state, best controls, and one tolerance block
                # remain live after a committed stage.
                del output, device_output, member_records, tolerance_rows
                gc.collect()

            store.complete_batch(execution_id, run_ids.values())
        return {
            "batch_id": batch.batch_id,
            "cases": len(batch.cases),
            "runs": len(batch.cases) * initialisations,
            "random_initialisations": random_initialisations,
            "queried_initialisations": queried_initialisations,
            "steps": start_step,
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
) -> list[dict]:
    """Run one config; compatible points batch while shape-changing points queue."""

    document = load_config(config) if isinstance(config, (str, Path)) else config
    queue_id = random_id() if queue_id is None else int(queue_id)
    jax.config.update("jax_enable_x64", document.runtime.use_x64)
    device = _device(document.runtime.device)
    store = ResultStore(_database_path(document))
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
    query_starts = _load_query_starts(document, store)
    batches = _selected_batches(document, batch_indices)
    workers = min(document.runtime.concurrent_workers, len(batches))
    if workers == 1:
        return [
            _run_batch(
                document,
                batch,
                queue_id=queue_id,
                store=store,
                config_document_id=config_document_id,
                query_starts=query_starts,
                device=device,
            )
            for batch in batches
        ]
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
                device=device,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            completed.append((futures[future].batch_index, future.result()))
    return [result for _, result in sorted(completed)]


def run_configs(
    configs: Iterable[str | Path], *, batch_indices=None, queue_id: int | None = None
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
    output = []
    for document in documents:
        output.append(
            {
                "config_id": document.config_id,
                "config_file": document.config_file,
                "batches": run_config(
                    document, queue_id=queue_id, batch_indices=batch_indices
                ),
            }
        )
    return {"queue_id": queue_id, "configs": output}
