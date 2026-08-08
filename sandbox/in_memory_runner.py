"""Small, storage-free runner used only by the sandbox notebook.

This module deliberately imports no production runner, storage, or results
adapter.  It executes one scalar config entirely in memory and returns the
same plain run mappings consumed by ``ofc.plotting.plot_standard_figures``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ofc.analysis import ToleranceTracker
from ofc.config import ConfigDocument, load_config
from ofc.initialization import random_fourier_controls
from ofc.optimization import BatchedAdamOptimizer, BatchedLBFGSOptimizer
from ofc.physics import Physics


def _member_parameters(case, count: int, dtype) -> dict[str, jax.Array]:
    values = {
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
    return {
        name: jnp.full((count,), value, dtype=dtype)
        for name, value in values.items()
    }


def _bounded_batch(physics, raw, parameters, use_jit: bool):
    bounded = jax.vmap(physics.bounded_controls)
    return jax.jit(bounded)(raw, parameters) if use_jit else bounded(raw, parameters)


def _passed_tolerance(row, case) -> bool:
    return bool(
        row["score_tolerance"] < case.J_tol
        and (case.u_tol is None or row["u_tolerance"] < case.u_tol)
        and (case.v_tol is None or row["v_tolerance"] < case.v_tol)
    )


def run_config_in_memory(config: ConfigDocument | str | Path):
    """Optimize one scalar sandbox config without opening or writing a database."""

    document = load_config(config) if isinstance(config, (str, Path)) else config
    if document.query is not None:
        raise ValueError(
            "The storage-free sandbox cannot resolve an initialization query; "
            "run queried configs through run.py or Slurm."
        )
    batches = document.batches()
    if len(batches) != 1 or len(batches[0].cases) != 1:
        raise ValueError(
            "The cheap in-memory notebook accepts exactly one scalar case; "
            "use the production runner for sweeps."
        )

    runtime = document.runtime
    batch = batches[0]
    case = batch.cases[0]
    count = runtime.initialisations
    dtype = jnp.float64 if runtime.use_x64 else jnp.float32
    jax.config.update("jax_enable_x64", runtime.use_x64)
    devices = jax.devices("cpu" if runtime.device == "auto" else runtime.device)
    if not devices:
        raise RuntimeError(f"No JAX {runtime.device} device is available.")

    with jax.default_device(devices[0]):
        physics = Physics(
            case.N,
            dtype=dtype,
            u_isbound=case.u_isbound,
            v_isbound=case.v_isbound,
            u_sharp_active=case.effective_u_sharp != 0.0,
            v_sharp_active=case.effective_v_sharp != 0.0,
        )
        raw = random_fourier_controls(
            batch.seed,
            count,
            case.N,
            num_modes=runtime.fourier_num_modes,
            rms_amplitude=runtime.fourier_rms_amplitude,
            intensity_fraction=runtime.fourier_intensity_fraction,
            dtype=dtype,
        )
        parameters = _member_parameters(case, count, dtype)
        if case.optimizer == "adam":
            optimizer = BatchedAdamOptimizer(
                physics,
                block_size=case.block_size,
                use_jit=runtime.use_jit,
            )
        else:
            optimizer = BatchedLBFGSOptimizer(
                physics,
                block_size=case.block_size,
                history_size=case.lbfgs_history_size,
                max_linesearch_steps=case.lbfgs_max_linesearch_steps,
                use_jit=runtime.use_jit,
            )
        state = optimizer.initialise(raw, parameters)
        initial_raw = jax.device_get(raw)
        initial_controls = jax.device_get(
            _bounded_batch(physics, raw, parameters, runtime.use_jit)
        )
        tracker = ToleranceTracker(case.block_size, initial_raw)

        history_chunks = {name: [] for name in ("score", "objective", "penalty")}
        tolerance_rows = [[] for _ in range(count)]
        stage_starts = []
        stage_rates = []
        optimizer_step_sizes = (
            parameters["adam_learning_rate"]
            if case.optimizer == "adam"
            else jnp.ones_like(parameters["lbfgs_tolerance"])
        )
        start_step = 0

        for stage_index, (steps, multiplier) in enumerate(case.schedule):
            if case.optimizer == "adam":
                optimizer_step_sizes = optimizer_step_sizes * multiplier
            stage_starts.append(start_step)
            state, output = optimizer.run_stage(
                state,
                parameters,
                steps=steps,
                start_step=start_step,
                learning_rate=optimizer_step_sizes,
            )
            jax.block_until_ready(output["score_history"])
            stage_rates.append(
                float(
                    jax.device_get(
                        optimizer_step_sizes[0]
                        if case.optimizer == "adam"
                        else output["optimizer_step_size"][0]
                    )
                )
            )
            host = jax.device_get(
                {
                    name: output[name]
                    for name in (
                        "score_history",
                        "objective_history",
                        "penalty_history",
                        "checkpoint_raw",
                    )
                }
            )
            for name in history_chunks:
                values = host[f"{name}_history"]
                history_chunks[name].append(
                    values if stage_index == 0 else values[:, 1:]
                )
            rows = tracker.consume_stage(
                start_step=start_step,
                score_history=host["score_history"],
                checkpoint_raw=host["checkpoint_raw"],
            )
            for row in rows:
                tolerance_rows[row["member"]].append(
                    {**row, "passed": _passed_tolerance(row, case)}
                )
            start_step += steps

        histories = {
            name: np.concatenate(chunks, axis=1)
            for name, chunks in history_chunks.items()
        }
        best_values = jax.device_get(
            {
                "score": state.best_score,
                "objective": state.best_objective,
                "penalty": state.best_penalty,
                "step": state.best_step,
            }
        )
        best_controls = jax.device_get(
            _bounded_batch(physics, state.best_raw, parameters, runtime.use_jit)
        )
        final_controls = jax.device_get(
            _bounded_batch(physics, state.raw, parameters, runtime.use_jit)
        )

    learning_rate_history = np.empty(start_step + 1, dtype=float)
    for stage_index, ((steps, _), stage_start) in enumerate(
        zip(case.schedule, stage_starts)
    ):
        learning_rate_history[stage_start : stage_start + steps] = stage_rates[
            stage_index
        ]
    learning_rate_history[-1] = stage_rates[-1]

    case_values = asdict(case)
    runs = []
    for member in range(count):
        member_tolerances = tolerance_rows[member]
        tolerance = {
            "step": np.asarray(
                [row["step"] for row in member_tolerances], dtype=int
            ),
            "score_tolerance": np.asarray(
                [row["score_tolerance"] for row in member_tolerances], dtype=float
            ),
            "u_tolerance": np.asarray(
                [row["u_tolerance"] for row in member_tolerances], dtype=float
            ),
            "v_tolerance": np.asarray(
                [row["v_tolerance"] for row in member_tolerances], dtype=float
            ),
            "passed": np.asarray(
                [row["passed"] for row in member_tolerances], dtype=np.uint8
            ),
        }
        runs.append(
            {
                **case_values,
                "run_id": member + 1,
                "config_id": document.config_id,
                "config_name": document.name,
                "initialization_index": member,
                "status": "complete",
                "best_score": float(best_values["score"][member]),
                "best_objective": float(best_values["objective"][member]),
                "best_penalty": float(best_values["penalty"][member]),
                "best_step": int(best_values["step"][member]),
                "final_score": float(histories["score"][member, -1]),
                "final_objective": float(histories["objective"][member, -1]),
                "final_penalty": float(histories["penalty"][member, -1]),
                "history": {
                    "step": np.arange(start_step + 1, dtype=int),
                    "score": histories["score"][member],
                    "objective": histories["objective"][member],
                    "penalty": histories["penalty"][member],
                    "learning_rate": learning_rate_history.copy(),
                    "learning_rate_change_steps": np.asarray(
                        stage_starts, dtype=int
                    ),
                    "stage_learning_rates": np.asarray(stage_rates, dtype=float),
                },
                "tolerances": tolerance,
                "controls": {
                    "initial": {
                        name: np.asarray(values[member])
                        for name, values in initial_controls.items()
                    },
                    "best": {
                        name: np.asarray(values[member])
                        for name, values in best_controls.items()
                    },
                    "final": {
                        name: np.asarray(values[member])
                        for name, values in final_controls.items()
                    },
                },
            }
        )
    return document, runs
