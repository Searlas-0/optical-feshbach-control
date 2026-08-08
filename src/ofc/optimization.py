"""Batched JAX optimization kernels for Adam and L-BFGS.

Isolation boundary: this module knows only numerical arrays and scalar
parameters.  It never reads configs, opens the results database, invokes a CLI,
or plots.  The runner passes every argument explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
from jaxopt import LBFGS

from .physics import Physics


def _select(mask, candidate, incumbent):
    expanded = mask.reshape(mask.shape + (1,) * (candidate.ndim - mask.ndim))
    return jnp.where(expanded, candidate, incumbent)


@dataclass
class OptimizerState:
    raw: dict
    count: jax.Array
    first_moment: dict
    second_moment: dict
    best_raw: dict
    best_score: jax.Array
    best_objective: jax.Array
    best_penalty: jax.Array
    best_step: jax.Array


class BatchedAdamOptimizer:
    """Compile and cache fixed-shape Adam schedule-stage executables."""

    def __init__(self, physics: Physics, *, block_size: int, use_jit: bool = True):
        self.physics = physics
        self.block_size = int(block_size)
        self.use_jit = bool(use_jit)
        self._stage_cache = {}
        self._batched_metrics = jax.vmap(physics.metrics)
        if self.use_jit:
            self._batched_metrics = jax.jit(self._batched_metrics)

    def initialise(self, raw, parameters) -> OptimizerState:
        scores, objectives, penalties = self._batched_metrics(raw, parameters)
        zeros = jax.tree.map(jnp.zeros_like, raw)
        return OptimizerState(
            raw=raw,
            count=jnp.asarray(0, dtype=jnp.int32),
            first_moment=zeros,
            second_moment=jax.tree.map(jnp.zeros_like, raw),
            best_raw=jax.tree.map(lambda value: value, raw),
            best_score=scores,
            best_objective=objectives,
            best_penalty=penalties,
            best_step=jnp.zeros_like(scores, dtype=jnp.int32),
        )

    def _build_stage(self, steps: int, start_modulo: int):
        physics = self.physics
        block_size = self.block_size
        checkpoint_count = (start_modulo + steps) // block_size
        loss_and_grad = jax.vmap(
            jax.value_and_grad(physics.minimization_target, has_aux=True)
        )
        metrics = jax.vmap(physics.metrics)

        def run_stage(
            raw,
            count,
            first_moment,
            second_moment,
            best_raw,
            best_score,
            best_objective,
            best_penalty,
            best_step,
            parameters,
            learning_rate,
            stage_start,
        ):
            snapshot_shape = (max(checkpoint_count, 1),) + raw["u"].shape
            snapshots = {
                "u": jnp.zeros(snapshot_shape, dtype=raw["u"].dtype),
                "v": jnp.zeros(snapshot_shape, dtype=raw["v"].dtype),
            }

            def scan_step(carry, offset):
                (
                    current_raw,
                    current_count,
                    current_first,
                    current_second,
                    current_best_raw,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    current_snapshots,
                ) = carry
                (losses, (objectives, penalties)), gradients = loss_and_grad(
                    current_raw, parameters
                )
                scores = -losses
                global_step = stage_start + offset
                better = jnp.isfinite(scores) & (
                    ~jnp.isfinite(current_best_score) | (scores > current_best_score)
                )
                current_best_raw = jax.tree.map(
                    lambda candidate, incumbent: _select(
                        better, candidate, incumbent
                    ),
                    current_raw,
                    current_best_raw,
                )
                current_best_score = jnp.where(
                    better, scores, current_best_score
                )
                current_best_objective = jnp.where(
                    better, objectives, current_best_objective
                )
                current_best_penalty = jnp.where(
                    better, penalties, current_best_penalty
                )
                current_best_step = jnp.where(
                    better, global_step, current_best_step
                )

                next_count = current_count + 1
                beta1 = parameters["adam_beta1"][:, None]
                beta2 = parameters["adam_beta2"][:, None]
                eps = parameters["adam_eps"][:, None]
                next_first = jax.tree.map(
                    lambda moment, gradient: beta1 * moment + (1.0 - beta1) * gradient,
                    current_first,
                    gradients,
                )
                next_second = jax.tree.map(
                    lambda moment, gradient: beta2 * moment + (1.0 - beta2) * gradient**2,
                    current_second,
                    gradients,
                )
                bias1 = 1.0 - beta1**next_count
                bias2 = 1.0 - beta2**next_count
                rate = learning_rate[:, None]
                next_raw = jax.tree.map(
                    lambda values, first, second: values
                    - rate * (first / bias1) / (jnp.sqrt(second / bias2) + eps),
                    current_raw,
                    next_first,
                    next_second,
                )

                completed_step = global_step + 1
                is_checkpoint = completed_step % block_size == 0
                snapshot_index = (start_modulo + offset + 1) // block_size - 1
                snapshot_index = jnp.clip(snapshot_index, 0, max(checkpoint_count, 1) - 1)
                next_snapshots = jax.tree.map(
                    lambda stored, values: lax.cond(
                        is_checkpoint,
                        lambda array: array.at[snapshot_index].set(values),
                        lambda array: array,
                        stored,
                    ),
                    current_snapshots,
                    next_raw,
                )
                return (
                    next_raw,
                    next_count,
                    next_first,
                    next_second,
                    current_best_raw,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    next_snapshots,
                ), (scores, objectives, penalties)

            (
                final_raw,
                final_count,
                final_first,
                final_second,
                final_best_raw,
                final_best_score,
                final_best_objective,
                final_best_penalty,
                final_best_step,
                snapshots,
            ), histories = lax.scan(
                scan_step,
                (
                    raw,
                    count,
                    first_moment,
                    second_moment,
                    best_raw,
                    best_score,
                    best_objective,
                    best_penalty,
                    best_step,
                    snapshots,
                ),
                jnp.arange(steps, dtype=jnp.int32),
            )
            final_scores, final_objectives, final_penalties = metrics(
                final_raw, parameters
            )
            better = jnp.isfinite(final_scores) & (
                ~jnp.isfinite(final_best_score) | (final_scores > final_best_score)
            )
            final_best_raw = jax.tree.map(
                lambda candidate, incumbent: _select(better, candidate, incumbent),
                final_raw,
                final_best_raw,
            )
            final_best_score = jnp.where(better, final_scores, final_best_score)
            final_best_objective = jnp.where(
                better, final_objectives, final_best_objective
            )
            final_best_penalty = jnp.where(
                better, final_penalties, final_best_penalty
            )
            final_best_step = jnp.where(
                better, stage_start + steps, final_best_step
            )
            score_history, objective_history, penalty_history = histories
            return {
                "raw": final_raw,
                "count": final_count,
                "first_moment": final_first,
                "second_moment": final_second,
                "best_raw": final_best_raw,
                "best_score": final_best_score,
                "best_objective": final_best_objective,
                "best_penalty": final_best_penalty,
                "best_step": final_best_step,
                "score_history": jnp.concatenate(
                    [jnp.swapaxes(score_history, 0, 1), final_scores[:, None]], axis=1
                ),
                "objective_history": jnp.concatenate(
                    [jnp.swapaxes(objective_history, 0, 1), final_objectives[:, None]], axis=1
                ),
                "penalty_history": jnp.concatenate(
                    [jnp.swapaxes(penalty_history, 0, 1), final_penalties[:, None]], axis=1
                ),
                "checkpoint_raw": jax.tree.map(
                    lambda value: value[:checkpoint_count], snapshots
                ),
            }

        return jax.jit(run_stage) if self.use_jit else run_stage

    def run_stage(
        self,
        state: OptimizerState,
        parameters,
        *,
        steps: int,
        start_step: int,
        learning_rate,
    ):
        key = (int(steps), int(start_step) % self.block_size)
        runner = self._stage_cache.get(key)
        if runner is None:
            # Only the offset modulo block_size affects snapshot placement, but
            # the absolute offset is required for persisted best-step values.
            runner = self._build_stage(int(steps), key[1])
            self._stage_cache[key] = runner
        output = runner(
            state.raw,
            state.count,
            state.first_moment,
            state.second_moment,
            state.best_raw,
            state.best_score,
            state.best_objective,
            state.best_penalty,
            state.best_step,
            parameters,
            learning_rate,
            jnp.asarray(start_step, dtype=jnp.int32),
        )
        next_state = OptimizerState(
            raw=output["raw"],
            count=output["count"],
            first_moment=output["first_moment"],
            second_moment=output["second_moment"],
            best_raw=output["best_raw"],
            best_score=output["best_score"],
            best_objective=output["best_objective"],
            best_penalty=output["best_penalty"],
            best_step=output["best_step"],
        )
        return next_state, output


@dataclass
class LBFGSOptimizerState:
    raw: dict
    solver_state: object
    count: jax.Array
    best_raw: dict
    best_score: jax.Array
    best_objective: jax.Array
    best_penalty: jax.Array
    best_step: jax.Array


class BatchedLBFGSOptimizer:
    """Batched JAX-native L-BFGS with per-member line searches."""

    def __init__(
        self,
        physics: Physics,
        *,
        block_size: int,
        history_size: int,
        max_linesearch_steps: int,
        use_jit: bool = True,
    ):
        self.physics = physics
        self.block_size = int(block_size)
        self.use_jit = bool(use_jit)
        self._stage_cache = {}
        self._solver = LBFGS(
            fun=physics.minimization_target,
            has_aux=True,
            history_size=int(history_size),
            maxls=int(max_linesearch_steps),
            tol=0.0,
            jit=self.use_jit,
        )
        self._batched_metrics = jax.vmap(physics.metrics)
        self._batched_init = jax.vmap(self._solver.init_state)
        if self.use_jit:
            self._batched_metrics = jax.jit(self._batched_metrics)
            self._batched_init = jax.jit(self._batched_init)

    def initialise(self, raw, parameters) -> LBFGSOptimizerState:
        scores, objectives, penalties = self._batched_metrics(raw, parameters)
        solver_state = self._batched_init(raw, parameters)
        return LBFGSOptimizerState(
            raw=raw,
            solver_state=solver_state,
            count=jnp.asarray(0, dtype=jnp.int32),
            best_raw=jax.tree.map(lambda value: value, raw),
            best_score=scores,
            best_objective=objectives,
            best_penalty=penalties,
            best_step=jnp.zeros_like(scores, dtype=jnp.int32),
        )

    def _build_stage(self, steps: int, start_modulo: int):
        block_size = self.block_size
        checkpoint_count = (start_modulo + steps) // block_size
        metrics = jax.vmap(self.physics.metrics)
        update = jax.vmap(self._solver.update)

        def run_stage(
            raw,
            solver_state,
            count,
            best_raw,
            best_score,
            best_objective,
            best_penalty,
            best_step,
            parameters,
            stage_start,
        ):
            snapshot_shape = (max(checkpoint_count, 1),) + raw["u"].shape
            snapshots = {
                "u": jnp.zeros(snapshot_shape, dtype=raw["u"].dtype),
                "v": jnp.zeros(snapshot_shape, dtype=raw["v"].dtype),
            }

            def scan_step(carry, offset):
                (
                    current_raw,
                    current_solver_state,
                    current_best_raw,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    current_snapshots,
                ) = carry
                scores = -current_solver_state.value
                objectives, penalties = current_solver_state.aux
                global_step = stage_start + offset
                better = jnp.isfinite(scores) & (
                    ~jnp.isfinite(current_best_score) | (scores > current_best_score)
                )
                current_best_raw = jax.tree.map(
                    lambda candidate, incumbent: _select(better, candidate, incumbent),
                    current_raw,
                    current_best_raw,
                )
                current_best_score = jnp.where(
                    better, scores, current_best_score
                )
                current_best_objective = jnp.where(
                    better, objectives, current_best_objective
                )
                current_best_penalty = jnp.where(
                    better, penalties, current_best_penalty
                )
                current_best_step = jnp.where(
                    better, global_step, current_best_step
                )

                candidate = update(current_raw, current_solver_state, parameters)
                # JAXopt deliberately initializes error to +inf. It must remain
                # active for the first update; only NaN or a previous line-search
                # failure makes a member ineligible to take another step.
                active = (
                    ~jnp.isnan(current_solver_state.error)
                    & (current_solver_state.error > parameters["lbfgs_tolerance"])
                    & ~current_solver_state.failed_linesearch
                )
                failed = active & (
                    candidate.state.failed_linesearch
                    | ~jnp.isfinite(candidate.state.value)
                    | ~jnp.isfinite(candidate.state.error)
                )
                accepted = active & ~failed
                next_raw = jax.tree.map(
                    lambda new, old: _select(accepted, new, old),
                    candidate.params,
                    current_raw,
                )
                failed_solver_state = current_solver_state._replace(
                    failed_linesearch=jnp.where(
                        failed,
                        jnp.ones_like(current_solver_state.failed_linesearch),
                        current_solver_state.failed_linesearch,
                    )
                )
                next_solver_state = jax.tree.map(
                    lambda new, old: _select(accepted, new, old),
                    candidate.state,
                    failed_solver_state,
                )
                completed_step = global_step + 1
                is_checkpoint = completed_step % block_size == 0
                snapshot_index = (start_modulo + offset + 1) // block_size - 1
                snapshot_index = jnp.clip(
                    snapshot_index, 0, max(checkpoint_count, 1) - 1
                )
                next_snapshots = jax.tree.map(
                    lambda stored, values: lax.cond(
                        is_checkpoint,
                        lambda array: array.at[snapshot_index].set(values),
                        lambda array: array,
                        stored,
                    ),
                    current_snapshots,
                    next_raw,
                )
                return (
                    next_raw,
                    next_solver_state,
                    current_best_raw,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    next_snapshots,
                ), (scores, objectives, penalties)

            (
                final_raw,
                final_solver_state,
                final_best_raw,
                final_best_score,
                final_best_objective,
                final_best_penalty,
                final_best_step,
                snapshots,
            ), histories = lax.scan(
                scan_step,
                (
                    raw,
                    solver_state,
                    best_raw,
                    best_score,
                    best_objective,
                    best_penalty,
                    best_step,
                    snapshots,
                ),
                jnp.arange(steps, dtype=jnp.int32),
            )
            final_scores, final_objectives, final_penalties = metrics(
                final_raw, parameters
            )
            better = jnp.isfinite(final_scores) & (
                ~jnp.isfinite(final_best_score) | (final_scores > final_best_score)
            )
            final_best_raw = jax.tree.map(
                lambda candidate, incumbent: _select(better, candidate, incumbent),
                final_raw,
                final_best_raw,
            )
            final_best_score = jnp.where(better, final_scores, final_best_score)
            final_best_objective = jnp.where(
                better, final_objectives, final_best_objective
            )
            final_best_penalty = jnp.where(
                better, final_penalties, final_best_penalty
            )
            final_best_step = jnp.where(
                better, stage_start + steps, final_best_step
            )
            score_history, objective_history, penalty_history = histories
            return {
                "raw": final_raw,
                "solver_state": final_solver_state,
                "count": count + steps,
                "best_raw": final_best_raw,
                "best_score": final_best_score,
                "best_objective": final_best_objective,
                "best_penalty": final_best_penalty,
                "best_step": final_best_step,
                "optimizer_step_size": final_solver_state.stepsize,
                "score_history": jnp.concatenate(
                    [jnp.swapaxes(score_history, 0, 1), final_scores[:, None]], axis=1
                ),
                "objective_history": jnp.concatenate(
                    [jnp.swapaxes(objective_history, 0, 1), final_objectives[:, None]], axis=1
                ),
                "penalty_history": jnp.concatenate(
                    [jnp.swapaxes(penalty_history, 0, 1), final_penalties[:, None]], axis=1
                ),
                "checkpoint_raw": jax.tree.map(
                    lambda value: value[:checkpoint_count], snapshots
                ),
            }

        return jax.jit(run_stage) if self.use_jit else run_stage

    def run_stage(
        self,
        state: LBFGSOptimizerState,
        parameters,
        *,
        steps: int,
        start_step: int,
        learning_rate=None,
    ):
        del learning_rate
        key = (int(steps), int(start_step) % self.block_size)
        runner = self._stage_cache.get(key)
        if runner is None:
            runner = self._build_stage(int(steps), key[1])
            self._stage_cache[key] = runner
        output = runner(
            state.raw,
            state.solver_state,
            state.count,
            state.best_raw,
            state.best_score,
            state.best_objective,
            state.best_penalty,
            state.best_step,
            parameters,
            jnp.asarray(start_step, dtype=jnp.int32),
        )
        next_state = LBFGSOptimizerState(
            raw=output["raw"],
            solver_state=output["solver_state"],
            count=output["count"],
            best_raw=output["best_raw"],
            best_score=output["best_score"],
            best_objective=output["best_objective"],
            best_penalty=output["best_penalty"],
            best_step=output["best_step"],
        )
        return next_state, output
