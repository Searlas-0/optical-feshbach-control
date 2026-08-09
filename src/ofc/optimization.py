"""Batched JAX optimization kernels for Adam and L-BFGS.

Isolation boundary: this module knows only numerical arrays and scalar
parameters.  It never reads configs, opens the results database, invokes a CLI,
or plots.  The runner passes every argument explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxopt import LBFGSB

from .physics import Physics


def _select(mask, candidate, incumbent):
    expanded = mask.reshape(mask.shape + (1,) * (candidate.ndim - mask.ndim))
    return jnp.where(expanded, candidate, incumbent)


def projected_gradient_rms(
    physics: Physics,
    controls,
    gradients,
    *,
    test_step,
):
    """Return ``||G_alpha|| / sqrt(2P)`` in normalized control space."""

    alpha = jnp.asarray(test_step, dtype=controls["u"].dtype)
    trial = jax.tree.map(
        lambda values, gradient: values - alpha * gradient,
        controls,
        gradients,
    )
    projected = physics.project_normalised_controls(trial)
    mapping = jax.tree.map(
        lambda values, feasible: (values - feasible) / alpha,
        controls,
        projected,
    )
    squared_norm = sum(jnp.sum(values**2) for values in mapping.values())
    variable_count = sum(values.size for values in mapping.values())
    return jnp.sqrt(squared_norm / variable_count)


def normalised_control_rms(previous, current):
    """Return RMS movement across both normalized control vectors."""

    squared_norm = sum(
        jnp.sum((current[name] - previous[name]) ** 2) for name in current
    )
    variable_count = sum(values.size for values in current.values())
    return jnp.sqrt(squared_norm / variable_count)


class StabilityState(NamedTuple):
    previous_controls: dict
    previous_score: jax.Array
    block_score_sum: jax.Array
    block_score_count: jax.Array
    consecutive_blocks: jax.Array
    last_step: jax.Array
    values: dict
    halted: jax.Array


class StabilityMonitor:
    """Device-side block stability with statically pruned optional metrics."""

    required_blocks = 3

    def __init__(
        self,
        physics: Physics,
        *,
        block_size: int,
        score_enabled: bool = False,
        u_enabled: bool = False,
        v_enabled: bool = False,
        projected_gradient_enabled: bool = False,
        auto_halt: bool = False,
    ):
        self.physics = physics
        self.block_size = int(block_size)
        self.score_enabled = bool(score_enabled)
        self.u_enabled = bool(u_enabled)
        self.v_enabled = bool(v_enabled)
        self.control_enabled = self.u_enabled or self.v_enabled
        self.projected_gradient_enabled = bool(projected_gradient_enabled)
        self.any_enabled = (
            self.score_enabled
            or self.control_enabled
            or self.projected_gradient_enabled
        )
        self.auto_halt = bool(auto_halt) and self.any_enabled
        self._normalised = jax.vmap(physics.normalised_controls)
        self._projected_loss_and_grad = None
        if self.projected_gradient_enabled:
            self._projected_loss_and_grad = jax.vmap(
                jax.value_and_grad(
                    physics.normalised_minimization_target,
                    has_aux=True,
                )
            )

    def initialise(self, raw, parameters, scores) -> StabilityState:
        member_count = scores.shape[0]
        dtype = scores.dtype
        previous_controls = (
            self._normalised(raw) if self.control_enabled else {}
        )
        values = {}
        if self.score_enabled:
            values["score_tolerance"] = jnp.full((member_count,), jnp.inf, dtype=dtype)
        if self.u_enabled:
            values["u_tolerance"] = jnp.full((member_count,), jnp.inf, dtype=dtype)
        if self.v_enabled:
            values["v_tolerance"] = jnp.full((member_count,), jnp.inf, dtype=dtype)
        if self.control_enabled:
            values["control_tolerance"] = jnp.full(
                (member_count,), jnp.inf, dtype=dtype
            )
        if self.projected_gradient_enabled:
            values["projected_gradient_tolerance"] = jnp.full(
                (member_count,), jnp.inf, dtype=dtype
            )
        return StabilityState(
            previous_controls=previous_controls,
            previous_score=scores,
            block_score_sum=jnp.zeros_like(scores),
            block_score_count=jnp.asarray(0, dtype=jnp.int32),
            consecutive_blocks=jnp.zeros((member_count,), dtype=jnp.int32),
            last_step=jnp.asarray(0, dtype=jnp.int32),
            values=values,
            halted=jnp.asarray(False),
        )

    def projected_gradient_values(self, normalised, parameters):
        """Evaluate projected-gradient RMS for each normalized batch member."""

        if not self.projected_gradient_enabled:
            raise RuntimeError("Projected-gradient stability is disabled.")
        (_, _), gradients = self._projected_loss_and_grad(normalised, parameters)
        return jax.vmap(
            lambda controls, gradient, alpha: projected_gradient_rms(
                self.physics,
                controls,
                gradient,
                test_step=alpha,
            )
        )(
            normalised,
            gradients,
            parameters["projected_gradient_alpha"],
        )

    def advance(self, state, next_raw, scores, parameters, completed_step):
        if self.score_enabled:
            state = state._replace(
                block_score_sum=state.block_score_sum + scores,
                block_score_count=state.block_score_count + 1,
            )
        if not self.any_enabled:
            return state

        def checkpoint(current):
            values = dict(current.values)
            stable = jnp.ones_like(current.consecutive_blocks, dtype=bool)
            previous_controls = current.previous_controls
            if self.score_enabled:
                block_mean = current.block_score_sum / jnp.maximum(
                    current.block_score_count, 1
                )
                score_tolerance = jnp.abs(block_mean - current.previous_score) / jnp.maximum(
                    jnp.abs(block_mean), 1e-12
                )
                values["score_tolerance"] = score_tolerance
                stable = stable & (score_tolerance < parameters["J_tol"])
                previous_score = block_mean
                block_score_sum = jnp.zeros_like(current.block_score_sum)
                block_score_count = jnp.asarray(0, dtype=jnp.int32)
            else:
                previous_score = current.previous_score
                block_score_sum = current.block_score_sum
                block_score_count = current.block_score_count

            normalised = None
            if self.control_enabled:
                normalised = self._normalised(next_raw)
                u_delta = jnp.sqrt(
                    jnp.mean(
                        (normalised["u"] - current.previous_controls["u"]) ** 2,
                        axis=1,
                    )
                )
                v_delta = jnp.sqrt(
                    jnp.mean(
                        (normalised["v"] - current.previous_controls["v"]) ** 2,
                        axis=1,
                    )
                )
                if self.u_enabled:
                    values["u_tolerance"] = u_delta
                    stable = stable & (u_delta < parameters["u_tol"])
                if self.v_enabled:
                    values["v_tolerance"] = v_delta
                    stable = stable & (v_delta < parameters["v_tol"])
                values["control_tolerance"] = jax.vmap(normalised_control_rms)(
                    current.previous_controls, normalised
                )
                previous_controls = normalised

            if self.projected_gradient_enabled:
                if normalised is None:
                    normalised = self._normalised(next_raw)
                projected = self.projected_gradient_values(
                    normalised, parameters
                )
                values["projected_gradient_tolerance"] = projected
                stable = stable & (
                    projected < parameters["projected_gradient_tol"]
                )

            consecutive = jnp.where(
                stable, current.consecutive_blocks + 1, 0
            )
            halted = (
                jnp.all(consecutive >= self.required_blocks)
                if self.auto_halt
                else current.halted
            )
            return StabilityState(
                previous_controls=previous_controls,
                previous_score=previous_score,
                block_score_sum=block_score_sum,
                block_score_count=block_score_count,
                consecutive_blocks=consecutive,
                last_step=completed_step,
                values=values,
                halted=halted,
            )

        is_checkpoint = completed_step % self.block_size == 0
        return lax.cond(is_checkpoint, checkpoint, lambda current: current, state)


@dataclass
class OptimizerState:
    raw: dict
    count: jax.Array
    first_moment: dict
    second_moment: dict
    best_raw: dict
    best_count: jax.Array
    best_first_moment: dict
    best_second_moment: dict
    best_score: jax.Array
    best_objective: jax.Array
    best_penalty: jax.Array
    best_step: jax.Array
    stability: StabilityState


class BatchedAdamOptimizer:
    """Compile and cache fixed-shape Adam schedule-stage executables."""

    def __init__(
        self,
        physics: Physics,
        *,
        block_size: int,
        score_tolerance: bool = False,
        u_tolerance: bool = False,
        v_tolerance: bool = False,
        projected_gradient_tolerance: bool = False,
        auto_halt: bool = False,
        use_jit: bool = True,
    ):
        self.physics = physics
        self.block_size = int(block_size)
        self.use_jit = bool(use_jit)
        self._stage_cache = {}
        self.stability_monitor = StabilityMonitor(
            physics,
            block_size=block_size,
            score_enabled=score_tolerance,
            u_enabled=u_tolerance,
            v_enabled=v_tolerance,
            projected_gradient_enabled=projected_gradient_tolerance,
            auto_halt=auto_halt,
        )
        self._batched_metrics = jax.vmap(physics.metrics)
        if self.use_jit:
            self._batched_metrics = jax.jit(self._batched_metrics)

    def initialise(
        self,
        raw,
        parameters,
        *,
        count=None,
        first_moment=None,
        second_moment=None,
    ) -> OptimizerState:
        """Initialize fresh members or restore their Adam counter and moments."""

        scores, objectives, penalties = self._batched_metrics(raw, parameters)
        member_count = int(raw["u"].shape[0])
        if count is None:
            count = jnp.zeros((member_count,), dtype=jnp.int32)
        else:
            count = jnp.asarray(count, dtype=jnp.int32)
            if count.ndim == 0:
                count = jnp.broadcast_to(count, (member_count,))
            if count.shape != (member_count,):
                raise ValueError(
                    f"Adam count must have shape ({member_count},), got {count.shape}."
                )
            if bool(jnp.any(count < 0)):
                raise ValueError("Adam count values must be non-negative.")
        if (first_moment is None) != (second_moment is None):
            raise ValueError(
                "Adam first_moment and second_moment must be supplied together."
            )
        if first_moment is None:
            first_moment = jax.tree.map(jnp.zeros_like, raw)
            second_moment = jax.tree.map(jnp.zeros_like, raw)
        else:
            if set(first_moment) != set(raw) or set(second_moment) != set(raw):
                raise ValueError("Adam moment names must match the raw controls.")
            for moment_name, moment in (
                ("first_moment", first_moment),
                ("second_moment", second_moment),
            ):
                for name, values in raw.items():
                    if jnp.shape(moment[name]) != values.shape:
                        raise ValueError(
                            f"Adam {moment_name}[{name!r}] must have shape "
                            f"{values.shape}, got {jnp.shape(moment[name])}."
                        )
            first_moment = jax.tree.map(jnp.asarray, first_moment)
            second_moment = jax.tree.map(jnp.asarray, second_moment)
        return OptimizerState(
            raw=raw,
            count=count,
            first_moment=first_moment,
            second_moment=second_moment,
            best_raw=jax.tree.map(lambda value: value, raw),
            best_count=count,
            best_first_moment=jax.tree.map(lambda value: value, first_moment),
            best_second_moment=jax.tree.map(lambda value: value, second_moment),
            best_score=scores,
            best_objective=objectives,
            best_penalty=penalties,
            best_step=jnp.zeros_like(scores, dtype=jnp.int32),
            stability=self.stability_monitor.initialise(raw, parameters, scores),
        )

    def _build_stage(self, steps: int):
        physics = self.physics
        stability_monitor = self.stability_monitor
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
            best_count,
            best_first_moment,
            best_second_moment,
            best_score,
            best_objective,
            best_penalty,
            best_step,
            stability,
            parameters,
            learning_rate,
            stage_start,
        ):
            member_count = raw["u"].shape[0]
            history_shape = (member_count, steps + 1)
            score_history = jnp.full(history_shape, jnp.nan, dtype=raw["u"].dtype)
            objective_history = jnp.full_like(score_history, jnp.nan)
            penalty_history = jnp.full_like(score_history, jnp.nan)

            def condition(carry):
                offset = carry[0]
                current_stability = carry[13]
                return (offset < steps) & ~current_stability.halted

            def take_step(carry):
                (
                    offset,
                    current_raw,
                    current_count,
                    current_first,
                    current_second,
                    current_best_raw,
                    current_best_count,
                    current_best_first,
                    current_best_second,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    current_stability,
                    current_score_history,
                    current_objective_history,
                    current_penalty_history,
                ) = carry
                (losses, (objectives, penalties)), gradients = loss_and_grad(
                    current_raw, parameters
                )
                scores = -losses
                global_step = stage_start + offset
                current_score_history = current_score_history.at[:, offset].set(scores)
                current_objective_history = current_objective_history.at[:, offset].set(
                    objectives
                )
                current_penalty_history = current_penalty_history.at[:, offset].set(
                    penalties
                )
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
                current_best_count = jnp.where(
                    better, current_count, current_best_count
                )
                current_best_first = jax.tree.map(
                    lambda candidate, incumbent: _select(
                        better, candidate, incumbent
                    ),
                    current_first,
                    current_best_first,
                )
                current_best_second = jax.tree.map(
                    lambda candidate, incumbent: _select(
                        better, candidate, incumbent
                    ),
                    current_second,
                    current_best_second,
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
                bias1 = 1.0 - beta1 ** next_count[:, None]
                bias2 = 1.0 - beta2 ** next_count[:, None]
                rate = learning_rate[:, None]
                next_raw = jax.tree.map(
                    lambda values, first, second: values
                    - rate * (first / bias1) / (jnp.sqrt(second / bias2) + eps),
                    current_raw,
                    next_first,
                    next_second,
                )
                completed_step = global_step + 1
                next_stability = stability_monitor.advance(
                    current_stability,
                    next_raw,
                    scores,
                    parameters,
                    completed_step,
                )
                return (
                    offset + 1,
                    next_raw,
                    next_count,
                    next_first,
                    next_second,
                    current_best_raw,
                    current_best_count,
                    current_best_first,
                    current_best_second,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    next_stability,
                    current_score_history,
                    current_objective_history,
                    current_penalty_history,
                )

            (
                actual_steps,
                final_raw,
                final_count,
                final_first,
                final_second,
                final_best_raw,
                final_best_count,
                final_best_first,
                final_best_second,
                final_best_score,
                final_best_objective,
                final_best_penalty,
                final_best_step,
                final_stability,
                score_history,
                objective_history,
                penalty_history,
            ) = lax.while_loop(
                condition,
                take_step,
                (
                    jnp.asarray(0, dtype=jnp.int32),
                    raw,
                    count,
                    first_moment,
                    second_moment,
                    best_raw,
                    best_count,
                    best_first_moment,
                    best_second_moment,
                    best_score,
                    best_objective,
                    best_penalty,
                    best_step,
                    stability,
                    score_history,
                    objective_history,
                    penalty_history,
                ),
            )
            final_scores, final_objectives, final_penalties = metrics(
                final_raw, parameters
            )
            score_history = score_history.at[:, actual_steps].set(final_scores)
            objective_history = objective_history.at[:, actual_steps].set(
                final_objectives
            )
            penalty_history = penalty_history.at[:, actual_steps].set(final_penalties)
            better = jnp.isfinite(final_scores) & (
                ~jnp.isfinite(final_best_score) | (final_scores > final_best_score)
            )
            final_best_raw = jax.tree.map(
                lambda candidate, incumbent: _select(better, candidate, incumbent),
                final_raw,
                final_best_raw,
            )
            final_best_count = jnp.where(better, final_count, final_best_count)
            final_best_first = jax.tree.map(
                lambda candidate, incumbent: _select(better, candidate, incumbent),
                final_first,
                final_best_first,
            )
            final_best_second = jax.tree.map(
                lambda candidate, incumbent: _select(better, candidate, incumbent),
                final_second,
                final_best_second,
            )
            final_best_score = jnp.where(better, final_scores, final_best_score)
            final_best_objective = jnp.where(
                better, final_objectives, final_best_objective
            )
            final_best_penalty = jnp.where(
                better, final_penalties, final_best_penalty
            )
            final_best_step = jnp.where(
                better, stage_start + actual_steps, final_best_step
            )
            best_stability_values = {}
            if stability_monitor.projected_gradient_enabled:
                best_stability_values["best_projected_gradient_rms"] = (
                    stability_monitor.projected_gradient_values(
                        stability_monitor._normalised(final_best_raw), parameters
                    )
                )
            return {
                "raw": final_raw,
                "count": final_count,
                "first_moment": final_first,
                "second_moment": final_second,
                "best_raw": final_best_raw,
                "best_count": final_best_count,
                "best_first_moment": final_best_first,
                "best_second_moment": final_best_second,
                "best_score": final_best_score,
                "best_objective": final_best_objective,
                "best_penalty": final_best_penalty,
                "best_step": final_best_step,
                "score_history": score_history,
                "objective_history": objective_history,
                "penalty_history": penalty_history,
                "actual_steps": actual_steps,
                "stability_values": final_stability.values,
                "best_stability_values": best_stability_values,
                "stability_step": final_stability.last_step,
                "stability_consecutive_blocks": final_stability.consecutive_blocks,
                "halted": final_stability.halted,
                "stability": final_stability,
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
        key = int(steps)
        runner = self._stage_cache.get(key)
        if runner is None:
            runner = self._build_stage(key)
            self._stage_cache[key] = runner
        output = runner(
            state.raw,
            state.count,
            state.first_moment,
            state.second_moment,
            state.best_raw,
            state.best_count,
            state.best_first_moment,
            state.best_second_moment,
            state.best_score,
            state.best_objective,
            state.best_penalty,
            state.best_step,
            state.stability,
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
            best_count=output["best_count"],
            best_first_moment=output["best_first_moment"],
            best_second_moment=output["best_second_moment"],
            best_score=output["best_score"],
            best_objective=output["best_objective"],
            best_penalty=output["best_penalty"],
            best_step=output["best_step"],
            stability=output["stability"],
        )
        return next_state, output


@dataclass
class LBFGSOptimizerState:
    raw: dict
    normalised: dict
    solver_state: object
    count: jax.Array
    best_raw: dict
    best_normalised: dict
    best_score: jax.Array
    best_objective: jax.Array
    best_penalty: jax.Array
    best_step: jax.Array
    stability: StabilityState


class BatchedLBFGSOptimizer:
    """Batched bound-aware L-BFGS-B in normalized feasible coordinates."""

    def __init__(
        self,
        physics: Physics,
        *,
        block_size: int,
        history_size: int,
        max_linesearch_steps: int,
        score_tolerance: bool = False,
        u_tolerance: bool = False,
        v_tolerance: bool = False,
        projected_gradient_tolerance: bool = False,
        auto_halt: bool = False,
        use_jit: bool = True,
    ):
        self.physics = physics
        self.block_size = int(block_size)
        self.use_jit = bool(use_jit)
        self._stage_cache = {}
        self.stability_monitor = StabilityMonitor(
            physics,
            block_size=block_size,
            score_enabled=score_tolerance,
            u_enabled=u_tolerance,
            v_enabled=v_tolerance,
            projected_gradient_enabled=projected_gradient_tolerance,
            auto_halt=auto_halt,
        )
        self._solver = LBFGSB(
            fun=physics.normalised_minimization_target,
            has_aux=True,
            history_size=int(history_size),
            maxls=int(max_linesearch_steps),
            tol=0.0,
            jit=self.use_jit,
        )
        self._batched_metrics = jax.vmap(
            lambda controls, parameters: physics.metrics_from_controls(
                physics.controls_from_normalised(controls, parameters), parameters
            )
        )
        self._to_normalised = jax.vmap(physics.normalised_controls)
        self._to_raw = jax.vmap(physics.raw_from_normalised)
        lower = {
            "u": jnp.full((physics.N + 1,), 0.0 if physics.u_isbound else -jnp.inf),
            "v": jnp.full((physics.N + 1,), -1.0 if physics.v_isbound else -jnp.inf),
        }
        upper = {
            "u": jnp.full((physics.N + 1,), 1.0 if physics.u_isbound else jnp.inf),
            "v": jnp.full((physics.N + 1,), 1.0 if physics.v_isbound else jnp.inf),
        }
        self._bounds = (lower, upper)
        self._batched_init = jax.vmap(
            self._solver.init_state, in_axes=(0, None, 0)
        )
        if self.use_jit:
            self._batched_metrics = jax.jit(self._batched_metrics)
            self._to_normalised = jax.jit(self._to_normalised)
            self._to_raw = jax.jit(self._to_raw)
            self._batched_init = jax.jit(self._batched_init)

    def initialise(self, raw, parameters) -> LBFGSOptimizerState:
        normalised = self._to_normalised(raw)
        scores, objectives, penalties = self._batched_metrics(
            normalised, parameters
        )
        solver_state = self._batched_init(normalised, self._bounds, parameters)
        return LBFGSOptimizerState(
            raw=raw,
            normalised=normalised,
            solver_state=solver_state,
            count=jnp.asarray(0, dtype=jnp.int32),
            best_raw=jax.tree.map(lambda value: value, raw),
            best_normalised=jax.tree.map(lambda value: value, normalised),
            best_score=scores,
            best_objective=objectives,
            best_penalty=penalties,
            best_step=jnp.zeros_like(scores, dtype=jnp.int32),
            stability=self.stability_monitor.initialise(raw, parameters, scores),
        )

    def _build_stage(self, steps: int):
        stability_monitor = self.stability_monitor
        metrics = jax.vmap(
            lambda controls, parameters: self.physics.metrics_from_controls(
                self.physics.controls_from_normalised(controls, parameters),
                parameters,
            )
        )
        to_raw = jax.vmap(self.physics.raw_from_normalised)
        bounds = self._bounds
        update = jax.vmap(self._solver.update, in_axes=(0, 0, None, 0))

        def run_stage(
            raw,
            normalised,
            solver_state,
            count,
            best_raw,
            best_normalised,
            best_score,
            best_objective,
            best_penalty,
            best_step,
            stability,
            parameters,
            stage_start,
        ):
            member_count = raw["u"].shape[0]
            history_shape = (member_count, steps + 1)
            score_history = jnp.full(history_shape, jnp.nan, dtype=raw["u"].dtype)
            objective_history = jnp.full_like(score_history, jnp.nan)
            penalty_history = jnp.full_like(score_history, jnp.nan)

            def condition(carry):
                return (carry[0] < steps) & ~carry[10].halted

            def take_step(carry):
                (
                    offset,
                    current_raw,
                    current_normalised,
                    current_solver_state,
                    current_best_raw,
                    current_best_normalised,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    current_stability,
                    current_score_history,
                    current_objective_history,
                    current_penalty_history,
                ) = carry
                scores = -current_solver_state.value
                objectives, penalties = current_solver_state.aux
                global_step = stage_start + offset
                current_score_history = current_score_history.at[:, offset].set(scores)
                current_objective_history = current_objective_history.at[:, offset].set(
                    objectives
                )
                current_penalty_history = current_penalty_history.at[:, offset].set(
                    penalties
                )
                better = jnp.isfinite(scores) & (
                    ~jnp.isfinite(current_best_score) | (scores > current_best_score)
                )
                current_best_raw = jax.tree.map(
                    lambda candidate, incumbent: _select(better, candidate, incumbent),
                    current_raw,
                    current_best_raw,
                )
                current_best_normalised = jax.tree.map(
                    lambda candidate, incumbent: _select(better, candidate, incumbent),
                    current_normalised,
                    current_best_normalised,
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

                candidate = update(
                    current_normalised, current_solver_state, bounds, parameters
                )
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
                next_normalised = jax.tree.map(
                    lambda new, old: _select(accepted, new, old),
                    candidate.params,
                    current_normalised,
                )
                next_raw = to_raw(next_normalised)
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
                next_stability = stability_monitor.advance(
                    current_stability,
                    next_raw,
                    -next_solver_state.value,
                    parameters,
                    completed_step,
                )
                return (
                    offset + 1,
                    next_raw,
                    next_normalised,
                    next_solver_state,
                    current_best_raw,
                    current_best_normalised,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    next_stability,
                    current_score_history,
                    current_objective_history,
                    current_penalty_history,
                )

            (
                actual_steps,
                final_raw,
                final_normalised,
                final_solver_state,
                final_best_raw,
                final_best_normalised,
                final_best_score,
                final_best_objective,
                final_best_penalty,
                final_best_step,
                final_stability,
                score_history,
                objective_history,
                penalty_history,
            ) = lax.while_loop(
                condition,
                take_step,
                (
                    jnp.asarray(0, dtype=jnp.int32),
                    raw,
                    normalised,
                    solver_state,
                    best_raw,
                    best_normalised,
                    best_score,
                    best_objective,
                    best_penalty,
                    best_step,
                    stability,
                    score_history,
                    objective_history,
                    penalty_history,
                ),
            )
            final_scores, final_objectives, final_penalties = metrics(
                final_normalised, parameters
            )
            score_history = score_history.at[:, actual_steps].set(final_scores)
            objective_history = objective_history.at[:, actual_steps].set(
                final_objectives
            )
            penalty_history = penalty_history.at[:, actual_steps].set(final_penalties)
            better = jnp.isfinite(final_scores) & (
                ~jnp.isfinite(final_best_score) | (final_scores > final_best_score)
            )
            final_best_raw = jax.tree.map(
                lambda candidate, incumbent: _select(better, candidate, incumbent),
                final_raw,
                final_best_raw,
            )
            final_best_normalised = jax.tree.map(
                lambda candidate, incumbent: _select(better, candidate, incumbent),
                final_normalised,
                final_best_normalised,
            )
            final_best_score = jnp.where(better, final_scores, final_best_score)
            final_best_objective = jnp.where(
                better, final_objectives, final_best_objective
            )
            final_best_penalty = jnp.where(
                better, final_penalties, final_best_penalty
            )
            final_best_step = jnp.where(
                better, stage_start + actual_steps, final_best_step
            )
            best_stability_values = {}
            if stability_monitor.projected_gradient_enabled:
                best_stability_values["best_projected_gradient_rms"] = (
                    stability_monitor.projected_gradient_values(
                        final_best_normalised, parameters
                    )
                )
            return {
                "raw": final_raw,
                "normalised": final_normalised,
                "solver_state": final_solver_state,
                "count": count + actual_steps,
                "best_raw": final_best_raw,
                "best_normalised": final_best_normalised,
                "best_score": final_best_score,
                "best_objective": final_best_objective,
                "best_penalty": final_best_penalty,
                "best_step": final_best_step,
                "optimizer_step_size": final_solver_state.stepsize,
                "score_history": score_history,
                "objective_history": objective_history,
                "penalty_history": penalty_history,
                "actual_steps": actual_steps,
                "stability_values": final_stability.values,
                "best_stability_values": best_stability_values,
                "stability_step": final_stability.last_step,
                "stability_consecutive_blocks": final_stability.consecutive_blocks,
                "halted": final_stability.halted,
                "stability": final_stability,
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
        key = int(steps)
        runner = self._stage_cache.get(key)
        if runner is None:
            runner = self._build_stage(key)
            self._stage_cache[key] = runner
        output = runner(
            state.raw,
            state.normalised,
            state.solver_state,
            state.count,
            state.best_raw,
            state.best_normalised,
            state.best_score,
            state.best_objective,
            state.best_penalty,
            state.best_step,
            state.stability,
            parameters,
            jnp.asarray(start_step, dtype=jnp.int32),
        )
        next_state = LBFGSOptimizerState(
            raw=output["raw"],
            normalised=output["normalised"],
            solver_state=output["solver_state"],
            count=output["count"],
            best_raw=output["best_raw"],
            best_normalised=output["best_normalised"],
            best_score=output["best_score"],
            best_objective=output["best_objective"],
            best_penalty=output["best_penalty"],
            best_step=output["best_step"],
            stability=output["stability"],
        )
        return next_state, output


@dataclass
class PeakRefinementOptimizerState:
    """State for monotone projected-gradient peak refinement."""

    raw: dict
    normalised: dict
    count: jax.Array
    step_size: jax.Array
    best_raw: dict
    best_normalised: dict
    best_score: jax.Array
    best_objective: jax.Array
    best_penalty: jax.Array
    best_step: jax.Array
    stability: StabilityState


class BatchedPeakRefinementOptimizer:
    """Careful bound-aware ascent with per-member Armijo backtracking.

    Each direction is a normalized projected-gradient direction. A candidate
    is accepted only when it gives sufficient improvement in the regularized
    score. Failed line searches leave that member exactly at its incumbent
    controls and reduce its next trial step, so momentum can never carry a
    member through a narrow peak.
    """

    def __init__(
        self,
        physics: Physics,
        *,
        block_size: int,
        max_linesearch_steps: int,
        score_tolerance: bool = False,
        u_tolerance: bool = False,
        v_tolerance: bool = False,
        projected_gradient_tolerance: bool = False,
        auto_halt: bool = False,
        use_jit: bool = True,
    ):
        self.physics = physics
        self.block_size = int(block_size)
        self.max_linesearch_steps = int(max_linesearch_steps)
        self.use_jit = bool(use_jit)
        self._stage_cache = {}
        self.stability_monitor = StabilityMonitor(
            physics,
            block_size=block_size,
            score_enabled=score_tolerance,
            u_enabled=u_tolerance,
            v_enabled=v_tolerance,
            projected_gradient_enabled=projected_gradient_tolerance,
            auto_halt=auto_halt,
        )
        self._to_normalised = jax.vmap(physics.normalised_controls)
        self._to_raw = jax.vmap(physics.raw_from_normalised)
        self._batched_metrics = jax.vmap(
            lambda controls, parameters: physics.metrics_from_controls(
                physics.controls_from_normalised(controls, parameters), parameters
            )
        )
        if self.use_jit:
            self._to_normalised = jax.jit(self._to_normalised)
            self._to_raw = jax.jit(self._to_raw)
            self._batched_metrics = jax.jit(self._batched_metrics)

    def initialise(self, raw, parameters) -> PeakRefinementOptimizerState:
        normalised = self._to_normalised(raw)
        scores, objectives, penalties = self._batched_metrics(
            normalised, parameters
        )
        step_size = jnp.clip(
            parameters["peak_initial_step_size"],
            parameters["peak_min_step_size"],
            parameters["peak_max_step_size"],
        )
        return PeakRefinementOptimizerState(
            raw=raw,
            normalised=normalised,
            count=jnp.asarray(0, dtype=jnp.int32),
            step_size=step_size,
            best_raw=jax.tree.map(lambda value: value, raw),
            best_normalised=jax.tree.map(lambda value: value, normalised),
            best_score=scores,
            best_objective=objectives,
            best_penalty=penalties,
            best_step=jnp.zeros_like(scores, dtype=jnp.int32),
            stability=self.stability_monitor.initialise(raw, parameters, scores),
        )

    def _build_stage(self, steps: int):
        physics = self.physics
        stability_monitor = self.stability_monitor
        max_linesearch_steps = self.max_linesearch_steps
        loss_and_grad = jax.vmap(
            jax.value_and_grad(
                physics.normalised_minimization_target,
                has_aux=True,
            )
        )
        metrics = jax.vmap(
            lambda controls, parameters: physics.metrics_from_controls(
                physics.controls_from_normalised(controls, parameters), parameters
            )
        )
        project = jax.vmap(physics.project_normalised_controls)
        to_raw = jax.vmap(physics.raw_from_normalised)
        variable_count = float(2 * (physics.N + 1))

        def run_stage(
            raw,
            normalised,
            count,
            step_size,
            best_raw,
            best_normalised,
            best_score,
            best_objective,
            best_penalty,
            best_step,
            stability,
            parameters,
            stage_start,
        ):
            member_count = raw["u"].shape[0]
            history_shape = (member_count, steps + 1)
            score_history = jnp.full(history_shape, jnp.nan, dtype=raw["u"].dtype)
            objective_history = jnp.full_like(score_history, jnp.nan)
            penalty_history = jnp.full_like(score_history, jnp.nan)

            def condition(carry):
                return (carry[0] < steps) & ~carry[11].halted

            def take_step(carry):
                (
                    offset,
                    current_raw,
                    current_normalised,
                    current_count,
                    current_step_size,
                    current_best_raw,
                    current_best_normalised,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    current_stability,
                    current_score_history,
                    current_objective_history,
                    current_penalty_history,
                ) = carry
                (losses, (objectives, penalties)), gradients = loss_and_grad(
                    current_normalised, parameters
                )
                scores = -losses
                global_step = stage_start + offset
                current_score_history = current_score_history.at[:, offset].set(scores)
                current_objective_history = current_objective_history.at[:, offset].set(
                    objectives
                )
                current_penalty_history = current_penalty_history.at[:, offset].set(
                    penalties
                )
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
                current_best_normalised = jax.tree.map(
                    lambda candidate, incumbent: _select(
                        better, candidate, incumbent
                    ),
                    current_normalised,
                    current_best_normalised,
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

                gradient_squared_norm = sum(
                    jnp.sum(values**2, axis=1) for values in gradients.values()
                )
                gradient_rms = jnp.sqrt(gradient_squared_norm / variable_count)
                direction = jax.tree.map(
                    lambda gradient: -gradient
                    / jnp.maximum(gradient_rms[:, None], 1e-30),
                    gradients,
                )
                initial_rate = jnp.clip(
                    current_step_size,
                    parameters["peak_min_step_size"],
                    parameters["peak_max_step_size"],
                )
                accepted = jnp.zeros((member_count,), dtype=bool)

                def search_condition(search):
                    return (search[0] < max_linesearch_steps) & ~jnp.all(search[2])

                def search_step(search):
                    (
                        attempt,
                        trial_rate,
                        already_accepted,
                        accepted_controls,
                        accepted_losses,
                        accepted_objectives,
                        accepted_penalties,
                        accepted_rates,
                    ) = search
                    proposed = project(
                        jax.tree.map(
                            lambda values, update: values
                            + trial_rate[:, None] * update,
                            current_normalised,
                            direction,
                        )
                    )
                    candidate = jax.tree.map(
                        lambda proposed_values, accepted_values: _select(
                            already_accepted, accepted_values, proposed_values
                        ),
                        proposed,
                        accepted_controls,
                    )
                    candidate_losses, candidate_aux = jax.vmap(
                        physics.normalised_minimization_target,
                        out_axes=(0, (0, 0)),
                    )(candidate, parameters)
                    candidate_objectives, candidate_penalties = candidate_aux
                    displacement = jax.tree.map(
                        lambda candidate_values, current_values: (
                            candidate_values - current_values
                        ),
                        candidate,
                        current_normalised,
                    )
                    directional_change = sum(
                        jnp.sum(gradient * delta, axis=1)
                        for gradient, delta in zip(
                            gradients.values(), displacement.values()
                        )
                    )
                    movement = sum(
                        jnp.sum(delta**2, axis=1)
                        for delta in displacement.values()
                    )
                    sufficient_decrease = candidate_losses <= (
                        losses
                        + parameters["peak_armijo"] * directional_change
                    )
                    newly_accepted = (
                        ~already_accepted
                        & jnp.isfinite(candidate_losses)
                        & (directional_change < 0.0)
                        & (movement > 0.0)
                        & sufficient_decrease
                    )
                    accepted_controls = jax.tree.map(
                        lambda candidate_values, incumbent: _select(
                            newly_accepted, candidate_values, incumbent
                        ),
                        candidate,
                        accepted_controls,
                    )
                    accepted_losses = jnp.where(
                        newly_accepted, candidate_losses, accepted_losses
                    )
                    accepted_objectives = jnp.where(
                        newly_accepted,
                        candidate_objectives,
                        accepted_objectives,
                    )
                    accepted_penalties = jnp.where(
                        newly_accepted, candidate_penalties, accepted_penalties
                    )
                    accepted_rates = jnp.where(
                        newly_accepted, trial_rate, accepted_rates
                    )
                    now_accepted = already_accepted | newly_accepted
                    next_trial_rate = jnp.where(
                        now_accepted,
                        trial_rate,
                        jnp.maximum(
                            trial_rate * parameters["peak_backtracking_factor"],
                            parameters["peak_min_step_size"],
                        ),
                    )
                    return (
                        attempt + 1,
                        next_trial_rate,
                        now_accepted,
                        accepted_controls,
                        accepted_losses,
                        accepted_objectives,
                        accepted_penalties,
                        accepted_rates,
                    )

                search = lax.while_loop(
                    search_condition,
                    search_step,
                    (
                        jnp.asarray(0, dtype=jnp.int32),
                        initial_rate,
                        accepted,
                        current_normalised,
                        losses,
                        objectives,
                        penalties,
                        initial_rate,
                    ),
                )
                (
                    _,
                    final_trial_rate,
                    accepted,
                    next_normalised,
                    next_losses,
                    next_objectives,
                    next_penalties,
                    accepted_rates,
                ) = search
                next_raw = to_raw(next_normalised)
                next_scores = -next_losses
                next_step_size = jnp.where(
                    accepted,
                    jnp.minimum(
                        accepted_rates * parameters["peak_step_growth"],
                        parameters["peak_max_step_size"],
                    ),
                    jnp.maximum(
                        final_trial_rate,
                        parameters["peak_min_step_size"],
                    ),
                )
                completed_step = global_step + 1
                next_stability = stability_monitor.advance(
                    current_stability,
                    next_raw,
                    next_scores,
                    parameters,
                    completed_step,
                )
                return (
                    offset + 1,
                    next_raw,
                    next_normalised,
                    current_count + 1,
                    next_step_size,
                    current_best_raw,
                    current_best_normalised,
                    current_best_score,
                    current_best_objective,
                    current_best_penalty,
                    current_best_step,
                    next_stability,
                    current_score_history,
                    current_objective_history,
                    current_penalty_history,
                )

            (
                actual_steps,
                final_raw,
                final_normalised,
                final_count,
                final_step_size,
                final_best_raw,
                final_best_normalised,
                final_best_score,
                final_best_objective,
                final_best_penalty,
                final_best_step,
                final_stability,
                score_history,
                objective_history,
                penalty_history,
            ) = lax.while_loop(
                condition,
                take_step,
                (
                    jnp.asarray(0, dtype=jnp.int32),
                    raw,
                    normalised,
                    count,
                    step_size,
                    best_raw,
                    best_normalised,
                    best_score,
                    best_objective,
                    best_penalty,
                    best_step,
                    stability,
                    score_history,
                    objective_history,
                    penalty_history,
                ),
            )
            final_scores, final_objectives, final_penalties = metrics(
                final_normalised, parameters
            )
            score_history = score_history.at[:, actual_steps].set(final_scores)
            objective_history = objective_history.at[:, actual_steps].set(
                final_objectives
            )
            penalty_history = penalty_history.at[:, actual_steps].set(final_penalties)
            better = jnp.isfinite(final_scores) & (
                ~jnp.isfinite(final_best_score) | (final_scores > final_best_score)
            )
            final_best_raw = jax.tree.map(
                lambda candidate, incumbent: _select(
                    better, candidate, incumbent
                ),
                final_raw,
                final_best_raw,
            )
            final_best_normalised = jax.tree.map(
                lambda candidate, incumbent: _select(
                    better, candidate, incumbent
                ),
                final_normalised,
                final_best_normalised,
            )
            final_best_score = jnp.where(
                better, final_scores, final_best_score
            )
            final_best_objective = jnp.where(
                better, final_objectives, final_best_objective
            )
            final_best_penalty = jnp.where(
                better, final_penalties, final_best_penalty
            )
            final_best_step = jnp.where(
                better, stage_start + actual_steps, final_best_step
            )
            best_stability_values = {}
            if stability_monitor.projected_gradient_enabled:
                best_stability_values["best_projected_gradient_rms"] = (
                    stability_monitor.projected_gradient_values(
                        final_best_normalised, parameters
                    )
                )
            return {
                "raw": final_raw,
                "normalised": final_normalised,
                "count": final_count,
                "step_size": final_step_size,
                "best_raw": final_best_raw,
                "best_normalised": final_best_normalised,
                "best_score": final_best_score,
                "best_objective": final_best_objective,
                "best_penalty": final_best_penalty,
                "best_step": final_best_step,
                "optimizer_step_size": final_step_size,
                "score_history": score_history,
                "objective_history": objective_history,
                "penalty_history": penalty_history,
                "actual_steps": actual_steps,
                "stability_values": final_stability.values,
                "best_stability_values": best_stability_values,
                "stability_step": final_stability.last_step,
                "stability_consecutive_blocks": final_stability.consecutive_blocks,
                "halted": final_stability.halted,
                "stability": final_stability,
            }

        return jax.jit(run_stage) if self.use_jit else run_stage

    def run_stage(
        self,
        state: PeakRefinementOptimizerState,
        parameters,
        *,
        steps: int,
        start_step: int,
        learning_rate=None,
    ):
        del learning_rate
        key = int(steps)
        runner = self._stage_cache.get(key)
        if runner is None:
            runner = self._build_stage(key)
            self._stage_cache[key] = runner
        output = runner(
            state.raw,
            state.normalised,
            state.count,
            state.step_size,
            state.best_raw,
            state.best_normalised,
            state.best_score,
            state.best_objective,
            state.best_penalty,
            state.best_step,
            state.stability,
            parameters,
            jnp.asarray(start_step, dtype=jnp.int32),
        )
        next_state = PeakRefinementOptimizerState(
            raw=output["raw"],
            normalised=output["normalised"],
            count=output["count"],
            step_size=output["step_size"],
            best_raw=output["best_raw"],
            best_normalised=output["best_normalised"],
            best_score=output["best_score"],
            best_objective=output["best_objective"],
            best_penalty=output["best_penalty"],
            best_step=output["best_step"],
            stability=output["stability"],
        )
        return next_state, output
