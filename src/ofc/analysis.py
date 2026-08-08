"""Pure post-processing for tolerances and control-gradient diagnostics.

Isolation boundary: functions receive arrays and return arrays/dictionaries.
This module never knows config paths, databases, runners, CLIs, or plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def score_stability(current_scores, previous_scores=None, eps=1e-12) -> float:
    """Reference block-to-block relative mean-score stability."""

    current = np.asarray(current_scores)
    current_mean = np.mean(current)
    previous_mean = current.reshape(-1)[0] if previous_scores is None else np.mean(previous_scores)
    return float(abs(current_mean - previous_mean) / max(abs(current_mean), eps))


def control_stability(previous_raw, current_raw, eps=1e-12) -> float:
    """Reference relative Euclidean movement in unconstrained control space."""

    previous = np.asarray(previous_raw)
    current = np.asarray(current_raw)
    return float(np.linalg.norm(current - previous) / (np.linalg.norm(current) + eps))


@dataclass
class ToleranceTracker:
    """Maintain only one score block and control snapshot between stages."""

    block_size: int
    initial_raw: dict[str, np.ndarray]

    def __post_init__(self):
        self.previous_raw = {name: np.array(value, copy=True) for name, value in self.initial_raw.items()}
        self.previous_scores = None
        self.pending_scores = None

    def consume_stage(self, *, start_step, score_history, checkpoint_raw):
        """Return per-member diagnostics at checkpoint steps in this stage."""

        scores = np.asarray(score_history)
        if self.pending_scores is None:
            combined = scores
            combined_start = start_step
        else:
            combined = np.concatenate([self.pending_scores, scores[:, 1:]], axis=1)
            combined_start = start_step - (self.pending_scores.shape[1] - 1)

        checkpoint_steps = list(
            range(
                ((start_step // self.block_size) + 1) * self.block_size,
                start_step + scores.shape[1],
                self.block_size,
            )
        )
        rows = []
        consumed_index = 0
        for checkpoint_index, step in enumerate(checkpoint_steps):
            end = step - combined_start
            current_scores = combined[:, consumed_index : end + 1]
            raw_u = np.asarray(checkpoint_raw["u"])[checkpoint_index]
            raw_v = np.asarray(checkpoint_raw["v"])[checkpoint_index]
            for member in range(scores.shape[0]):
                rows.append(
                    {
                        "member": member,
                        "step": step,
                        "score_tolerance": score_stability(
                            current_scores[member],
                            None if self.previous_scores is None else self.previous_scores[member],
                        ),
                        "u_tolerance": control_stability(
                            self.previous_raw["u"][member], raw_u[member]
                        ),
                        "v_tolerance": control_stability(
                            self.previous_raw["v"][member], raw_v[member]
                        ),
                    }
                )
            self.previous_scores = np.array(current_scores, copy=True)
            self.previous_raw = {"u": np.array(raw_u, copy=True), "v": np.array(raw_v, copy=True)}
            consumed_index = end
        self.pending_scores = np.array(combined[:, consumed_index:], copy=True)
        return rows


def best_control_derivatives(
    controls,
    *,
    dt,
    u_sharp_active: bool,
    v_sharp_active: bool,
):
    """Return physical-time derivative maxima for a bounded best control.

    Second differences are evaluated only for controls whose sharpness
    coefficient is active. This function runs on the best controls already
    copied at a stage boundary, avoiding another physics/objective evaluation.
    """

    dt = float(dt)
    u = np.asarray(controls["u"], dtype=float)
    v = np.asarray(controls["v"], dtype=float)
    output = {
        "best_max_abs_du_dt": float(np.max(np.abs(np.diff(u) / dt))),
        "best_max_abs_dv_dt": float(np.max(np.abs(np.diff(v) / dt))),
    }
    if u_sharp_active:
        output["best_max_abs_d2u_dt2"] = float(
            np.max(np.abs(np.diff(u, n=2) / dt**2))
        )
    if v_sharp_active:
        output["best_max_abs_d2v_dt2"] = float(
            np.max(np.abs(np.diff(v, n=2) / dt**2))
        )
    return output
