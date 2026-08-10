#!/usr/bin/env python3
"""Generate the isolated CPU configs for the N=100→500 resolution ladder.

This file only emits immutable config documents. Runtime orchestration stays
in ``slurm/run_resolution_ladder.slurm`` and numerical work stays in ``ofc``.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofc.config import make_document, write_config


CAPS = {
    40: {"learning_rate": 0.1, "smoothness": 1.25e-7, "sharpness": 5e-8},
    160: {"learning_rate": 0.02, "smoothness": 1.25e-7, "sharpness": 1.25e-8},
    1280: {"learning_rate": 0.1, "smoothness": 2.5e-7, "sharpness": 2.5e-8},
}
TARGET_RESOLUTIONS = (200, 300, 400, 500)


def stage_name(N: int, cap: int, tolerance: str) -> str:
    return f"N{N}_u{cap}_resolution_ladder_{tolerance}_cpu"


def database_name(cap: int) -> str:
    return f"results/slurm_isolated/N100_u{cap}_top_peak_refinement_strict.sqlite3"


def source_names(N: int, cap: int) -> list[str]:
    if N == TARGET_RESOLUTIONS[0]:
        return [f"N100_u{cap}_top_peak_refinement_strict_slurm_isolated_cpu"]
    previous = N - 100
    return [
        stage_name(previous, cap, "loose"),
        stage_name(previous, cap, "strict"),
    ]


def parameters(N: int, cap: int, *, strict: bool) -> dict:
    settings = CAPS[cap]
    return {
        "N": N,
        "t_interval": 4.0,
        "r_bg": -0.008716,
        "u_isbound": True,
        "v_isbound": True,
        "u_max": float(cap),
        "v_max": 1000.0,
        "slew_limit": 0.05,
        "optimizer": "peak_refinement",
        "schedule": [(250, 1.0)],
        "adam_learning_rate": settings["learning_rate"],
        "adam_beta1": 0.7,
        "adam_beta2": 0.99,
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
        "smoothness": settings["smoothness"],
        "u_smooth": None,
        "v_smooth": None,
        "sharpness": settings["sharpness"],
        "u_sharp": None,
        "v_sharp": None,
        "block_size": 25,
        "J_tol": 1e-6 if strict else 1e-5,
        "u_tol": 1e-5 if strict else 1e-4,
        "v_tol": 1e-5 if strict else 1e-4,
        "projected_gradient_tol": 1e-5 if strict else 1e-4,
        "projected_gradient_alpha": 1.0,
        "grid_refinement_tol": 1e-3 if strict else 1e-2,
        "grid_refinement_y_floor": 1e-12,
    }


def runtime(cap: int, *, strict: bool) -> dict:
    return {
        "initialisations": 0 if strict else 10,
        "fourier_num_modes": 5,
        "fourier_rms_amplitude": 0.3,
        "fourier_intensity_fraction": 0.3,
        "use_jit": True,
        "use_x64": True,
        "device": "cpu",
        "concurrent_workers": 1,
        "max_cases_per_batch": None,
        "max_initialisations_per_batch": None,
        "max_steps_per_chunk": 250,
        "max_batch_elapsed_seconds": 4 * 60 * 60,
        "max_elapsed_seconds": 4 * 60 * 60,
        "distribute_max_elapsed_across_batches": False,
        "repeat_schedule_until_stable": True,
        "auto_halt": True,
        "database": database_name(cap),
    }


def query(N: int, cap: int, *, strict: bool) -> dict:
    names = [stage_name(N, cap, "loose")] if strict else source_names(N, cap)
    return {
        "where": {
            "status": "complete",
            # Prefer stable runs by score when present, but a saved four-hour
            # time-limit result is still a valid promotion/refinement source.
            "termination_reason": ["stability", "time_limit"],
            "config_name": names,
            "N": N if strict else N - 100,
            "u_max": float(cap),
            "smoothness": CAPS[cap]["smoothness"],
            "sharpness": CAPS[cap]["sharpness"],
        },
        "limit": 1 if strict else 5,
        "order_by": "best_score",
        "descending": True,
        "control_kind": "best",
        "resume_optimizer": False,
        "perturbed": False,
        # A case-matched query requires exactly ``limit`` rows. Promotion
        # therefore requires five saved predecessors, not five stable ones.
        "match_parameters": ["u_max"],
    }


def documents():
    for cap in CAPS:
        for N in TARGET_RESOLUTIONS:
            for strict in (False, True):
                tolerance = "strict" if strict else "loose"
                name = stage_name(N, cap, tolerance)
                yield make_document(
                    name=name,
                    description=(
                        f"N={N} u_max={cap} {tolerance} CPU peak refinement in "
                        "the isolated resolution ladder. Loose stages combine "
                        "five interpolated stable predecessors with ten fresh "
                        "Fourier starts; strict stages refine the best stable "
                        "loose result."
                    ),
                    parameters=parameters(N, cap, strict=strict),
                    runtime=runtime(cap, strict=strict),
                    query=query(N, cap, strict=strict),
                )


def main() -> tuple[Path, ...]:
    paths = []
    for document in documents():
        path = ROOT / "run_config" / f"{document.name}.yaml"
        paths.append(write_config(document, path))
        print(path)
    return tuple(paths)


if __name__ == "__main__":
    main()
