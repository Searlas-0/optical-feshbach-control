#!/usr/bin/env python3
"""Generate the repeating ten-hour best-per-cap refinement ladder."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofc.config import make_document, write_config


CAPS = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
TOP_COUNTS = (10, 50, 100)
DATABASE = "results/results.sqlite3"
MANIFEST = ROOT / "run_config/N100_all_caps_best_refinement_10h_v1.manifest"
PER_CONFIG_SECONDS = 60 * 60
TOLERANCE_LEVELS = (
    {
        "name": "loose",
        "optimizer": "lbfgs",
        "schedule": ((50, 1.0),),
        "J_tol": 1e-5,
        "control_tol": 1e-4,
        "gradient_tol": 1e-4,
        "grid_tol": 1e-2,
    },
    {
        "name": "tighter",
        "optimizer": "peak_refinement",
        "schedule": ((250, 1.0),),
        "J_tol": 3e-6,
        "control_tol": 3e-5,
        "gradient_tol": 3e-5,
        "grid_tol": 3e-3,
    },
    {
        "name": "strict",
        "optimizer": "peak_refinement",
        "schedule": ((250, 1.0),),
        "J_tol": 1e-6,
        "control_tol": 1e-5,
        "gradient_tol": 1e-5,
        "grid_tol": 1e-3,
    },
)


def _document(level: dict, top_count: int):
    name = f"N100_all_caps_top{top_count}_{level['name']}_refinement_10h_v1_gpu"
    return make_document(
        name=name,
        description=(
            f"Refine the current top {top_count} best-score controls independently "
            f"at each of the nine N=100 caps using {level['name']} convergence "
            "thresholds. The one-hour guard is distributed across caps and query "
            "shards so no cap can monopolize the repeating ten-hour ladder."
        ),
        parameters={
            "N": 100,
            "t_interval": 4.0,
            "r_bg": -0.008716,
            "u_isbound": True,
            "v_isbound": True,
            "u_max": float(CAPS[0]),
            "v_max": 1000.0,
            "slew_limit": 0.05,
            "optimizer": level["optimizer"],
            "schedule": level["schedule"],
            "adam_learning_rate": 0.05,
            "adam_beta1": 0.9,
            "adam_beta2": 0.99,
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
            "smoothness": 3.952847075210474e-9,
            "u_smooth": None,
            "v_smooth": None,
            "sharpness": 3.952847075210474e-10,
            "u_sharp": None,
            "v_sharp": None,
            "block_size": 10 if level["optimizer"] == "lbfgs" else 25,
            "J_tol": level["J_tol"],
            "u_tol": level["control_tol"],
            "v_tol": level["control_tol"],
            "projected_gradient_tol": level["gradient_tol"],
            "projected_gradient_alpha": 1.0,
            "grid_refinement_tol": level["grid_tol"],
            "grid_refinement_y_floor": 1e-12,
        },
        runtime={
            "initialisations": 0,
            "fourier_num_modes": 6,
            "fourier_rms_amplitude": 0.8,
            "fourier_intensity_fraction": 0.3,
            "fourier_intensity_auto_database": None,
            "use_jit": True,
            "use_x64": True,
            "device": "gpu",
            "concurrent_workers": 1,
            "max_cases_per_batch": None,
            "max_initialisations_per_batch": 25,
            "max_steps_per_chunk": 50,
            "max_batch_elapsed_seconds": PER_CONFIG_SECONDS,
            "max_elapsed_seconds": PER_CONFIG_SECONDS,
            "distribute_max_elapsed_across_batches": True,
            "repeat_schedule_until_stable": True,
            "auto_halt": True,
            "database": DATABASE,
        },
        query={
            "where": {
                "status": "complete",
                "N": 100,
                "t_interval": 4.0,
                "r_bg": -0.008716,
                "u_max": [float(cap) for cap in CAPS],
            },
            "database": DATABASE,
            "limit": top_count,
            "order_by": "best_score",
            "descending": True,
            "control_kind": "best",
            "resume_optimizer": False,
            "perturbed": False,
            "match_parameters": ["u_max"],
            "discover_parameters": ["u_max"],
            "discover_group_parameters": ["u_max"],
        },
    )


def main() -> tuple[Path, ...]:
    if MANIFEST.exists():
        paths = tuple(
            ROOT / line.strip()
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if paths and all(path.is_file() for path in paths):
            return paths
        raise FileNotFoundError("The existing timed-refinement manifest is incomplete.")
    paths = []
    for level in TOLERANCE_LEVELS:
        for top_count in TOP_COUNTS:
            document = _document(level, top_count)
            path = ROOT / "run_config" / f"{document.name}.yaml"
            paths.append(write_config(document, path))
    MANIFEST.write_text(
        "".join(f"{path.relative_to(ROOT)}\n" for path in paths),
        encoding="utf-8",
    )
    return tuple(paths)


if __name__ == "__main__":
    for generated in main():
        print(generated.relative_to(ROOT))
