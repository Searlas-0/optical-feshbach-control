"""CLI adapter for the runner; no calculation or storage logic belongs here.

Isolation boundary: this entry point is the sole local-process composition
root.  It resolves config-name arguments, passes them to ``run_configs``, and
prints the returned summary.  Keep future calculations in their named modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import PROJECT_ROOT, run_configs


def resolve_config(value: str) -> Path:
    supplied = Path(value).expanduser()
    candidates = [supplied]
    if supplied.suffix not in {".yaml", ".yml"}:
        candidates.extend([supplied.with_suffix(".yaml"), supplied.with_suffix(".yml")])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        in_config_dir = PROJECT_ROOT / "run_config" / candidate
        if in_config_dir.is_file():
            return in_config_dir.resolve()
    raise FileNotFoundError(
        f"Cannot find config {value!r}; bare names are resolved inside {PROJECT_ROOT / 'run_config'}."
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one or more configs in argument order; sweeps execute within each config."
    )
    parser.add_argument("configs", nargs="+", help="Config path or filename in run_config/")
    parser.add_argument(
        "--batch-index",
        action="append",
        type=int,
        help="Run only this zero-based batch index; repeat to select several.",
    )
    parser.add_argument(
        "--queue-id",
        type=int,
        help="Use an explicit shared queue ID (for example a Slurm array job ID).",
    )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    summary = run_configs(
        [resolve_config(value) for value in arguments.configs],
        batch_indices=arguments.batch_index,
        queue_id=arguments.queue_id,
    )
    print(json.dumps(summary, indent=2))
    return 0
