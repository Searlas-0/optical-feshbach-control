"""Independent read-only result-query command.

Isolation boundary: this command knows only the Results API.  It never imports
the runner, config generator, numerical modules, or plotting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .filtering import parse_filters
from .results import Results


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Search result metadata. Numerical ranges use NAME=MIN:MAX."
    )
    parser.add_argument("--database", default=str(PROJECT_ROOT / "results/results.sqlite3"))
    parser.add_argument("--where", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--order-by", default="run_id")
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort the selected field from largest/newest to smallest/oldest.",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Return only the run with the highest regularized best_score.",
    )
    parser.add_argument(
        "--config-run",
        type=int,
        metavar="N",
        help=(
            "Restrict results to the Nth most recent matching configuration "
            "execution (1 is latest, 2 is second latest)."
        ),
    )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if arguments.config_run is not None and arguments.config_run < 1:
        raise ValueError("--config-run must be a positive integer.")
    order_by = "best_score" if arguments.best else arguments.order_by
    descending = True if arguments.best else arguments.descending
    limit = 1 if arguments.best else arguments.limit
    results = Results(arguments.database)
    search = results.search if arguments.config_run is None else results.search_config_run
    options = {
        "limit": limit,
        "order_by": order_by,
        "descending": descending,
        **parse_filters(arguments.where),
    }
    if arguments.config_run is not None:
        options["rank"] = arguments.config_run
    matches = search(**options)
    print(json.dumps(matches, indent=2))
    return 0
