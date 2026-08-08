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
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    matches = Results(arguments.database).search(
        limit=arguments.limit,
        order_by=arguments.order_by,
        **parse_filters(arguments.where),
    )
    print(json.dumps(matches, indent=2))
    return 0
