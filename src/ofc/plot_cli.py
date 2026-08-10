"""Read-only adapter for the isolated standard plotting functions.

With no query, the most recently stored completed config is selected. This
adapter retrieves complete mappings and passes them to plotting; plotting never
opens the database and Results never imports plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .filtering import parse_filters
from .plotting import save_standard_figures
from .results import Results


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create the unified score, objective-strip, and control summary."
    )
    parser.add_argument("--database", default=str(PROJECT_ROOT / "results/results.sqlite3"))
    parser.add_argument("--run-id", action="append", type=int, default=[])
    parser.add_argument("--where", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of matching initializations to plot.",
    )
    parser.add_argument(
        "--config-run",
        type=int,
        metavar="N",
        help=(
            "Plot the Nth most recent matching configuration execution "
            "(1 is latest, 2 is second latest)."
        ),
    )
    parser.add_argument(
        "--sweep-parameter",
        help=(
            "Parameter used to split the unified summary into sweep rows. "
            "It is inferred when exactly one configuration parameter varies; "
            "otherwise initializations are numbered as a categorical sweep."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory; defaults to figures/<selected-config>.",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("png", "pdf"),
        help="Repeat to save multiple formats; defaults to png.",
    )
    return parser


def select_rows(results: Results, *, run_ids, filters, limit, config_run=None):
    if run_ids:
        rows = [results.search(run_id=run_id, limit=1) for run_id in run_ids]
        missing = [run_id for run_id, matches in zip(run_ids, rows) if not matches]
        if missing:
            raise KeyError(f"Unknown run_id values: {missing}")
        return [matches[0] for matches in rows]
    if config_run is not None:
        return results.search_config_run(
            rank=config_run,
            limit=limit,
            **filters,
        )
    if filters:
        return results.search(limit=limit, **filters)

    completed = results.search_config_run(rank=1, status="complete", limit=limit)
    if not completed:
        raise ValueError("The results database contains no completed runs to plot.")
    return completed


def _default_output_dir(rows):
    names = list(dict.fromkeys(str(row["config_name"]) for row in rows))
    directory = names[0] if len(names) == 1 else "selection"
    return PROJECT_ROOT / "figures" / directory


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    results = Results(arguments.database)
    rows = select_rows(
        results,
        run_ids=arguments.run_id,
        filters=parse_filters(arguments.where),
        limit=arguments.limit,
        config_run=arguments.config_run,
    )
    if not rows:
        raise ValueError("The query matched no runs.")
    runs = [results.get(row["run_id"]) for row in rows]
    saved = save_standard_figures(
        runs,
        arguments.output_dir or _default_output_dir(rows),
        formats=arguments.formats or ("png",),
        sweep_parameter=arguments.sweep_parameter,
    )
    for name, formats in saved.items():
        for file_format, path in formats.items():
            print(f"{name} ({file_format}): {path}")
    return 0
