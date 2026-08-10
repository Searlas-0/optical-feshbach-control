#!/usr/bin/env python3
"""Build the compact cross-cap database downloaded by the laptop worker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ofc.auto_initialization import (
    AUTO_EXACT_CAP_LIMIT,
    AUTO_GLOBAL_LIMIT,
    priors_from_results,
    write_prior_snapshot,
)


DEFAULT_DATABASES = (
    "results/results.sqlite3",
    "results/bar_seed_sensitivity.sqlite3",
    "results/bar_endpoint_seed_sensitivity.sqlite3",
    "results/bar_endpoint_seed_sensitivity_u320.sqlite3",
    "results/bar_u320_crossover_screen.sqlite3",
    "results/bar_endpoint_seed1000_loose_u320.sqlite3",
    "results/slurm_endpoint_strict_u320.sqlite3",
)
DEFAULT_CAPS = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0)
DEFAULT_OUTPUT = ROOT / "results" / "auto_fourier_intensity_priors.sqlite3"
DEFAULT_TRANSFER = ROOT / "results" / "auto_fourier_intensity_priors-transfer.zip"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", action="append", dest="databases")
    parser.add_argument("--cap", action="append", type=float, dest="caps")
    parser.add_argument("--t-interval", type=float, default=4.0)
    parser.add_argument("--r-bg", type=float, default=-0.008716)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transfer", type=Path, default=DEFAULT_TRANSFER)
    arguments = parser.parse_args(argv)

    databases = tuple(
        path
        for path in map(_project_path, arguments.databases or DEFAULT_DATABASES)
        if path.is_file()
    )
    if not databases:
        parser.error("none of the requested source databases exist")
    caps = tuple(arguments.caps or DEFAULT_CAPS)
    priors = {}
    for database in databases:
        candidates = list(
            priors_from_results(
                database,
                t_interval=arguments.t_interval,
                r_bg=arguments.r_bg,
                u_max=None,
                limit=AUTO_GLOBAL_LIMIT,
            )
        )
        for cap in caps:
            candidates.extend(
                priors_from_results(
                    database,
                    t_interval=arguments.t_interval,
                    r_bg=arguments.r_bg,
                    u_max=cap,
                    limit=AUTO_EXACT_CAP_LIMIT,
                )
            )
        priors.update({candidate.source_key: candidate for candidate in candidates})
        print(
            f"PRIORS {database.name} | accumulated_unique={len(priors)}",
            flush=True,
        )
    if not priors:
        raise RuntimeError("No complete matching solutions with best controls were found.")

    created = datetime.now(timezone.utc).isoformat()
    output = _project_path(arguments.output)
    write_prior_snapshot(
        output,
        priors.values(),
        metadata={
            "created_utc": created,
            "t_interval": arguments.t_interval,
            "r_bg": arguments.r_bg,
            "caps": caps,
            "source_databases": [str(path) for path in databases],
        },
    )
    transfer = _project_path(arguments.transfer)
    transfer.parent.mkdir(parents=True, exist_ok=True)
    temporary = transfer.with_suffix(transfer.suffix + ".tmp")
    metadata = {
        "created_utc": created,
        "database_file": output.name,
        "row_count": len(priors),
        "t_interval": arguments.t_interval,
        "r_bg": arguments.r_bg,
        "caps": caps,
    }
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output, output.name)
        archive.writestr(
            "auto_fourier_intensity_priors_metadata.json",
            json.dumps(metadata, indent=2) + "\n",
        )
    temporary.replace(transfer)
    print(f"WROTE {output} | rows={len(priors)}", flush=True)
    print(f"TRANSFER {transfer}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
