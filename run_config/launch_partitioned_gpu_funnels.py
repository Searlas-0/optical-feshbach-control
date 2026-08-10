#!/usr/bin/env python3
"""Safely replace one queue with two memory-partitioned GPU queues."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofc.config import random_id
from ofc.resilient_queue import (
    mark_interrupted_runs_failed,
    wait_for_config_then_stop,
)


def _manifest_paths(path: Path) -> tuple[Path, ...]:
    entries = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not entries:
        raise ValueError(f"{path} is empty.")
    paths = tuple((ROOT / entry).resolve() for entry in entries)
    missing = [str(config) for config in paths if not config.is_file()]
    if missing:
        raise FileNotFoundError("Missing manifest configs: " + ", ".join(missing))
    return paths


def _queue_command(
    *, queue_id: int, database: str, configs: tuple[Path, ...]
) -> list[str]:
    command = [
        str(Path(sys.executable).resolve()),
        "-m",
        "ofc.resilient_queue",
        "--project-root",
        str(ROOT),
        "--python",
        str(Path(sys.executable).resolve()),
        "--database",
        database,
    ]
    for config in configs:
        command.extend(("--run", str(queue_id), str(config)))
    return command


def _refresh_wait_database(database: Path) -> int:
    """Refresh diagnostics written by the still-running pre-feature process."""

    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(f"Refreshing finalized grid diagnostics in {database}.", flush=True)
    result = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-m",
            "ofc.grid_refinement_backfill",
            "--database",
            str(database),
            "--overwrite",
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    print(
        f"Grid diagnostic refresh exited with code {result.returncode}.", flush=True
    )
    return result.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--wait-queue-id", type=int, required=True)
    parser.add_argument("--wait-config-id", type=int, required=True)
    parser.add_argument("--wait-database", type=Path, required=True)
    parser.add_argument("--track-a-manifest", type=Path, required=True)
    parser.add_argument("--track-a-database", required=True)
    parser.add_argument("--track-a-log", type=Path, required=True)
    parser.add_argument("--track-b-manifest", type=Path, required=True)
    parser.add_argument("--track-b-database", required=True)
    parser.add_argument("--track-b-log", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    arguments = parser.parse_args(argv)
    if not 0.0 < arguments.memory_fraction < 0.5:
        parser.error("--memory-fraction must be greater than zero and below 0.5")

    wait_database = arguments.wait_database
    if not wait_database.is_absolute():
        wait_database = ROOT / wait_database
    wait_for_config_then_stop(
        arguments.wait_pid,
        database=wait_database.resolve(),
        queue_id=arguments.wait_queue_id,
        config_id=arguments.wait_config_id,
    )
    repaired = mark_interrupted_runs_failed(
        wait_database.resolve(),
        queue_id=arguments.wait_queue_id,
        message="Superseded by the requested two-track GPU partition.",
    )
    refresh_exit_code = _refresh_wait_database(wait_database.resolve())

    tracks = (
        (
            "track_a",
            arguments.track_a_manifest,
            arguments.track_a_database,
            arguments.track_a_log,
        ),
        (
            "track_b",
            arguments.track_b_manifest,
            arguments.track_b_database,
            arguments.track_b_log,
        ),
    )
    environment = os.environ.copy()
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(arguments.memory_fraction)
    state = {
        "launched_utc": datetime.now(timezone.utc).isoformat(),
        "memory_fraction_per_track": arguments.memory_fraction,
        "superseded_pid": arguments.wait_pid,
        "superseded_queue_id": arguments.wait_queue_id,
        "closed_superseded_rows": repaired,
        "grid_diagnostic_refresh_exit_code": refresh_exit_code,
        "tracks": {},
    }
    for label, manifest, database, log_path in tracks:
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        configs = _manifest_paths(manifest.resolve())
        queue_id = random_id()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            _queue_command(queue_id=queue_id, database=database, configs=configs),
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_stream.close()
        state["tracks"][label] = {
            "pid": process.pid,
            "queue_id": queue_id,
            "database": database,
            "manifest": str(manifest.resolve()),
            "log": str(log_path.resolve()),
            "config_count": len(configs),
            "first_config": configs[0].name,
            "last_config": configs[-1].name,
        }
        print(
            f"LAUNCHED {label} | pid={process.pid} | queue_id={queue_id} | "
            f"configs={len(configs)} | log={log_path}",
            flush=True,
        )

    state_path = arguments.state
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(state_path)
    print(f"STATE {state_path} | closed_superseded_rows={repaired}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
