#!/usr/bin/env python3
"""Launch the repeating best-per-cap refinement ladder on the local GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofc.config import random_id


MANIFEST = ROOT / "run_config/N100_all_caps_best_refinement_10h_v1.manifest"
DATABASE = "results/results.sqlite3"
LOG = ROOT / "logs/N100_all_caps_best_refinement_10h_v1.log"
STATE = ROOT / "logs/N100_all_caps_best_refinement_10h_v1.state.json"


def _process_start(process_id: int) -> str | None:
    try:
        fields = (Path("/proc") / str(process_id) / "stat").read_text(
            encoding="utf-8"
        ).split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return fields[21] if len(fields) > 21 else None


def _manifest_paths() -> tuple[Path, ...]:
    paths = tuple(
        (ROOT / line.strip()).resolve()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("The timed cap-refinement manifest is incomplete.")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=10.0)
    arguments = parser.parse_args(argv)
    if arguments.hours <= 0.0:
        parser.error("--hours must be positive")
    if STATE.exists():
        previous = json.loads(STATE.read_text(encoding="utf-8"))
        if _process_start(int(previous["pid"])) == previous.get("process_start"):
            raise RuntimeError(
                f"Timed refinement PID {previous['pid']} is already running."
            )

    paths = _manifest_paths()
    queue_id = random_id()
    duration_seconds = arguments.hours * 60 * 60
    command = [
        str(Path(sys.executable).resolve()),
        "-m",
        "ofc.resilient_queue",
        "--project-root",
        str(ROOT),
        "--python",
        str(Path(sys.executable).resolve()),
        "--database",
        DATABASE,
        "--max-elapsed-seconds",
        str(duration_seconds),
        "--repeat-until-deadline",
    ]
    for path in paths:
        command.extend(("--run", str(queue_id), str(path)))

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (str(ROOT / "src"), existing_pythonpath)
        if item
    )
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launched = datetime.now(timezone.utc)
    state = {
        "pid": process.pid,
        "process_start": _process_start(process.pid),
        "queue_id": queue_id,
        "database": DATABASE,
        "manifest": str(MANIFEST),
        "log": str(LOG),
        "hours": arguments.hours,
        "launched_utc": launched.isoformat(),
        "deadline_utc": (launched + timedelta(seconds=duration_seconds)).isoformat(),
    }
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
