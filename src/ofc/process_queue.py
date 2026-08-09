"""Detached local process dependency used by the notebook workflow adapter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time


def _process_start_time(process_id: int) -> str | None:
    try:
        fields = (Path("/proc") / str(process_id) / "stat").read_text(
            encoding="utf-8"
        ).split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return fields[21] if len(fields) > 21 else None


def wait_then_exec(process_id: int, command: list[str], *, poll_seconds=30.0) -> None:
    """Wait for one exact Linux process lifetime, then replace this process."""

    if process_id < 1:
        raise ValueError("process_id must be positive.")
    if not command:
        raise ValueError("command cannot be empty.")
    expected_start = _process_start_time(process_id)
    if expected_start is not None:
        print(f"Waiting for process {process_id} before starting queued run.", flush=True)
    while expected_start is not None:
        current_start = _process_start_time(process_id)
        if current_start != expected_start:
            break
        time.sleep(poll_seconds)
    print(f"Process {process_id} finished; starting queued run.", flush=True)
    os.execvpe(command[0], command, os.environ)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    wait_then_exec(arguments.wait_pid, command)


if __name__ == "__main__":
    main()
