"""Detached, failure-tolerant queue for sequential local GPU configs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from .storage import ResultStore


def _process_start_time(process_id: int) -> str | None:
    try:
        fields = (Path("/proc") / str(process_id) / "stat").read_text(
            encoding="utf-8"
        ).split()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return fields[21] if len(fields) > 21 else None


def _config_is_physically_complete(
    database: Path,
    *,
    queue_id: int,
    config_id: int,
) -> bool:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        count, complete = connection.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN status='complete' THEN 1 ELSE 0 END)
                 FROM runs WHERE queue_id=? AND config_id=?""",
            (queue_id, config_id),
        ).fetchone()
    return bool(count) and int(complete or 0) == int(count)


def _config_progress(database: Path, *, queue_id: int, config_id: int):
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM runs WHERE queue_id=? AND config_id=?",
            (queue_id, config_id),
        ).fetchone()[0]
        halted = connection.execute(
            """SELECT COUNT(DISTINCT r.run_id)
                 FROM runs r JOIN physical_values value ON value.run_id=r.run_id
                WHERE r.queue_id=? AND r.config_id=?
                  AND value.name='termination_reason'
                  AND value.json_value='\"stability\"'""",
            (queue_id, config_id),
        ).fetchone()[0]
        step_range = connection.execute(
            """SELECT MIN(value.numeric_value), MAX(value.numeric_value)
                 FROM runs r JOIN physical_values value ON value.run_id=r.run_id
                WHERE r.queue_id=? AND r.config_id=?
                  AND value.name='completed_steps'""",
            (queue_id, config_id),
        ).fetchone()
    return int(total), int(halted), step_range


def _process_elapsed_seconds(process_id: int, start_time: str) -> float | None:
    try:
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        ticks = int(os.sysconf("SC_CLK_TCK"))
        return max(0.0, uptime - int(start_time) / ticks)
    except (OSError, ValueError, IndexError):
        return None


def _stop_exact_process(process_id: int, expected_start: str) -> None:
    if _process_start_time(process_id) != expected_start:
        return
    os.kill(process_id, signal.SIGTERM)
    for _ in range(300):
        if _process_start_time(process_id) != expected_start:
            return
        time.sleep(0.1)
    if _process_start_time(process_id) == expected_start:
        os.kill(process_id, signal.SIGKILL)


def wait_for_config_then_stop(
    process_id: int,
    *,
    database: Path,
    queue_id: int,
    config_id: int,
    poll_seconds: float = 1.0,
) -> None:
    """Let one loaded config finish, then stop the old multi-config process."""

    expected_start = _process_start_time(process_id)
    if expected_start is None:
        print(f"Process {process_id} already ended; continuing queue.", flush=True)
        return
    print(
        f"Waiting for config_id={config_id} in queue_id={queue_id} to finish "
        f"before replacing process {process_id}.",
        flush=True,
    )
    next_progress = 0.0
    while _process_start_time(process_id) == expected_start:
        if _config_is_physically_complete(
            database, queue_id=queue_id, config_id=config_id
        ):
            print(
                f"config_id={config_id} completed; stopping superseded process "
                f"{process_id} before its next config.",
                flush=True,
            )
            _stop_exact_process(process_id, expected_start)
            return
        now = time.monotonic()
        if now >= next_progress:
            total, halted, step_range = _config_progress(
                database, queue_id=queue_id, config_id=config_id
            )
            minimum, maximum = step_range
            step = (
                "not saved"
                if maximum is None
                else str(int(maximum))
                if minimum == maximum
                else f"{int(minimum)}-{int(maximum)}"
            )
            elapsed = _process_elapsed_seconds(process_id, expected_start)
            print(
                f"CURRENT | config_id={config_id} | halted {halted}/{total} | "
                f"step {step} | elapsed "
                f"{0.0 if elapsed is None else elapsed / 3600:.2f} h",
                flush=True,
            )
            next_progress = now + 60.0
        time.sleep(poll_seconds)
    print(
        f"Process {process_id} ended before the completion transition was needed.",
        flush=True,
    )


def mark_interrupted_runs_failed(
    database: Path,
    *,
    queue_id: int,
    message: str,
) -> int:
    """Close any records left running by a crashed or superseded subprocess."""

    store = ResultStore(database)
    with store.connect() as connection:
        rows = connection.execute(
            """SELECT execution_id, run_id FROM runs
               WHERE queue_id=? AND status='running'
               ORDER BY execution_id, run_id""",
            (queue_id,),
        ).fetchall()
    grouped: dict[int, list[int]] = {}
    for row in rows:
        grouped.setdefault(int(row["execution_id"]), []).append(int(row["run_id"]))
    for execution_id, run_ids in grouped.items():
        store.fail_batch(execution_id, run_ids, message)
    return sum(map(len, grouped.values()))


def _run_command(
    *,
    queue_id: int,
    config: Path,
    project_root: Path,
    python_executable: Path,
) -> list[str]:
    return [
        str(python_executable),
        str(project_root / "run.py"),
        "--queue-id",
        str(queue_id),
        str(config),
    ]


def run_queue(
    runs: list[tuple[int, Path]],
    *,
    project_root: Path,
    python_executable: Path,
    database: Path,
) -> int:
    """Launch every config in its own subprocess and continue after failures."""

    failures = 0
    total = len(runs)
    for index, (queue_id, config) in enumerate(runs, start=1):
        print(
            f"QUEUE {index}/{total} START | queue_id={queue_id} | "
            f"config={config.name} | utc={datetime.now(timezone.utc).isoformat()}",
            flush=True,
        )
        command = _run_command(
            queue_id=queue_id,
            config=config,
            project_root=project_root,
            python_executable=python_executable,
        )
        result = subprocess.run(command, cwd=project_root, check=False)
        if result.returncode:
            failures += 1
            repaired = mark_interrupted_runs_failed(
                database,
                queue_id=queue_id,
                message=(
                    f"Detached queue subprocess exited with code {result.returncode}; "
                    "the next config was still launched."
                ),
            )
            print(
                f"QUEUE {index}/{total} FAILED | config={config.name} | "
                f"exit={result.returncode} | repaired_running_rows={repaired}; "
                "continuing",
                flush=True,
            )
        else:
            print(
                f"QUEUE {index}/{total} COMPLETE | config={config.name} | "
                f"utc={datetime.now(timezone.utc).isoformat()}",
                flush=True,
            )
    print(
        f"QUEUE FINISHED | configs={total} | failures={failures} | "
        f"utc={datetime.now(timezone.utc).isoformat()}",
        flush=True,
    )
    return 1 if failures else 0


def _terminate_subprocess(process: subprocess.Popen, *, grace_seconds=30.0) -> int:
    """Stop one exact child, allowing Python/SQLite a short graceful exit first."""

    process.terminate()
    try:
        return int(process.wait(timeout=grace_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        return int(process.wait())


def run_timed_queue(
    runs: list[tuple[int, Path]],
    *,
    project_root: Path,
    python_executable: Path,
    database: Path,
    max_elapsed_seconds: float,
    repeat_until_deadline: bool = False,
) -> int:
    """Run configs sequentially and stop the active child at one wall deadline."""

    if not runs:
        raise ValueError("At least one queued config is required.")
    if max_elapsed_seconds <= 0.0:
        raise ValueError("max_elapsed_seconds must be positive.")
    deadline = time.monotonic() + float(max_elapsed_seconds)
    failures = 0
    cycle = 0
    while True:
        cycle += 1
        total = len(runs)
        for index, (queue_id, config) in enumerate(runs, start=1):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                print(
                    f"TIMED QUEUE FINISHED | cycles={cycle - 1} | failures={failures} | "
                    f"utc={datetime.now(timezone.utc).isoformat()}",
                    flush=True,
                )
                return 1 if failures else 0
            print(
                f"CYCLE {cycle} QUEUE {index}/{total} START | queue_id={queue_id} | "
                f"config={config.name} | remaining={remaining / 3600:.2f} h | "
                f"utc={datetime.now(timezone.utc).isoformat()}",
                flush=True,
            )
            command = _run_command(
                queue_id=queue_id,
                config=config,
                project_root=project_root,
                python_executable=python_executable,
            )
            process = subprocess.Popen(command, cwd=project_root)
            deadline_reached = False
            try:
                return_code = int(process.wait(timeout=remaining))
            except subprocess.TimeoutExpired:
                deadline_reached = True
                return_code = _terminate_subprocess(process)
            if return_code:
                if not deadline_reached:
                    failures += 1
                message = (
                    "Timed queue wall deadline reached; committed partial data "
                    "was preserved."
                    if deadline_reached
                    else f"Timed queue subprocess exited with code {return_code}."
                )
                repaired = mark_interrupted_runs_failed(
                    database,
                    queue_id=queue_id,
                    message=message,
                )
                print(
                    f"CYCLE {cycle} QUEUE {index}/{total} "
                    f"{'DEADLINE' if deadline_reached else 'FAILED'} | "
                    f"config={config.name} | exit={return_code} | "
                    f"repaired_running_rows={repaired}",
                    flush=True,
                )
            else:
                print(
                    f"CYCLE {cycle} QUEUE {index}/{total} COMPLETE | "
                    f"config={config.name} | "
                    f"utc={datetime.now(timezone.utc).isoformat()}",
                    flush=True,
                )
            if deadline_reached:
                print(
                    f"TIMED QUEUE FINISHED | cycles={cycle - 1} | failures={failures} | "
                    f"utc={datetime.now(timezone.utc).isoformat()}",
                    flush=True,
                )
                return 1 if failures else 0
        if not repeat_until_deadline:
            print(
                f"TIMED QUEUE FINISHED | cycles={cycle} | failures={failures} | "
                f"utc={datetime.now(timezone.utc).isoformat()}",
                flush=True,
            )
            return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-queue-id", type=int)
    parser.add_argument("--wait-config-id", type=int)
    parser.add_argument("--max-elapsed-seconds", type=float)
    parser.add_argument("--repeat-until-deadline", action="store_true")
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("QUEUE_ID", "CONFIG"),
        required=True,
    )
    arguments = parser.parse_args(argv)
    project_root = arguments.project_root.expanduser().resolve()
    database = arguments.database.expanduser()
    if not database.is_absolute():
        database = (project_root / database).resolve()
    transition = (
        arguments.wait_pid,
        arguments.wait_queue_id,
        arguments.wait_config_id,
    )
    if any(value is not None for value in transition):
        if any(value is None for value in transition):
            parser.error(
                "--wait-pid, --wait-queue-id, and --wait-config-id are required together"
            )
        wait_for_config_then_stop(
            arguments.wait_pid,
            database=database,
            queue_id=arguments.wait_queue_id,
            config_id=arguments.wait_config_id,
        )
        repaired = mark_interrupted_runs_failed(
            database,
            queue_id=arguments.wait_queue_id,
            message="Superseded after the requested preceding config completed.",
        )
        print(f"Closed {repaired} superseded running row(s).", flush=True)
    runs = [
        (int(queue_id), Path(config).expanduser().resolve())
        for queue_id, config in arguments.run
    ]
    if arguments.repeat_until_deadline and arguments.max_elapsed_seconds is None:
        parser.error("--repeat-until-deadline requires --max-elapsed-seconds")
    if arguments.max_elapsed_seconds is not None:
        return run_timed_queue(
            runs,
            project_root=project_root,
            python_executable=arguments.python.expanduser().resolve(),
            database=database,
            max_elapsed_seconds=arguments.max_elapsed_seconds,
            repeat_until_deadline=arguments.repeat_until_deadline,
        )
    return run_queue(
        runs,
        project_root=project_root,
        python_executable=arguments.python.expanduser().resolve(),
        database=database,
    )


if __name__ == "__main__":
    raise SystemExit(main())
