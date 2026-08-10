"""Replaceable Slurm strict refinements for bar endpoint challengers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

from .config import write_config
from .resilient_queue import mark_interrupted_runs_failed
from .results import Results
from .seed_sensitivity import (
    fixed_seed_stage_name,
    loose_stage_name,
    strict_refinement_document,
)


def _log(message: str) -> None:
    print(
        f"{datetime.now(timezone.utc).isoformat()} | {message}",
        flush=True,
    )


def _process_exists(process_id: int) -> bool:
    return (Path("/proc") / str(process_id) / "stat").is_file()


def _job_active(job_id: int) -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"squeue failed for {job_id}")
    return bool(result.stdout.strip())


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"keys": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _endpoint_key(N: int, cap: int, endpoint: str) -> str:
    return f"N{N}_u{cap}_{endpoint}"


def _candidate(
    results: Results,
    *,
    N: int,
    cap: int,
    endpoint: str,
    settings: Mapping[str, Any],
    exploration_initialisations: int,
    loose_count: int = 20,
    parameter_label_suffix: str = "_bar_v2",
) -> dict[str, Any] | None:
    label = f"{endpoint}_regularization{parameter_label_suffix}"
    config_names = [
        fixed_seed_stage_name(
            N,
            cap,
            stage,
            exploration_initialisations,
            parameter_label=label,
        )
        for stage in ("exploration", loose_stage_name(loose_count))
    ]
    rows = results.search(
        N=N,
        u_max=float(cap),
        smoothness=float(settings["smoothness"]),
        sharpness=float(settings["sharpness"]),
        config_name=config_names,
        order_by="best_objective",
        descending=True,
        limit=1,
    )
    if not rows or not math.isfinite(float(rows[0].get("best_objective", math.nan))):
        return None
    return rows[0]


def _strict_rows(
    database: Path,
    *,
    N: int,
    cap: int,
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not database.is_file():
        return []
    return Results(database).search(
        N=N,
        u_max=float(cap),
        smoothness=float(settings["smoothness"]),
        sharpness=float(settings["sharpness"]),
        order_by="best_objective",
        descending=True,
    )


def _pair_agreement(
    results: Results,
    low: Mapping[str, Any],
    high: Mapping[str, Any],
) -> dict[str, float]:
    low_controls = results.controls(int(low["run_id"]), "best")
    high_controls = results.controls(int(high["run_id"]), "best")
    cap = float(low["u_max"])
    v_max = float(low["v_max"])
    differences = np.concatenate(
        (
            (np.asarray(low_controls["u"]) - np.asarray(high_controls["u"])) / cap,
            (np.asarray(low_controls["v"]) - np.asarray(high_controls["v"])) / v_max,
        )
    )
    low_objective = float(low["best_objective"])
    high_objective = float(high["best_objective"])
    return {
        "objective_relative_difference": abs(low_objective - high_objective)
        / max(abs(low_objective), abs(high_objective), 1e-12),
        "normalized_control_rms_difference": float(
            np.sqrt(np.mean(np.square(differences)))
        ),
    }


def _final_agreement_report(
    database: Path,
    *,
    endpoint_settings: Mapping[int, Mapping[str, Mapping[str, Any]]],
    resolutions: tuple[int, ...],
    objective_tolerance: float,
    control_tolerance: float,
) -> dict[str, Any]:
    results = Results(database)
    report: dict[str, Any] = {}
    for N in resolutions:
        for cap, endpoints in endpoint_settings.items():
            best = {}
            for endpoint, settings in endpoints.items():
                rows = _strict_rows(
                    database,
                    N=N,
                    cap=cap,
                    settings=settings,
                )
                if rows:
                    best[endpoint] = rows[0]
            key = f"N{N}_u{cap}"
            if set(best) != {"low", "high"}:
                report[key] = {"agrees": False, "reason": "missing endpoint"}
                continue
            differences = _pair_agreement(results, best["low"], best["high"])
            agrees = (
                differences["objective_relative_difference"] <= objective_tolerance
                and differences["normalized_control_rms_difference"]
                <= control_tolerance
            )
            report[key] = {"agrees": agrees, **differences}
            _log(
                f"AGREEMENT | {key} | agrees={agrees} | "
                f"relative_J_mol={differences['objective_relative_difference']:.6g} | "
                f"normalized_control_rms="
                f"{differences['normalized_control_rms_difference']:.6g}"
            )
    return report


def _submit_strict(
    *,
    project_root: Path,
    python_executable: Path,
    bar_database: Path,
    strict_database: Path,
    state_entry: dict[str, Any],
    key: str,
    N: int,
    cap: int,
    endpoint: str,
    settings: Mapping[str, Any],
    candidate: Mapping[str, Any],
    strict_max_elapsed_seconds: float,
    partition: str,
    slurm_time: str,
    cpus: int,
    memory: str,
) -> None:
    active_job_id = state_entry.get("active_job_id")
    if active_job_id is not None:
        cancellation = subprocess.run(
            ["scancel", str(active_job_id)],
            check=False,
            capture_output=True,
            text=True,
        )
        if cancellation.returncode and _job_active(int(active_job_id)):
            raise RuntimeError(
                cancellation.stderr.strip()
                or f"could not cancel strict job {active_job_id}"
            )
        for _ in range(60):
            try:
                if not _job_active(int(active_job_id)):
                    break
            except RuntimeError:
                break
            time.sleep(0.5)
        if _job_active(int(active_job_id)):
            raise RuntimeError(
                f"strict job {active_job_id} did not stop within 30 seconds"
            )
        if strict_database.is_file():
            repaired = mark_interrupted_runs_failed(
                strict_database,
                queue_id=int(active_job_id),
                message=(
                    "Superseded by a higher-J_mol exploratory or loose-refinement "
                    "challenger; saved strict checkpoints retained."
                ),
            )
        else:
            repaired = 0
        _log(f"{key} | cancelled superseded strict job {active_job_id}")
        if repaired:
            _log(f"{key} | closed {repaired} superseded strict run row(s)")

    revision = int(state_entry.get("revision", 0)) + 1
    name = f"{key}_strict_revision{revision}_slurm_cpu"
    document = strict_refinement_document(
        name=name,
        N=N,
        cap=cap,
        settings=settings,
        source_database=str(bar_database),
        target_database=str(strict_database),
        source_run_id=int(candidate["run_id"]),
        max_elapsed_seconds=strict_max_elapsed_seconds,
    )
    config_path = project_root / "run_config" / f"{name}.yaml"
    while config_path.exists():
        revision += 1
        name = f"{key}_strict_revision{revision}_slurm_cpu"
        document = strict_refinement_document(
            name=name,
            N=N,
            cap=cap,
            settings=settings,
            source_database=str(bar_database),
            target_database=str(strict_database),
            source_run_id=int(candidate["run_id"]),
            max_elapsed_seconds=strict_max_elapsed_seconds,
        )
        config_path = project_root / "run_config" / f"{name}.yaml"
    write_config(document, config_path)
    log_stem = project_root / "logs" / f"{name}-%j"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={name[:80]}",
        f"--partition={partition}",
        f"--time={slurm_time}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--output={log_stem}.out",
        f"--error={log_stem}.err",
        f"--export=ALL,OFC_PYTHON={python_executable}",
        str(project_root / "slurm" / "run_config.slurm"),
        str(config_path),
    ]
    submission = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = int(submission.stdout.strip().splitlines()[-1].split(";", 1)[0])
    state_entry.update(
        {
            "revision": revision,
            "active_job_id": job_id,
            "active_config_name": name,
            "last_source_run_id": int(candidate["run_id"]),
            "last_source_objective": float(candidate["best_objective"]),
            "last_submission_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    _log(
        f"{key} | submitted strict job {job_id} from bar run "
        f"{candidate['run_id']} at J_mol={float(candidate['best_objective']):.12g}"
    )


def monitor(
    *,
    project_root: Path,
    python_executable: Path,
    bar_database: Path,
    strict_database: Path,
    state_path: Path,
    bar_process_id: int,
    endpoint_settings: Mapping[int, Mapping[str, Mapping[str, Any]]],
    resolutions: tuple[int, ...],
    exploration_initialisations: int,
    loose_count: int,
    parameter_label_suffix: str,
    strict_max_elapsed_seconds: float,
    poll_seconds: float,
    objective_epsilon: float,
    partition: str,
    slurm_time: str,
    cpus: int,
    memory: str,
    agreement_objective_tolerance: float,
    agreement_control_tolerance: float,
) -> int:
    """Monitor challengers until one current strict run completes for every key."""

    state = _load_state(state_path)
    state_entries = state.setdefault("keys", {})
    expected = [
        (N, cap, endpoint, settings)
        for N in resolutions
        for cap, endpoints in endpoint_settings.items()
        for endpoint, settings in endpoints.items()
    ]
    _log(
        f"monitor started | keys={len(expected)} | bar_pid={bar_process_id} | "
        f"bar_database={bar_database} | strict_database={strict_database}"
    )
    while True:
        bar_results = Results(bar_database) if bar_database.is_file() else None
        completed_keys = 0
        missing_keys = 0
        active_jobs = 0
        changed = False
        for N, cap, endpoint, settings in expected:
            key = _endpoint_key(N, cap, endpoint)
            entry = state_entries.setdefault(key, {})
            job_id = entry.get("active_job_id")
            if job_id is not None:
                try:
                    if _job_active(int(job_id)):
                        active_jobs += 1
                    else:
                        entry["active_job_id"] = None
                        changed = True
                except RuntimeError as error:
                    _log(f"{key} | scheduler check deferred: {error}")
                    active_jobs += 1

            try:
                candidate = (
                    None
                    if bar_results is None
                    else _candidate(
                        bar_results,
                        N=N,
                        cap=cap,
                        endpoint=endpoint,
                        settings=settings,
                        exploration_initialisations=exploration_initialisations,
                        loose_count=loose_count,
                        parameter_label_suffix=parameter_label_suffix,
                    )
                )
            except (OSError, sqlite3.Error, KeyError, ValueError) as error:
                _log(f"{key} | challenger query deferred: {error}")
                missing_keys += 1
                continue
            if candidate is None:
                missing_keys += 1
                continue

            try:
                strict_rows = _strict_rows(
                    strict_database,
                    N=N,
                    cap=cap,
                    settings=settings,
                )
            except (OSError, sqlite3.Error, KeyError, ValueError) as error:
                _log(f"{key} | strict-result query deferred: {error}")
                continue
            strict_best = max(
                (
                    float(row["best_objective"])
                    for row in strict_rows
                    if math.isfinite(float(row.get("best_objective", math.nan)))
                ),
                default=-math.inf,
            )
            last_source_objective = float(
                entry.get("last_source_objective", -math.inf)
            )
            challenger_objective = float(candidate["best_objective"])
            incumbent = max(strict_best, last_source_objective)
            active_job_id = entry.get("active_job_id")
            active_config_name = entry.get("active_config_name")
            active_rows = [
                row for row in strict_rows if row.get("config_name") == active_config_name
            ]
            active_completed = bool(active_rows) and all(
                row["status"] == "complete" for row in active_rows
            )

            should_submit = challenger_objective > incumbent + objective_epsilon
            if active_job_id is None and not active_completed:
                should_submit = True
            if should_submit:
                try:
                    _submit_strict(
                        project_root=project_root,
                        python_executable=python_executable,
                        bar_database=bar_database,
                        strict_database=strict_database,
                        state_entry=entry,
                        key=key,
                        N=N,
                        cap=cap,
                        endpoint=endpoint,
                        settings=settings,
                        candidate=candidate,
                        strict_max_elapsed_seconds=strict_max_elapsed_seconds,
                        partition=partition,
                        slurm_time=slurm_time,
                        cpus=cpus,
                        memory=memory,
                    )
                    changed = True
                    active_jobs += 1
                except (
                    OSError,
                    RuntimeError,
                    subprocess.SubprocessError,
                    ValueError,
                ) as error:
                    _log(f"{key} | strict submission failed; will retry: {error}")
                continue

            if active_job_id is None and active_completed:
                completed_keys += 1

        if changed:
            _save_state(state_path, state)
        bar_active = _process_exists(bar_process_id)
        _log(
            f"progress | strict_complete={completed_keys}/{len(expected)} | "
            f"strict_active={active_jobs} | candidates_missing={missing_keys} | "
            f"bar_active={bar_active}"
        )
        if not bar_active and completed_keys == len(expected):
            state["agreement"] = _final_agreement_report(
                strict_database,
                endpoint_settings=endpoint_settings,
                resolutions=resolutions,
                objective_tolerance=agreement_objective_tolerance,
                control_tolerance=agreement_control_tolerance,
            )
            _save_state(state_path, state)
            _log("all endpoint maxima have completed strict refinement; monitor halted")
            return 0
        time.sleep(poll_seconds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--bar-database", type=Path, required=True)
    parser.add_argument("--strict-database", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--bar-pid", type=int, required=True)
    parser.add_argument("--endpoint-settings", required=True)
    parser.add_argument("--resolutions", required=True)
    parser.add_argument("--exploration-initialisations", type=int, default=1_000)
    parser.add_argument("--loose-count", type=int, default=20)
    parser.add_argument("--parameter-label-suffix", default="_bar_v2")
    parser.add_argument("--strict-max-elapsed-seconds", type=float, default=14_400)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--objective-epsilon", type=float, default=1e-9)
    parser.add_argument("--partition", default="zen5,epyc")
    parser.add_argument("--slurm-time", default="04:15:00")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--memory", default="4G")
    parser.add_argument("--agreement-objective-tolerance", type=float, default=0.01)
    parser.add_argument("--agreement-control-tolerance", type=float, default=0.01)
    arguments = parser.parse_args(argv)
    project_root = arguments.project_root.expanduser().resolve()

    def resolved(path: Path) -> Path:
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    raw_settings = json.loads(arguments.endpoint_settings)
    endpoint_settings = {int(cap): endpoints for cap, endpoints in raw_settings.items()}
    resolutions = tuple(int(value) for value in json.loads(arguments.resolutions))
    return monitor(
        project_root=project_root,
        python_executable=arguments.python.expanduser().resolve(),
        bar_database=resolved(arguments.bar_database),
        strict_database=resolved(arguments.strict_database),
        state_path=resolved(arguments.state_path),
        bar_process_id=arguments.bar_pid,
        endpoint_settings=endpoint_settings,
        resolutions=resolutions,
        exploration_initialisations=arguments.exploration_initialisations,
        loose_count=arguments.loose_count,
        parameter_label_suffix=arguments.parameter_label_suffix,
        strict_max_elapsed_seconds=arguments.strict_max_elapsed_seconds,
        poll_seconds=arguments.poll_seconds,
        objective_epsilon=arguments.objective_epsilon,
        partition=arguments.partition,
        slurm_time=arguments.slurm_time,
        cpus=arguments.cpus,
        memory=arguments.memory,
        agreement_objective_tolerance=arguments.agreement_objective_tolerance,
        agreement_control_tolerance=arguments.agreement_control_tolerance,
    )


if __name__ == "__main__":
    raise SystemExit(main())
