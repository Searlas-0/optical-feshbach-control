"""Reusable orchestration for the repository's argument-only notebooks.

This is deliberately an adapter: configuration, execution, result retrieval,
and plotting remain isolated in their owning modules.  Notebooks only declare
arguments and call this API; they should not grow experiment-specific helper
functions or repeat subprocess/database/figure boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .config import (
    DEFAULT_RESULTS_DATABASE,
    ConfigDocument,
    load_config,
    make_document,
    random_id,
    write_config,
)
from .plotting import (
    FIGURE_DPI,
    PNG_DPI,
    INITIALIZATION_PARAMETER,
    _best_sweep_runs,
    _detect_sweep,
    _make_sweep_spec,
    plot_convergence,
    plot_controls,
    plot_double_sweep_summary,
    plot_single_sweep_summary,
    plot_standard_summary,
    plot_triple_sweep_summary,
    plot_yield_distribution,
    varying_sweep_parameters,
)
from .results import Results


def find_project_root(start: str | Path | None = None) -> Path:
    """Find this checkout from the repository root or any child directory."""

    current = Path.cwd() if start is None else Path(start).expanduser()
    current = current.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "ofc").is_dir() and (candidate / "run.py").is_file():
            return candidate
    raise FileNotFoundError("Could not find the optical-feshbach-control project root.")


def _display(value) -> None:
    try:
        from IPython.display import display
    except ImportError:
        print(value)
    else:
        display(value)


def _normalise_statuses(statuses) -> tuple[str, ...] | None:
    if statuses is None:
        return None
    values = (statuses,) if isinstance(statuses, str) else tuple(statuses)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("statuses must be None, one status, or a non-empty status sequence.")
    return values


@dataclass
class NotebookQuery:
    """One persisted config execution plus any requested sweep dimensions."""

    workflow: "RunNotebook"
    results: Results
    config_document: ConfigDocument | Mapping[str, Any] | None
    execution: Mapping[str, Any]
    rows: list[dict[str, Any]]
    registered_rows: list[dict[str, Any]]
    sweep_parameters: tuple[str, ...]
    _sweeps: dict[str, Any] = field(default_factory=dict)

    @property
    def config_parameters(self) -> Mapping[str, Any]:
        """Parameters recorded by the selected config, including its sweeps."""

        if isinstance(self.config_document, ConfigDocument):
            return self.config_document.parameters
        if isinstance(self.config_document, Mapping):
            parameters = self.config_document.get("parameters", {})
            return parameters if isinstance(parameters, Mapping) else {}
        return {}

    def sweep(self, name: str | None = None):
        """Return a named sweep, or the sole/inferred sweep when unambiguous."""

        if name is None:
            if len(self.sweep_parameters) == 1:
                name = self.sweep_parameters[0]
            elif not self.sweep_parameters:
                return _detect_sweep(self.rows)
            else:
                raise ValueError(
                    "Choose a sweep_parameter from " + ", ".join(self.sweep_parameters)
                )
        name = str(name)
        if name not in self._sweeps:
            # Explicit notebook plot dimensions may intentionally have one value
            # (for example a constant cap used as the row in a cap-specific
            # triple-summary cell). The plotting layer handles singleton axes.
            self._sweeps[name] = _make_sweep_spec(
                self.rows, name, allow_single=True
            )
        return self._sweeps[name]

    def best_runs(self, sweep_parameter: str | None = None) -> list[Mapping[str, Any]]:
        sweep = self.sweep(sweep_parameter)
        return [
            self.results.get(row["run_id"])
            for row in _best_sweep_runs(self.rows, sweep)
        ]

    def plot_convergence(self, *, sweep_parameter: str | None = None, **options):
        sweep = self.sweep(sweep_parameter)
        return plot_convergence(self.best_runs(sweep.name), sweep=sweep, **options)

    def plot_distribution(self, *, sweep_parameter: str | None = None, **options):
        sweep = self.sweep(sweep_parameter)
        return plot_yield_distribution(self.rows, sweep=sweep, **options)

    def plot_controls(self, *, sweep_parameter: str | None = None, **options):
        sweep = self.sweep(sweep_parameter)
        return plot_controls(self.best_runs(sweep.name), sweep=sweep, **options)

    def plot_summary(
        self,
        *,
        sweep_parameter: str | None = None,
        history_points: int = 1200,
    ):
        """Plot the queried rows as one unified score/strip/control summary."""

        sweep = self.sweep(sweep_parameter)
        return plot_standard_summary(
            self.rows,
            sweep=sweep,
            history_points=history_points,
            **self._summary_loaders,
        )

    @property
    def _summary_loaders(self):
        return {
            "load_history": self.results.history,
            "load_tolerances": self.results.tolerances,
            "load_controls": lambda run_id: self.results.controls(run_id, "best"),
        }

    def plot_single_sweep_summary(
        self,
        *,
        sweep_parameter: str,
        history_points: int = 1200,
    ):
        return plot_single_sweep_summary(
            self.rows,
            self.sweep(sweep_parameter),
            history_points=history_points,
            **self._summary_loaders,
        )

    def plot_double_sweep_summary(
        self,
        *,
        separate_sweep_parameter: str,
        colour_sweep_parameter: str,
        history_points: int = 1200,
    ):
        return plot_double_sweep_summary(
            self.rows,
            separate_sweep=self.sweep(separate_sweep_parameter),
            colour_sweep=self.sweep(colour_sweep_parameter),
            history_points=history_points,
            **self._summary_loaders,
        )

    def plot_triple_sweep_summary(
        self,
        *,
        row_sweep_parameter: str,
        column_sweep_parameter: str,
        colour_sweep_parameter: str,
        history_points: int = 1200,
    ):
        return plot_triple_sweep_summary(
            self.rows,
            row_sweep=self.sweep(row_sweep_parameter),
            column_sweep=self.sweep(column_sweep_parameter),
            colour_sweep=self.sweep(colour_sweep_parameter),
            history_points=history_points,
            **self._summary_loaders,
        )


class RunNotebook:
    """Generic state and actions shared by every optimization run notebook."""

    def __init__(self, run_name: str, *, project_root: str | Path | None = None):
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("run_name must be a non-empty filename-safe name.")
        self.project_root = find_project_root(project_root)
        self.run_name = run_name
        self.config_path = (
            self.project_root / "run_config" / f"{self.run_name}.yaml"
        ).resolve()
        self.active_queue_id: int | None = None
        self.local_process_id: int | None = None
        self.strict_monitor_process_id: int | None = None
        self.local_log_path: Path | None = None
        self.config_document: ConfigDocument | None = None
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/ofc-matplotlib")

    def show_context(self) -> None:
        _display(
            {
                "project_root": str(self.project_root),
                "run_name": self.run_name,
                "config_path": str(self.config_path.relative_to(self.project_root)),
                "config_exists": self.config_path.is_file(),
            }
        )

    def load_config(self, *, required: bool = True) -> ConfigDocument | None:
        if not self.config_path.is_file():
            if required:
                raise FileNotFoundError(
                    f"No existing config at {self.config_path}. "
                    "Check run_name or activate the config cell once."
                )
            return None
        self.config_document = load_config(self.config_path)
        return self.config_document

    def create_config(
        self,
        *,
        activated: bool,
        description: str,
        parameters: Mapping[str, Any],
        runtime: Mapping[str, Any],
        initialization_query: Mapping[str, Any] | None = None,
        reuse_existing: bool = False,
    ) -> ConfigDocument | None:
        if not isinstance(activated, bool):
            raise TypeError("Activated must be a bool.")
        if not activated:
            print("Config creation is disabled (Activated=False).")
            return self.load_config(required=False)
        if self.config_path.exists():
            if not reuse_existing:
                raise FileExistsError(
                    f"{self.config_path} already exists. Keep Activated=False, choose a "
                    "new run_name, or set reuse_existing=True without changing arguments."
                )
            document = self.load_config()
            print(f"Reusing immutable config: {self.config_path.relative_to(self.project_root)}")
            return document
        document = make_document(
            name=self.run_name,
            description=description,
            parameters=parameters,
            runtime=runtime,
            query=initialization_query,
        )
        write_config(document, self.config_path)
        self.config_document = load_config(self.config_path)
        print(f"Created config: {self.config_path.relative_to(self.project_root)}")
        return self.config_document

    def create_config_group(
        self,
        *,
        activated: bool,
        documents: Sequence[ConfigDocument],
        reuse_existing: bool = False,
    ) -> tuple["RunNotebook", ...]:
        """Create or load an ordered group of immutable dependent configs."""

        if not isinstance(activated, bool):
            raise TypeError("Activated must be a bool.")
        if not isinstance(reuse_existing, bool):
            raise TypeError("reuse_existing must be a bool.")
        selected_documents = tuple(documents)
        if not selected_documents:
            raise ValueError("documents cannot be empty.")
        if any(not isinstance(document, ConfigDocument) for document in selected_documents):
            raise TypeError("documents must contain ConfigDocument instances.")
        names = tuple(document.name for document in selected_documents)
        if len(set(names)) != len(names):
            raise ValueError("Grouped config document names must be unique.")
        workflows = tuple(
            RunNotebook(name, project_root=self.project_root) for name in names
        )
        existing = tuple(workflow.config_path.is_file() for workflow in workflows)
        if not activated:
            count = sum(existing)
            print(
                "Grouped config creation is disabled (Activated=False); "
                f"found {count}/{len(workflows)} existing configs."
            )
            return workflows
        if any(existing):
            if not all(existing):
                raise FileExistsError(
                    "Only part of the grouped config set exists. Choose a new "
                    "experiment name or complete/remove the partial set explicitly."
                )
            if not reuse_existing:
                raise FileExistsError(
                    "Every grouped config already exists. Keep Activated=False or "
                    "set reuse_existing=True without changing the arguments."
                )
            for workflow in workflows:
                workflow.load_config()
            print(f"Reusing {len(workflows)} immutable grouped configs.")
            return workflows
        for workflow, document in zip(workflows, selected_documents):
            write_config(document, workflow.config_path)
            workflow.config_document = load_config(workflow.config_path)
        print(f"Created {len(workflows)} grouped configs in run_config/.")
        return workflows

    def run_local(
        self,
        *,
        activated: bool,
        queue_id: int | None = None,
        python_executable: str | Path | None = None,
        extra_arguments: Sequence[str] = (),
        detached: bool = False,
        log_path: str | Path | None = None,
        environment_overrides: Mapping[str, str] | None = None,
    ) -> int | None:
        if not isinstance(activated, bool):
            raise TypeError("Activated must be a bool.")
        if not activated:
            print("Local execution is disabled (Activated=False).")
            return None
        if not isinstance(detached, bool):
            raise TypeError("detached must be a bool.")
        self.load_config()
        selected_queue_id = random_id() if queue_id is None else int(queue_id)
        command = [
            str(python_executable or sys.executable),
            str(self.project_root / "run.py"),
            "--queue-id",
            str(selected_queue_id),
            *map(str, extra_arguments),
            str(self.config_path),
        ]
        print(shlex.join(command))
        process_environment = None
        if environment_overrides is not None:
            process_environment = {
                **os.environ,
                **{str(name): str(value) for name, value in environment_overrides.items()},
            }
        process_id = None
        selected_log_path = None
        if detached:
            selected_log_path = Path(
                log_path
                or self.project_root
                / "logs"
                / f"{self.run_name}-local-{selected_queue_id}.log"
            ).expanduser()
            if not selected_log_path.is_absolute():
                selected_log_path = (self.project_root / selected_log_path).resolve()
            selected_log_path.parent.mkdir(parents=True, exist_ok=True)
            with selected_log_path.open("ab") as output:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=process_environment,
                )
            process_id = process.pid
        else:
            subprocess.run(
                command,
                cwd=self.project_root,
                check=True,
                env=process_environment,
            )
        self.active_queue_id = selected_queue_id
        self.local_process_id = process_id
        self.local_log_path = selected_log_path
        if detached:
            print(
                f"Detached local execution started; queue_id={selected_queue_id}; "
                f"pid={process_id}"
            )
            print(f"Log: {selected_log_path}")
        else:
            print(f"Local execution complete; queue_id={selected_queue_id}")
        return selected_queue_id

    def run_on_bar_gpu(
        self,
        *,
        activated: bool,
        queue_id: int | None = None,
        python_executable: str | Path | None = None,
        extra_arguments: Sequence[str] = (),
        detached: bool = True,
        log_path: str | Path | None = None,
    ) -> int | None:
        """Launch a config on bar's GPU, detached from the notebook session."""

        if not isinstance(activated, bool):
            raise TypeError("Activated must be a bool.")
        if not activated:
            print("Bar GPU execution is disabled (Activated=False).")
            return None
        if os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Bar GPU execution must run outside Slurm.")
        host = socket.gethostname().split(".", 1)[0]
        if host != "bar":
            raise RuntimeError(f"Bar GPU execution requires host 'bar', found {host!r}.")
        document = self.load_config()
        if document.runtime.device == "cpu":
            raise RuntimeError(
                "The immutable config requests device: cpu. Create a new config with "
                "device: gpu or device: auto before using the bar GPU run cell."
            )
        selected_python = str(python_executable or sys.executable)
        gpu_environment = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        probe = subprocess.run(
            [
                selected_python,
                "-c",
                "import jax; devices=jax.devices('gpu'); "
                "assert devices, 'No JAX GPU is visible'; print(devices[0])",
            ],
            cwd=self.project_root,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **gpu_environment},
        )
        print(f"Verified bar GPU: {probe.stdout.strip()}")
        return self.run_local(
            activated=True,
            queue_id=queue_id,
            python_executable=selected_python,
            extra_arguments=extra_arguments,
            detached=detached,
            log_path=log_path,
            environment_overrides=gpu_environment,
        )

    def run_on_bar_gpu_group(
        self,
        *,
        activated: bool,
        additional_workflows: Sequence["RunNotebook"] = (),
        queue_id: int | None = None,
        python_executable: str | Path | None = None,
        extra_arguments: Sequence[str] = (),
        detached: bool = True,
        log_path: str | Path | None = None,
        wait_for_process_id: int | None = None,
    ) -> int | None:
        """Verify bar's GPU, then run several explicit-GPU configs sequentially."""

        if not isinstance(activated, bool):
            raise TypeError("Activated must be a bool.")
        if not activated:
            print("Grouped bar GPU execution is disabled (Activated=False).")
            return None
        if os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Bar GPU execution must run outside Slurm.")
        host = socket.gethostname().split(".", 1)[0]
        if host != "bar":
            raise RuntimeError(f"Bar GPU execution requires host 'bar', found {host!r}.")
        workflows = (self, *tuple(additional_workflows))
        documents = [workflow.load_config() for workflow in workflows]
        non_gpu = [
            document.config_file
            for document in documents
            if document.runtime.device != "gpu"
        ]
        if non_gpu:
            raise RuntimeError(
                "Grouped bar execution requires every config to set device: gpu "
                f"explicitly; offending configs: {non_gpu}."
            )
        selected_python = str(python_executable or sys.executable)
        gpu_environment = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        if wait_for_process_id is None:
            probe = subprocess.run(
                [
                    selected_python,
                    "-c",
                    "import jax; devices=jax.devices('gpu'); "
                    "assert devices, 'No JAX GPU is visible'; print(devices[0])",
                ],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, **gpu_environment},
            )
            print(f"Verified bar GPU: {probe.stdout.strip()}")
        else:
            print(
                f"GPU verification deferred until process {wait_for_process_id} "
                "releases the card; queued configs explicitly require device: gpu."
            )
        return self.run_local_group(
            activated=True,
            additional_workflows=additional_workflows,
            queue_id=queue_id,
            python_executable=selected_python,
            extra_arguments=extra_arguments,
            detached=detached,
            log_path=log_path,
            wait_for_process_id=wait_for_process_id,
            environment_overrides=gpu_environment,
        )

    def run_local_group(
        self,
        *,
        activated: bool,
        additional_workflows: Sequence["RunNotebook"] = (),
        queue_id: int | None = None,
        python_executable: str | Path | None = None,
        extra_arguments: Sequence[str] = (),
        detached: bool = False,
        log_path: str | Path | None = None,
        wait_for_process_id: int | None = None,
        environment_overrides: Mapping[str, str] | None = None,
    ) -> int | None:
        """Run several configs sequentially in one local accelerator process."""

        if not isinstance(activated, bool):
            raise TypeError("Activated must be a bool.")
        if not activated:
            print("Grouped local execution is disabled (Activated=False).")
            return None
        workflows = (self, *tuple(additional_workflows))
        if any(not isinstance(workflow, RunNotebook) for workflow in workflows):
            raise TypeError("additional_workflows must contain RunNotebook instances.")
        if any(workflow.project_root != self.project_root for workflow in workflows):
            raise ValueError("Grouped workflows must belong to the same project root.")
        if not isinstance(detached, bool):
            raise TypeError("detached must be a bool.")
        config_paths = [workflow.load_config().config_file for workflow in workflows]
        if len(set(config_paths)) != len(config_paths):
            raise ValueError("Grouped workflows must identify distinct config files.")
        selected_queue_id = random_id() if queue_id is None else int(queue_id)
        run_command = [
            str(python_executable or sys.executable),
            str(self.project_root / "run.py"),
            "--queue-id",
            str(selected_queue_id),
            *map(str, extra_arguments),
            *(str(workflow.config_path) for workflow in workflows),
        ]
        if wait_for_process_id is not None:
            if (
                isinstance(wait_for_process_id, bool)
                or not isinstance(wait_for_process_id, int)
                or wait_for_process_id < 1
            ):
                raise ValueError("wait_for_process_id must be a positive integer or None.")
            command = [
                str(python_executable or sys.executable),
                "-m",
                "ofc.process_queue",
                "--wait-pid",
                str(wait_for_process_id),
                "--",
                *run_command,
            ]
        else:
            command = run_command
        print(shlex.join(command))
        process_environment = None
        if environment_overrides is not None:
            process_environment = {
                **os.environ,
                **{str(name): str(value) for name, value in environment_overrides.items()},
            }
        process_id = None
        selected_log_path = None
        if detached:
            selected_log_path = Path(
                log_path
                or self.project_root
                / "logs"
                / f"{self.run_name}-local-{selected_queue_id}.log"
            ).expanduser()
            if not selected_log_path.is_absolute():
                selected_log_path = (self.project_root / selected_log_path).resolve()
            selected_log_path.parent.mkdir(parents=True, exist_ok=True)
            with selected_log_path.open("ab") as output:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=process_environment,
                )
            process_id = process.pid
        else:
            subprocess.run(
                command,
                cwd=self.project_root,
                check=True,
                env=process_environment,
            )
        for workflow in workflows:
            workflow.active_queue_id = selected_queue_id
            workflow.local_process_id = process_id
            workflow.local_log_path = selected_log_path
        if detached:
            print(
                f"Detached grouped local execution "
                f"{'queued' if wait_for_process_id is not None else 'started'}; "
                f"queue_id={selected_queue_id}; "
                f"pid={process_id}; configs={len(workflows)}"
            )
            print(f"Log: {selected_log_path}")
        else:
            print(
                f"Grouped local execution complete; queue_id={selected_queue_id}; "
                f"configs={len(workflows)}"
            )
        return selected_queue_id

    def launch_strict_refinement_monitor(
        self,
        *,
        activated: bool,
        bar_process_id: int | None,
        endpoint_settings: Mapping[int, Mapping[str, Mapping[str, Any]]],
        resolutions: Sequence[int],
        bar_database: str | Path,
        strict_database: str | Path,
        state_path: str | Path,
        python_executable: str | Path | None = None,
        exploration_initialisations: int = 1_000,
        loose_count: int = 20,
        parameter_label_suffix: str = "_bar_v2",
        strict_max_elapsed_seconds: float = 4 * 60 * 60,
        poll_seconds: float = 60.0,
        objective_epsilon: float = 1e-9,
        partition: str = "zen5,epyc",
        slurm_time: str = "04:15:00",
        cpus: int = 2,
        memory: str = "4G",
        agreement_objective_tolerance: float = 0.01,
        agreement_control_tolerance: float = 0.01,
        log_path: str | Path = "logs/endpoint_strict_refinement_monitor.log",
    ) -> int | None:
        """Launch the detached bar-to-Slurm strict-incumbent monitor."""

        if not isinstance(activated, bool):
            raise TypeError("Activated must be a bool.")
        if not activated:
            print("Strict refinement monitoring is disabled (Activated=False).")
            return None
        if bar_process_id is None or isinstance(bar_process_id, bool) or bar_process_id < 1:
            raise ValueError("bar_process_id must identify the detached bar pipeline.")
        if os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("The strict refinement monitor must launch outside Slurm.")
        host = socket.gethostname().split(".", 1)[0]
        if host != "bar":
            raise RuntimeError(f"Strict monitoring requires host 'bar', found {host!r}.")
        if not endpoint_settings or not resolutions:
            raise ValueError("endpoint_settings and resolutions cannot be empty.")
        selected_python = str(python_executable or sys.executable)
        command = [
            selected_python,
            "-m",
            "ofc.strict_refinement_monitor",
            "--project-root",
            str(self.project_root),
            "--python",
            selected_python,
            "--bar-database",
            str(bar_database),
            "--strict-database",
            str(strict_database),
            "--state-path",
            str(state_path),
            "--bar-pid",
            str(bar_process_id),
            "--endpoint-settings",
            json.dumps(endpoint_settings, sort_keys=True, separators=(",", ":")),
            "--resolutions",
            json.dumps(list(resolutions), separators=(",", ":")),
            "--exploration-initialisations",
            str(exploration_initialisations),
            "--loose-count",
            str(loose_count),
            "--parameter-label-suffix",
            parameter_label_suffix,
            "--strict-max-elapsed-seconds",
            str(strict_max_elapsed_seconds),
            "--poll-seconds",
            str(poll_seconds),
            "--objective-epsilon",
            str(objective_epsilon),
            "--partition",
            partition,
            "--slurm-time",
            slurm_time,
            "--cpus",
            str(cpus),
            "--memory",
            memory,
            "--agreement-objective-tolerance",
            str(agreement_objective_tolerance),
            "--agreement-control-tolerance",
            str(agreement_control_tolerance),
        ]
        selected_log_path = Path(log_path).expanduser()
        if not selected_log_path.is_absolute():
            selected_log_path = (self.project_root / selected_log_path).resolve()
        selected_log_path.parent.mkdir(parents=True, exist_ok=True)
        with selected_log_path.open("ab") as output:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.strict_monitor_process_id = process.pid
        print(
            f"Detached strict refinement monitor started; pid={process.pid}; "
            f"bar_pid={bar_process_id}"
        )
        print(f"Log: {selected_log_path}")
        return process.pid

    def submit_slurm(
        self,
        *,
        activated: bool,
        partition: str = "zen5,epyc",
        time: str = "4-03:00:00",
        cpus: int = 32,
        memory: str = "64G",
        array: bool | None = None,
        array_max_concurrent: int | None = None,
        job_name: str | None = None,
        extra_arguments: Sequence[str] = (),
    ) -> int | None:
        if not isinstance(activated, bool):
            raise TypeError("Activated must be a bool.")
        if not activated:
            print("Slurm submission is disabled (Activated=False).")
            return None
        document = self.load_config()
        batch_count = len(document.batches())
        use_array = batch_count > 1 if array is None else array
        if array_max_concurrent is not None and (
            isinstance(array_max_concurrent, bool) or array_max_concurrent < 1
        ):
            raise ValueError("array_max_concurrent must be a positive integer or None.")
        selected_job_name = (job_name or self.run_name)[:80]
        log_token = "%A_%a" if use_array else "%j"
        log_stem = f"logs/{self.run_name}-{log_token}"
        (self.project_root / "logs").mkdir(parents=True, exist_ok=True)
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={selected_job_name}",
            f"--partition={partition}",
            f"--time={time}",
            f"--cpus-per-task={int(cpus)}",
            f"--mem={memory}",
        ]
        if use_array:
            array_value = f"0-{batch_count - 1}"
            if array_max_concurrent is not None:
                array_value += f"%{int(array_max_concurrent)}"
            command.append(f"--array={array_value}")
        command.extend(
            [
                f"--output={log_stem}.out",
                f"--error={log_stem}.err",
                *map(str, extra_arguments),
                str(self.project_root / "slurm" / "run_config.slurm"),
                str(self.config_path),
            ]
        )
        print(shlex.join(command))
        submission = subprocess.run(
            command,
            cwd=self.project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        if submission.stderr.strip():
            print(submission.stderr.strip())
        job_token = submission.stdout.strip().splitlines()[-1].split(";", 1)[0]
        self.active_queue_id = int(job_token)
        kind = "array" if use_array else "job"
        print(f"Submitted Slurm {kind} {self.active_queue_id}")
        print(f"Monitor: squeue -j {self.active_queue_id}")
        print(f"Output: {log_stem.replace(log_token, str(self.active_queue_id))}.out")
        return self.active_queue_id

    def query(
        self,
        *,
        inherit_config: bool = True,
        database: str | Path | None = None,
        queue_id: int | None = None,
        config_run_rank: int = 1,
        statuses: str | Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        sweep_parameters: Sequence[str] | None = None,
        require_saved_stage: bool = True,
        limit: int | None = None,
        order_by: str = "run_id",
        descending: bool = False,
    ) -> NotebookQuery:
        """Select one persisted execution with or without a local config file.

        With ``inherit_config=True``, the immutable YAML identified by
        ``run_name`` supplies the database and config identity automatically.
        With it disabled, the query operates directly on the historical
        database and ``filters`` determine which execution is selected.
        """

        if not isinstance(inherit_config, bool):
            raise TypeError("inherit_config must be a bool.")
        document = self.load_config() if inherit_config else None
        if (
            isinstance(config_run_rank, bool)
            or not isinstance(config_run_rank, int)
            or config_run_rank < 1
        ):
            raise ValueError("config_run_rank must be a positive integer.")
        if not isinstance(require_saved_stage, bool):
            raise TypeError("require_saved_stage must be a bool.")
        selected_statuses = _normalise_statuses(statuses)
        query_filters = dict(filters or {})
        if "queue_id" in query_filters:
            raise ValueError("Pass queue_id as its named query argument, not in filters.")
        if document is not None and "config_id" in query_filters:
            requested_config_id = int(query_filters["config_id"])
            if requested_config_id != document.config_id:
                raise ValueError(
                    "filters['config_id'] conflicts with the inherited config. "
                    "Set inherit_config=False to query a different config."
                )

        selected_database = (
            database
            if database is not None
            else (
                document.runtime.database
                if document is not None
                else DEFAULT_RESULTS_DATABASE
            )
        )
        database_path = Path(selected_database).expanduser()
        if not database_path.is_absolute():
            database_path = (self.project_root / database_path).resolve()
        if not database_path.is_file():
            raise FileNotFoundError(f"No results database exists yet at {database_path}")
        results = Results(database_path)
        execution_filters = dict(query_filters)
        if document is not None:
            execution_filters["config_id"] = document.config_id
        executions = results.config_runs(**execution_filters)
        selected_queue_id = queue_id
        if selected_queue_id is None:
            selected_queue_id = self.active_queue_id
        if selected_queue_id is None:
            if config_run_rank > len(executions):
                source = (
                    f"inherited config {document.name!r}"
                    if document is not None
                    else "historical filters"
                )
                raise RuntimeError(
                    f"Only {len(executions)} execution(s) match the {source}."
                )
            execution = executions[config_run_rank - 1]
            selected_queue_id = int(execution["queue_id"])
        else:
            matches = [
                execution
                for execution in executions
                if int(execution["queue_id"]) == int(selected_queue_id)
            ]
            if not matches:
                raise RuntimeError(
                    f"queue_id={selected_queue_id} does not match the selected query source."
                )
            execution = matches[0]

        if selected_statuses is not None:
            query_filters["status"] = (
                selected_statuses[0] if len(selected_statuses) == 1 else list(selected_statuses)
            )
        registered_rows = results.search(
            config_document_id=execution["config_document_id"],
            queue_id=selected_queue_id,
            order_by=order_by,
            descending=descending,
            **query_filters,
        )
        rows = (
            [row for row in registered_rows if "best_score" in row]
            if require_saved_stage
            else list(registered_rows)
        )
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("limit must be a positive integer or None.")
            rows = rows[:limit]
        if not rows:
            suffix = " with a saved optimization stage" if require_saved_stage else ""
            raise ValueError(f"The query matched no runs{suffix}.")
        rows.sort(
            key=lambda row: (
                row.get("batch_index", 0),
                row.get("case_index", 0),
                row["run_id"],
            )
        )

        varied = varying_sweep_parameters(rows)
        selected_sweeps = tuple(varied if sweep_parameters is None else sweep_parameters)
        if len(set(selected_sweeps)) != len(selected_sweeps):
            raise ValueError("sweep_parameters cannot contain duplicates.")
        sweeps = {name: _make_sweep_spec(rows, name) for name in selected_sweeps}
        selected_document: ConfigDocument | Mapping[str, Any] | None = document
        if selected_document is None:
            try:
                selected_document = results.config_document(rows[0]["run_id"])
            except (AttributeError, KeyError):
                selected_document = None
        status_counts = {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        }
        output = NotebookQuery(
            workflow=self,
            results=results,
            config_document=selected_document,
            execution=execution,
            rows=rows,
            registered_rows=registered_rows,
            sweep_parameters=selected_sweeps,
            _sweeps=sweeps,
        )
        _display(
            {
                "config_name": execution["config_name"],
                "config_id": execution["config_id"],
                "queue_id": selected_queue_id,
                "execution_status": execution["status"],
                "registered_runs": len(registered_rows),
                "queried_runs": len(rows),
                "run_statuses": status_counts,
                "varying_parameters": varied or (INITIALIZATION_PARAMETER,),
                "selected_sweeps": selected_sweeps,
                "query_source": "local config" if inherit_config else "historical database",
                "database": (
                    str(database_path.relative_to(self.project_root))
                    if database_path.is_relative_to(self.project_root)
                    else str(database_path)
                ),
            }
        )
        return output

    def query_group(
        self,
        *,
        additional_workflows: Sequence["RunNotebook"] = (),
        queue_id: int | None = None,
        config_run_rank: int = 1,
        statuses: str | Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        sweep_parameters: Sequence[str] | None = None,
        require_saved_stage: bool = True,
        limit: int | None = None,
        order_by: str = "run_id",
        descending: bool = False,
        allow_missing: bool = True,
    ) -> NotebookQuery:
        """Combine available same-queue executions from several run configs."""

        workflows = (self, *tuple(additional_workflows))
        if any(not isinstance(workflow, RunNotebook) for workflow in workflows):
            raise TypeError("additional_workflows must contain RunNotebook instances.")
        if not isinstance(allow_missing, bool):
            raise TypeError("allow_missing must be a bool.")
        queries = []
        missing = []
        selected_queue_id = queue_id
        for workflow in workflows:
            try:
                query = workflow.query(
                    inherit_config=True,
                    queue_id=selected_queue_id,
                    config_run_rank=config_run_rank,
                    statuses=statuses,
                    filters=filters,
                    sweep_parameters=(),
                    require_saved_stage=require_saved_stage,
                    limit=limit,
                    order_by=order_by,
                    descending=descending,
                )
            except (RuntimeError, ValueError) as error:
                expected_missing = (
                    "Only 0 execution(s) match" in str(error)
                    or "does not match the selected query source" in str(error)
                    or "The query matched no runs" in str(error)
                )
                if not allow_missing or not expected_missing:
                    raise
                missing.append(workflow.run_name)
                continue
            queries.append(query)
            if selected_queue_id is None:
                selected_queue_id = int(query.execution["queue_id"])
        if not queries:
            raise RuntimeError("None of the grouped configs has queryable run data yet.")
        databases = {query.results.database for query in queries}
        if len(databases) != 1:
            raise ValueError("Grouped notebook queries must use one results database.")
        queue_ids = {int(query.execution["queue_id"]) for query in queries}
        if len(queue_ids) != 1:
            raise RuntimeError(
                "The selected config executions do not share one queue_id; pass "
                "queue_id explicitly or choose a different config_run_rank."
            )
        rows = [row for query in queries for row in query.rows]
        registered_rows = [
            row for query in queries for row in query.registered_rows
        ]
        rows.sort(
            key=lambda row: (
                row.get("config_name", ""),
                row.get("batch_index", 0),
                row.get("case_index", 0),
                row["run_id"],
            )
        )
        varied = varying_sweep_parameters(rows)
        selected_sweeps = tuple(
            varied if sweep_parameters is None else sweep_parameters
        )
        if len(set(selected_sweeps)) != len(selected_sweeps):
            raise ValueError("sweep_parameters cannot contain duplicates.")
        sweeps = {
            name: _make_sweep_spec(rows, name, allow_single=bool(missing))
            for name in selected_sweeps
        }
        output = NotebookQuery(
            workflow=self,
            results=queries[0].results,
            config_document={
                "parameters": {
                    "config_names": [query.execution["config_name"] for query in queries]
                }
            },
            execution={
                "queue_id": queries[0].execution["queue_id"],
                "config_names": tuple(
                    query.execution["config_name"] for query in queries
                ),
            },
            rows=rows,
            registered_rows=registered_rows,
            sweep_parameters=selected_sweeps,
            _sweeps=sweeps,
        )
        _display(
            {
                "config_names": output.execution["config_names"],
                "queue_id": output.execution["queue_id"],
                "registered_runs": len(registered_rows),
                "queried_runs": len(rows),
                "varying_parameters": varied or (INITIALIZATION_PARAMETER,),
                "selected_sweeps": selected_sweeps,
                "not_started_or_unsaved": tuple(missing),
                "database": str(queries[0].results.database),
            }
        )
        return output

    def present_figure(
        self,
        plot,
        filename: str,
        *,
        save_figure: str | Path | None = None,
        figure_format: str = "png",
        preview_dpi: int | float = FIGURE_DPI,
        save_dpi: int | float = PNG_DPI,
    ) -> None:
        """Display one plot and optionally save it under ``figures/``."""

        figure = plot[0] if isinstance(plot, tuple) else plot
        if figure_format not in {"png", "pdf"}:
            raise ValueError("figure_format must be 'png' or 'pdf'.")
        for name, value in (("preview_dpi", preview_dpi), ("save_dpi", save_dpi)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive number.")
        if save_figure is not None:
            figures_root = (self.project_root / "figures").resolve()
            relative_directory = str(save_figure).strip().lstrip("/\\") or "."
            output_directory = (figures_root / relative_directory).resolve()
            try:
                output_directory.relative_to(figures_root)
            except ValueError as error:
                raise ValueError("save_figure must name a location inside figures/.") from error
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / f"{filename}.{figure_format}"
            save_options = {"dpi": float(save_dpi)} if figure_format == "png" else {}
            figure.savefig(output_path, bbox_inches="tight", **save_options)
            print(f"Saved {output_path.relative_to(self.project_root)}")
        preview = io.BytesIO()
        figure.savefig(
            preview,
            format="png",
            dpi=float(preview_dpi),
            bbox_inches="tight",
        )
        try:
            from IPython.display import Image, display
        except ImportError:
            return
        display(Image(data=preview.getvalue()))
