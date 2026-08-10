from types import SimpleNamespace

from ofc.config import load_config, make_document
from ofc.results import Results
from ofc.runner import run_config
from ofc.seed_sensitivity import fixed_seed_stage_name
from ofc.strict_refinement_monitor import _candidate, _pair_agreement, _submit_strict


SETTINGS = {
    "adam_learning_rate": 0.05,
    "adam_beta1": 0.95,
    "adam_beta2": 0.99,
    "smoothness": 1e-7,
    "sharpness": 1e-8,
}


def test_candidate_uses_only_matching_exploration_or_loose_objective(tmp_path):
    database = tmp_path / "bar.sqlite3"
    name = fixed_seed_stage_name(
        4,
        10,
        "exploration",
        3,
        parameter_label="low_regularization_bar_v2",
    )
    document = make_document(
        name=name,
        parameters={
            "N": 4,
            "r_bg": 0.75,
            "u_max": 10,
            "v_max": 20,
            "smoothness": SETTINGS["smoothness"],
            "sharpness": SETTINGS["sharpness"],
            "schedule": [(1, 1.0)],
            "block_size": 1,
        },
        runtime={
            "initialisations": 1,
            "use_jit": False,
            "device": "cpu",
            "database": str(database),
        },
    )
    run_config(document, queue_id=101)

    selected = _candidate(
        Results(database),
        N=4,
        cap=10,
        endpoint="low",
        settings=SETTINGS,
        exploration_initialisations=3,
    )

    assert selected is not None
    assert selected["config_name"] == name
    assert selected["best_objective"] is not None
    assert _pair_agreement(Results(database), selected, selected) == {
        "objective_relative_difference": 0.0,
        "normalized_control_rms_difference": 0.0,
    }


def test_submit_strict_writes_cross_database_config_and_records_job(tmp_path, monkeypatch):
    project_root = tmp_path
    (project_root / "run_config").mkdir()
    (project_root / "logs").mkdir()
    (project_root / "slurm").mkdir()
    (project_root / "slurm" / "run_config.slurm").touch()
    calls = []

    def fake_run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(stdout="4567\n", stderr="", returncode=0)

    monkeypatch.setattr(
        "ofc.strict_refinement_monitor.subprocess.run", fake_run
    )
    state = {}
    _submit_strict(
        project_root=project_root,
        python_executable=project_root / "python",
        bar_database=project_root / "bar.sqlite3",
        strict_database=project_root / "strict.sqlite3",
        state_entry=state,
        key="N100_u320_low",
        N=100,
        cap=320,
        endpoint="low",
        settings=SETTINGS,
        candidate={"run_id": 123, "best_objective": 99.5},
        strict_max_elapsed_seconds=14_400,
        partition="zen5,epyc",
        slurm_time="04:15:00",
        cpus=2,
        memory="4G",
    )

    assert state["active_job_id"] == 4567
    assert state["last_source_run_id"] == 123
    assert state["last_source_objective"] == 99.5
    config = load_config(
        project_root / "run_config" / "N100_u320_low_strict_revision1_slurm_cpu.yaml"
    )
    assert config.runtime.database == str(project_root / "strict.sqlite3")
    assert config.query.database == str(project_root / "bar.sqlite3")
    assert config.query.where["run_id"] == 123
    assert calls[0][0][0] == "sbatch"
    assert "--cpus-per-task=2" in calls[0][0]
