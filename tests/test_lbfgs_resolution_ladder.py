from pathlib import Path

import pytest

from ofc.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cap", [40, 160])
def test_n100_lbfgs_handoff_uses_cancelled_saved_peak_checkpoints(cap):
    document = load_config(
        ROOT / "run_config" / f"N100_u{cap}_lbfgs_handoff_cpu.yaml"
    )
    case = document.scalar_cases()[0]

    assert case.optimizer == "lbfgs"
    assert case.N == 100
    assert case.lbfgs_history_size == 100
    assert case.lbfgs_max_linesearch_steps == 200
    assert case.lbfgs_tolerance == pytest.approx(1e-12)
    assert case.J_tol == pytest.approx(1e-6)
    assert case.u_tol == case.v_tol == pytest.approx(1e-5)
    assert case.projected_gradient_tol == pytest.approx(1e-5)
    assert document.runtime.initialisations == 0
    assert document.runtime.max_elapsed_seconds == 4 * 60 * 60
    assert document.query.limit == 5
    assert document.query.where["status"] == ["running", "complete", "failed"]
    assert document.query.resume_optimizer is False


@pytest.mark.parametrize("cap", [40, 160])
@pytest.mark.parametrize("N", [200, 300, 400, 500])
def test_lbfgs_resolution_stages_promote_saved_results_and_repeat_strict_tolerances(cap, N):
    loose = load_config(
        ROOT / "run_config" / f"N{N}_u{cap}_lbfgs_ladder_loose_cpu.yaml"
    )
    strict = load_config(
        ROOT / "run_config" / f"N{N}_u{cap}_lbfgs_ladder_strict_cpu.yaml"
    )
    loose_case = loose.scalar_cases()[0]
    strict_case = strict.scalar_cases()[0]

    assert loose_case.optimizer == strict_case.optimizer == "lbfgs"
    assert loose_case.N == strict_case.N == N
    assert loose.runtime.initialisations == 10
    assert strict.runtime.initialisations == 0
    assert loose.query.limit == 5
    assert strict.query.limit == 1
    assert loose.query.where["termination_reason"] == ["stability", "time_limit"]
    assert strict.query.where["termination_reason"] == ["stability", "time_limit"]
    assert loose_case.J_tol == pytest.approx(10 * strict_case.J_tol)
    assert loose_case.u_tol == pytest.approx(10 * strict_case.u_tol)
    assert loose_case.v_tol == pytest.approx(10 * strict_case.v_tol)
    assert loose_case.projected_gradient_tol == pytest.approx(
        10 * strict_case.projected_gradient_tol
    )
    assert strict_case.J_tol == pytest.approx(1e-6)
    assert strict_case.u_tol == strict_case.v_tol == pytest.approx(1e-5)
    assert strict_case.projected_gradient_tol == pytest.approx(1e-5)
    assert loose_case.grid_refinement_tol == pytest.approx(1e-2)
    assert strict_case.grid_refinement_tol == pytest.approx(1e-3)


def test_lbfgs_slurm_driver_runs_handoff_then_every_resolution_resiliently():
    source = (ROOT / "slurm" / "run_lbfgs_resolution_ladder.slurm").read_text()

    assert "N100_u${CAP}_lbfgs_handoff_cpu.yaml" in source
    assert "for resolution in 200 300 400 500" in source
    assert source.count("srun --cpu-bind=cores") == 3
    assert "continuing safely" in source
