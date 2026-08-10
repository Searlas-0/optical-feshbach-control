from pathlib import Path

import pytest

from ofc.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cap", [40, 160, 1280])
@pytest.mark.parametrize("N", [200, 300, 400, 500])
def test_resolution_ladder_configs_are_isolated_gated_and_one_decade_apart(cap, N):
    loose = load_config(
        ROOT / "run_config" / f"N{N}_u{cap}_resolution_ladder_loose_cpu.yaml"
    )
    strict = load_config(
        ROOT / "run_config" / f"N{N}_u{cap}_resolution_ladder_strict_cpu.yaml"
    )

    assert loose.scalar_cases()[0].N == strict.scalar_cases()[0].N == N
    assert loose.scalar_cases()[0].u_max == strict.scalar_cases()[0].u_max == cap
    assert loose.runtime.database == strict.runtime.database
    assert loose.runtime.database.endswith(
        f"N100_u{cap}_top_peak_refinement_strict.sqlite3"
    )
    assert loose.runtime.device == strict.runtime.device == "cpu"
    assert loose.runtime.initialisations == 10
    assert strict.runtime.initialisations == 0
    assert loose.runtime.max_elapsed_seconds == 4 * 60 * 60
    assert strict.runtime.max_elapsed_seconds == 4 * 60 * 60
    assert loose.runtime.repeat_schedule_until_stable is True
    assert strict.runtime.repeat_schedule_until_stable is True

    loose_case = loose.scalar_cases()[0]
    strict_case = strict.scalar_cases()[0]
    assert loose_case.J_tol == pytest.approx(1e-5)
    assert loose_case.J_tol == pytest.approx(10 * strict_case.J_tol)
    assert loose_case.u_tol == pytest.approx(1e-4)
    assert loose_case.u_tol == pytest.approx(10 * strict_case.u_tol)
    assert loose_case.v_tol == pytest.approx(1e-4)
    assert loose_case.v_tol == pytest.approx(10 * strict_case.v_tol)
    assert loose_case.projected_gradient_tol == pytest.approx(1e-4)
    assert loose_case.projected_gradient_tol == pytest.approx(
        10 * strict_case.projected_gradient_tol
    )
    assert loose_case.grid_refinement_tol == pytest.approx(1e-2)
    assert strict_case.grid_refinement_tol == pytest.approx(1e-3)
    assert loose.query.limit == 5
    assert strict.query.limit == 1
    assert loose.query.match_parameters == strict.query.match_parameters == ("u_max",)
    assert loose.query.where["termination_reason"] == ["stability", "time_limit"]
    assert strict.query.where["termination_reason"] == ["stability", "time_limit"]
    assert loose.query.resume_optimizer is strict.query.resume_optimizer is False
    assert loose.query.perturbed is strict.query.perturbed is False


def test_resolution_ladder_slurm_driver_is_serial_and_error_resilient():
    source = (ROOT / "slurm" / "run_resolution_ladder.slurm").read_text()

    assert "for resolution in 200 300 400 500" in source
    assert "5 promoted + 10 random" in source
    assert source.count("srun --cpu-bind=cores") == 2
    assert "failures=$((failures + 1))" in source
    assert "continuing safely" in source
