from pathlib import Path

import pytest

from ofc.auto_initialization import (
    IntensityPrior,
    resolve_auto_intensity_center,
    write_prior_snapshot,
)
from ofc.config import make_document
from ofc.results import Results
from ofc.runner import run_config
from ofc.storage import ResultStore


def _prior(key: str, *, cap: float, objective: float, mean_u: float):
    return IntensityPrior(
        source_key=key,
        source_database="source.sqlite3",
        source_run_id=int(key.removeprefix("run-")),
        t_interval=4.0,
        r_bg=-0.008716,
        u_max=cap,
        best_objective=objective,
        mean_u=mean_u,
    )


def test_auto_center_combines_unique_global_and_exact_cap_solutions(tmp_path):
    output = tmp_path / "output.sqlite3"
    ResultStore(output)
    snapshot = write_prior_snapshot(
        tmp_path / "priors.sqlite3",
        (
            _prior("run-1", cap=40.0, objective=3.0, mean_u=10.0),
            _prior("run-2", cap=160.0, objective=2.0, mean_u=80.0),
            _prior("run-3", cap=160.0, objective=1.0, mean_u=40.0),
        ),
    )

    center = resolve_auto_intensity_center(
        output_database=output,
        prior_database=snapshot,
        t_interval=4.0,
        r_bg=-0.008716,
        u_max=160.0,
    )

    # The two exact-cap rows also occur in the global top ten, but each actual
    # solution is counted only once. Cross-cap means are normalized by their
    # own caps: 160 * (0.3 + 10/40 + 80/160 + 40/160) / 4.
    assert center.bounded_center == pytest.approx(52.0)
    assert center.fraction == pytest.approx(0.325)
    assert center.source_count == 3
    assert center.global_source_count == 3
    assert center.exact_cap_source_count == 2


def test_auto_center_falls_back_to_thirty_percent_without_data(tmp_path):
    output = tmp_path / "empty.sqlite3"
    ResultStore(output)

    center = resolve_auto_intensity_center(
        output_database=output,
        prior_database=None,
        t_interval=4.0,
        r_bg=-0.008716,
        u_max=2560.0,
    )

    assert center.bounded_center == pytest.approx(768.0)
    assert center.fraction == pytest.approx(0.3)
    assert center.source_count == 0


def test_runner_persists_resolved_auto_center_provenance(tmp_path):
    output = tmp_path / "run.sqlite3"
    snapshot = write_prior_snapshot(
        tmp_path / "priors.sqlite3",
        (_prior("run-7", cap=10.0, objective=1.0, mean_u=4.0),),
    )
    document = make_document(
        name="auto_center",
        parameters={
            "N": 4,
            "t_interval": 4.0,
            "r_bg": -0.008716,
            "u_max": 10.0,
            "v_max": 10.0,
            "schedule": ((1, 1.0),),
            "block_size": 1,
            "J_tol": None,
            "u_tol": None,
            "v_tol": None,
            "projected_gradient_tol": None,
        },
        runtime={
            "initialisations": 1,
            "fourier_intensity_fraction": "auto",
            "fourier_intensity_auto_database": str(snapshot),
            "fourier_rms_amplitude": 0.1,
            "device": "cpu",
            "concurrent_workers": 1,
            "use_jit": False,
            "database": str(output),
        },
    )

    run_config(document, queue_id=123)
    row = Results(output).search(config_id=document.config_id, limit=1)[0]

    assert row["fourier_u_center_mode"] == "auto"
    assert row["fourier_u_center"] == pytest.approx((3.0 + 4.0) / 2.0)
    assert row["fourier_u_center_fraction"] == pytest.approx(0.35)
    assert row["fourier_u_center_source_count"] == 1
    assert row["fourier_u_center_source_keys"] == ["run-7"]
