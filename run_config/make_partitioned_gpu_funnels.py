#!/usr/bin/env python3
"""Generate the server and RTX 4060 progressive GPU experiment tracks.

Track A establishes strict, independently challenged solutions for the low and
high regularization endpoints of u_max=40, 320, and 1280 at N=100..500.  Track
B explores the three larger remaining caps on ``bar``.  A matching local track
offloads the three smaller caps to an 8 GB RTX 4060 Laptop GPU.  The generated
manifests are ordered by increasing computational commitment so useful coarse
data is persisted before the expensive convergence stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofc.config import ConfigDocument, load_config, make_document, write_config


TRACK_A_DATABASE = "results/bar_strict_three_cap_multigrid_v1.sqlite3"
TRACK_B_DATABASE = "results/bar_underexplored_progressive_v2.sqlite3"
LOCAL_DATABASE = "results/local_rtx4060_underexplored_v1.sqlite3"
TRACK_A_MANIFEST = ROOT / "run_config" / "partitioned_gpu_track_a.manifest"
TRACK_B_MANIFEST = ROOT / "run_config" / "partitioned_gpu_track_b.manifest"
LOCAL_MANIFEST = ROOT / "scripts" / "local" / "rtx4060.manifest"
TWO_HOURS = 2 * 60 * 60
FINAL_STRICT_HOURS = 12 * 60 * 60
RESOLUTIONS = (100, 200, 300, 400, 500)
SERVER_UNDEREXPLORED_CAPS = (2560, 640, 160)
LOCAL_UNDEREXPLORED_CAPS = (80, 20, 10)


def _settings(
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    smoothness: float,
    sharpness: float,
    intensity_fraction: float = 0.5,
) -> dict[str, float]:
    return {
        "adam_learning_rate": learning_rate,
        "adam_beta1": beta1,
        "adam_beta2": beta2,
        "smoothness": smoothness,
        "sharpness": sharpness,
        "fourier_intensity_fraction": intensity_fraction,
    }


STRICT_ENDPOINTS = {
    40: {
        "low": _settings(
            learning_rate=0.15,
            beta1=0.9,
            beta2=0.99,
            smoothness=3.952847075210474e-9,
            sharpness=1.5811388300841896e-9,
        ),
        "high": _settings(
            learning_rate=0.15,
            beta1=0.9,
            beta2=0.99,
            smoothness=3.9528470752104736e-6,
            sharpness=1.5811388300841896e-6,
        ),
    },
    320: {
        "low": _settings(
            learning_rate=0.05,
            beta1=0.95,
            beta2=0.99,
            smoothness=3.952847075210474e-9,
            sharpness=3.952847075210474e-10,
        ),
        "high": _settings(
            learning_rate=0.05,
            beta1=0.95,
            beta2=0.99,
            smoothness=3.9528470752104736e-6,
            sharpness=3.952847075210474e-7,
        ),
    },
    1280: {
        "low": _settings(
            learning_rate=0.15,
            beta1=0.95,
            beta2=0.999,
            smoothness=7.905694150420948e-9,
            sharpness=7.905694150420948e-10,
        ),
        "high": _settings(
            learning_rate=0.15,
            beta1=0.95,
            beta2=0.999,
            smoothness=7.905694150420947e-6,
            sharpness=7.905694150420948e-7,
        ),
    },
}


def _underexplored_settings(cap: int, endpoint: str) -> dict[str, float]:
    learning_rate = {
        2560: 0.00625,
        640: 0.025,
        160: 0.1,
        80: 0.15,
        20: 0.15,
        10: 0.15,
    }[cap]
    intensity_fraction = {2560: 0.0625, 640: 0.25}.get(cap, 0.5)
    beta1 = 0.95 if cap > 40 else 0.9
    beta2 = 0.999 if cap >= 1280 else 0.99
    if endpoint == "low":
        smoothness = 3.952847075210474e-9
        sharpness = 3.952847075210474e-10
    else:
        smoothness = 3.9528470752104736e-6
        sharpness = 3.952847075210474e-7
    return _settings(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        smoothness=smoothness,
        sharpness=sharpness,
        intensity_fraction=intensity_fraction,
    )


UNDEREXPLORED_ENDPOINTS = {
    cap: {
        endpoint: _underexplored_settings(cap, endpoint)
        for endpoint in ("low", "high")
    }
    for cap in (2560, 640, 160, 80, 20, 10)
}


def _parameters(
    N: int,
    cap: int,
    settings: Mapping[str, float],
    *,
    optimizer: str,
    schedule: tuple[tuple[int, float], ...],
    strictness: str | None = None,
) -> dict:
    if strictness == "loose":
        tolerances = (1e-5, 1e-4, 1e-4)
    elif strictness == "strict":
        tolerances = (1e-6, 1e-5, 1e-5)
    else:
        tolerances = (None, None, None)
    score_tolerance, control_tolerance, gradient_tolerance = tolerances
    return {
        "N": N,
        "t_interval": 4.0,
        "r_bg": -0.008716,
        "u_isbound": True,
        "v_isbound": True,
        "u_max": float(cap),
        "v_max": 1000.0,
        "slew_limit": 0.05,
        "optimizer": optimizer,
        "schedule": schedule,
        "adam_learning_rate": settings["adam_learning_rate"],
        "adam_beta1": settings["adam_beta1"],
        "adam_beta2": settings["adam_beta2"],
        "adam_eps": 1e-8,
        "lbfgs_history_size": 100,
        "lbfgs_max_linesearch_steps": 100,
        "lbfgs_tolerance": 1e-12,
        "peak_initial_step_size": 1e-2,
        "peak_min_step_size": 1e-12,
        "peak_max_step_size": 0.1,
        "peak_backtracking_factor": 0.5,
        "peak_step_growth": 1.5,
        "peak_armijo": 1e-4,
        "peak_max_linesearch_steps": 24,
        "smoothness": settings["smoothness"],
        "u_smooth": None,
        "v_smooth": None,
        "sharpness": settings["sharpness"],
        "u_sharp": None,
        "v_sharp": None,
        "block_size": 10 if optimizer == "lbfgs" else 25,
        "J_tol": score_tolerance,
        "u_tol": control_tolerance,
        "v_tol": control_tolerance,
        "projected_gradient_tol": gradient_tolerance,
        "projected_gradient_alpha": 1.0,
        "grid_refinement_tol": 1e-3 if strictness == "strict" else 1e-2,
        "grid_refinement_y_floor": 1e-12,
    }


def _runtime(
    database: str,
    *,
    initialisations: int,
    batch_size: int,
    max_elapsed_seconds: float | None = None,
    distribute_deadline: bool = False,
    repeat_until_stable: bool = False,
    auto_halt: bool = False,
    intensity_fraction: float = 0.5,
) -> dict:
    return {
        "initialisations": initialisations,
        "fourier_num_modes": 6,
        "fourier_rms_amplitude": 0.8,
        "fourier_intensity_fraction": intensity_fraction,
        "use_jit": True,
        "use_x64": True,
        "device": "gpu",
        "concurrent_workers": 1,
        "max_cases_per_batch": None,
        "max_initialisations_per_batch": batch_size,
        "max_steps_per_chunk": 50 if repeat_until_stable else 1_000,
        "max_batch_elapsed_seconds": max_elapsed_seconds,
        "max_elapsed_seconds": max_elapsed_seconds,
        "distribute_max_elapsed_across_batches": distribute_deadline,
        "repeat_schedule_until_stable": repeat_until_stable,
        "auto_halt": auto_halt,
        "database": database,
    }


def _source_query(
    source: ConfigDocument,
    *,
    source_N: int,
    cap: int,
    settings: Mapping[str, float],
    limit: int | None,
    source_database: str | None = None,
    terminal_only: bool = False,
) -> dict:
    where = {
        "status": "complete",
        "config_id": source.config_id,
        "config_name": source.name,
        "N": source_N,
        "u_max": float(cap),
        "smoothness": settings["smoothness"],
        "sharpness": settings["sharpness"],
    }
    if terminal_only:
        where["termination_reason"] = ["stability", "time_limit"]
    return {
        "database": source_database,
        "where": where,
        "limit": limit,
        "order_by": "best_objective",
        "descending": True,
        "control_kind": "best",
        "resume_optimizer": False,
        "perturbed": False,
        "match_parameters": [],
    }


def _document(
    *,
    name: str,
    description: str,
    N: int,
    cap: int,
    settings: Mapping[str, float],
    database: str,
    optimizer: str,
    schedule: tuple[tuple[int, float], ...],
    initialisations: int,
    batch_size: int,
    strictness: str | None = None,
    source: ConfigDocument | None = None,
    source_N: int | None = None,
    source_limit: int | None = None,
    source_database: str | None = None,
    terminal_source: bool = False,
    max_elapsed_seconds: float | None = None,
    distribute_deadline: bool = False,
    repeat_until_stable: bool = False,
    auto_halt: bool = False,
) -> ConfigDocument:
    query = None
    if source is not None:
        query = _source_query(
            source,
            source_N=N if source_N is None else source_N,
            cap=cap,
            settings=settings,
            limit=source_limit,
            source_database=source_database,
            terminal_only=terminal_source,
        )
    return make_document(
        name=name,
        description=description,
        parameters=_parameters(
            N,
            cap,
            settings,
            optimizer=optimizer,
            schedule=schedule,
            strictness=strictness,
        ),
        runtime=_runtime(
            database,
            initialisations=initialisations,
            batch_size=batch_size,
            max_elapsed_seconds=max_elapsed_seconds,
            distribute_deadline=distribute_deadline,
            repeat_until_stable=repeat_until_stable,
            auto_halt=auto_halt,
            intensity_fraction=settings["fourier_intensity_fraction"],
        ),
        query=query,
    )


def _n100_funnel(
    *,
    prefix: str,
    cap: int,
    endpoint: str,
    settings: Mapping[str, float],
    database: str,
) -> tuple[ConfigDocument, ...]:
    base = f"N100_u{cap}_{endpoint}_{prefix}"
    scout = _document(
        name=f"{base}_scout500_2k_gpu",
        description=(
            f"N=100 u_max={cap} {endpoint} endpoint: 500 broad Fourier seeds "
            "receive an inexpensive 2,000-step Adam scout."
        ),
        N=100,
        cap=cap,
        settings=settings,
        database=database,
        optimizer="adam",
        schedule=((2_000, 1.0),),
        initialisations=500,
        batch_size=50,
    )
    broad = _document(
        name=f"{base}_broad500_10k_gpu",
        description=(
            f"N=100 u_max={cap} {endpoint} endpoint: continue all 500 scouts "
            "for 8,000 steps so every seed reaches at least 10,000 steps."
        ),
        N=100,
        cap=cap,
        settings=settings,
        database=database,
        optimizer="adam",
        schedule=((8_000, 0.5),),
        initialisations=0,
        batch_size=50,
        source=scout,
        source_limit=None,
    )
    deep = _document(
        name=f"{base}_deep250_20k_gpu",
        description=(
            f"N=100 u_max={cap} {endpoint} endpoint: spend at most two hours "
            "taking the best 250 broad controls to 20,000 cumulative steps."
        ),
        N=100,
        cap=cap,
        settings=settings,
        database=database,
        optimizer="adam",
        schedule=((10_000, 0.25),),
        initialisations=0,
        batch_size=50,
        source=broad,
        source_limit=250,
        max_elapsed_seconds=TWO_HOURS,
        distribute_deadline=True,
    )
    polish = _document(
        name=f"{base}_polish100_1k_gpu",
        description=(
            f"N=100 u_max={cap} {endpoint} endpoint: two-hour, 1,000-step "
            "monotone polish of the best 100 deep candidates."
        ),
        N=100,
        cap=cap,
        settings=settings,
        database=database,
        optimizer="peak_refinement",
        schedule=((1_000, 1.0),),
        initialisations=0,
        batch_size=25,
        source=deep,
        source_limit=100,
        max_elapsed_seconds=TWO_HOURS,
        distribute_deadline=True,
    )
    loose = _document(
        name=f"{base}_loose10_gpu",
        description=(
            f"N=100 u_max={cap} {endpoint} endpoint: loose L-BFGS-B "
            "convergence of the best ten polished candidates for up to two hours."
        ),
        N=100,
        cap=cap,
        settings=settings,
        database=database,
        optimizer="lbfgs",
        schedule=((50, 1.0),),
        initialisations=0,
        batch_size=10,
        strictness="loose",
        source=polish,
        source_limit=10,
        max_elapsed_seconds=TWO_HOURS,
        repeat_until_stable=True,
        auto_halt=True,
    )
    preliminary_strict = _document(
        name=f"{base}_strict3_gpu",
        description=(
            f"N=100 u_max={cap} {endpoint} endpoint: independently strict-"
            "refine the best three loose challengers for up to two hours."
        ),
        N=100,
        cap=cap,
        settings=settings,
        database=database,
        optimizer="peak_refinement",
        schedule=((250, 1.0),),
        initialisations=0,
        batch_size=3,
        strictness="strict",
        source=loose,
        source_limit=3,
        terminal_source=True,
        max_elapsed_seconds=TWO_HOURS,
        repeat_until_stable=True,
        auto_halt=True,
    )
    return scout, broad, deep, polish, loose, preliminary_strict


def _strict_from_existing_u1280() -> ConfigDocument:
    source_path = (
        ROOT
        / "run_config"
        / "N100_u1280_fixed_parameter_low_regularization_bar_v3_seed1000_top10_loose_gpu.yaml"
    )
    source = load_config(source_path)
    settings = STRICT_ENDPOINTS[1280]["low"]
    return _document(
        name="N100_u1280_low_strictgrid_v1_strict3_gpu",
        description=(
            "Strictly refine the top three of the existing u_max=1280 low-"
            "endpoint top-ten handoff; its 1,000 seeds already received 20,000 "
            "Adam steps and a 2,000-step monotone polish."
        ),
        N=100,
        cap=1280,
        settings=settings,
        database=TRACK_A_DATABASE,
        optimizer="peak_refinement",
        schedule=((250, 1.0),),
        initialisations=0,
        batch_size=3,
        strictness="strict",
        source=source,
        source_limit=3,
        source_database=source.runtime.database,
        terminal_source=True,
        max_elapsed_seconds=TWO_HOURS,
        repeat_until_stable=True,
        auto_halt=True,
    )


def _grid_promotion(
    *,
    cap: int,
    endpoint: str,
    settings: Mapping[str, float],
    N: int,
    source: ConfigDocument,
) -> tuple[ConfigDocument, ConfigDocument]:
    base = f"N{N}_u{cap}_{endpoint}_strictgrid_v1"
    loose = _document(
        name=f"{base}_loose23_gpu",
        description=(
            f"Promote three independent N={N - 100} challengers to N={N}, "
            "add 20 fresh Fourier starts, and run loose L-BFGS-B for two hours."
        ),
        N=N,
        cap=cap,
        settings=settings,
        database=TRACK_A_DATABASE,
        optimizer="lbfgs",
        schedule=((50, 1.0),),
        initialisations=20,
        batch_size=23,
        strictness="loose",
        source=source,
        source_N=N - 100,
        source_limit=3,
        terminal_source=True,
        max_elapsed_seconds=TWO_HOURS,
        repeat_until_stable=True,
        auto_halt=True,
    )
    strict = _document(
        name=f"{base}_strict3_gpu",
        description=(
            f"Strictly refine the best three independent N={N} challengers "
            "for up to two hours before promoting to the next grid."
        ),
        N=N,
        cap=cap,
        settings=settings,
        database=TRACK_A_DATABASE,
        optimizer="peak_refinement",
        schedule=((250, 1.0),),
        initialisations=0,
        batch_size=3,
        strictness="strict",
        source=loose,
        source_limit=3,
        terminal_source=True,
        max_elapsed_seconds=TWO_HOURS,
        repeat_until_stable=True,
        auto_halt=True,
    )
    return loose, strict


def _final_strict(
    *,
    cap: int,
    endpoint: str,
    settings: Mapping[str, float],
    N: int,
    source: ConfigDocument,
) -> ConfigDocument:
    return _document(
        name=f"N{N}_u{cap}_{endpoint}_strictgrid_v1_final_strict1_gpu",
        description=(
            f"Final N={N} u_max={cap} {endpoint}-endpoint strict convergence "
            "of the best independently challenged solution, with a 12-hour guard."
        ),
        N=N,
        cap=cap,
        settings=settings,
        database=TRACK_A_DATABASE,
        optimizer="peak_refinement",
        schedule=((250, 1.0),),
        initialisations=0,
        batch_size=1,
        strictness="strict",
        source=source,
        source_limit=1,
        terminal_source=True,
        max_elapsed_seconds=FINAL_STRICT_HOURS,
        repeat_until_stable=True,
        auto_halt=True,
    )


def track_a_documents() -> tuple[ConfigDocument, ...]:
    lanes: dict[tuple[int, str], tuple[ConfigDocument, ...]] = {}
    for cap, endpoints in STRICT_ENDPOINTS.items():
        for endpoint, settings in endpoints.items():
            if (cap, endpoint) == (1280, "low"):
                continue
            lanes[(cap, endpoint)] = _n100_funnel(
                prefix="strictgrid_v1",
                cap=cap,
                endpoint=endpoint,
                settings=settings,
                database=TRACK_A_DATABASE,
            )

    existing_u1280_strict = _strict_from_existing_u1280()
    ordered: list[ConfigDocument] = [existing_u1280_strict]
    # Finish each low-cost stage for all lanes before increasing commitment.
    for stage_index in range(6):
        ordered.extend(lane[stage_index] for lane in lanes.values())

    preliminary_by_grid: dict[tuple[int, str, int], ConfigDocument] = {
        (1280, "low", 100): existing_u1280_strict
    }
    preliminary_by_grid.update(
        {
            (cap, endpoint, 100): lane[-1]
            for (cap, endpoint), lane in lanes.items()
        }
    )
    lane_order = tuple(
        (cap, endpoint)
        for cap in STRICT_ENDPOINTS
        for endpoint in ("low", "high")
    )
    for N in RESOLUTIONS[1:]:
        promoted = []
        for cap, endpoint in lane_order:
            loose, strict = _grid_promotion(
                cap=cap,
                endpoint=endpoint,
                settings=STRICT_ENDPOINTS[cap][endpoint],
                N=N,
                source=preliminary_by_grid[(cap, endpoint, N - 100)],
            )
            promoted.append((cap, endpoint, loose, strict))
        ordered.extend(item[2] for item in promoted)
        ordered.extend(item[3] for item in promoted)
        preliminary_by_grid.update(
            {(cap, endpoint, N): strict for cap, endpoint, _, strict in promoted}
        )

    # Deep strict convergence is deliberately last: preliminary results at all
    # grids are saved before any single hard tolerance can monopolize Track A.
    for N in RESOLUTIONS:
        for cap, endpoint in lane_order:
            ordered.append(
                _final_strict(
                    cap=cap,
                    endpoint=endpoint,
                    settings=STRICT_ENDPOINTS[cap][endpoint],
                    N=N,
                    source=preliminary_by_grid[(cap, endpoint, N)],
                )
            )
    return tuple(ordered)


def _progressive_documents(
    *,
    caps: tuple[int, ...],
    prefix: str,
    database: str,
) -> tuple[ConfigDocument, ...]:
    lanes = {
        (cap, endpoint): _n100_funnel(
            prefix=prefix,
            cap=cap,
            endpoint=endpoint,
            settings=settings,
            database=database,
        )
        for cap in caps
        for endpoints in (UNDEREXPLORED_ENDPOINTS[cap],)
        for endpoint, settings in endpoints.items()
    }
    ordered: list[ConfigDocument] = []
    for stage_index in range(6):
        ordered.extend(lane[stage_index] for lane in lanes.values())
    # The sixth stage already gives Track B a top-three strict comparison; one
    # final strict solution per endpoint narrows 3 -> 1 for the last level.
    for (cap, endpoint), lane in lanes.items():
        ordered.append(
            _document(
                name=f"N100_u{cap}_{endpoint}_{prefix}_final_strict1_gpu",
                description=(
                    f"Final two-hour strict refinement of the best u_max={cap} "
                    f"{endpoint}-endpoint candidate after the 500→250→100→10→3 funnel."
                ),
                N=100,
                cap=cap,
                settings=UNDEREXPLORED_ENDPOINTS[cap][endpoint],
                database=database,
                optimizer="peak_refinement",
                schedule=((250, 1.0),),
                initialisations=0,
                batch_size=1,
                strictness="strict",
                source=lane[-1],
                source_limit=1,
                terminal_source=True,
                max_elapsed_seconds=TWO_HOURS,
                repeat_until_stable=True,
                auto_halt=True,
            )
        )
    return tuple(ordered)


def track_b_documents() -> tuple[ConfigDocument, ...]:
    """Return the half-P100 server lanes that remain after laptop offload."""

    return _progressive_documents(
        caps=SERVER_UNDEREXPLORED_CAPS,
        prefix="progressive_v2",
        database=TRACK_B_DATABASE,
    )


def local_track_documents() -> tuple[ConfigDocument, ...]:
    """Return independent N=100 lanes sized for the 8 GB RTX 4060 worker."""

    return _progressive_documents(
        caps=LOCAL_UNDEREXPLORED_CAPS,
        prefix="local4060_v1",
        database=LOCAL_DATABASE,
    )


def _write_documents(
    documents: tuple[ConfigDocument, ...], manifest: Path
) -> tuple[Path, ...]:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        write_config(document, ROOT / "run_config" / f"{document.name}.yaml")
        for document in documents
    )
    _write_manifest(paths, manifest)
    return paths


def _write_manifest(paths: tuple[Path, ...], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(f"{path.relative_to(ROOT)}\n" for path in paths),
        encoding="utf-8",
    )


def _existing_manifest_paths(manifest: Path) -> tuple[Path, ...]:
    return tuple(
        (ROOT / line.strip()).resolve()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _paths_for_caps(paths: tuple[Path, ...], caps: tuple[int, ...]) -> tuple[Path, ...]:
    selected = []
    for path in paths:
        document = load_config(path)
        if int(document.scalar_cases()[0].u_max) in caps:
            selected.append(path)
    return tuple(selected)


def main() -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    track_a = (
        _existing_manifest_paths(TRACK_A_MANIFEST)
        if TRACK_A_MANIFEST.is_file()
        else _write_documents(track_a_documents(), TRACK_A_MANIFEST)
    )
    if TRACK_B_MANIFEST.is_file():
        track_b = _paths_for_caps(
            _existing_manifest_paths(TRACK_B_MANIFEST), SERVER_UNDEREXPLORED_CAPS
        )
        _write_manifest(track_b, TRACK_B_MANIFEST)
    else:
        track_b = _write_documents(track_b_documents(), TRACK_B_MANIFEST)
    local_track = (
        _existing_manifest_paths(LOCAL_MANIFEST)
        if LOCAL_MANIFEST.is_file()
        else _write_documents(local_track_documents(), LOCAL_MANIFEST)
    )
    print(f"Track A: {len(track_a)} configs -> {TRACK_A_MANIFEST}")
    print(f"Track B: {len(track_b)} configs -> {TRACK_B_MANIFEST}")
    print(f"RTX 4060: {len(local_track)} configs -> {LOCAL_MANIFEST}")
    return track_a, track_b, local_track


if __name__ == "__main__":
    main()
