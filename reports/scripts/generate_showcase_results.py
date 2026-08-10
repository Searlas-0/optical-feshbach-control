#!/usr/bin/env python3
"""Generate the data figure and auditable numbers used by the showcase.

The plot deliberately separates three kinds of evidence:

* all fixed-method seeds at every cap in the completed sweep;
* fixed-control transfer of the best fixed-method pulse at every cap to finer
  time grids; and
* the best fixed-method controls grouped into tight, crossover, and loose-cap
  regimes.

This distinction prevents an unfinished high-cap iterate from being presented
as a converged physical optimum.  Every source is resolved to an immutable
``config_document_id`` and ``queue_id`` before its rows are summarised.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPORT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ofc.physical import HBAR  # noqa: E402
from ofc.results import Results  # noqa: E402


DEFAULT_DATABASE = PROJECT_ROOT / "results" / "results.sqlite3"
DEFAULT_OUTPUT = REPORT_DIR / "figures" / "showcase_provisional_results.pdf"
DEFAULT_METADATA = REPORT_DIR / "figures" / "showcase_results_metadata.json"
DEFAULT_MACROS = REPORT_DIR / "figures" / "showcase_results_macros.tex"
DEFAULT_TARGET_DATABASES = {
    40: PROJECT_ROOT
    / "results"
    / "slurm_isolated"
    / "N100_u40_top_peak_refinement_strict.sqlite3",
    320: PROJECT_ROOT / "results" / "bar_u320_crossover_screen.sqlite3",
    1280: PROJECT_ROOT
    / "results"
    / "slurm_isolated"
    / "N100_u1280_top_peak_refinement_strict.sqlite3",
}

BASELINE_CONFIG = "88_Sr_N_100_u_max_sweep"
STRICT_CONFIGS = {
    40: "N100_u40_top_peak_refinement_strict_slurm_isolated_cpu",
    320: None,
    1280: "N100_u1280_top_peak_refinement_strict_slurm_isolated_cpu",
}
SELECTED_CAPS = (40, 320, 1280)
CONTROL_CAP_GROUPS = {
    "tight": (10, 20, 40, 80, 160),
    "crossover": (320, 640),
    "loose": (1280, 2560),
}
CONTROL_CAPS = tuple(
    cap for group in CONTROL_CAP_GROUPS.values() for cap in group
)
GRID_REFINEMENT_N = (100, 200, 400, 800)

# Dimensionalisation requested for the showcase.  The project defines g_2(0)
# as the unnormalised equal-position pair density (units m^-6), not the
# dimensionless pair-correlation function convention often denoted g^(2)(0).
BOHR_RADIUS_M = 5.291_772_109_03e-11
T_STAR_S = 1.0e-7
A_BG_M = -1.4 * BOHR_RADIUS_M
ATOM_MASS_KG = 1.459_707e-25
GAMMA_HZ = 40_000.0
G2_INITIAL_M6 = 1.0
L_STAR_M = float(np.sqrt(HBAR * T_STAR_S / ATOM_MASS_KG))
R_BG = A_BG_M / L_STAR_M
MOLECULAR_DENSITY_PER_OBJECTIVE_M3 = G2_INITIAL_M6 * L_STAR_M**3

FREQUENCY_DISPLAY_HZ = 1.0e6
TIME_DISPLAY_S = 1.0e-6
DENSITY_DISPLAY_M3 = 1.0e-21

NAVY = "#143D66"
BLUE = "#2474A6"
CYAN = "#69AEC9"
RED = "#B6403A"
GOLD = "#D49A22"
GREY = "#626B76"

# The notebook cap-sweep figures use a viridis progression.  Fix the colours
# explicitly so cap identity is stable across the strip, refinement, and
# control panels and across Matplotlib releases.
CAP_COLOURS = {
    10: "#440154",
    20: "#472D7B",
    40: "#3B528B",
    80: "#2C728E",
    160: "#21918C",
    320: "#28AE80",
    640: "#5EC962",
    1280: "#ADDC30",
    2560: "#E4C91A",
}


@dataclass(frozen=True)
class Execution:
    database: Path
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class TargetResult:
    cap: int
    database: Path
    row: dict[str, Any]
    controls: dict[str, np.ndarray]
    tolerances: dict[str, np.ndarray]
    strict_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class ShowcaseData:
    generated_utc: str
    baseline: Execution
    baseline_by_cap: dict[int, list[dict[str, Any]]]
    targets: dict[int, TargetResult]
    control_targets: dict[int, TargetResult]
    grid_refinement: dict[int, dict[int, float]]
    control_grid_refinement: dict[int, dict[int, float]]


def _validate_background_scale(rows: list[dict[str, Any]]) -> None:
    """Refuse to label results with a dimensionalisation they do not use."""

    for row in rows:
        observed = float(row["r_bg"])
        if not np.isclose(observed, R_BG, rtol=0.0, atol=5.0e-7):
            raise RuntimeError(
                "supplied a_bg, mass, and t_star imply "
                f"r_bg={R_BG:.9g}, but run {row['run_id']} stores {observed:.9g}"
            )


def _finite_objective(row: dict[str, Any]) -> bool:
    value = row.get("best_objective")
    return value is not None and np.isfinite(float(value))


def _completed_execution(results: Results, config_name: str) -> Execution:
    summaries = results.config_runs(config_name=config_name)
    completed = [summary for summary in summaries if summary["status"] == "complete"]
    if not completed:
        raise RuntimeError(f"no completed execution found for {config_name!r}")
    summary = completed[0]
    rows = results.search(
        config_document_id=summary["config_document_id"],
        queue_id=summary["queue_id"],
    )
    if not rows or any(row["status"] != "complete" for row in rows):
        raise RuntimeError(f"completed execution {config_name!r} has incomplete rows")
    return Execution(results.database, summary, rows)


def _strict_rows(results: Results, config_name: str | None) -> list[dict[str, Any]]:
    if config_name is None:
        return []
    summaries = results.config_runs(config_name=config_name)
    if not summaries:
        return []
    summary = summaries[0]
    return results.search(
        config_document_id=summary["config_document_id"],
        queue_id=summary["queue_id"],
    )


def _latest_target(database: Path, cap: int) -> TargetResult:
    if not database.is_file():
        raise FileNotFoundError(f"target database not found: {database}")
    results = Results(database)
    candidates = [
        row
        for row in results.search(N=100, u_max=float(cap))
        if _finite_objective(row)
    ]
    if not candidates:
        raise RuntimeError(f"no saved N=100 target result for u_max={cap} in {database}")
    row = max(candidates, key=lambda item: float(item["best_score"]))
    controls = {
        name: np.asarray(values, dtype=float)
        for name, values in results.controls(int(row["run_id"]), "best").items()
    }
    strict_rows = _strict_rows(results, STRICT_CONFIGS[cap])
    tolerances = results.tolerances(int(row["run_id"]))
    return TargetResult(cap, database, row, controls, tolerances, strict_rows)


def _best_fixed_method_targets(
    results: Results,
    execution: Execution,
    baseline_by_cap: dict[int, list[dict[str, Any]]],
) -> dict[int, TargetResult]:
    """Load the highest-density control from the same method at every cap."""

    missing = sorted(set(CONTROL_CAPS) - set(baseline_by_cap))
    if missing:
        raise RuntimeError(f"fixed-method sweep is missing control caps {missing}")
    targets: dict[int, TargetResult] = {}
    for cap in CONTROL_CAPS:
        candidates = [
            row
            for row in baseline_by_cap[cap]
            if _finite_objective(row) and np.isfinite(float(row["best_score"]))
        ]
        if not candidates:
            raise RuntimeError(f"no finite fixed-method control at cap {cap}")
        row = max(candidates, key=lambda item: float(item["best_objective"]))
        controls = {
            name: np.asarray(values, dtype=float)
            for name, values in results.controls(int(row["run_id"]), "best").items()
        }
        tolerances = results.tolerances(int(row["run_id"]))
        targets[cap] = TargetResult(
            cap=cap,
            database=execution.database,
            row=row,
            controls=controls,
            tolerances=tolerances,
            strict_rows=[],
        )
    return targets


def _fixed_control_objective(target: TargetResult, N: int) -> float:
    """Evaluate one stored control after linear transfer to ``N`` intervals."""

    old_u = np.asarray(target.controls["u"], dtype=float)
    old_v = np.asarray(target.controls["v"], dtype=float)
    if old_u.shape != old_v.shape:
        raise RuntimeError(f"u and v controls have different shapes for cap {target.cap}")
    old_grid = np.linspace(0.0, 1.0, old_u.size)
    new_grid = np.linspace(0.0, 1.0, N + 1)
    u = np.interp(new_grid, old_grid, old_u)
    v = np.interp(new_grid, old_grid, old_v)
    r_bg = float(target.row["r_bg"])
    dt = float(target.row["t_interval"]) / N
    sign = np.sign(r_bg)
    scattering = r_bg * (1.0 + sign * u / (-sign * u - v + 0.5j))

    eta = np.zeros(N + 1, dtype=complex)
    eta[0] = -4.0 * np.pi * scattering[0]
    kernel_prefactor = -1.0 / (4.0 * np.pi ** 1.5 * np.sqrt(1j))
    l1_prefactor = 2.0 * kernel_prefactor / np.sqrt(dt)
    indices = np.arange(N)
    for k in range(1, N + 1):
        valid = indices < (k - 1)
        distance = np.maximum(k - indices, 1)
        weights = np.where(
            valid, np.sqrt(distance) - np.sqrt(distance - 1), 0.0
        )
        known_history = np.sum(weights * np.diff(eta))
        numerator = -1.0 + l1_prefactor * (eta[k - 1] - known_history)
        denominator = 1.0 / (4.0 * np.pi * scattering[k]) + l1_prefactor
        eta[k] = numerator / denominator

    contact = np.imag(1.0 / scattering) * np.abs(eta) ** 2
    return float(np.sum(0.5 * dt * (contact[:-1] + contact[1:])) / (2.0 * np.pi))


def _grid_refinement(targets: dict[int, TargetResult]) -> dict[int, dict[int, float]]:
    refinement = {
        cap: {N: _fixed_control_objective(target, N) for N in GRID_REFINEMENT_N}
        for cap, target in targets.items()
    }
    for cap, values in refinement.items():
        stored = float(targets[cap].row["best_objective"])
        if not np.isclose(values[100], stored, rtol=2.0e-9, atol=2.0e-9):
            raise RuntimeError(
                f"cap {cap} N=100 reevaluation {values[100]:.12g} "
                f"does not reproduce stored objective {stored:.12g}"
            )
    return refinement


def load_data(
    database: Path,
    target_databases: dict[int, Path],
) -> ShowcaseData:
    if not database.is_file():
        raise FileNotFoundError(f"result database not found: {database}")
    results = Results(database)
    baseline = _completed_execution(results, BASELINE_CONFIG)

    baseline_by_cap: dict[int, list[dict[str, Any]]] = {}
    for row in baseline.rows:
        cap = int(round(float(row["u_max"])))
        baseline_by_cap.setdefault(cap, []).append(row)

    targets = {
        cap: _latest_target(target_databases[cap], cap) for cap in SELECTED_CAPS
    }
    control_targets = _best_fixed_method_targets(
        results, baseline, baseline_by_cap
    )
    data = ShowcaseData(
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        baseline=baseline,
        baseline_by_cap=baseline_by_cap,
        targets=targets,
        control_targets=control_targets,
        grid_refinement=_grid_refinement(targets),
        control_grid_refinement=_grid_refinement(control_targets),
    )
    _validate_background_scale(
        [
            baseline.rows[0],
            *(target.row for target in targets.values()),
            *(target.row for target in control_targets.values()),
        ]
    )
    return data


def _objective_values(rows: list[dict[str, Any]]) -> np.ndarray:
    values = [float(row["best_objective"]) for row in rows if _finite_objective(row)]
    if not values:
        raise RuntimeError("a selected result group contains no finite objectives")
    return np.asarray(values)


def _format_objective(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _frequency_mhz(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * GAMMA_HZ / FREQUENCY_DISPLAY_HZ


def _density_m3(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * MOLECULAR_DENSITY_PER_OBJECTIVE_M3


def _density_display(value: float | np.ndarray) -> float | np.ndarray:
    return _density_m3(value) / DENSITY_DISPLAY_M3


def _format_density_display(value: float) -> str:
    """Format a density expressed in units of 10^-21 m^-3."""

    return f"{float(_density_display(value)):.3g}"


def _format_scientific_latex(value: float) -> str:
    """Return a compact numeric value that renders as mathematics in LaTeX."""

    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.3g}\times10^{{{exponent}}}"


def _display_path(path: Path) -> str:
    """Return a repository-relative path when the source is local."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _style_axis(axis: plt.Axes, *, grid: str = "both") -> None:
    axis.grid(True, which=grid, color="#DDE2E7", lw=0.45, alpha=0.78)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#8A929C")
        spine.set_linewidth(0.62)


def _draw_cap_sweep(axis: plt.Axes, data: ShowcaseData) -> None:
    """Draw the notebook-style categorical seed strip in physical units."""

    caps = list(CONTROL_CAPS)
    all_values: list[float] = []
    for cap_index, cap in enumerate(caps):
        values = np.asarray(
            _density_display(_objective_values(data.baseline_by_cap[cap]))
        )
        all_values.extend(values.tolist())
        position = float(cap_index)
        jitter = (
            np.zeros(1)
            if values.size == 1
            else np.linspace(-0.19, 0.19, values.size)
        )
        axis.scatter(
            position + jitter,
            values,
            s=11.5,
            color=CAP_COLOURS[cap],
            alpha=0.78,
            edgecolors="#FFFFFF",
            linewidths=0.22,
            rasterized=True,
            zorder=3,
        )
        median = float(np.median(values))
        minimum, maximum = float(np.min(values)), float(np.max(values))
        axis.plot(
            [position, position],
            [minimum, maximum],
            color="#202020",
            lw=0.62,
            alpha=0.44,
            solid_capstyle="round",
            zorder=2,
        )
        axis.plot(
            [
                position - 0.20,
                position + 0.20,
                np.nan,
                position - 0.20,
                position + 0.20,
            ],
            [minimum, minimum, np.nan, maximum, maximum],
            color="#202020",
            lw=0.62,
            alpha=0.44,
            solid_capstyle="round",
            zorder=2.1,
        )
        axis.plot(
            [position - 0.13, position + 0.13],
            [median, median],
            color="#202020",
            lw=0.72,
            alpha=0.68,
            solid_capstyle="round",
            zorder=4,
        )
        selected = data.control_targets[cap].row
        axis.scatter(
            [position],
            [_density_display(float(selected["best_objective"]))],
            s=22.0,
            color=CAP_COLOURS[cap],
            edgecolor="#202020",
            lw=0.70,
            zorder=5,
        )

    axis.set_yscale("log")
    positive_values = np.asarray([value for value in all_values if value > 0.0])
    axis.set_xlim(-0.45, len(caps) - 0.55)
    axis.set_ylim(
        float(np.min(positive_values)) / 2.2,
        float(np.max(positive_values)) * 10.0,
    )
    axis.set_xticks(np.arange(len(caps), dtype=float))
    axis.set_xticklabels(
        [f"{float(_frequency_mhz(cap)):g}" for cap in caps]
    )
    axis.set_xlabel(r"$\Gamma_{\max}\;(\mathrm{MHz})$", labelpad=2)
    axis.set_ylabel(
        r"$n_{\rm mol}\;(10^{-21}\,\mathrm{m}^{-3})$", labelpad=2
    )
    axis.text(
        -0.015,
        1.025,
        "(a)",
        transform=axis.transAxes,
        color=NAVY,
        fontweight="bold",
        fontsize=7.4,
        ha="left",
        va="bottom",
    )
    axis.text(
        0.03,
        0.985,
        "50 seeds/cap; coloured dots: every run; outlined dot: max $n_{\\rm mol}$\n"
        "thin whisker: min–max; short thin line: median",
        transform=axis.transAxes,
        fontsize=5.05,
        linespacing=0.85,
        color=GREY,
        va="top",
    )
    _style_axis(axis)


def _terminal_tolerance_ratio(
    target: TargetResult, value_name: str, tolerance_name: str
) -> float:
    values = np.asarray(target.tolerances.get(value_name, []), dtype=float)
    finite = values[np.isfinite(values)]
    tolerance = target.row.get(tolerance_name)
    if finite.size == 0 or tolerance is None or float(tolerance) <= 0.0:
        return np.nan
    return float(finite[-1] / float(tolerance))


def _compact_number(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1.0e-2 or magnitude >= 1.0e4):
        return f"{value:.1e}"
    return f"{value:.3g}"


def _draw_cap_table(axis: plt.Axes, data: ShowcaseData) -> None:
    """Tabulate dimensional yield, convergence, stability, and slew data."""

    axis.axis("off")
    gamma_mhz = GAMMA_HZ / FREQUENCY_DISPLAY_HZ
    t_star_us = T_STAR_S / TIME_DISPLAY_S
    d1_scale = gamma_mhz / t_star_us
    d2_scale = gamma_mhz / t_star_us**2
    rows: list[list[str]] = []
    for cap in CONTROL_CAPS:
        target = data.control_targets[cap]
        row = target.row
        grid_error = row.get("best_grid_refinement_relative_error")
        grid_passed = bool(row.get("best_grid_refinement_passed"))
        grid_cell = (
            "—"
            if grid_error is None or not np.isfinite(float(grid_error))
            else f"{100.0 * float(grid_error):.3g} {'P' if grid_passed else 'F'}"
        )
        d1_u = d1_scale * float(row["best_max_abs_du_dt"])
        d1_v = d1_scale * float(row["best_max_abs_dv_dt"])
        d2_u = d2_scale * float(row["best_max_abs_d2u_dt2"])
        d2_v = d2_scale * float(row["best_max_abs_d2v_dt2"])
        rows.append(
            [
                f"{float(_frequency_mhz(cap)):g}",
                _compact_number(
                    float(_density_display(float(row["best_objective"])))
                ),
                str(int(row["N"])),
                grid_cell,
                _compact_number(
                    _terminal_tolerance_ratio(target, "score_tolerance", "J_tol")
                ),
                _compact_number(
                    _terminal_tolerance_ratio(target, "u_tolerance", "u_tol")
                ),
                _compact_number(
                    _terminal_tolerance_ratio(target, "v_tolerance", "v_tol")
                ),
                f"{_compact_number(d1_u)}/{_compact_number(d1_v)}",
                f"{_compact_number(d2_u)}/{_compact_number(d2_v)}",
            ]
        )

    labels = [
        r"$\Gamma_{\max}$",
        r"$n_{\rm mol}^{\star}$",
        r"$N$",
        r"$\epsilon_g\;(\%)$",
        r"$\epsilon_J/J_{\rm tol}$",
        r"$\epsilon_u/u_{\rm tol}$",
        r"$\epsilon_v/v_{\rm tol}$",
        r"$\|\dot\Gamma\|_\infty/\|\dot\nu\|_\infty$",
        r"$\|\ddot\Gamma\|_\infty/\|\ddot\nu\|_\infty$",
    ]
    widths = [0.075, 0.095, 0.045, 0.085, 0.10, 0.10, 0.10, 0.19, 0.21]
    table = axis.table(
        cellText=rows,
        colLabels=labels,
        colWidths=widths,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.12, 1.0, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(4.25)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#C8CED5")
        cell.set_linewidth(0.35)
        if row_index == 0:
            cell.set_facecolor("#E9F0F6")
            cell.set_text_props(color=NAVY, fontweight="bold")
        elif column_index == 0:
            cap = CONTROL_CAPS[row_index - 1]
            cell.set_facecolor(CAP_COLOURS[cap])
            cell.set_text_props(
                color="white" if row_index <= 6 else "#202020",
                fontweight="bold",
            )
        elif row_index % 2 == 0:
            cell.set_facecolor("#F7F9FB")

    axis.text(
        0.0,
        0.965,
        "(b)",
        transform=axis.transAxes,
        color=NAVY,
        fontweight="bold",
        fontsize=7.4,
        ha="left",
        va="bottom",
    )
    axis.text(
        0.0,
        0.015,
        r"$\epsilon_g=|\mathcal{J}_{\mathrm{m},2N-1}-\mathcal{J}_{\mathrm{m},N}|/"
        r"\max(|\mathcal{J}_{\mathrm{m},2N-1}|,y_{\rm floor})$; P/F at $1\%$. "
        r"$\epsilon_J/J_{\rm tol},\epsilon_u/u_{\rm tol},\epsilon_v/v_{\rm tol}<1$: stable.\n"
        r"$n_{\rm mol}^{\star}$: $10^{-21}\,\mathrm{m}^{-3}$; "
        r"$\dot\Gamma,\dot\nu$: $\mathrm{MHz}\,\mu\mathrm{s}^{-1}$; "
        r"$\ddot\Gamma,\ddot\nu$: $\mathrm{MHz}\,\mu\mathrm{s}^{-2}$.",
        transform=axis.transAxes,
        fontsize=3.95,
        linespacing=1.15,
        color=GREY,
        ha="left",
        va="bottom",
    )


def _common_physical_horizon_microseconds(data: ShowcaseData) -> float:
    horizons = [
        float(target.row["t_interval"]) * T_STAR_S / TIME_DISPLAY_S
        for target in data.targets.values()
    ]
    if not np.allclose(horizons, horizons[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("selected controls do not share one physical time horizon")
    return horizons[0]


def _draw_control_group(
    axis: plt.Axes,
    data: ShowcaseData,
    group_name: str,
    caps: tuple[int, ...],
    index: int,
) -> None:
    horizons: list[float] = []
    for cap in caps:
        target = data.control_targets[cap]
        u = np.asarray(target.controls["u"], dtype=float)
        v = np.asarray(target.controls["v"], dtype=float)
        if u.shape != v.shape:
            raise RuntimeError(
                f"u and v controls have different shapes for cap {target.cap}"
            )
        horizon = float(target.row["t_interval"])
        horizons.append(horizon)
        time_us = np.linspace(
            0.0, horizon * T_STAR_S / TIME_DISPLAY_S, u.size
        )
        colour = CAP_COLOURS[cap]
        gamma_mhz = np.asarray(_frequency_mhz(u), dtype=float)
        gamma_max_mhz = float(_frequency_mhz(cap))
        highlighted = cap in SELECTED_CAPS
        axis.plot(
            time_us,
            gamma_mhz / gamma_max_mhz,
            color=colour,
            lw=1.30 if highlighted else 0.78,
            alpha=1.0 if highlighted else 0.30,
            label=f"{gamma_max_mhz:g}",
            zorder=4 if highlighted else 2,
        )

    if not np.allclose(horizons, horizons[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"{group_name} controls do not share one time horizon")

    horizon_us = horizons[0] * T_STAR_S / TIME_DISPLAY_S
    axis.set_xlim(0.0, horizon_us)
    axis.set_ylim(0.0, 1.06)
    axis.set_xticks([0.0, horizon_us / 2.0, horizon_us])
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_xlabel(r"$t\;(\mu\mathrm{s})$", labelpad=2)
    if index == 0:
        axis.set_ylabel(r"$\Gamma/\Gamma_{\max}$", labelpad=2)
    axis.text(
        0.98,
        0.96,
        group_name,
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=NAVY,
        fontsize=6.0,
        fontweight="bold",
    )
    axis.legend(
        title=r"$\Gamma_{\max}\;(\mathrm{MHz})$",
        loc="lower left",
        ncols=min(3, len(caps)),
        frameon=False,
        fontsize=4.5,
        title_fontsize=4.7,
        handlelength=1.25,
        columnspacing=0.55,
        labelspacing=0.15,
        borderaxespad=0.25,
    )
    _style_axis(axis, grid="major")
    if index == 0:
        axis.text(
            -0.015,
            1.035,
            "(c)",
            transform=axis.transAxes,
            color=NAVY,
            fontweight="bold",
            fontsize=7.4,
            ha="left",
            va="bottom",
        )


def build_figure(data: ShowcaseData) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.30, 3.52))
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.32, 0.78],
        hspace=0.34,
        left=0.062,
        right=0.992,
        bottom=0.145,
        top=0.965,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.92, 1.30], wspace=0.24)
    bottom = outer[1].subgridspec(1, 3, wspace=0.22)

    cap_axis = figure.add_subplot(top[0])
    table_axis = figure.add_subplot(top[1])
    _draw_cap_sweep(cap_axis, data)
    _draw_cap_table(table_axis, data)

    control_axes = [figure.add_subplot(bottom[index]) for index in range(3)]
    for index, (group_name, caps) in enumerate(CONTROL_CAP_GROUPS.items()):
        _draw_control_group(
            control_axes[index], data, group_name, caps, index
        )
    snapshot = datetime.fromisoformat(data.generated_utc).strftime("%Y-%m-%d %H:%M UTC")
    figure.text(
        0.992,
        0.018,
        f"Data snapshot {snapshot}. Density remains a counting-convention-dependent proxy.",
        ha="right",
        va="bottom",
        fontsize=5.35,
        color=GREY,
    )
    return figure


def _source_record(execution: Execution) -> dict[str, Any]:
    return {
        "database": _display_path(execution.database),
        "config_name": execution.summary["config_name"],
        "config_document_id": int(execution.summary["config_document_id"]),
        "queue_id": int(execution.summary["queue_id"]),
        "status": execution.summary["status"],
        "run_count": len(execution.rows),
    }


def metadata_record(data: ShowcaseData) -> dict[str, Any]:
    baseline_stats = {}
    for cap, rows in sorted(data.baseline_by_cap.items()):
        values = _objective_values(rows)
        baseline_stats[str(cap)] = {
            "run_count": int(values.size),
            "minimum": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25, method="linear")),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75, method="linear")),
            "maximum": float(np.max(values)),
            "molecular_density_m^-3": {
                "minimum": float(_density_m3(np.min(values))),
                "q25": float(
                    _density_m3(np.quantile(values, 0.25, method="linear"))
                ),
                "median": float(_density_m3(np.median(values))),
                "q75": float(
                    _density_m3(np.quantile(values, 0.75, method="linear"))
                ),
                "maximum": float(_density_m3(np.max(values))),
            },
        }

    target_records = {}
    for cap, target in data.targets.items():
        target_records[str(cap)] = {
            "database": _display_path(target.database),
            "run_id": int(target.row["run_id"]),
            "config_name": target.row["config_name"],
            "config_document_id": int(target.row["config_document_id"]),
            "queue_id": int(target.row["queue_id"]),
            "status": target.row["status"],
            "termination_reason": target.row.get("termination_reason"),
            "best_objective": float(target.row["best_objective"]),
            "molecular_density_m^-3": float(
                _density_m3(float(target.row["best_objective"]))
            ),
            "Gamma_max_hz": float(target.cap * GAMMA_HZ),
            "time_horizon_s": float(target.row["t_interval"]) * T_STAR_S,
            "best_score": float(target.row["best_score"]),
            "best_penalty": float(target.row["best_penalty"]),
            "completed_steps": int(target.row["completed_steps"]),
            "strict_status_counts": {
                status: sum(row["status"] == status for row in target.strict_rows)
                for status in sorted({row["status"] for row in target.strict_rows})
            },
            "strict_termination_counts": {
                str(reason): sum(
                    row.get("termination_reason") == reason for row in target.strict_rows
                )
                for reason in sorted(
                    {row.get("termination_reason") for row in target.strict_rows},
                    key=str,
                )
            },
        }

    control_records = {}
    gamma_mhz = GAMMA_HZ / FREQUENCY_DISPLAY_HZ
    t_star_us = T_STAR_S / TIME_DISPLAY_S
    for group_name, caps in CONTROL_CAP_GROUPS.items():
        for cap in caps:
            target = data.control_targets[cap]
            row = target.row
            control_records[str(cap)] = {
                "regime": group_name,
                "colour": CAP_COLOURS[cap],
                "database": _display_path(target.database),
                "run_id": int(target.row["run_id"]),
                "config_name": target.row["config_name"],
                "config_document_id": int(target.row["config_document_id"]),
                "queue_id": int(target.row["queue_id"]),
                "selection": "maximum finite best_objective at this cap",
                "best_score": float(row["best_score"]),
                "best_objective": float(row["best_objective"]),
                "best_density_proxy_m^-3": float(
                    _density_m3(float(row["best_objective"]))
                ),
                "N": int(row["N"]),
                "grid_refinement": {
                    "refined_N": int(row["best_grid_refinement_refined_N"]),
                    "relative_error": float(
                        row["best_grid_refinement_relative_error"]
                    ),
                    "tolerance": float(row["best_grid_refinement_tolerance"]),
                    "passed": bool(row["best_grid_refinement_passed"]),
                },
                "terminal_stability_ratios": {
                    "epsilon_J_over_J_tol": _terminal_tolerance_ratio(
                        target, "score_tolerance", "J_tol"
                    ),
                    "epsilon_u_over_u_tol": _terminal_tolerance_ratio(
                        target, "u_tolerance", "u_tol"
                    ),
                    "epsilon_v_over_v_tol": _terminal_tolerance_ratio(
                        target, "v_tolerance", "v_tol"
                    ),
                },
                "dimensional_control_derivatives": {
                    "max_abs_dGamma_dt_MHz_per_us": float(
                        row["best_max_abs_du_dt"]
                    )
                    * gamma_mhz
                    / t_star_us,
                    "max_abs_dnu_dt_MHz_per_us": float(
                        row["best_max_abs_dv_dt"]
                    )
                    * gamma_mhz
                    / t_star_us,
                    "max_abs_d2Gamma_dt2_MHz_per_us2": float(
                        row["best_max_abs_d2u_dt2"]
                    )
                    * gamma_mhz
                    / t_star_us**2,
                    "max_abs_d2nu_dt2_MHz_per_us2": float(
                        row["best_max_abs_d2v_dt2"]
                    )
                    * gamma_mhz
                    / t_star_us**2,
                },
            }

    return {
        "generated_utc": data.generated_utc,
        "objective_label": "dimensionless molecular objective J_m",
        "plotted_quantity": (
            "dimensional density proxy n_mol, dimensional cap Gamma_max and "
            "time t, plus the explicitly requested ratio Gamma/Gamma_max"
        ),
        "dimensionalisation": {
            "t_star_s": T_STAR_S,
            "bohr_radius_m": BOHR_RADIUS_M,
            "a_bg_m": A_BG_M,
            "atom_mass_kg": ATOM_MASS_KG,
            "gamma_hz": GAMMA_HZ,
            "g_2_initial_m^-6": G2_INITIAL_M6,
            "l_star_m": L_STAR_M,
            "r_bg": R_BG,
            "molecular_density_per_objective_m^-3": (
                MOLECULAR_DENSITY_PER_OBJECTIVE_M3
            ),
            "frequency_convention": (
                "gamma, Gamma, and nu are treated in the same Hz units; no 2*pi "
                "factor is introduced"
            ),
            "g_2_convention": (
                "unnormalised initial equal-position pair density, not "
                "dimensionless g^(2)(0)"
            ),
        },
        "baseline_source": _source_record(data.baseline),
        "baseline_spread_method": {
            "population": "all finite fixed-method runs at each cap",
            "plotted_spread": "thin whisker from minimum to maximum",
            "plotted_centre": "short thin line at the median",
            "quartile_statistics": (
                "q25 and q75 remain tabulated using "
                "numpy.quantile(method='linear') but are not drawn"
            ),
            "outlined_point": "run with the maximum finite best_objective",
        },
        "baseline_statistics": baseline_stats,
        "grid_refinement": {
            "method": (
                "linear interpolation of each plotted best fixed-method N=100 "
                "control followed by forward evaluation; no reoptimisation"
            ),
            "N": list(GRID_REFINEMENT_N),
            "results": {
                str(cap): {
                    str(N): {
                        "objective": float(value),
                        "molecular_density_m^-3": float(_density_m3(value)),
                    }
                    for N, value in sorted(values.items())
                }
                for cap, values in data.control_grid_refinement.items()
            },
        },
        "targeted_grid_refinement": {
            "method": (
                "linear interpolation of the separately targeted N=100 controls "
                "followed by forward evaluation; no reoptimisation"
            ),
            "N": list(GRID_REFINEMENT_N),
            "results": {
                str(cap): {
                    str(N): {
                        "objective": float(value),
                        "molecular_density_m^-3": float(_density_m3(value)),
                    }
                    for N, value in sorted(values.items())
                }
                for cap, values in data.grid_refinement.items()
            },
        },
        "plotted_control_results": control_records,
        "target_results": target_records,
    }


def write_macros(path: Path, data: ShowcaseData) -> None:
    strict_rows = [
        row for target in data.targets.values() for row in target.strict_rows
    ]
    strict_complete = sum(
        row["status"] == "complete" and row.get("termination_reason") == "stability"
        for row in strict_rows
    )
    target_values = {
        cap: float(data.targets[cap].row["best_objective"]) for cap in SELECTED_CAPS
    }
    grid_change_percent = {
        cap: 100.0
        * (
            data.grid_refinement[cap][GRID_REFINEMENT_N[-1]]
            / data.grid_refinement[cap][GRID_REFINEMENT_N[0]]
            - 1.0
        )
        for cap in SELECTED_CAPS
    }
    high_cap_fine_grid_change_percent = 100.0 * (
        data.grid_refinement[1280][800] / data.grid_refinement[1280][400] - 1.0
    )
    high_cap_steps = int(data.targets[1280].row["completed_steps"])
    baseline_count = len(data.baseline.rows)
    starts_per_cap = min(len(rows) for rows in data.baseline_by_cap.values())
    displayed_count = baseline_count
    baseline_1280 = _objective_values(data.baseline_by_cap[1280])
    baseline_320 = _objective_values(data.baseline_by_cap[320])
    baseline_2560 = _objective_values(data.baseline_by_cap[2560])
    horizon_us = _common_physical_horizon_microseconds(data)
    lines = [
        "% Generated by reports/scripts/generate_showcase_results.py; do not edit.",
        rf"\newcommand{{\ShowcaseSweepRuns}}{{{baseline_count}}}",
        rf"\newcommand{{\ShowcaseSweepCaps}}{{{len(data.baseline_by_cap)}}}",
        rf"\newcommand{{\ShowcaseStartsPerCap}}{{{starts_per_cap}}}",
        rf"\newcommand{{\ShowcaseDisplayedRuns}}{{{displayed_count}}}",
        rf"\newcommand{{\ShowcaseDisplayedCaps}}{{{len(data.baseline_by_cap)}}}",
        rf"\newcommand{{\ShowcaseFortyObjective}}{{{_format_objective(target_values[40])}}}",
        rf"\newcommand{{\ShowcaseThreeTwentyObjective}}{{{_format_objective(target_values[320])}}}",
        rf"\newcommand{{\ShowcaseTwelveEightyObjective}}{{{_format_objective(target_values[1280])}}}",
        rf"\newcommand{{\ShowcaseFortyDensityScaled}}{{{_format_density_display(target_values[40])}}}",
        rf"\newcommand{{\ShowcaseThreeTwentyDensityScaled}}{{{_format_density_display(target_values[320])}}}",
        rf"\newcommand{{\ShowcaseTwelveEightyDensityScaled}}{{{_format_density_display(target_values[1280])}}}",
        rf"\newcommand{{\ShowcaseFortyGridChangePercent}}{{{grid_change_percent[40]:+.2f}}}",
        rf"\newcommand{{\ShowcaseThreeTwentyGridChangePercent}}{{{grid_change_percent[320]:+.2f}}}",
        rf"\newcommand{{\ShowcaseTwelveEightyGridChangePercent}}{{{grid_change_percent[1280]:+.2f}}}",
        rf"\newcommand{{\ShowcaseTwelveEightyFineGridChangePercent}}{{{high_cap_fine_grid_change_percent:+.2f}}}",
        rf"\newcommand{{\ShowcaseLengthScaleNm}}{{{L_STAR_M * 1e9:.4f}}}",
        rf"\newcommand{{\ShowcasePhysicalHorizonMicroseconds}}{{{horizon_us:.1f}}}",
        rf"\newcommand{{\ShowcaseFortyCapMHz}}{{{float(_frequency_mhz(40)):g}}}",
        rf"\newcommand{{\ShowcaseThreeTwentyCapMHz}}{{{float(_frequency_mhz(320)):g}}}",
        rf"\newcommand{{\ShowcaseTwelveEightyCapMHz}}{{{float(_frequency_mhz(1280)):g}}}",
        rf"\newcommand{{\ShowcaseHighCapSteps}}{{{high_cap_steps:,}}}",
        rf"\newcommand{{\ShowcaseStrictComplete}}{{{strict_complete}/{len(strict_rows)}}}",
        rf"\newcommand{{\ShowcaseBaselineTwelveEightyMedian}}{{{np.median(baseline_1280):.2f}}}",
        rf"\newcommand{{\ShowcaseBaselineTwelveEightyBest}}{{{np.max(baseline_1280):,.0f}}}",
        rf"\newcommand{{\ShowcaseBaselineThreeTwentyMedian}}{{{np.median(baseline_320):.2f}}}",
        rf"\newcommand{{\ShowcaseBaselineThreeTwentyBest}}{{{np.max(baseline_320):,.0f}}}",
        rf"\newcommand{{\ShowcaseBaselineTwentyFiveSixtyMedian}}{{{np.median(baseline_2560):.6f}}}",
        rf"\newcommand{{\ShowcaseBaselineTwelveEightyMedianDensityScaled}}{{{_format_density_display(float(np.median(baseline_1280)))}}}",
        rf"\newcommand{{\ShowcaseBaselineTwelveEightyBestDensityScaled}}{{{_format_density_display(float(np.max(baseline_1280)))}}}",
        rf"\newcommand{{\ShowcaseBaselineTwentyFiveSixtyMedianDensityScaled}}{{{_format_density_display(float(np.median(baseline_2560)))}}}",
        rf"\newcommand{{\ShowcaseBaselineTwentyFiveSixtyMedianDensity}}{{{_format_scientific_latex(float(_density_m3(np.median(baseline_2560))))}}}",
        rf"\newcommand{{\ShowcaseSnapshotDate}}{{{datetime.fromisoformat(data.generated_utc).strftime('%-d %B %Y')} UTC}}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--u40-database", type=Path, default=DEFAULT_TARGET_DATABASES[40])
    parser.add_argument("--u320-database", type=Path, default=DEFAULT_TARGET_DATABASES[320])
    parser.add_argument("--u1280-database", type=Path, default=DEFAULT_TARGET_DATABASES[1280])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--macros", type=Path, default=DEFAULT_MACROS)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_databases = {
        40: args.u40_database.expanduser().resolve(),
        320: args.u320_database.expanduser().resolve(),
        1280: args.u1280_database.expanduser().resolve(),
    }
    data = load_data(args.database.expanduser().resolve(), target_databases)

    output = args.output.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    macros = args.macros.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)

    figure = build_figure(data)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    metadata.write_text(json.dumps(metadata_record(data), indent=2) + "\n", encoding="utf-8")
    write_macros(macros, data)
    print(f"Wrote {output}")
    print(f"Wrote {metadata}")
    print(f"Wrote {macros}")


if __name__ == "__main__":
    main()
