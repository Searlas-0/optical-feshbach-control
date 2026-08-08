#!/usr/bin/env python3
"""Generate the provisional-results graphic used on showcase page 2.

The figure reads the completed default multi-start run from the archived
SQLite database.  It deliberately reports the optimizer's dimensionless
objective, not an absolute molecule count.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sqlite3
import zlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = REPORT_DIR.parent / "results" / "results.sqlite3"
DEFAULT_OUTPUT = REPORT_DIR / "figures" / "showcase_provisional_results.pdf"


def decode_array(blob: bytes) -> np.ndarray:
    """Decode the compressed NumPy-array format used by the run database."""

    return np.load(BytesIO(zlib.decompress(blob)), allow_pickle=False)


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a database snapshot without creating journals beside it."""

    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_history(connection: sqlite3.Connection, run_id: int) -> np.ndarray:
    rows = connection.execute(
        """SELECT objective_blob
           FROM history_chunks
           WHERE run_id=?
           ORDER BY stage_index""",
        (run_id,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"run {run_id} has no optimization history")
    return np.concatenate([decode_array(row["objective_blob"]) for row in rows])


def load_controls(connection: sqlite3.Connection, run_id: int) -> dict[str, np.ndarray]:
    rows = connection.execute(
        """SELECT name, values_blob
           FROM controls
           WHERE run_id=? AND kind='best'""",
        (run_id,),
    ).fetchall()
    controls = {row["name"]: decode_array(row["values_blob"]) for row in rows}
    if set(controls) != {"u", "v"}:
        raise RuntimeError(f"run {run_id} does not contain both best controls")
    return controls


def load_completed_default_runs(
    connection: sqlite3.Connection,
) -> tuple[list[sqlite3.Row], dict[int, np.ndarray]]:
    rows = connection.execute(
        """SELECT r.run_id, r.best_objective, r.best_score,
                  json_extract(c.parameters_json, '$.N') AS N,
                  json_extract(c.parameters_json, '$.u_max') AS u_max,
                  json_extract(c.parameters_json, '$.v_max') AS v_max,
                  r.completed_steps
           FROM runs r
           JOIN cases c ON c.case_id=r.case_id
           JOIN batches b ON b.batch_id=r.batch_id
           JOIN configs g ON g.config_id=b.config_id
           WHERE r.status='complete' AND g.config_name='default_config'
           ORDER BY r.run_id"""
    ).fetchall()
    if not rows:
        raise RuntimeError("the archive contains no completed default-config runs")
    histories = {int(row["run_id"]): load_history(connection, int(row["run_id"])) for row in rows}
    return rows, histories


def padded_history_matrix(histories: list[np.ndarray]) -> np.ndarray:
    width = max(len(history) for history in histories)
    matrix = np.full((len(histories), width), np.nan)
    for index, history in enumerate(histories):
        matrix[index, : len(history)] = history
        if len(history) < width:
            matrix[index, len(history) :] = history[-1]
    return matrix


def build_figure(database: Path) -> plt.Figure:
    with connect_read_only(database) as connection:
        rows, history_by_run = load_completed_default_runs(connection)
        best_row = max(rows, key=lambda row: float(row["best_objective"]))
        controls = load_controls(connection, int(best_row["run_id"]))

    objectives = np.asarray([float(row["best_objective"]) for row in rows])
    best_objective = float(np.max(objectives))
    shortfall_ppm = 1.0e6 * (best_objective - objectives) / abs(best_objective)
    histories = [history_by_run[int(row["run_id"])] for row in rows]
    history_matrix = padded_history_matrix(histories)

    navy = "#143D66"
    blue = "#2474A6"
    cyan = "#69AEC9"
    red = "#B6403A"
    gold = "#D49A22"
    grey = "#626B76"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.38),
        gridspec_kw={"width_ratios": [1.05, 1.2, 0.82], "wspace": 0.34},
    )

    # (a) Best normalized physical controls.
    axis = axes[0]
    time = np.linspace(0.0, 1.0, len(controls["u"]))
    u_max = float(best_row["u_max"])
    v_max = float(best_row["v_max"])
    u_normalized = np.asarray(controls["u"], dtype=float) / u_max
    v_normalized = np.asarray(controls["v"], dtype=float) / v_max
    axis.fill_between(time, 0.0, u_normalized, color=cyan, alpha=0.20, lw=0)
    axis.plot(time, u_normalized, color=blue, lw=1.8, label=r"intensity $u/u_{\max}$")
    axis.plot(time, v_normalized, color=red, lw=1.8, label=r"detuning $v/v_{\max}$")
    axis.axhline(0.0, color="#8A929C", lw=0.6)
    axis.set(xlim=(0, 1), ylim=(-1.07, 1.07), xlabel=r"normalized time $t/T$")
    axis.set_ylabel("normalized control", labelpad=2)
    axis.set_title("(a) Best laser protocol", loc="left", color=navy, fontweight="bold")
    axis.legend(loc="center right", frameon=False, handlelength=1.7)
    axis.set_xticks([0, 0.5, 1])
    axis.set_yticks([-1, 0, 1])

    # (b) Objective histories for all 40 starts.
    axis = axes[1]
    steps = np.arange(history_matrix.shape[1]) / 1000.0
    for history in history_matrix:
        axis.plot(steps, history / best_objective, color=cyan, alpha=0.20, lw=0.55)
    median_history = np.nanmedian(history_matrix, axis=0)
    best_history = history_by_run[int(best_row["run_id"])]
    axis.plot(steps, median_history / best_objective, color=navy, lw=1.8, label="median")
    axis.plot(
        np.arange(len(best_history)) / 1000.0,
        best_history / best_objective,
        color=gold,
        lw=1.25,
        label="best endpoint",
    )
    for boundary in (5, 10, 17.5):
        axis.axvline(boundary, color="#9AA1A9", ls=(0, (2, 2)), lw=0.6)
    axis.set(xlim=(0, 25), ylim=(-0.03, 1.025), xlabel="optimization steps (thousands)")
    axis.set_ylabel(r"objective $\mathcal{J}_{\rm m}/\mathcal{J}_{\rm best}$", labelpad=2)
    axis.set_title("(b) 40 independent starts", loc="left", color=navy, fontweight="bold")
    axis.legend(loc="lower right", frameon=False, handlelength=1.7)
    axis.set_xticks([0, 5, 10, 15, 20, 25])
    axis.set_yticks([0, 0.5, 1])

    # (c) Endpoint spread in parts per million avoids implying false precision
    # in the still-unvalidated absolute normalization.
    axis = axes[2]
    ordered = np.sort(shortfall_ppm)
    jitter = 0.06 * np.sin(np.arange(len(ordered)) * 2.399963)
    axis.scatter(ordered, jitter, s=17, color=blue, alpha=0.80, edgecolor="white", linewidth=0.35)
    median_shortfall = float(np.median(shortfall_ppm))
    axis.axvline(median_shortfall, color=red, lw=1.3, label=f"median {median_shortfall:.1f} ppm")
    axis.axvline(0.0, color=navy, lw=0.7)
    axis.set_ylim(-0.20, 0.20)
    axis.set_xlim(-0.8, max(25.0, float(np.max(shortfall_ppm)) + 1.0))
    axis.set_yticks([])
    axis.set_xlabel("shortfall from best [ppm]")
    axis.set_title("(c) Endpoint agreement", loc="left", color=navy, fontweight="bold")
    axis.legend(loc="upper right", frameon=False, handlelength=1.4)
    axis.text(
        0.03,
        0.10,
        f"{len(rows)}/{len(rows)} complete\nrange {np.ptp(shortfall_ppm):.1f} ppm",
        transform=axis.transAxes,
        fontsize=7.0,
        color=grey,
        va="bottom",
    )

    for axis in axes:
        axis.grid(color="#DDE2E7", lw=0.45, alpha=0.75)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#8A929C")
            spine.set_linewidth(0.65)

    figure.text(
        0.5,
        0.012,
        rf"Archived provisional run: $N={int(best_row['N'])}$, "
        rf"$u_{{\max}}=v_{{\max}}={u_max:g}$, "
        rf"{int(best_row['completed_steps']):,} Adam steps; objective normalization remains under validation.",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color=grey,
    )
    figure.subplots_adjust(left=0.062, right=0.995, bottom=0.22, top=0.91)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = args.database.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"result database not found: {database}")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = build_figure(database)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
