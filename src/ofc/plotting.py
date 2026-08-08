"""The standard three-figure view for completed optimization results.

Isolation boundary: plotting accepts already-retrieved result mappings and
never opens a database, loads a config, or launches calculations.  The visual
structure and styling mirror the old codebase's ``plot_run()`` output:
convergence, yield distribution, and optimized control overlays.

Selections containing only different initializations retain that view. When
one configuration parameter varies, the plots switch to a colour-keyed sweep
view: Figures 1 and 3 use the run with the highest regularized score at each
value, while Figure 2 scatters every initialization's objective recorded at
its own best-score checkpoint against the swept value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ofc-matplotlib"))


SWEEP_PARAMETERS = (
    "N",
    "t_interval",
    "r_bg",
    "u_isbound",
    "v_isbound",
    "u_max",
    "v_max",
    "slew_limit",
    "optimizer",
    "schedule",
    "adam_learning_rate",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "lbfgs_history_size",
    "lbfgs_max_linesearch_steps",
    "lbfgs_tolerance",
    "smoothness",
    "u_smooth",
    "v_smooth",
    "sharpness",
    "u_sharp",
    "v_sharp",
    "block_size",
    "J_tol",
    "u_tol",
    "v_tol",
)

SWEEP_LABELS = {
    "N": r"$N$",
    "t_interval": r"$T$",
    "r_bg": r"$r_{\mathrm{bg}}$",
    "u_isbound": r"bounded $u$",
    "v_isbound": r"bounded $\nu$",
    "u_max": r"$u_{\max}$",
    "v_max": r"$\nu_{\max}$",
    "slew_limit": r"$t_{\mathrm{slew}}/T$",
    "optimizer": "optimizer",
    "schedule": "learning-rate schedule",
    "adam_learning_rate": r"$\alpha_I$",
    "adam_beta1": r"$\beta_1$",
    "adam_beta2": r"$\beta_2$",
    "adam_eps": r"$\epsilon_{\mathrm{Adam}}$",
    "lbfgs_history_size": "L-BFGS history size",
    "lbfgs_max_linesearch_steps": "L-BFGS max line-search steps",
    "lbfgs_tolerance": "L-BFGS gradient tolerance",
    "smoothness": r"$\lambda$",
    "u_smooth": r"$\lambda_u$",
    "v_smooth": r"$\lambda_\nu$",
    "sharpness": r"$\kappa$",
    "u_sharp": r"$\kappa_u$",
    "v_sharp": r"$\kappa_\nu$",
    "block_size": "checkpoint block size",
    "J_tol": r"$J_{\mathrm{tol}}$",
    "u_tol": r"$u_{\mathrm{tol}}$",
    "v_tol": r"$\nu_{\mathrm{tol}}$",
}


@dataclass(frozen=True)
class SweepSpec:
    """One explicitly selected or unambiguously inferred parameter sweep."""

    name: str
    keys: tuple
    display_values: tuple
    numeric_values: tuple[float, ...] | None
    scale: str
    linthresh: float | None = None


def _plot_modules():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    return plt, Line2D, AnchoredOffsetbox, HPacker, TextArea, VPacker, Patch, FuncFormatter


def _display_number(value, precision=8, scientific_below=1e-3):
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0.0"
    if abs(value) < scientific_below:
        exponent = int(math.floor(math.log10(abs(value))))
        coefficient = value / 10.0**exponent
        text = np.format_float_positional(
            coefficient,
            precision=max(1, precision),
            unique=False,
            fractional=False,
            trim="-",
        )
        return rf"${text}\times 10^{{{exponent}}}$"
    return np.format_float_positional(
        value,
        precision=precision,
        unique=False,
        fractional=False,
        trim="-",
    )


def _format_axis_tick(value, _position):
    rounded = round(float(value))
    if math.isclose(float(value), rounded, rel_tol=0.0, abs_tol=1e-10):
        return str(rounded)
    return _display_number(value, precision=3)


def _freeze_value(value):
    """Return a hashable, stable representation of a stored config value."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _sweep_value_label(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float, np.number)):
        return _display_number(value, precision=5).strip("$")
    return str(value)


def _geometrically_spaced(values):
    magnitudes = np.unique(np.abs(np.asarray(values, dtype=float)))
    magnitudes = magnitudes[magnitudes > 0.0]
    if magnitudes.size < 3:
        return False
    log_gaps = np.diff(np.log10(magnitudes))
    return bool(
        np.all(log_gaps > 0.0)
        and np.allclose(log_gaps, log_gaps[0], rtol=2e-3, atol=1e-10)
    )


def _make_sweep_spec(runs, name):
    if name not in SWEEP_PARAMETERS:
        raise ValueError(
            f"Unknown sweep parameter {name!r}; choose one of {SWEEP_PARAMETERS}."
        )
    if not runs or any(name not in run for run in runs):
        raise ValueError(f"Sweep parameter {name!r} is missing from one or more runs.")

    unique = {}
    for run in runs:
        value = run[name]
        unique.setdefault(_freeze_value(value), value)
    if len(unique) < 2:
        raise ValueError(
            f"Sweep parameter {name!r} has only one value in the selected runs."
        )

    numeric = all(
        isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        for value in unique.values()
    )
    if numeric:
        ordered = sorted(unique.items(), key=lambda item: float(item[1]))
        numbers = tuple(float(value) for _, value in ordered)
        if _geometrically_spaced(numbers):
            if all(value > 0.0 for value in numbers):
                scale, linthresh = "log", None
            else:
                nonzero = np.abs(np.asarray(numbers))[np.asarray(numbers) != 0.0]
                scale, linthresh = "symlog", float(np.min(nonzero) / 2.0)
        else:
            scale, linthresh = "linear", None
        return SweepSpec(
            name=name,
            keys=tuple(key for key, _ in ordered),
            display_values=tuple(value for _, value in ordered),
            numeric_values=numbers,
            scale=scale,
            linthresh=linthresh,
        )

    ordered = sorted(unique.items(), key=lambda item: _sweep_value_label(item[1]))
    return SweepSpec(
        name=name,
        keys=tuple(key for key, _ in ordered),
        display_values=tuple(value for _, value in ordered),
        numeric_values=None,
        scale="categorical",
    )


def _detect_sweep(runs, sweep_parameter=None):
    """Return no sweep for initialization-only data, otherwise one sweep spec."""

    if sweep_parameter is not None:
        return _make_sweep_spec(runs, str(sweep_parameter))
    varied = []
    for name in SWEEP_PARAMETERS:
        if not runs or any(name not in run for run in runs):
            continue
        if len({_freeze_value(run[name]) for run in runs}) > 1:
            varied.append(name)
    if not varied:
        return None
    if len(varied) > 1:
        raise ValueError(
            "The selected runs vary multiple configuration parameters: "
            + ", ".join(varied)
            + ". Filter the query to one sweep or pass --sweep-parameter NAME."
        )
    return _make_sweep_spec(runs, varied[0])


def _sweep_groups(runs, sweep):
    groups = {key: [] for key in sweep.keys}
    for run in runs:
        groups[_freeze_value(run[sweep.name])].append(run)
    return groups


def _best_sweep_runs(runs, sweep):
    selected = []
    for key, members in _sweep_groups(runs, sweep).items():
        finite = [run for run in members if np.isfinite(run["best_score"])]
        if finite:
            selected.append(max(finite, key=lambda run: run["best_score"]))
    return selected


def _sweep_visuals(sweep):
    plt, _, _, _, _, _, _, _ = _plot_modules()
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    colour_map = plt.get_cmap("viridis")
    if sweep.numeric_values is None:
        coordinates = np.arange(len(sweep.keys), dtype=float)
        normalisation = Normalize(vmin=-0.5, vmax=len(sweep.keys) - 0.5)
    else:
        coordinates = np.asarray(sweep.numeric_values, dtype=float)
        if sweep.scale == "log":
            normalisation = LogNorm(vmin=float(coordinates.min()), vmax=float(coordinates.max()))
        elif sweep.scale == "symlog":
            normalisation = SymLogNorm(
                linthresh=float(sweep.linthresh),
                vmin=float(coordinates.min()),
                vmax=float(coordinates.max()),
                base=10,
            )
        else:
            normalisation = Normalize(
                vmin=float(coordinates.min()), vmax=float(coordinates.max())
            )
    colours = {
        key: colour_map(normalisation(coordinate))
        for key, coordinate in zip(sweep.keys, coordinates)
    }
    mappable = plt.cm.ScalarMappable(norm=normalisation, cmap=colour_map)
    return colours, mappable, coordinates


def _add_sweep_colourbar(figure, axes, sweep, mappable, coordinates, **kwargs):
    colourbar = figure.colorbar(mappable, ax=axes, **kwargs)
    colourbar.set_label(SWEEP_LABELS.get(sweep.name, sweep.name.replace("_", " ")))
    if sweep.numeric_values is None or len(sweep.keys) <= 12:
        colourbar.set_ticks(coordinates)
        colourbar.set_ticklabels(
            [_sweep_value_label(value) for value in sweep.display_values]
        )
    return colourbar


def _nice_zero_based_ticks(data_max, target_intervals=6):
    data_max = float(data_max)
    if not np.isfinite(data_max) or data_max <= 0.0:
        data_max = 1.0
    raw_step = data_max / target_intervals
    exponent = int(math.floor(math.log10(raw_step)))
    candidates = sorted(
        {
            multiplier * 10.0**candidate_exponent
            for candidate_exponent in range(exponent - 1, exponent + 3)
            for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0)
        }
    )

    def layout(step):
        intervals = max(1, int(math.ceil(data_max / step - 1e-12)))
        return (
            max(5 - intervals, 0, intervals - 8),
            abs(intervals - target_intervals),
            -step,
            intervals,
        )

    step = min(candidates, key=layout)
    intervals = layout(step)[-1]
    if math.isclose(intervals * step, data_max, rel_tol=1e-12, abs_tol=1e-12):
        intervals += 1
    return step * np.arange(intervals + 1, dtype=float), step


def _history_envelope(histories):
    histories = [
        np.asarray(history, dtype=float).reshape(-1)
        for history in histories
        if np.asarray(history).size
    ]
    if not histories:
        return None
    width = max(map(len, histories))
    matrix = np.full((len(histories), width), np.nan, dtype=float)
    for row, history in enumerate(histories):
        matrix[row, : len(history)] = history
        finite = np.flatnonzero(np.isfinite(history))
        if finite.size:
            final_index = int(finite[-1])
            matrix[row, final_index + 1 :] = history[final_index]
    return (
        histories,
        np.nanpercentile(matrix, 10, axis=0),
        np.nanpercentile(matrix, 90, axis=0),
        np.nanmedian(matrix, axis=0),
    )


def _relative_mean_change(current, previous=None, eps=1e-12):
    current_mean = float(np.mean(current))
    previous_mean = float(current[0] if previous is None else np.mean(previous))
    return abs(current_mean - previous_mean) / max(abs(current_mean), eps)


def _checkpoint_observations(run, history_name):
    history = np.asarray(run["history"][history_name], dtype=float)
    steps = np.asarray(run["tolerances"]["step"], dtype=int)
    steps = steps[(steps > 0) & (steps < len(history))]
    if history_name == "score" and len(steps):
        values = np.asarray(run["tolerances"]["score_tolerance"], dtype=float)
        return list(zip(steps, values[: len(steps)]))
    observations = []
    previous = None
    start = 0
    for step in steps:
        current = history[start : step + 1]
        observations.append((int(step), _relative_mean_change(current, previous)))
        previous = current
        start = int(step)
    return observations


def _history_stable_at_end(run, history_name):
    observations = _checkpoint_observations(run, history_name)
    if not observations:
        return False
    return bool(observations[-1][1] < float(run["J_tol"]))


def _control_metric(run, name):
    values = np.asarray(run["tolerances"][f"{name}_tolerance"], dtype=float)
    return float(values[-1]) if values.size else math.nan


def _control_stable(run, name):
    tolerance = run[f"{name}_tol"]
    value = _control_metric(run, name)
    return tolerance is None or (np.isfinite(value) and value < float(tolerance))


def _stability_fraction_text(runs, predicate):
    lines = []
    grid_sizes = sorted({int(run["N"]) for run in runs})
    for N in grid_sizes:
        members = [run for run in runs if int(run["N"]) == N]
        stable = sum(bool(predicate(run)) for run in members)
        lines.append(f"$N={N}$: {stable}/{len(members)} stable")
    return "\n".join(lines) if lines else "0/0 stable"


def _add_stability_fraction(axis, runs, predicate):
    axis.text(
        0.015,
        0.975,
        _stability_fraction_text(runs, predicate),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        linespacing=1.3,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#d2d2ce",
            "alpha": 0.92,
        },
        zorder=8,
    )


def _draw_checkpoint_history(axis, run, history_name, width, *, best=False):
    history = np.asarray(run["history"][history_name], dtype=float)
    observations = _checkpoint_observations(run, history_name)

    def draw(start, end, stable):
        if end < start:
            return
        colour = ("#237a4b" if best else "#4b9668") if stable else (
            "#b33f36" if best else "#c45d55"
        )
        axis.plot(
            np.arange(start, end + 1),
            history[start : end + 1],
            color=colour,
            alpha=1.0 if best else (0.45 if stable else 0.55),
            linewidth=1.2 if best else 0.8,
            zorder=4 if best else 1,
        )

    start = 0
    stable = False
    for boundary, metric in observations:
        draw(start, boundary, stable)
        start = boundary
        stable = bool(np.isfinite(metric) and metric < float(run["J_tol"]))
    draw(start, len(history) - 1, stable)
    if len(history) < width:
        axis.plot(
            np.arange(len(history) - 1, width),
            np.full(width - len(history) + 1, history[-1]),
            color=("#237a4b" if best else "#4b9668") if stable else (
                "#b33f36" if best else "#c45d55"
            ),
            alpha=1.0 if best else 0.45,
            linewidth=1.2 if best else 0.8,
            linestyle="--" if best else "-",
        )


def _distinct_display(runs, name, *, optional=False):
    values = []
    for run in runs:
        value = run[name]
        displayed = (
            "none"
            if optional and value is None
            else str(value)
            if isinstance(value, str)
            else _display_number(value, 3)
        )
        if displayed not in values:
            values.append(displayed)
    return " / ".join(values)


def _configuration_lines(runs, *, exclude=()):
    excluded = set(exclude)
    smoothness = []
    sharpness = []
    for run in runs:
        u_value = run["smoothness"] if run["u_smooth"] is None else run["u_smooth"]
        v_value = run["smoothness"] if run["v_smooth"] is None else run["v_smooth"]
        value = (
            _display_number(u_value, 3)
            if math.isclose(float(u_value), float(v_value))
            else rf"$\lambda_u$={_display_number(u_value, 3)}, $\lambda_\nu$={_display_number(v_value, 3)}"
        )
        if value not in smoothness:
            smoothness.append(value)
        u_value = run["sharpness"] if run["u_sharp"] is None else run["u_sharp"]
        v_value = run["sharpness"] if run["v_sharp"] is None else run["v_sharp"]
        value = (
            _display_number(u_value, 3)
            if math.isclose(float(u_value), float(v_value))
            else rf"$\kappa_u$={_display_number(u_value, 3)}, $\kappa_\nu$={_display_number(v_value, 3)}"
        )
        if value not in sharpness:
            sharpness.append(value)
    entries = [
        ({"t_interval"}, rf"$T$ = {_distinct_display(runs, 't_interval')}"),
        ({"r_bg"}, rf"$r_{{\mathrm{{bg}}}}$ = {_distinct_display(runs, 'r_bg')}"),
        ({"N"}, rf"$N$ = {_distinct_display(runs, 'N')}"),
        (
            {"slew_limit", "t_interval"},
            rf"$t_{{\mathrm{{slew}}}}$ = "
            + " / ".join(
                dict.fromkeys(
                    _display_number(run["slew_limit"] * run["t_interval"], 3)
                    for run in runs
                )
            ),
        ),
        ({"smoothness", "u_smooth", "v_smooth"}, rf"$\lambda$ = {' / '.join(smoothness)}"),
        ({"sharpness", "u_sharp", "v_sharp"}, rf"$\kappa$ = {' / '.join(sharpness)}"),
        ({"optimizer"}, f"optimizer = {_distinct_display(runs, 'optimizer')}"),
        ({"u_max"}, rf"$u_{{\max}}$ = {_distinct_display(runs, 'u_max')}"),
        ({"v_max"}, rf"$\nu_{{\max}}$ = {_distinct_display(runs, 'v_max')}"),
        ({"J_tol"}, rf"$J_{{\mathrm{{tol}}}}$ = {_distinct_display(runs, 'J_tol')}"),
        ({"u_tol"}, rf"$u_{{\mathrm{{tol}}}}$ = {_distinct_display(runs, 'u_tol', optional=True)}"),
        ({"v_tol"}, rf"$\nu_{{\mathrm{{tol}}}}$ = {_distinct_display(runs, 'v_tol', optional=True)}"),
    ]
    optimizers = {run["optimizer"] for run in runs}
    if "adam" in optimizers:
        entries.insert(
            7,
            ({"adam_learning_rate"}, rf"$\alpha_I$ = {_distinct_display(runs, 'adam_learning_rate')}"),
        )
    if "lbfgs" in optimizers:
        entries.insert(
            7,
            (
                {"lbfgs_history_size", "lbfgs_max_linesearch_steps", "lbfgs_tolerance"},
                "L-BFGS = "
                + _distinct_display(runs, "lbfgs_history_size")
                + " history, "
                + _distinct_display(runs, "lbfgs_max_linesearch_steps")
                + " line search, tol "
                + _distinct_display(runs, "lbfgs_tolerance"),
            ),
        )
    return [line for dependencies, line in entries if dependencies.isdisjoint(excluded)]


def _add_configuration_box(figure, runs, *, y=0.985, exclude=()):
    _, _, AnchoredOffsetbox, HPacker, TextArea, VPacker, _, _ = _plot_modules()
    lines = _configuration_lines(runs, exclude=exclude)
    split = (len(lines) + 1) // 2
    title = TextArea(
        "Configuration:",
        textprops={"fontsize": 10.2, "fontweight": "semibold", "ha": "left"},
    )
    columns = HPacker(
        children=[
            TextArea(
                "\n".join(items),
                textprops={"fontsize": 9.0, "linespacing": 1.3, "ha": "left"},
            )
            for items in (lines[:split], lines[split:])
        ],
        align="top",
        pad=0,
        sep=36,
    )
    content = VPacker(children=[title, columns], align="left", pad=0, sep=5)
    box = AnchoredOffsetbox(
        loc="upper center",
        child=content,
        bbox_to_anchor=(0.5, y),
        bbox_transform=figure.transFigure,
        frameon=True,
        borderpad=0.0,
        pad=0.55,
    )
    box.patch.set_boxstyle("round,pad=0.35")
    box.patch.set_facecolor("#fbfbfa")
    box.patch.set_edgecolor("#c9c9c5")
    box.patch.set_alpha(0.97)
    figure.add_artist(box)
    figure._configuration_box = box


def _shared_exponent_ratio(value, tolerance):
    if value is None or not np.isfinite(value):
        return "not available"
    if tolerance is None:
        return f"{_display_number(value, 6)} (no limit)"
    exponent = int(math.floor(math.log10(abs(float(tolerance)))))
    scale = 10.0**exponent
    ratio = rf"${value / scale:.3g}\times10^{{{exponent}}}\,/\,{float(tolerance) / scale:.3g}\times10^{{{exponent}}}$"
    return f"{ratio} ({'stable' if value < tolerance else 'unstable'})"


def plot_convergence(runs):
    """Return Figure 1: score, molecular objective, and penalty convergence."""

    plt, Line2D, _, _, _, _, _, FuncFormatter = _plot_modules()
    if not runs:
        raise ValueError("At least one retrieved run is required for plotting.")
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14.0, 11.5),
        dpi=180,
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )
    finite = [run for run in runs if np.isfinite(run["best_score"])]
    best = max(finite, key=lambda run: run["best_score"]) if finite else None
    score_median = math.nan
    height_ratios = []
    panels = (
        ("score", r"$J_{\mathrm{reg}}$"),
        ("objective", r"$J_{\mathrm{mol}}$"),
        ("penalty", "Penalty"),
    )
    for panel_index, (axis, (name, ylabel)) in enumerate(zip(axes, panels)):
        envelope = _history_envelope([run["history"][name] for run in runs])
        if envelope is not None:
            histories, percentile_10, percentile_90, median = envelope
            if name == "score":
                score_median = float(median[-1])
            for run in runs:
                if run is not best:
                    _draw_checkpoint_history(axis, run, name, len(median))
            steps = np.arange(len(median))
            axis.fill_between(
                steps, percentile_10, percentile_90, color="#8ab8d8", alpha=0.3, linewidth=0.0
            )
            axis.plot(steps, median, color="#2474b5", linewidth=1.2, zorder=3)
            if best is not None:
                _draw_checkpoint_history(axis, best, name, len(median), best=True)
        axis.set_ylabel(ylabel)
        data_min, data_max = float(axis.dataLim.ymin), float(axis.dataLim.ymax)
        if np.isfinite(data_min) and data_min >= 0.0:
            ticks, tick_step = _nice_zero_based_ticks(data_max)
            separating = 0 if panel_index == 0 else 1
            axis.set_ylim(0.0, ticks[-1] + separating * tick_step)
            axis.set_yticks(ticks)
            height_ratios.append(len(ticks) - 1 + separating)
        else:
            height_ratios.append(6)
        axis.yaxis.set_major_formatter(FuncFormatter(_format_axis_tick))
        axis.yaxis.get_offset_text().set_visible(False)
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
        _add_stability_fraction(axis, runs, lambda run, metric=name: _history_stable_at_end(run, metric))

    axes[0].get_gridspec().set_height_ratios(height_ratios)
    maximum_step = max(len(run["history"]["score"]) - 1 for run in runs)
    adam_view = all(run["optimizer"] == "adam" for run in runs)
    change_steps = sorted(
        {
            int(step)
            for run in runs
            for step in run["history"].get("learning_rate_change_steps", [])
            if 0 < int(step) <= maximum_step
        }
    )
    if not adam_view:
        change_steps = []
    for step in change_steps:
        for axis in axes:
            axis.axvline(step, color="#666666", linestyle="--", linewidth=0.9, alpha=0.55)
    if adam_view:
        axes[0].annotate(
            r"$\alpha_1=\alpha_I$",
            xy=(0, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=9.0,
            color="#444444",
            clip_on=False,
        )
    for index, step in enumerate(change_steps, start=2):
        factors = []
        for run in runs:
            starts = np.asarray(run["history"].get("learning_rate_change_steps", []))
            if step not in starts:
                continue
            position = int(np.where(starts == step)[0][0])
            rates = np.asarray(run["history"].get("stage_learning_rates", []), dtype=float)
            if position and rates[position - 1] != 0:
                factors.append(rates[position] / rates[position - 1])
        factor = float(np.median(factors)) if factors else 1.0
        relation = rf"$\alpha_{{{index}}}={factor:.3g}\,\alpha_{{{index - 1}}}$"
        axes[0].annotate(
            relation,
            xy=(step, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9.0,
            color="#444444",
            clip_on=False,
        )
    for axis in axes[:-1]:
        axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for axis in axes:
        axis.spines["bottom"].set_visible(True)
        axis.spines["bottom"].set_color("#555555")
        axis.spines["bottom"].set_linewidth(0.8)
        axis.margins(x=0)
    for axis in axes[1:]:
        axis.spines["top"].set_visible(False)
    ticks = [0, *change_steps]
    if maximum_step not in ticks:
        ticks.append(maximum_step)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xlim(0, maximum_step if maximum_step else 1)
    axes[-1].set_xlabel("Cumulative optimisation step")

    if best is not None:
        score_tolerance = (
            float(best["tolerances"]["score_tolerance"][-1])
            if len(best["tolerances"]["score_tolerance"])
            else math.nan
        )
        axes[0].text(
            0.985,
            0.035,
            (
                f"Median: {_display_number(score_median, 3)}\n"
                "Best result:\n"
                f"Score: {_display_number(best['best_score'], 3)}\n"
                f"Obj: {_display_number(best['best_objective'], 3)}\n"
                rf"$\epsilon_J$: {_shared_exponent_ratio(score_tolerance, best['J_tol'])}"
            ),
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            multialignment="left",
            fontsize=9.0,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#fbfbfa",
                "edgecolor": "#c9c9c5",
                "alpha": 0.97,
            },
        )
    axes[1].legend(
        handles=[
            Line2D([], [], color="#b33f36", linewidth=0.8, label="Unstable"),
            Line2D([], [], color="#237a4b", linewidth=0.8, label="Stable"),
        ],
        loc="lower right",
        ncol=2,
        frameon=True,
        fontsize=8.5,
    )
    figure.suptitle(
        "Figure 1 — Optimisation Convergence",
        x=0.5,
        fontsize=15,
        fontweight="semibold",
        y=0.86,
    )
    _add_configuration_box(figure, runs, y=0.985)
    figure.subplots_adjust(top=0.78, bottom=0.07, left=0.08, right=0.97, hspace=0.0)
    return figure, axes


def plot_yield_distribution(runs):
    """Return Figure 2: molecular-yield strip/box distribution grouped by N."""

    plt, Line2D, _, _, _, _, Patch, FuncFormatter = _plot_modules()
    if not runs:
        raise ValueError("At least one retrieved run is required for plotting.")
    figure, axis = plt.subplots(figsize=(10.5, 6.6), dpi=180)
    finite = [run for run in runs if np.isfinite(run["best_objective"])]
    grid_sizes = sorted({int(run["N"]) for run in finite})
    positions = {N: index for index, N in enumerate(grid_sizes, start=1)}
    for N in grid_sizes:
        members = [run for run in finite if int(run["N"]) == N]
        values = np.asarray([run["best_objective"] for run in members], dtype=float)
        axis.boxplot(
            [values],
            positions=[positions[N]],
            widths=0.24,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#264653", "linewidth": 1.6},
            boxprops={"facecolor": "#4fb7c5", "edgecolor": "#4fb7c5", "alpha": 0.68},
            whiskerprops={"color": "#34495e", "linewidth": 0.9},
            capprops={"color": "#34495e", "linewidth": 0.9},
        )
        axis.scatter([positions[N]], [float(np.mean(values))], s=22, color="#111111", zorder=6)
        jitter = np.zeros(1) if len(members) == 1 else np.linspace(-0.12, 0.12, len(members))
        for offset, run in zip(jitter, members):
            stable = _history_stable_at_end(run, "objective")
            axis.scatter(
                positions[N] + offset,
                run["best_objective"],
                s=23,
                color="#c44e52",
                alpha=0.95 if stable else 0.60,
                linewidths=0.0,
                zorder=5 if stable else 4,
            )
        p10, p90 = np.percentile(values, [10.0, 90.0])
        median = float(np.median(values))
        spread = (p90 - p10) / abs(median) if median else math.inf
        axis.text(
            positions[N],
            1.015,
            rf"$S_J={_display_number(spread, 5).strip('$')}$",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9.0,
            clip_on=False,
        )
    axis.set_xticks(range(1, len(grid_sizes) + 1), [f"$N={N}$" for N in grid_sizes])
    axis.set_ylabel(r"$\max J_{\mathrm{mol}}$")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _display_number(value, 6)))
    axis.yaxis.get_offset_text().set_visible(False)
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.5, alpha=0.35)
    _add_stability_fraction(axis, finite, lambda run: _history_stable_at_end(run, "objective"))
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="#c44e52", label="Fourier start"),
            Line2D([], [], marker="o", linestyle="none", color="#111111", label="Fourier mean"),
            Patch(facecolor="#4fb7c5", edgecolor="#4fb7c5", alpha=0.68, label="10–90% spread"),
        ],
        loc="best",
        ncol=2,
        fontsize=8.0,
        frameon=True,
    )
    figure.suptitle(
        "Figure 2 — Yield Distribution by Time-grid Size",
        x=0.5,
        fontsize=15,
        fontweight="semibold",
        y=0.965,
    )
    figure.subplots_adjust(top=0.82, bottom=0.13, left=0.1, right=0.97)
    return figure, axis


def plot_controls(runs):
    """Return Figure 3: normalized optimized u/v control overlays."""

    plt, _, _, _, _, _, _, _ = _plot_modules()
    if not runs:
        raise ValueError("At least one retrieved run is required for plotting.")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=180, squeeze=False)
    axes = axes.reshape(-1)
    finite = [run for run in runs if np.isfinite(run["best_score"])]
    best = max(finite, key=lambda run: run["best_score"]) if finite else None
    labels = {"u": r"$u/u_{\max}$", "v": r"$\nu/\nu_{\max}$"}
    for axis, name in zip(axes, ("u", "v")):
        labelled = set()
        for run in finite:
            values = np.asarray(run["controls"]["best"][name], dtype=float) / float(run[f"{name}_max"])
            time = np.linspace(0.0, float(run["t_interval"]), len(values))
            stable = _control_stable(run, name)
            status = "Stable" if stable else "Unstable"
            axis.plot(
                time,
                values,
                color="#4b9668" if stable else "#c45d55",
                alpha=0.45 if stable else 0.55,
                linewidth=0.65,
                label=f"{status} optimised result" if status not in labelled else None,
            )
            labelled.add(status)
        if best is not None:
            values = np.asarray(best["controls"]["best"][name], dtype=float) / float(best[f"{name}_max"])
            time = np.linspace(0.0, float(best["t_interval"]), len(values))
            stable = _control_stable(best, name)
            axis.plot(
                time,
                values,
                color="#237a4b" if stable else "#b33f36",
                linewidth=1.0,
                label=f"Best score ({'stable' if stable else 'unstable'})",
                zorder=4,
            )
            display_name = r"$u$" if name == "u" else r"$\nu$"
            axis.text(
                0.98,
                0.97,
                f"Best {display_name} stability: "
                + _shared_exponent_ratio(_control_metric(best, name), best[f"{name}_tol"]),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.8,
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": "#d2d2ce",
                    "alpha": 0.92,
                },
            )
        axis.set_xlabel("Dimensionless time")
        axis.set_ylabel(labels[name])
        axis.set_xlim(
            min(0.0, *(float(run["t_interval"]) for run in finite)),
            max(float(run["t_interval"]) for run in finite),
        )
        axis.margins(x=0)
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
        _add_stability_fraction(axis, finite, lambda run, control=name: _control_stable(run, control))
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="upper right", bbox_to_anchor=(0.985, 0.89), frameon=False, fontsize=8.2)
    figure.suptitle(
        "Figure 3 — Optimised Control Overlays",
        x=0.5,
        fontsize=15,
        fontweight="semibold",
        y=0.965,
    )
    figure.subplots_adjust(top=0.84, bottom=0.13, left=0.08, right=0.97, wspace=0.26)
    return figure, axes


def _plot_sweep_convergence(runs, sweep):
    """Plot the best-score history at every value of one parameter sweep."""

    plt, Line2D, _, _, _, _, _, FuncFormatter = _plot_modules()
    selected = _best_sweep_runs(runs, sweep)
    if not selected:
        raise ValueError("The parameter sweep contains no finite best scores.")
    colours, mappable, coordinates = _sweep_visuals(sweep)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14.0, 11.5),
        dpi=180,
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )
    panels = (
        ("score", r"$J_{\mathrm{reg}}$"),
        ("objective", r"$J_{\mathrm{mol}}$"),
        ("penalty", "Penalty"),
    )
    for run in selected:
        colour = colours[_freeze_value(run[sweep.name])]
        alpha = 1.0 if _history_stable_at_end(run, "score") else 0.55
        for axis, (name, ylabel) in zip(axes, panels):
            history = np.asarray(run["history"][name], dtype=float)
            axis.plot(
                np.arange(history.size),
                history,
                color=colour,
                alpha=alpha,
                linewidth=1.15,
            )
            axis.set_ylabel(ylabel)

    maximum_step = max(len(run["history"]["score"]) - 1 for run in selected)
    axes[-1].set_xlim(0, maximum_step if maximum_step else 1)
    axes[-1].set_xlabel("Cumulative optimisation step")
    for axis in axes:
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
        axis.margins(x=0)
        axis.yaxis.set_major_formatter(FuncFormatter(_format_axis_tick))
        axis.yaxis.get_offset_text().set_visible(False)
    for axis in axes[:-1]:
        axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes[0].legend(
        handles=[
            Line2D([], [], color="#555555", linewidth=1.2, alpha=1.0, label="Stable"),
            Line2D([], [], color="#555555", linewidth=1.2, alpha=0.55, label="Unstable"),
        ],
        loc="best",
        frameon=True,
        fontsize=8.5,
    )
    figure.subplots_adjust(top=0.78, bottom=0.07, left=0.08, right=0.86, hspace=0.0)
    _add_sweep_colourbar(
        figure,
        list(axes),
        sweep,
        mappable,
        coordinates,
        pad=0.015,
        aspect=34,
        shrink=0.8,
    )
    label = SWEEP_LABELS.get(sweep.name, sweep.name.replace("_", " "))
    figure.suptitle(
        f"Figure 1 — Best Optimisation Convergence by {label}",
        x=0.5,
        fontsize=15,
        fontweight="semibold",
        y=0.86,
    )
    _add_configuration_box(figure, selected, y=0.985, exclude=(sweep.name,))
    figure._sweep_parameter = sweep.name
    figure._plotted_run_ids = tuple(run["run_id"] for run in selected)
    return figure, axes


def _jittered_sweep_positions(base, count, sweep, coordinates):
    if count <= 1:
        return np.asarray([base], dtype=float)
    offsets = np.linspace(-1.0, 1.0, count)
    if sweep.scale in {"log", "symlog"}:
        magnitudes = np.unique(np.abs(np.asarray(coordinates, dtype=float)))
        gaps = np.diff(np.log10(magnitudes[magnitudes > 0.0]))
        width = 0.12 * float(np.min(gaps)) if gaps.size else 0.04
        return float(base) * 10.0 ** (width * offsets)
    if sweep.numeric_values is not None:
        gaps = np.diff(np.sort(np.asarray(coordinates, dtype=float)))
        width = 0.12 * float(np.min(gaps[gaps > 0.0])) if np.any(gaps > 0.0) else 0.1
        return float(base) + width * offsets
    return float(base) + 0.12 * offsets


def _plot_sweep_yield_distribution(runs, sweep):
    """Plot each initialization's objective at its own best-score checkpoint."""

    plt, Line2D, _, _, _, _, _, FuncFormatter = _plot_modules()
    colours, mappable, coordinates = _sweep_visuals(sweep)
    groups = _sweep_groups(runs, sweep)
    figure, axis = plt.subplots(figsize=(11.2, 6.8), dpi=180)
    best_x, best_y, best_colours = [], [], []
    scatter_count = 0
    for index, key in enumerate(sweep.keys):
        members = [
            run for run in groups[key] if np.isfinite(run["best_objective"])
        ]
        if not members:
            continue
        base = float(coordinates[index])
        x_values = _jittered_sweep_positions(base, len(members), sweep, coordinates)
        y_values = np.asarray(
            [run["best_objective"] for run in members], dtype=float
        )
        axis.scatter(
            x_values,
            y_values,
            s=24,
            color=colours[key],
            alpha=0.72,
            linewidths=0.0,
            zorder=3,
        )
        scatter_count += len(members)
        finite_scores = [run for run in members if np.isfinite(run["best_score"])]
        if finite_scores:
            best = max(finite_scores, key=lambda run: run["best_score"])
            best_x.append(base)
            best_y.append(float(best["best_objective"]))
            best_colours.append(colours[key])

    if best_x:
        axis.plot(
            best_x,
            best_y,
            color="#202020",
            linewidth=1.25,
            zorder=4,
        )
        axis.scatter(
            best_x,
            best_y,
            s=34,
            color=best_colours,
            edgecolor="#202020",
            linewidth=0.65,
            zorder=5,
        )
    if sweep.scale == "log":
        axis.set_xscale("log")
    elif sweep.scale == "symlog":
        axis.set_xscale("symlog", linthresh=float(sweep.linthresh), base=10)
    elif sweep.numeric_values is None:
        axis.set_xticks(coordinates)
        axis.set_xticklabels(
            [_sweep_value_label(value) for value in sweep.display_values]
        )
    if sweep.numeric_values is not None and len(sweep.keys) <= 12:
        axis.set_xticks(coordinates)
        axis.set_xticklabels(
            [_sweep_value_label(value) for value in sweep.display_values]
        )

    label = SWEEP_LABELS.get(sweep.name, sweep.name.replace("_", " "))
    axis.set_xlabel(label)
    axis.set_ylabel(r"Best molecular objective $J_{\mathrm{mol}}$")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _display_number(value, 6)))
    axis.yaxis.get_offset_text().set_visible(False)
    axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="#777777", label="All initializations"),
            Line2D([], [], marker="o", color="#202020", label="Best score at each value"),
        ],
        loc="best",
        frameon=True,
        fontsize=8.5,
    )
    _add_sweep_colourbar(
        figure,
        axis,
        sweep,
        mappable,
        coordinates,
        pad=0.025,
        aspect=28,
    )
    figure.suptitle(
        f"Figure 2 — Yield Distribution by {label}",
        x=0.5,
        fontsize=15,
        fontweight="semibold",
        y=0.965,
    )
    figure.subplots_adjust(top=0.88, bottom=0.13, left=0.1, right=0.86)
    figure._sweep_parameter = sweep.name
    figure._scatter_run_count = scatter_count
    return figure, axis


def _plot_sweep_controls(runs, sweep):
    """Overlay best-checkpoint controls from the best score at each sweep value."""

    plt, Line2D, _, _, _, _, _, _ = _plot_modules()
    selected = _best_sweep_runs(runs, sweep)
    if not selected:
        raise ValueError("The parameter sweep contains no finite best scores.")
    colours, mappable, coordinates = _sweep_visuals(sweep)
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.7), dpi=180, squeeze=False)
    axes = axes.reshape(-1)
    labels = {"u": r"$u/u_{\max}$", "v": r"$\nu/\nu_{\max}$"}
    for axis, name in zip(axes, ("u", "v")):
        for run in selected:
            values = np.asarray(run["controls"]["best"][name], dtype=float) / float(
                run[f"{name}_max"]
            )
            time = np.linspace(0.0, float(run["t_interval"]), len(values))
            stable = _control_stable(run, name)
            axis.plot(
                time,
                values,
                color=colours[_freeze_value(run[sweep.name])],
                alpha=1.0 if stable else 0.55,
                linewidth=1.15,
            )
        axis.set_xlabel("Dimensionless time")
        axis.set_ylabel(labels[name])
        axis.set_xlim(
            min(0.0, *(float(run["t_interval"]) for run in selected)),
            max(float(run["t_interval"]) for run in selected),
        )
        axis.margins(x=0)
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
    axes[0].legend(
        handles=[
            Line2D([], [], color="#555555", linewidth=1.2, alpha=1.0, label="Stable"),
            Line2D([], [], color="#555555", linewidth=1.2, alpha=0.55, label="Unstable"),
        ],
        loc="best",
        frameon=True,
        fontsize=8.5,
    )
    _add_sweep_colourbar(
        figure,
        list(axes),
        sweep,
        mappable,
        coordinates,
        orientation="horizontal",
        pad=0.2,
        aspect=42,
        fraction=0.07,
    )
    label = SWEEP_LABELS.get(sweep.name, sweep.name.replace("_", " "))
    figure.suptitle(
        f"Figure 3 — Best Optimised Controls by {label}",
        x=0.5,
        fontsize=15,
        fontweight="semibold",
        y=0.965,
    )
    figure.subplots_adjust(top=0.86, bottom=0.28, left=0.08, right=0.97, wspace=0.25)
    figure._sweep_parameter = sweep.name
    figure._plotted_run_ids = tuple(run["run_id"] for run in selected)
    return figure, axes


def plot_standard_figures(runs, *, sweep_parameter=None):
    """Return the initialization view or an inferred one-parameter sweep view."""

    runs = list(runs)
    sweep = _detect_sweep(runs, sweep_parameter)
    if sweep is not None:
        return {
            "convergence": _plot_sweep_convergence(runs, sweep),
            "distribution": _plot_sweep_yield_distribution(runs, sweep),
            "controls": _plot_sweep_controls(runs, sweep),
        }

    return {
        "convergence": plot_convergence(runs),
        "distribution": plot_yield_distribution(runs),
        "controls": plot_controls(runs),
    }


def save_standard_figures(
    runs,
    output_dir: str | Path,
    *,
    formats=("png",),
    close=True,
    sweep_parameter=None,
):
    """Create and save the three standard figures with stable filenames."""

    plt, _, _, _, _, _, _, _ = _plot_modules()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = plot_standard_figures(runs, sweep_parameter=sweep_parameter)
    saved = {}
    for index, (name, (figure, _)) in enumerate(figures.items(), start=1):
        saved[name] = {}
        for file_format in formats:
            file_format = str(file_format).lower()
            if file_format not in {"png", "pdf"}:
                raise ValueError("figure formats must be png or pdf.")
            path = output_dir / f"{index:02d}_{name}.{file_format}"
            options = {"dpi": 300} if file_format == "png" else {}
            figure.savefig(path, bbox_inches="tight", **options)
            saved[name][file_format] = path
        if close:
            plt.close(figure)
    return saved
