"""The standard three-figure view for completed optimization results.

Isolation boundary: plotting accepts already-retrieved result mappings and
never opens a database, loads a config, or launches calculations.  The visual
structure and styling mirror the old codebase's ``plot_run()`` output:
convergence, yield distribution, and optimized control overlays.

Every selection uses one colour-keyed sweep view. If no configuration
parameter varies, initializations become a categorical, one-based numbered
sweep. Figures 1 and 3 use the run with the highest regularized score at each
sweep value, while Figure 2 scatters every matching result's objective recorded
at its own best-score checkpoint against that value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ofc-matplotlib"))


FIGURE_DPI = 600
PNG_DPI = 600


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
INITIALIZATION_PARAMETER = "initialization_index"

SWEEP_LABELS = {
    INITIALIZATION_PARAMETER: "Initialization",
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
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": PNG_DPI,
        }
    )
    return plt, Line2D, AnchoredOffsetbox, HPacker, TextArea, VPacker, Patch, FuncFormatter


def _validate_log_base(value, name):
    if value is None:
        return None
    try:
        base = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be None or a finite logarithm base.") from error
    if not np.isfinite(base) or base <= 1.0:
        raise ValueError(f"{name} must be None or a finite value greater than 1.")
    return base


def _validate_axis_range(value, name):
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be None or a two-value sequence.")
    try:
        limits = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be None or a two-value sequence.") from error
    if len(limits) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    try:
        lower, upper = (float(limit) for limit in limits)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be finite numbers.") from error
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(
            f"{name} values must be finite and ordered from lower to upper."
        )
    return lower, upper


def _axis_labels(value, defaults, name):
    """Resolve one label or one label per panel; math-text strings pass through."""

    defaults = tuple(defaults)
    if value is None:
        return defaults
    if isinstance(value, str):
        return (value,) * len(defaults)
    try:
        labels = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a string or one string per panel.") from error
    if len(labels) != len(defaults) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError(f"{name} must be a string or one string per panel.")
    return labels


def _panel_values(value, count, name):
    """Expand one scalar setting or validate one value per plot panel."""

    if isinstance(value, (str, bytes)):
        return (value,) * count
    try:
        values = tuple(value)
    except TypeError:
        return (value,) * count
    if len(values) != count:
        raise ValueError(f"{name} must be one value or one value per panel.")
    return values


def _validate_multiplier(value, name):
    try:
        multiplier = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number.") from error
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return multiplier


def _format_log_tick(value, _, representation_base, multiplier=1.0):
    """Format one signed-log major tick as a plain numerical value."""

    value = float(value)
    if not np.isfinite(value):
        return ""
    if value == 0.0:
        return "0"
    return _display_number(value, precision=8)


def _minor_log_subs(base):
    """Return quarter-unit numeric subdivisions within one log-base decade."""

    integer_base = round(base)
    if not math.isclose(base, integer_base) or not 2 <= integer_base <= 16:
        return None
    return np.arange(1.25, integer_base, 0.25, dtype=float)


def _scaled_log_major_ticks(limits, base, multiplier, signed):
    """Return ``multiplier * base**n`` ticks, starting at exponent zero."""

    lower, upper = sorted(float(value) for value in limits)
    maximum = max(abs(lower), abs(upper), multiplier)
    if maximum <= multiplier:
        exponent_maximum = 0
    else:
        exponent_maximum = max(
            0,
            int(math.ceil(math.log(maximum / multiplier) / math.log(base))),
        )
    positive = multiplier * np.power(
        base, np.arange(exponent_maximum + 1, dtype=float)
    )
    if signed:
        return np.concatenate((-positive[::-1], [0.0], positive))
    return positive


def _scaled_log_minor_ticks(limits, base, multiplier, signed):
    """Return decade-local subdivisions and their subtly hierarchical lengths."""

    subs = _minor_log_subs(base)
    if subs is None:
        empty = np.asarray([], dtype=float)
        return empty, empty
    lower, upper = sorted(float(value) for value in limits)
    maximum = max(abs(lower), abs(upper))
    candidates = []

    # Four quarter-unit intervals replace each former integer interval. Their
    # plotted gaps compress logarithmically and reset in the next decade.
    subdivision_step = 0.25
    maximum_gap = math.log(1.0 + subdivision_step) / math.log(base)
    subdivision_lengths = {
        float(subdivision): 1.8
        + 0.6
        * (
            (
                math.log(subdivision / (subdivision - subdivision_step))
                / math.log(base)
            )
            / maximum_gap
        )
        + (0.75 if math.isclose(subdivision, round(subdivision)) else 0.0)
        for subdivision in subs
    }
    exponent = 0
    if not signed and lower > 0.0 and lower < multiplier:
        exponent = min(
            0,
            int(math.floor(math.log(lower / multiplier) / math.log(base))),
        )
    while multiplier * base**exponent < max(maximum, multiplier):
        decade = multiplier * base**exponent
        if exponent < 0:
            # Powers below the labelled multiplier lattice remain visible as
            # longer, unlabeled minor marks instead of leaving part of a
            # positive log axis visually blank.
            candidates.append((decade, 3.25))
        for subdivision in subs:
            value = decade * subdivision
            candidates.append((value, subdivision_lengths[float(subdivision)]))
            if signed:
                candidates.append((-value, subdivision_lengths[float(subdivision)]))
        exponent += 1

    if signed:
        integer_base = int(round(base))
        for subdivision in range(1, integer_base):
            value = multiplier * subdivision / integer_base
            candidates.append((value, 2.2))
            candidates.append((-value, 2.2))

    visible = sorted(
        (value, length)
        for value, length in candidates
        if lower < value < upper
    )
    return (
        np.asarray([value for value, _ in visible], dtype=float),
        np.asarray([length for _, length in visible], dtype=float),
    )


def _style_minor_tick_lengths(matplotlib_axis, lengths):
    """Apply per-tick lengths after installing a fixed minor locator."""

    ticks = matplotlib_axis.get_minor_ticks(len(lengths))
    for tick, length in zip(ticks, lengths):
        tick.tick1line.set_markersize(float(length))
        tick.tick2line.set_markersize(float(length))


def _set_axis_data_values(axis, dimension, values):
    """Record the semantic plotted values used to tighten a log-range envelope."""

    flattened = np.asarray(values, dtype=float).reshape(-1)
    setattr(
        axis,
        f"_ofc_{dimension}_data_values",
        flattened[np.isfinite(flattened)],
    )


def _axis_data_values(axis, dimension):
    """Return semantic plot data, falling back to artist data when unspecified."""

    recorded = getattr(axis, f"_ofc_{dimension}_data_values", None)
    if recorded is not None:
        return np.asarray(recorded, dtype=float)

    coordinate = 0 if dimension == "x" else 1
    values = []
    for line in axis.lines:
        data_dimensions = line.get_transform().contains_branch_seperately(
            axis.transData
        )
        if data_dimensions[coordinate]:
            values.extend(
                np.asarray(
                    line.get_xdata() if dimension == "x" else line.get_ydata(),
                    dtype=float,
                ).reshape(-1)
            )
    for collection in axis.collections:
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        if offsets.ndim == 2 and offsets.shape[1] == 2:
            values.extend(offsets[:, coordinate])
    if not values:
        values.extend(
            np.asarray(
                getattr(axis, f"{dimension}axis").get_data_interval(), dtype=float
            )
        )
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _next_log_major_tick(value, base, multiplier, *, minimum_exponent=0):
    """Return the first labelled logarithmic lattice value at or above value."""

    value = abs(float(value))
    if value <= 0.0:
        return float(multiplier) * base**minimum_exponent
    exponent = int(
        math.ceil(
            math.log(value / multiplier) / math.log(base) - 1e-12
        )
    )
    return float(multiplier) * base ** max(minimum_exponent, exponent)


def _cutoff_log_limits(axis, dimension, requested_range, multiplier, base):
    """Resolve a log range as a zero cutoff and tick-aligned upper envelope."""

    requested_minimum, requested_maximum = requested_range
    if requested_minimum < 0.0:
        raise ValueError(
            f"{dimension}_range for a logarithmic axis must contain positive "
            "absolute magnitudes."
        )
    values = _axis_data_values(axis, dimension)
    nonzero_magnitudes = np.abs(values[~np.isclose(values, 0.0)])
    visible_magnitudes = nonzero_magnitudes[
        nonzero_magnitudes <= requested_maximum
    ]
    if visible_magnitudes.size:
        cutoff = max(requested_minimum, float(np.min(visible_magnitudes)))
    else:
        cutoff = requested_minimum if requested_minimum > 0.0 else multiplier
    if cutoff <= 0.0:
        cutoff = min(multiplier, requested_maximum)
    data_maximum = (
        float(np.max(visible_magnitudes))
        if visible_magnitudes.size
        else cutoff
    )
    if cutoff > data_maximum and requested_minimum < data_maximum:
        cutoff = max(requested_minimum, np.finfo(float).tiny)

    def outward_limit(side_values):
        side_values = np.asarray(side_values, dtype=float)
        side_maximum = (
            float(np.max(side_values)) if side_values.size else cutoff
        )
        return min(
            requested_maximum,
            _next_log_major_tick(
                side_maximum,
                base,
                cutoff,
                minimum_exponent=1,
            ),
        )

    signed = bool(np.any(values < 0.0))
    if signed:
        negative = np.abs(
            values[(values < -cutoff) & (values >= -requested_maximum)]
        )
        positive = values[(values > cutoff) & (values <= requested_maximum)]
        negative_limit = outward_limit(negative)
        positive_limit = outward_limit(positive)
        lower = -negative_limit if np.any(values < 0.0) else 0.0
        upper = positive_limit if np.any(values > 0.0) else min(
            requested_maximum, cutoff * base
        )
        if math.isclose(lower, upper):
            upper = min(requested_maximum, cutoff * base)
    else:
        lower = cutoff
        upper = outward_limit(visible_magnitudes)
        if upper <= lower:
            upper = min(requested_maximum, lower * base)
    return cutoff, signed, (lower, upper)


def _cutoff_log_scale(matplotlib_axis, base, cutoff, signed):
    """Return a scale that collapses magnitudes through ``cutoff`` onto zero."""

    from matplotlib.scale import ScaleBase
    from matplotlib.ticker import NullFormatter, NullLocator
    from matplotlib.transforms import Transform

    class CutoffLogTransform(Transform):
        input_dims = output_dims = 1
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, values):
            masked = np.ma.asarray(values, dtype=float)
            data = np.asarray(np.ma.getdata(masked), dtype=float)
            magnitude = np.abs(data)
            transformed = np.zeros_like(data, dtype=float)
            outside = magnitude > cutoff
            transformed[outside] = np.log(magnitude[outside] / cutoff) / math.log(
                base
            )
            if signed:
                transformed = np.copysign(transformed, data)
            return np.ma.array(transformed, mask=np.ma.getmaskarray(masked))

        def inverted(self):
            return InvertedCutoffLogTransform()

    class InvertedCutoffLogTransform(Transform):
        input_dims = output_dims = 1
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, values):
            masked = np.ma.asarray(values, dtype=float)
            data = np.asarray(np.ma.getdata(masked), dtype=float)
            magnitude = cutoff * np.power(base, np.abs(data))
            if signed:
                inverted = np.where(
                    np.isclose(data, 0.0),
                    0.0,
                    np.copysign(magnitude, data),
                )
            else:
                inverted = magnitude
            return np.ma.array(inverted, mask=np.ma.getmaskarray(masked))

        def inverted(self):
            return CutoffLogTransform()

    class CutoffLogScale(ScaleBase):
        name = "symlog" if signed else "log"

        def __init__(self, axis):
            super().__init__(axis)
            self.base = base
            self.cutoff = cutoff
            self.signed = signed

        def get_transform(self):
            return CutoffLogTransform()

        def set_default_locators_and_formatters(self, axis):
            axis.set_major_locator(NullLocator())
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(NullFormatter())
            axis.set_minor_formatter(NullFormatter())

    return CutoffLogScale(matplotlib_axis)


def _cutoff_log_major_ticks(limits, base, cutoff, signed):
    """Return zero, cutoff-relative decades, and any hard-cap endpoint."""

    lower, upper = sorted(float(value) for value in limits)
    maximum = max(abs(lower), abs(upper))
    exponent_maximum = max(
        0,
        int(math.floor(math.log(maximum / cutoff) / math.log(base) + 1e-12)),
    )
    positive = cutoff * np.power(
        base, np.arange(1, exponent_maximum + 1, dtype=float)
    )
    positive = positive[positive <= maximum * (1.0 + 1e-12)]
    if signed:
        ticks = [
            *(-positive[::-1])[(-positive[::-1] >= lower)],
            0.0,
            *positive[positive <= upper],
        ]
        if lower < -cutoff and not any(np.isclose(lower, ticks)):
            ticks.append(lower)
        if upper > cutoff and not any(np.isclose(upper, ticks)):
            ticks.append(upper)
        return np.asarray(sorted(ticks), dtype=float)
    ticks = [cutoff, *positive[positive <= upper]]
    if upper > cutoff and not any(np.isclose(upper, ticks)):
        ticks.append(upper)
    return np.asarray(sorted(ticks), dtype=float)


def _cutoff_log_minor_ticks(limits, base, cutoff, signed):
    """Return the usual decade subdivisions above a collapsed zero cutoff."""

    subs = _minor_log_subs(base)
    if subs is None:
        empty = np.asarray([], dtype=float)
        return empty, empty
    lower, upper = sorted(float(value) for value in limits)
    maximum = max(abs(lower), abs(upper))
    candidates = []
    subdivision_step = 0.25
    maximum_gap = math.log(1.0 + subdivision_step) / math.log(base)
    lengths = {
        float(subdivision): 1.8
        + 0.6
        * (
            math.log(subdivision / (subdivision - subdivision_step))
            / math.log(base)
            / maximum_gap
        )
        + (0.75 if math.isclose(subdivision, round(subdivision)) else 0.0)
        for subdivision in subs
    }
    decade = cutoff
    while decade < maximum:
        for subdivision in subs:
            value = decade * subdivision
            candidates.append((value, lengths[float(subdivision)]))
            if signed:
                candidates.append((-value, lengths[float(subdivision)]))
        decade *= base
    visible = sorted(
        (value, length)
        for value, length in candidates
        if lower < value < upper
    )
    return (
        np.asarray([value for value, _ in visible], dtype=float),
        np.asarray([length for _, length in visible], dtype=float),
    )


def _apply_log_scales(
    axis,
    *,
    log_base_x=None,
    log_base_y=None,
    base_x="axis",
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
):
    """Apply optional log scales and format their major values.

    Without an explicit range, strictly positive values use an ordinary log
    scale and signed values use symmetric log. Logarithmic limits extend to the
    next labelled major tick. For a logarithmic axis, a range is an absolute
    clipping envelope: its minimum becomes the zero cutoff and its maximum is
    a hard cap; a cap between regular ticks becomes a labelled endpoint. The default
    ``"axis"`` labels major ticks as powers of that same base, ``None`` labels
    them with their actual values, and a numeric value displays them as powers
    of that base.
    """

    from matplotlib.ticker import (
        FixedLocator,
        FuncFormatter,
        NullFormatter,
    )

    for dimension, supplied_base, supplied_value_base, supplied_multiplier, supplied_range in (
        ("x", log_base_x, base_x, x_multiplier, x_range),
        ("y", log_base_y, base_y, y_multiplier, y_range),
    ):
        base = _validate_log_base(supplied_base, f"log_base_{dimension}")
        multiplier = _validate_multiplier(
            supplied_multiplier, f"{dimension}_multiplier"
        )
        axis_range = _validate_axis_range(supplied_range, f"{dimension}_range")
        if supplied_value_base == "axis":
            value_base = "axis"
        else:
            value_base = _validate_log_base(
                supplied_value_base, f"base_{dimension}"
            )
        matplotlib_axis = getattr(axis, f"{dimension}axis")
        current_scale = getattr(axis, f"get_{dimension}scale")()
        if base is None and current_scale in {"log", "symlog"}:
            base = float(matplotlib_axis._scale.base)
        cutoff = None
        if base is not None:
            if axis_range is not None:
                cutoff, signed, effective_limits = _cutoff_log_limits(
                    axis,
                    dimension,
                    axis_range,
                    multiplier,
                    base,
                )
                scale = _cutoff_log_scale(
                    matplotlib_axis,
                    base,
                    cutoff,
                    signed,
                )
                getattr(axis, f"set_{dimension}scale")(scale)
                getattr(axis, f"set_{dimension}lim")(*effective_limits)
            else:
                visible_limits = getattr(axis, f"get_{dimension}lim")()
                values = _axis_data_values(axis, dimension)
                data_minimum = (
                    float(np.nanmin(values)) if values.size else math.nan
                )
                signed = current_scale == "symlog" or (
                    data_minimum <= 0.0 and min(visible_limits) <= 0.0
                )
                scale_options = {"base": base}
                if signed:
                    scale_options["linthresh"] = multiplier
                    # Match the constant-scale 0..multiplier interval to the
                    # local logarithmic distance from multiplier to twice it.
                    scale_options["linscale"] = (
                        (1.0 - base**-1) * math.log(2.0) / math.log(base)
                    )
                scale_name = "symlog" if signed else "log"
                getattr(axis, f"set_{dimension}scale")(
                    scale_name, **scale_options
                )
                if signed:
                    linthresh = float(scale_options["linthresh"])
                    negative = np.abs(values[values < 0.0])
                    positive = values[values > 0.0]
                    lower = (
                        -_next_log_major_tick(
                            float(np.max(negative)), base, multiplier
                        )
                        if negative.size
                        else 0.0
                    )
                    upper = (
                        _next_log_major_tick(
                            float(np.max(positive)), base, multiplier
                        )
                        if positive.size
                        else linthresh
                    )
                    getattr(axis, f"set_{dimension}lim")(lower, upper)
                else:
                    positive = values[values > 0.0]
                    if positive.size:
                        lower = getattr(axis, f"get_{dimension}lim")()[0]
                        upper = _next_log_major_tick(
                            float(np.max(positive)), base, multiplier
                        )
                        if upper <= lower:
                            lower = min(float(np.min(positive)), upper / base)
                        getattr(axis, f"set_{dimension}lim")(lower, upper)
        if axis_range is not None and base is None:
            getattr(axis, f"set_{dimension}lim")(*axis_range)
        if getattr(axis, f"get_{dimension}scale")() not in {"log", "symlog"}:
            if value_base not in (None, "axis"):
                raise ValueError(
                    f"base_{dimension} requires a logarithmic {dimension}-axis."
                )
            if not math.isclose(multiplier, 1.0):
                raise ValueError(
                    f"{dimension}_multiplier requires a logarithmic {dimension}-axis."
                )
            continue
        if value_base == "axis":
            value_base = float(matplotlib_axis._scale.base)
        axis_scale = getattr(axis, f"get_{dimension}scale")()
        axis_base = float(matplotlib_axis._scale.base)
        limits = getattr(axis, f"get_{dimension}lim")()
        signed = axis_scale == "symlog"
        if cutoff is None:
            major_ticks = _scaled_log_major_ticks(
                limits,
                axis_base,
                multiplier,
                signed,
            )
            minor_ticks, minor_lengths = _scaled_log_minor_ticks(
                limits,
                axis_base,
                multiplier,
                signed,
            )
            zero_location = 0.0
            formatter_multiplier = multiplier
        else:
            major_ticks = _cutoff_log_major_ticks(
                limits,
                axis_base,
                cutoff,
                signed,
            )
            minor_ticks, minor_lengths = _cutoff_log_minor_ticks(
                limits,
                axis_base,
                cutoff,
                signed,
            )
            zero_location = 0.0 if signed else cutoff
            cutoff_exponent = math.log(cutoff) / math.log(axis_base)
            formatter_multiplier = (
                1.0
                if math.isclose(
                    cutoff_exponent,
                    round(cutoff_exponent),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                else cutoff
            )
        if minor_ticks.size and major_ticks.size:
            minor_only = ~np.any(
                np.isclose(
                    minor_ticks[:, np.newaxis],
                    major_ticks[np.newaxis, :],
                    rtol=1e-12,
                    atol=1e-15,
                ),
                axis=1,
            )
            minor_ticks = minor_ticks[minor_only]
            minor_lengths = minor_lengths[minor_only]
        matplotlib_axis.set_major_locator(FixedLocator(major_ticks))
        matplotlib_axis.set_minor_locator(FixedLocator(minor_ticks))
        matplotlib_axis.set_major_formatter(
            FuncFormatter(
                lambda value, position, display_base=value_base, scale=formatter_multiplier, zero=zero_location: (
                    "0"
                    if math.isclose(
                        float(value), zero, rel_tol=1e-12, abs_tol=1e-15
                    )
                    else _format_log_tick(value, position, display_base, scale)
                )
            )
        )
        matplotlib_axis.set_minor_formatter(NullFormatter())
        axis.tick_params(
            axis=dimension,
            which="major",
            length=5.0,
            width=0.65,
            color="#555555",
        )
        axis.tick_params(
            axis=dimension,
            which="minor",
            length=2.2,
            width=0.45,
            color="#777777",
        )
        _style_minor_tick_lengths(matplotlib_axis, minor_lengths)
        # Minor ticks remain unlabeled and do not create a grid.
        axis.grid(False, axis=dimension, which="minor")


def _validate_point_size(value):
    try:
        size = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("point_size must be a finite positive number.") from error
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("point_size must be a finite positive number.")
    return size


def _validate_alpha(value, name):
    try:
        alpha = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite value between 0 and 1.") from error
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1.")
    return alpha


def _seed_sensitivity_tolerances(runs):
    """Return every distinct objective tolerance stored in queried run data."""

    tolerances = []
    for run in runs:
        if "J_tol" not in run or run["J_tol"] is None:
            continue
        try:
            tolerance = float(run["J_tol"])
        except (TypeError, ValueError) as error:
            raise ValueError("Queried J_tol values must be finite and non-negative.") from error
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Queried J_tol values must be finite and non-negative.")
        if not any(
            math.isclose(tolerance, existing, rel_tol=1e-12, abs_tol=0.0)
            for existing in tolerances
        ):
            tolerances.append(tolerance)
    return tuple(sorted(tolerances))


def _resolve_seed_sensitivity_tolerances(runs, value):
    """Use stored ``J_tol`` values or an explicit scalar/sequence override."""

    if value is None:
        return _seed_sensitivity_tolerances(runs)
    values = (value,) if isinstance(value, (int, float, np.number)) else tuple(value)
    tolerances = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError(
                "seed_sensitivity_tolerance must contain finite non-negative numbers."
            )
        try:
            tolerance = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "seed_sensitivity_tolerance must be None, one number, or a number sequence."
            ) from error
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "seed_sensitivity_tolerance must contain finite non-negative numbers."
            )
        if not any(
            math.isclose(tolerance, existing, rel_tol=1e-12, abs_tol=0.0)
            for existing in tolerances
        ):
            tolerances.append(tolerance)
    return tuple(sorted(tolerances))


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
        return f"{text}e{exponent}"
    return np.format_float_positional(
        value,
        precision=precision,
        unique=False,
        fractional=False,
        trim="-",
    )


def _set_control_time_ticks(axis, final_time, x_range=None):
    """Keep time tick marks while labeling only zero and the float final time."""

    if hasattr(axis.xaxis._scale, "cutoff"):
        return
    final_time = float(final_time)
    lower, upper = axis.get_xlim()
    ticks = [
        float(value)
        for value in axis.get_xticks()
        if lower <= float(value) <= upper
    ]
    ticks.extend((0.0, final_time))
    ticks = sorted(set(ticks))
    labels = []
    for value in ticks:
        if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12):
            labels.append("0.0")
        elif math.isclose(value, final_time, rel_tol=1e-12, abs_tol=1e-12):
            labels.append(str(final_time))
        else:
            labels.append("")
    axis.set_xticks(ticks, labels)
    if x_range is not None:
        axis.set_xlim(*_validate_axis_range(x_range, "x_range"))


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
        return _display_number(value, precision=5)
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


def _make_sweep_spec(runs, name, *, allow_single=False):
    if name in {INITIALIZATION_PARAMETER, "initialisation_index"}:
        return _make_initialization_sweep(runs)
    if name not in SWEEP_PARAMETERS:
        raise ValueError(
            f"Unknown sweep parameter {name!r}; choose one of "
            f"{SWEEP_PARAMETERS + (INITIALIZATION_PARAMETER,)}."
        )
    if not runs or any(name not in run for run in runs):
        raise ValueError(f"Sweep parameter {name!r} is missing from one or more runs.")

    unique = {}
    for run in runs:
        value = run[name]
        unique.setdefault(_freeze_value(value), value)
    if len(unique) < 2 and not allow_single:
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


def _initialization_value(run):
    """Return a stable identifier for one initialization result."""

    if INITIALIZATION_PARAMETER in run:
        return run[INITIALIZATION_PARAMETER]
    if "initialisation_index" in run:
        return run["initialisation_index"]
    if "run_id" in run:
        return run["run_id"]
    raise ValueError(
        "Every run needs initialization_index, initialisation_index, or run_id "
        "to construct the initialization sweep."
    )


def _make_initialization_sweep(runs):
    """Represent repeated starts as a categorical, one-based numbered sweep."""

    if not runs:
        raise ValueError("At least one retrieved run is required for plotting.")
    unique = {}
    for run in runs:
        value = _initialization_value(run)
        unique.setdefault(_freeze_value(value), value)

    def order(item):
        value = item[1]
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            return (0, float(value))
        return (1, _sweep_value_label(value))

    ordered = sorted(unique.items(), key=order)
    return SweepSpec(
        name=INITIALIZATION_PARAMETER,
        keys=tuple(key for key, _ in ordered),
        display_values=tuple(range(1, len(ordered) + 1)),
        numeric_values=None,
        scale="categorical",
    )


def _detect_sweep(runs, sweep_parameter=None):
    """Return a physical-parameter sweep or a numbered initialization sweep."""

    if sweep_parameter is not None:
        return _make_sweep_spec(runs, str(sweep_parameter))
    varied = []
    for name in SWEEP_PARAMETERS:
        if not runs or any(name not in run for run in runs):
            continue
        if len({_freeze_value(run[name]) for run in runs}) > 1:
            varied.append(name)
    if not varied:
        return _make_initialization_sweep(runs)
    if len(varied) > 1:
        raise ValueError(
            "The selected runs vary multiple configuration parameters: "
            + ", ".join(varied)
            + ". Filter the query to one sweep or pass --sweep-parameter NAME."
        )
    return _make_sweep_spec(runs, varied[0])


def varying_sweep_parameters(runs):
    """Return every supported configuration parameter that varies in ``runs``."""

    runs = list(runs)
    return tuple(
        name
        for name in SWEEP_PARAMETERS
        if runs
        and all(name in run for run in runs)
        and len({_freeze_value(run[name]) for run in runs}) > 1
    )


def _sweep_groups(runs, sweep):
    groups = {key: [] for key in sweep.keys}
    for run in runs:
        value = (
            _initialization_value(run)
            if sweep.name == INITIALIZATION_PARAMETER
            else run[sweep.name]
        )
        groups[_freeze_value(value)].append(run)
    return groups


def _best_sweep_runs(runs, sweep):
    selected = []
    for key, members in _sweep_groups(runs, sweep).items():
        finite = [run for run in members if np.isfinite(run["best_score"])]
        if finite:
            selected.append(max(finite, key=lambda run: run["best_score"]))
    return selected


def _score_reference_values(runs, *, include_median=True):
    scores = np.asarray(
        [run["best_score"] for run in runs if np.isfinite(run.get("best_score", np.nan))],
        dtype=float,
    )
    if not scores.size:
        return ()
    values = [float(np.max(scores))]
    if include_median:
        values.insert(0, float(np.median(scores)))
    unique = []
    for value in values:
        if not any(math.isclose(value, item, rel_tol=1e-12, abs_tol=1e-12) for item in unique):
            unique.append(value)
    return tuple(unique)


def _add_score_reference_ticks(
    axis,
    runs,
    *,
    include_median=True,
    show_ticks=True,
):
    """Mark final median/best scores on the right and draw the best reference."""

    values = _score_reference_values(runs, include_median=include_median)
    if not values:
        return None
    best = max(values)
    line = axis.axhline(
        best,
        color="#b33f36",
        linewidth=0.85,
        linestyle=(0, (4, 3)),
        alpha=0.45,
        zorder=1.5,
    )
    axis._best_score_reference_line = line
    axis._score_reference_values = values
    if not show_ticks:
        return None
    reference_axis = axis.secondary_yaxis("right")
    reference_axis.set_yticks(values)
    reference_axis.set_yticklabels([str(int(np.rint(value))) for value in values])
    reference_axis.tick_params(axis="y", labelsize=7.2, length=3.0, pad=2.0)
    reference_axis.set_ylabel("")
    axis._score_reference_axis = reference_axis
    return reference_axis


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


def _add_sweep_colourbar(
    figure,
    axes,
    sweep,
    mappable,
    coordinates,
    *,
    show_label=True,
    **kwargs,
):
    colourbar = figure.colorbar(mappable, ax=axes, **kwargs)
    colourbar.minorticks_off()
    colourbar.ax.tick_params(which="minor", length=0)
    if show_label:
        colourbar.set_label(
            SWEEP_LABELS.get(sweep.name, sweep.name.replace("_", " "))
        )
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


def _nice_convergence_ticks(data_min, data_max, target_intervals=6):
    """Return uniform linear ticks that include zero and any negative data."""

    data_min = float(data_min)
    data_max = float(data_max)
    if data_min >= 0.0:
        return _nice_zero_based_ticks(data_max, target_intervals)

    lower_data = min(data_min, 0.0)
    upper_data = max(data_max, 0.0)
    span = upper_data - lower_data
    if not np.isfinite(span) or span <= 0.0:
        span = max(abs(lower_data), 1.0)
    raw_step = span / target_intervals
    exponent = int(math.floor(math.log10(raw_step)))
    candidates = sorted(
        {
            multiplier * 10.0**candidate_exponent
            for candidate_exponent in range(exponent - 1, exponent + 3)
            for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0)
        }
    )

    def layout(step):
        lower_tick = step * math.floor(lower_data / step + 1e-12)
        upper_tick = step * math.ceil(upper_data / step - 1e-12)
        intervals = max(1, int(round((upper_tick - lower_tick) / step)))
        return (
            max(5 - intervals, 0, intervals - 8),
            abs(intervals - target_intervals),
            (lower_data - lower_tick) + (upper_tick - upper_data),
            -step,
            lower_tick,
            upper_tick,
            intervals,
        )

    chosen = min(candidates, key=layout)
    _, _, _, _, lower_tick, upper_tick, intervals = layout(chosen)
    if math.isclose(upper_tick, data_max, rel_tol=1e-12, abs_tol=1e-12):
        upper_tick += chosen
        intervals += 1
    ticks = np.linspace(lower_tick, upper_tick, intervals + 1)
    ticks[np.isclose(ticks, 0.0, rtol=0.0, atol=1e-14)] = 0.0
    return ticks, chosen


def _style_convergence_separators(axes, panel_top_ticks):
    """Draw scale-independent panel joins and share linear boundary ticks."""

    for index, axis in enumerate(axes):
        axis.margins(x=0)
        axis.spines["bottom"].set_visible(index == len(axes) - 1)
        axis.spines["bottom"].set_color("#555555")
        axis.spines["bottom"].set_linewidth(0.8)
        if index < len(axes) - 1 and panel_top_ticks[index] is not None:
            axis.yaxis.get_major_ticks()[0].label1.set_visible(False)
        if index == 0:
            continue
        top_tick = panel_top_ticks[index]
        axis.spines["top"].set_visible(True)
        if top_tick is not None:
            axis.spines["top"].set_position(("data", top_tick))
        else:
            # Axes coordinates remain correct when the y scale or range changes.
            axis.spines["top"].set_position(("axes", 1.0))
        axis.spines["top"].set_color("#555555")
        axis.spines["top"].set_linewidth(0.8)


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
    if (
        history_name == "score"
        and len(steps)
        and "score_tolerance" in run["tolerances"]
    ):
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
    if run.get("J_tol") is None:
        return True
    observations = _checkpoint_observations(run, history_name)
    if not observations:
        return False
    return bool(observations[-1][1] < float(run["J_tol"]))


def _control_metric(run, name):
    values = np.asarray(
        run["tolerances"].get(f"{name}_tolerance", []), dtype=float
    )
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


def _draw_history_by_terminal_stability(axis, run, history_name, width, *, best=False):
    history = np.asarray(run["history"][history_name], dtype=float)
    stable = _history_stable_at_end(run, history_name)
    colour = ("#237a4b" if best else "#4b9668") if stable else (
        "#b33f36" if best else "#c45d55"
    )
    if len(history) < width:
        history = np.concatenate(
            [history, np.full(width - len(history), history[-1], dtype=float)]
        )
    axis.plot(
        np.arange(len(history)),
        history,
        color=colour,
        alpha=1.0 if best else (0.45 if stable else 0.55),
        linewidth=1.2 if best else 0.8,
        zorder=4 if best else 1,
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
        ({"J_tol"}, rf"$J_{{\mathrm{{tol}}}}$ = {_distinct_display(runs, 'J_tol', optional=True)}"),
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
    if "peak_refinement" in optimizers:
        entries.insert(
            7,
            (
                {
                    "peak_initial_step_size",
                    "peak_min_step_size",
                    "peak_max_step_size",
                    "peak_backtracking_factor",
                    "peak_max_linesearch_steps",
                },
                "peak refinement = step "
                + _distinct_display(runs, "peak_initial_step_size")
                + ", backtrack "
                + _distinct_display(runs, "peak_backtracking_factor")
                + ", max line search "
                + _distinct_display(runs, "peak_max_linesearch_steps"),
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
    ratio = (
        f"{_display_number(value, 6)} / "
        f"{_display_number(float(tolerance), 6)}"
    )
    return f"{ratio} ({'stable' if value < tolerance else 'unstable'})"


def _schedule_change_steps(runs, histories=None, maximum_step=None):
    """Return steps where the configured Adam learning rate actually changes."""

    runs = list(runs)
    if histories is None:
        histories = [run.get("history", {}) for run in runs]
    else:
        histories = list(histories)
    if len(histories) != len(runs):
        raise ValueError("histories must contain one item per run.")

    change_steps = set()
    for run, history in zip(runs, histories):
        persisted_steps = history.get(
            "optimizer_step_size_change_steps",
            history.get("learning_rate_change_steps", ()),
        )
        persisted_steps = np.asarray(persisted_steps, dtype=int).reshape(-1)
        if persisted_steps.size:
            change_steps.update(int(step) for step in persisted_steps if int(step) > 0)
            continue

        if run.get("optimizer") != "adam":
            continue

        cumulative_step = 0
        schedule = run.get("schedule", ())
        for stage_index, stage in enumerate(schedule):
            try:
                stage_steps = int(stage[0])
            except (TypeError, ValueError, IndexError):
                break
            cumulative_step += stage_steps
            if stage_index >= len(schedule) - 1:
                continue
            try:
                next_multiplier = float(schedule[stage_index + 1][1])
            except (TypeError, ValueError, IndexError):
                break
            if next_multiplier != 1.0:
                change_steps.add(cumulative_step)

    if maximum_step is not None:
        change_steps = {
            step for step in change_steps if step <= int(maximum_step)
        }
    return tuple(sorted(change_steps))


def _draw_schedule_change_lines(axes, runs, *, histories=None, maximum_step=None):
    """Draw configured Adam learning-rate updates on optimisation-step axes."""

    change_steps = _schedule_change_steps(
        runs,
        histories=histories,
        maximum_step=maximum_step,
    )
    for axis in np.atleast_1d(axes).reshape(-1):
        lines = []
        for step in change_steps:
            lines.append(
                axis.axvline(
                    step,
                    color="#666666",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.55,
                    zorder=1.6,
                )
            )
        axis._schedule_change_steps = change_steps
        axis._schedule_change_lines = tuple(lines)
    return change_steps


def _plot_initialization_convergence_legacy(
    runs,
    *,
    log_base_x=None,
    log_base_y=None,
    base_x="axis",
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
):
    """Return Figure 1: score, molecular objective, and penalty convergence."""

    plt, Line2D, _, _, _, _, _, FuncFormatter = _plot_modules()
    if not runs:
        raise ValueError("At least one retrieved run is required for plotting.")
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14.0, 11.5),
        dpi=FIGURE_DPI,
        sharex=True,
        gridspec_kw={"hspace": 0.02},
    )
    finite = [run for run in runs if np.isfinite(run["best_score"])]
    best = max(finite, key=lambda run: run["best_score"]) if finite else None
    score_median = math.nan
    height_ratios = []
    panel_top_ticks = []
    y_labels = _axis_labels(
        y_label,
        (r"$J_{\mathrm{reg}}$", r"$J_{\mathrm{mol}}$", "Penalty"),
        "y_label",
    )
    panels = tuple(zip(("score", "objective", "penalty"), y_labels))
    for axis, (name, ylabel) in zip(axes, panels):
        envelope = _history_envelope([run["history"][name] for run in runs])
        if envelope is not None:
            histories, percentile_10, percentile_90, median = envelope
            if name == "score":
                score_median = float(median[-1])
            for run in runs:
                if run is not best:
                    _draw_history_by_terminal_stability(axis, run, name, len(median))
            steps = np.arange(len(median))
            axis.fill_between(
                steps, percentile_10, percentile_90, color="#8ab8d8", alpha=0.3, linewidth=0.0
            )
            axis.plot(steps, median, color="#2474b5", linewidth=1.2, zorder=3)
            if best is not None:
                _draw_history_by_terminal_stability(
                    axis, best, name, len(median), best=True
                )
        axis.set_ylabel(ylabel)
        data_min, data_max = float(axis.dataLim.ymin), float(axis.dataLim.ymax)
        if (
            log_base_y is None
            and y_range is None
            and np.isfinite(data_min)
            and np.isfinite(data_max)
        ):
            ticks, _ = _nice_convergence_ticks(data_min, data_max)
            axis.set_ylim(ticks[0], ticks[-1])
            axis.set_yticks(ticks)
            height_ratios.append(len(ticks) - 1)
            panel_top_ticks.append(float(ticks[-1]))
        else:
            height_ratios.append(6)
            panel_top_ticks.append(None)
        axis.yaxis.set_major_formatter(FuncFormatter(_format_axis_tick))
        axis.yaxis.get_offset_text().set_visible(False)
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
        _add_stability_fraction(axis, runs, lambda run, metric=name: _history_stable_at_end(run, metric))

    axes[0].get_gridspec().set_height_ratios(height_ratios)
    maximum_step = max(len(run["history"]["score"]) - 1 for run in runs)
    adam_view = all(run["optimizer"] == "adam" for run in runs)
    change_steps = list(_schedule_change_steps(runs, maximum_step=maximum_step))
    if not adam_view:
        change_steps = []
    for axis in axes:
        lines = [
            axis.axvline(
                step,
                color="#666666",
                linestyle="--",
                linewidth=0.9,
                alpha=0.55,
            )
            for step in change_steps
        ]
        axis._schedule_change_steps = tuple(change_steps)
        axis._schedule_change_lines = tuple(lines)
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
    _style_convergence_separators(axes, panel_top_ticks)
    if log_base_x is None:
        ticks = [0, *change_steps]
        if maximum_step not in ticks:
            ticks.append(maximum_step)
        axes[-1].set_xticks(ticks)
    if log_base_x is None:
        axes[-1].set_xlim(0, maximum_step if maximum_step else 1)
    else:
        axes[-1].set_xlim(1, maximum_step if maximum_step > 1 else 10)
    axes[-1].set_xlabel(
        _axis_labels(x_label, ("Cumulative optimisation step",), "x_label")[0]
    )

    if best is not None:
        stored_score_tolerances = best["tolerances"].get(
            "score_tolerance", []
        )
        score_tolerance = (
            float(stored_score_tolerances[-1])
            if len(stored_score_tolerances)
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
    _add_configuration_box(figure, runs, y=0.985)
    for axis, (history_name, _), panel_y_multiplier in zip(
        axes,
        panels,
        _panel_values(y_multiplier, len(axes), "y_multiplier"),
    ):
        _set_axis_data_values(
            axis,
            "x",
            np.concatenate(
                [
                    np.arange(len(run["history"][history_name]), dtype=float)
                    for run in runs
                ]
            ),
        )
        _set_axis_data_values(
            axis,
            "y",
            np.concatenate(
                [
                    np.asarray(run["history"][history_name], dtype=float)
                    for run in runs
                ]
            ),
        )
        _apply_log_scales(
            axis,
            log_base_x=log_base_x,
            log_base_y=log_base_y,
            base_x=base_x,
            base_y=base_y,
            x_multiplier=x_multiplier,
            y_multiplier=panel_y_multiplier,
            x_range=x_range,
            y_range=y_range,
        )
    figure.subplots_adjust(top=0.78, bottom=0.07, left=0.08, right=0.97, hspace=0.02)
    return figure, axes


def _plot_initialization_yield_distribution_legacy(
    runs,
    *,
    log_base_x=None,
    log_base_y=None,
    base_x="axis",
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    point_size=23.0,
):
    """Return Figure 2: molecular-yield strip/box distribution grouped by N."""

    plt, Line2D, _, _, _, _, _, FuncFormatter = _plot_modules()
    if not runs:
        raise ValueError("At least one retrieved run is required for plotting.")
    point_size = _validate_point_size(point_size)
    figure, axis = plt.subplots(figsize=(10.5, 6.6), dpi=FIGURE_DPI)
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
        axis.scatter(
            [positions[N]],
            [float(np.mean(values))],
            s=0.95 * point_size,
            color="#111111",
            zorder=6,
        )
        jitter = np.zeros(1) if len(members) == 1 else np.linspace(-0.12, 0.12, len(members))
        for offset, run in zip(jitter, members):
            stable = _history_stable_at_end(run, "objective")
            axis.scatter(
                positions[N] + offset,
                run["best_objective"],
                s=point_size,
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
    axis.set_xlabel(_axis_labels(x_label, ("",), "x_label")[0])
    axis.set_ylabel(
        _axis_labels(y_label, (r"$\max J_{\mathrm{mol}}$",), "y_label")[0]
    )
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
    _apply_log_scales(
        axis,
        log_base_x=log_base_x,
        log_base_y=log_base_y,
        base_x=base_x,
        base_y=base_y,
        x_multiplier=x_multiplier,
        y_multiplier=y_multiplier,
        x_range=x_range,
        y_range=y_range,
    )
    # Preserve the categorical N labels if a logarithmic x-axis was requested.
    axis.set_xticks(range(1, len(grid_sizes) + 1), [f"$N={N}$" for N in grid_sizes])
    if x_range is not None:
        axis.set_xlim(*_validate_axis_range(x_range, "x_range"))
    figure.subplots_adjust(top=0.82, bottom=0.13, left=0.1, right=0.97)
    return figure, axis


def _plot_initialization_controls_legacy(
    runs,
    *,
    log_base_x=None,
    log_base_y=None,
    base_x="axis",
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
):
    """Return Figure 3: normalized optimized u/v control overlays."""

    plt, _, _, _, _, _, _, _ = _plot_modules()
    if not runs:
        raise ValueError("At least one retrieved run is required for plotting.")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.4),
        dpi=FIGURE_DPI,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    finite = [run for run in runs if np.isfinite(run["best_score"])]
    best = max(finite, key=lambda run: run["best_score"]) if finite else None
    final_time = max(float(run["t_interval"]) for run in finite)
    x_labels = _axis_labels(
        x_label, ("Dimensionless time", "Dimensionless time"), "x_label"
    )
    y_labels = _axis_labels(
        y_label, (r"$u/u_{\max}$", r"$\nu/\nu_{\max}$"), "y_label"
    )
    for axis, name, xlabel, ylabel in zip(
        axes, ("u", "v"), x_labels, y_labels
    ):
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
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xlim(
            min(0.0, *(float(run["t_interval"]) for run in finite)),
            final_time,
        )
        axis.margins(x=0)
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
        _add_stability_fraction(axis, finite, lambda run, control=name: _control_stable(run, control))
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="upper right", bbox_to_anchor=(0.985, 0.89), frameon=False, fontsize=8.2)
    for axis, panel_y_multiplier in zip(
        axes, _panel_values(y_multiplier, len(axes), "y_multiplier")
    ):
        _apply_log_scales(
            axis,
            log_base_x=log_base_x,
            log_base_y=log_base_y,
            base_x=base_x,
            base_y=base_y,
            x_multiplier=x_multiplier,
            y_multiplier=panel_y_multiplier,
            x_range=x_range,
            y_range=y_range,
        )
        _set_control_time_ticks(axis, final_time, x_range)
    figure.subplots_adjust(top=0.84, bottom=0.13, left=0.08, right=0.97, wspace=0.26)
    return figure, axes


def _plot_sweep_convergence(
    runs,
    sweep,
    *,
    log_base_x=None,
    log_base_y=None,
    base_x="axis",
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
):
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
        dpi=FIGURE_DPI,
        sharex=True,
        gridspec_kw={"hspace": 0.02},
    )
    y_labels = _axis_labels(
        y_label,
        (r"$J_{\mathrm{reg}}$", r"$J_{\mathrm{mol}}$", "Penalty"),
        "y_label",
    )
    panels = tuple(zip(("score", "objective", "penalty"), y_labels))
    panel_height_intervals = []
    panel_top_ticks = []
    for run in selected:
        colour = colours[
            _freeze_value(
                _initialization_value(run)
                if sweep.name == INITIALIZATION_PARAMETER
                else run[sweep.name]
            )
        ]
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
    _draw_schedule_change_lines(
        axes,
        selected,
        maximum_step=maximum_step,
    )
    if log_base_x is None:
        axes[-1].set_xlim(0, maximum_step if maximum_step else 1)
    else:
        axes[-1].set_xlim(1, maximum_step if maximum_step > 1 else 10)
    axes[-1].set_xlabel(
        _axis_labels(x_label, ("Cumulative optimisation step",), "x_label")[0]
    )
    for axis in axes:
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
        axis.margins(x=0)
        data_min, data_max = float(axis.dataLim.ymin), float(axis.dataLim.ymax)
        if (
            log_base_y is None
            and y_range is None
            and np.isfinite(data_min)
            and np.isfinite(data_max)
        ):
            ticks, _ = _nice_convergence_ticks(data_min, data_max)
            axis.set_ylim(ticks[0], ticks[-1])
            axis.set_yticks(ticks)
            panel_height_intervals.append(len(ticks) - 1)
            panel_top_ticks.append(float(ticks[-1]))
        else:
            panel_height_intervals.append(6)
            panel_top_ticks.append(None)
        axis.yaxis.set_major_formatter(FuncFormatter(_format_axis_tick))
        axis.yaxis.get_offset_text().set_visible(False)
    # Every vertical interval has the same pixel height across all panels. The
    # lower panel labels each boundary while the upper panel's matching minimum
    # label stays hidden across the narrow physical gutter.
    axes[0].get_gridspec().set_height_ratios(panel_height_intervals)
    _style_convergence_separators(axes, panel_top_ticks)
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
    figure.subplots_adjust(top=0.78, bottom=0.07, left=0.08, right=0.86, hspace=0.02)
    _add_sweep_colourbar(
        figure,
        list(axes),
        sweep,
        mappable,
        coordinates,
        pad=0.015,
        aspect=34,
        shrink=1.0,
    )
    _add_configuration_box(figure, selected, y=0.985, exclude=(sweep.name,))
    for axis, (history_name, _), panel_y_multiplier in zip(
        axes,
        panels,
        _panel_values(y_multiplier, len(axes), "y_multiplier"),
    ):
        _set_axis_data_values(
            axis,
            "x",
            np.concatenate(
                [
                    np.arange(len(run["history"][history_name]), dtype=float)
                    for run in selected
                ]
            ),
        )
        _set_axis_data_values(
            axis,
            "y",
            np.concatenate(
                [
                    np.asarray(run["history"][history_name], dtype=float)
                    for run in selected
                ]
            ),
        )
        _apply_log_scales(
            axis,
            log_base_x=log_base_x,
            log_base_y=log_base_y,
            base_x=base_x,
            base_y=base_y,
            x_multiplier=x_multiplier,
            y_multiplier=panel_y_multiplier,
            x_range=x_range,
            y_range=y_range,
        )
    figure._score_reference_axis = _add_score_reference_ticks(axes[0], selected)
    figure._sweep_parameter = sweep.name
    figure._plotted_run_ids = tuple(run["run_id"] for run in selected)
    return figure, axes


def _jittered_strip_positions(base, count):
    """Spread runs narrowly around one categorical strip position."""

    if count <= 1:
        return np.asarray([base], dtype=float)
    return float(base) + 0.12 * np.linspace(-1.0, 1.0, count)


def _set_categorical_sweep_axis(axis, sweep, positions):
    """Label equally spaced strip positions with their real sweep values."""

    positions = np.asarray(positions, dtype=float)
    axis.set_xscale("linear")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [_sweep_value_label(value) for value in sweep.display_values]
    )
    if positions.size:
        axis.set_xlim(float(positions[0] - 0.38), float(positions[-1] + 0.38))


def _plot_sweep_yield_distribution(
    runs,
    sweep,
    *,
    log_base_x=None,
    log_base_y=None,
    base_x=None,
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    point_size=24.0,
    line_alpha=0.22,
    seed_sensitivity_log_base_y=10,
    seed_sensitivity_base_y=None,
    seed_sensitivity_y_multiplier=1.0,
    seed_sensitivity_y_range=None,
    seed_sensitivity_tolerance=None,
):
    """Plot objectives and seed sensitivity at equally spaced sweep categories.

    The legacy x-axis style arguments remain accepted for call compatibility,
    but the strip positions are always categorical and therefore never use a
    numerical or logarithmic x scale.
    """

    plt, Line2D, _, _, _, _, _, FuncFormatter = _plot_modules()
    _validate_log_base(log_base_x, "log_base_x")
    if base_x != "axis":
        _validate_log_base(base_x, "base_x")
    _validate_multiplier(x_multiplier, "x_multiplier")
    if x_range is not None:
        _validate_axis_range(x_range, "x_range")
    point_size = _validate_point_size(point_size)
    line_alpha = _validate_alpha(line_alpha, "line_alpha")
    seed_sensitivity_log_base_y = _validate_log_base(
        seed_sensitivity_log_base_y,
        "seed_sensitivity_log_base_y",
    )
    if seed_sensitivity_base_y != "axis":
        seed_sensitivity_base_y = _validate_log_base(
            seed_sensitivity_base_y,
            "seed_sensitivity_base_y",
        )
    seed_sensitivity_y_multiplier = _validate_multiplier(
        seed_sensitivity_y_multiplier,
        "seed_sensitivity_y_multiplier",
    )
    if seed_sensitivity_y_range is not None:
        seed_sensitivity_y_range = _validate_axis_range(
            seed_sensitivity_y_range,
            "seed_sensitivity_y_range",
        )
    seed_sensitivity_tolerances = _resolve_seed_sensitivity_tolerances(
        runs, seed_sensitivity_tolerance
    )
    # The strip axis is deliberately categorical. Sweep values remain the tick
    # labels and colour semantics, but cannot distort horizontal spacing.
    colours, _, _ = _sweep_visuals(sweep)
    strip_positions = np.arange(len(sweep.keys), dtype=float)
    groups = _sweep_groups(runs, sweep)
    figure, (axis, sensitivity_axis) = plt.subplots(
        2,
        1,
        figsize=(11.2, 8.2),
        dpi=FIGURE_DPI,
        sharex=True,
        gridspec_kw={"height_ratios": (3.4, 1.0), "hspace": 0.06},
    )
    best_x, best_y, best_colours = [], [], []
    spread_x, spread_bounds, percentile_10, percentile_50, percentile_90, sensitivities = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    scatter_count = 0
    objective_values = []
    for index, key in enumerate(sweep.keys):
        members = [
            run for run in groups[key] if np.isfinite(run["best_objective"])
        ]
        if not members:
            continue
        base = float(strip_positions[index])
        x_values = _jittered_strip_positions(base, len(members))
        linear_half_width = 0.138

        def cap_bounds(factor):
            return (
                base - factor * linear_half_width,
                base + factor * linear_half_width,
            )

        spread_bounds.append((cap_bounds(1.68), cap_bounds(0.9)))
        y_values = np.asarray(
            [run["best_objective"] for run in members], dtype=float
        )
        objective_values.extend(y_values)
        axis.scatter(
            x_values,
            y_values,
            s=point_size,
            color=colours[key],
            alpha=0.72,
            linewidths=0.0,
            zorder=3,
        )
        scatter_count += len(members)
        lower, median, upper = np.percentile(y_values, [10.0, 50.0, 90.0])
        sensitivity = (
            float((upper - lower) / abs(median))
            if not math.isclose(float(median), 0.0)
            else math.inf
        )
        spread_x.append(base)
        percentile_10.append(float(lower))
        percentile_50.append(float(median))
        percentile_90.append(float(upper))
        sensitivities.append(sensitivity)
        finite_scores = [run for run in members if np.isfinite(run["best_score"])]
        if finite_scores:
            best = max(finite_scores, key=lambda run: run["best_score"])
            best_x.append(base)
            best_y.append(float(best["best_objective"]))
            best_colours.append(colours[key])

    if best_x:
        axis.scatter(
            best_x,
            best_y,
            s=1.4 * point_size,
            color=best_colours,
            edgecolor="#202020",
            linewidth=0.65,
            zorder=5,
        )
    spread_colour = "#202020"
    for x_value, (endpoint_bounds, median_bounds), lower, median, upper in zip(
        spread_x,
        spread_bounds,
        percentile_10,
        percentile_50,
        percentile_90,
    ):
        axis.plot(
            [x_value, x_value],
            [lower, upper],
            color=spread_colour,
            linewidth=0.9,
            alpha=line_alpha,
            solid_capstyle="round",
            zorder=2,
        )
        axis.plot(
            [
                endpoint_bounds[0],
                endpoint_bounds[1],
                math.nan,
                endpoint_bounds[0],
                endpoint_bounds[1],
            ],
            [lower, lower, math.nan, upper, upper],
            linewidth=1.25,
            color=spread_colour,
            alpha=line_alpha,
            solid_capstyle="round",
            zorder=2.1,
        )
        axis.plot(
            median_bounds,
            [median, median],
            linewidth=1.0,
            color=spread_colour,
            alpha=line_alpha,
            solid_capstyle="round",
            zorder=2.2,
        )

    finite_sensitivity = np.asarray(sensitivities, dtype=float)
    finite_mask = np.isfinite(finite_sensitivity)
    if np.any(finite_mask):
        finite_x = np.asarray(spread_x, dtype=float)[finite_mask]
        finite_values = finite_sensitivity[finite_mask]
        sensitivity_axis.plot(
            finite_x,
            finite_values,
            color="#444444",
            linewidth=1.0,
            alpha=0.32,
            zorder=2,
        )
        sensitivity_axis.scatter(
            finite_x,
            finite_values,
            s=0.9 * point_size,
            color="#555555",
            edgecolor="none",
            zorder=3,
        )
    for seed_sensitivity_tolerance in seed_sensitivity_tolerances:
        sensitivity_axis.axhline(
            seed_sensitivity_tolerance,
            color="#555555",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            alpha=0.9,
            zorder=5,
            clip_on=False,
        )
    sensitivity_maximum = max(
        [
            value
            for value in sensitivities
            if np.isfinite(value)
        ]
        + list(seed_sensitivity_tolerances)
        + [0.0]
    )
    if sensitivity_maximum > 0.0 and seed_sensitivity_log_base_y is not None:
        sensitivity_floor = min(
            [
                value
                for value in [
                    *sensitivities,
                    *seed_sensitivity_tolerances,
                ]
                if np.isfinite(value) and value > 0.0
            ]
            or [seed_sensitivity_y_multiplier]
        )
        sensitivity_upper = _next_log_major_tick(
            sensitivity_maximum,
            seed_sensitivity_log_base_y,
            sensitivity_floor,
            minimum_exponent=1,
        )
    else:
        sensitivity_upper = (
            1.15 * sensitivity_maximum if sensitivity_maximum > 0.0 else 1.0
        )
    label = SWEEP_LABELS.get(sweep.name, sweep.name.replace("_", " "))
    sensitivity_axis.set_xlabel(_axis_labels(x_label, (label,), "x_label")[0])
    axis.set_ylabel(
        _axis_labels(
            y_label,
            (r"Best molecular objective $J_{\mathrm{mol}}$",),
            "y_label",
        )[0]
    )
    sensitivity_axis.set_ylabel(r"Seed sensitivity $S_J$")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _display_number(value, 6)))
    axis.yaxis.get_offset_text().set_visible(False)
    sensitivity_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: _display_number(value, 4))
    )
    sensitivity_axis.yaxis.get_offset_text().set_visible(False)
    for plot_axis in (axis, sensitivity_axis):
        plot_axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
    _set_axis_data_values(axis, "x", spread_x)
    _set_axis_data_values(axis, "y", objective_values)
    _set_axis_data_values(sensitivity_axis, "x", spread_x)
    _set_axis_data_values(
        sensitivity_axis,
        "y",
        [
            *[value for value in sensitivities if np.isfinite(value)],
            *seed_sensitivity_tolerances,
        ],
    )
    _apply_log_scales(
        axis,
        log_base_x=None,
        log_base_y=log_base_y,
        base_x=None,
        base_y=base_y,
        x_multiplier=1.0,
        y_multiplier=y_multiplier,
        x_range=None,
        y_range=y_range,
    )
    _apply_log_scales(
        sensitivity_axis,
        log_base_x=None,
        log_base_y=seed_sensitivity_log_base_y,
        base_x=None,
        base_y=seed_sensitivity_base_y,
        x_multiplier=1.0,
        y_multiplier=seed_sensitivity_y_multiplier,
        x_range=None,
        y_range=(
            (0.0, sensitivity_upper)
            if seed_sensitivity_y_range is None
            else seed_sensitivity_y_range
        ),
    )
    _set_categorical_sweep_axis(
        sensitivity_axis,
        sweep,
        strip_positions,
    )
    axis.tick_params(axis="x", which="minor", bottom=False)
    axis.tick_params(axis="x", which="major", bottom=True, labelbottom=False)
    sensitivity_axis.tick_params(axis="x", which="minor", bottom=False)
    sensitivity_axis.tick_params(axis="x", which="major", bottom=True)
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color="#777777", label="All initializations"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="#777777",
            markeredgecolor="#202020",
            label="Best score at each value",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="best",
        frameon=True,
        fontsize=8.5,
    )
    figure.subplots_adjust(top=0.94, bottom=0.11, left=0.1, right=0.97, hspace=0.06)
    figure._sweep_parameter = sweep.name
    figure._scatter_run_count = scatter_count
    figure._seed_sensitivity_axis = sensitivity_axis
    figure._strip_positions = tuple(strip_positions)
    figure._seed_sensitivity = tuple(
        zip(
            sweep.display_values,
            sensitivities,
            percentile_10,
            percentile_50,
            percentile_90,
        )
    )
    figure._seed_sensitivity_tolerances = seed_sensitivity_tolerances
    return figure, axis


def _plot_sweep_controls(
    runs,
    sweep,
    *,
    log_base_x=None,
    log_base_y=None,
    base_x="axis",
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
):
    """Overlay best-checkpoint controls from the best score at each sweep value."""

    plt, Line2D, _, _, _, _, _, _ = _plot_modules()
    selected = _best_sweep_runs(runs, sweep)
    if not selected:
        raise ValueError("The parameter sweep contains no finite best scores.")
    colours, mappable, coordinates = _sweep_visuals(sweep)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.7),
        dpi=FIGURE_DPI,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    final_time = max(float(run["t_interval"]) for run in selected)
    x_labels = _axis_labels(
        x_label, ("Dimensionless time", "Dimensionless time"), "x_label"
    )
    y_labels = _axis_labels(
        y_label, (r"$u/u_{\max}$", r"$\nu/\nu_{\max}$"), "y_label"
    )
    for axis, name, xlabel, ylabel, panel_y_multiplier in zip(
        axes,
        ("u", "v"),
        x_labels,
        y_labels,
        _panel_values(y_multiplier, len(axes), "y_multiplier"),
    ):
        for run in selected:
            values = np.asarray(run["controls"]["best"][name], dtype=float) / float(
                run[f"{name}_max"]
            )
            time = np.linspace(0.0, float(run["t_interval"]), len(values))
            stable = _control_stable(run, name)
            axis.plot(
                time,
                values,
                color=colours[
                    _freeze_value(
                        _initialization_value(run)
                        if sweep.name == INITIALIZATION_PARAMETER
                        else run[sweep.name]
                    )
                ],
                alpha=1.0 if stable else 0.55,
                linewidth=1.15,
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xlim(
            min(0.0, *(float(run["t_interval"]) for run in selected)),
            final_time,
        )
        axis.margins(x=0)
        axis.grid(color="#d8d8d8", linewidth=0.5, alpha=0.35)
        _set_axis_data_values(
            axis,
            "x",
            np.concatenate(
                [
                    np.linspace(
                        0.0,
                        float(run["t_interval"]),
                        len(run["controls"]["best"][name]),
                    )
                    for run in selected
                ]
            ),
        )
        _set_axis_data_values(
            axis,
            "y",
            np.concatenate(
                [
                    np.asarray(run["controls"]["best"][name], dtype=float)
                    / float(run[f"{name}_max"])
                    for run in selected
                ]
            ),
        )
        _apply_log_scales(
            axis,
            log_base_x=log_base_x,
            log_base_y=log_base_y,
            base_x=base_x,
            base_y=base_y,
            x_multiplier=x_multiplier,
            y_multiplier=panel_y_multiplier,
            x_range=x_range,
            y_range=y_range,
        )
        _set_control_time_ticks(axis, final_time, x_range)
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
    figure.subplots_adjust(top=0.86, bottom=0.28, left=0.08, right=0.97, wspace=0.25)
    figure._sweep_parameter = sweep.name
    figure._plotted_run_ids = tuple(run["run_id"] for run in selected)
    return figure, axes


def _sample_summary_history(values, steps):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = np.flatnonzero(np.isfinite(values))
    if not finite.size:
        return np.full(steps.size, np.nan)
    return values[np.minimum(steps, int(finite[-1]))]


def _last_finite_tolerance(tolerances, name):
    values = np.asarray(tolerances.get(name, []), dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else math.nan


def _plot_sweep_run_summaries_legacy(
    runs,
    sweep,
    *,
    load_history,
    load_tolerances,
    load_controls,
    history_points=1200,
):
    """Plot a compact convergence/control row for every value in one sweep.

    The loader callbacks keep large histories and controls out of the initial
    query while preserving the plotting module's database isolation boundary.
    """

    if isinstance(history_points, bool) or not isinstance(history_points, int):
        raise ValueError("history_points must be a positive integer.")
    if history_points < 1:
        raise ValueError("history_points must be a positive integer.")
    runs = list(runs)
    groups = _sweep_groups(runs, sweep)
    entries = [
        (key, display_value, groups[key])
        for key, display_value in zip(sweep.keys, sweep.display_values)
        if any(np.isfinite(run["best_score"]) for run in groups[key])
    ]
    if not entries:
        raise ValueError("The parameter sweep contains no finite best scores.")

    plt, Line2D, _, _, _, _, _, _ = _plot_modules()
    figure = plt.figure(
        figsize=(16.0, 2.05 * len(entries) + 0.8),
        dpi=FIGURE_DPI,
        layout="constrained",
    )
    outer = figure.add_gridspec(
        len(entries),
        2,
        width_ratios=(1.25, 1.75),
        hspace=0.10,
        wspace=0.12,
    )
    sweep_label = SWEEP_LABELS.get(sweep.name, sweep.name.replace("_", " "))
    summary_records = []

    for row_index, (_, display_value, members) in enumerate(entries):
        members = [run for run in members if np.isfinite(run["best_score"])]
        best = max(members, key=lambda run: run["best_score"])
        best_index = members.index(best)
        histories = [load_history(run["run_id"]) for run in members]
        tolerances = [load_tolerances(run["run_id"]) for run in members]
        controls = [load_controls(run["run_id"]) for run in members]

        maximum_step = max(len(history["score"]) - 1 for history in histories)
        sample_steps = np.unique(
            np.linspace(
                0,
                maximum_step,
                min(history_points, maximum_step + 1),
            ).astype(int)
        )
        score_matrix = np.asarray(
            [
                _sample_summary_history(history["score"], sample_steps)
                for history in histories
            ],
            dtype=float,
        )
        lower, median, upper = np.nanpercentile(
            score_matrix,
            [10.0, 50.0, 90.0],
            axis=0,
        )
        score_axis = figure.add_subplot(outer[row_index, 0])
        score_axis.fill_between(
            sample_steps,
            lower,
            upper,
            color="#8ab8d8",
            alpha=0.30,
            linewidth=0.0,
        )
        score_axis.plot(
            sample_steps,
            median,
            color="#2474b5",
            linewidth=0.85,
            label="Median (10–90%)",
        )
        score_axis.plot(
            sample_steps,
            score_matrix[best_index],
            color="#b33f36",
            linewidth=1.0,
            zorder=3,
            label="Best score",
        )
        _draw_schedule_change_lines(
            score_axis,
            members,
            histories=histories,
            maximum_step=maximum_step,
        )
        score_axis.set_ylabel(r"$J_{\mathrm{reg}}$", fontsize=8.0)
        score_axis.grid(color="#d8d8d8", linewidth=0.4, alpha=0.35)
        score_axis.tick_params(labelsize=6.8)
        score_axis.margins(x=0)
        if maximum_step > 1:
            _set_axis_data_values(
                score_axis,
                "x",
                [0.0, 1.0, float(maximum_step)],
            )
            _apply_log_scales(
                score_axis,
                log_base_x=10,
                base_x=None,
                x_range=(0.0, float(maximum_step)),
            )
        if row_index == len(entries) - 1:
            score_axis.set_xlabel("Optimisation step", fontsize=8.0)
        else:
            score_axis.tick_params(axis="x", labelbottom=False)

        score_metrics = [
            _last_finite_tolerance(item, "score_tolerance")
            for item in tolerances
        ]
        score_stable = [
            run.get("J_tol") is None
            or (np.isfinite(metric) and metric < float(run["J_tol"]))
            for run, metric in zip(members, score_metrics)
        ]
        best_scores = np.asarray(
            [run["best_score"] for run in members], dtype=float
        )
        score_axis.text(
            0.012,
            0.975,
            (
                f"{sweep_label}={_sweep_value_label(display_value)}\n"
                f"Best score: {_display_number(best['best_score'], 5)}; "
                f"median: {_display_number(np.nanmedian(best_scores), 5)}\n"
                f"Score stable: {sum(score_stable)}/{len(score_stable)}"
            ),
            transform=score_axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#d2d2ce",
                "alpha": 0.90,
            },
        )
        if row_index == 0:
            score_axis.legend(fontsize=6.8, loc="best", frameon=False)

        controls_grid = outer[row_index, 1].subgridspec(1, 2, wspace=0.08)
        control_axes = [
            figure.add_subplot(controls_grid[0, control_index])
            for control_index in range(2)
        ]
        control_summary = {}
        for control_axis, name, label in zip(
            control_axes,
            ("u", "v"),
            (r"$u$", r"$\nu$"),
        ):
            metric_name = f"{name}_tolerance"
            threshold_name = f"{name}_tol"
            control_metrics = [
                _last_finite_tolerance(item, metric_name) for item in tolerances
            ]
            control_stable = [
                run[threshold_name] is None
                or (
                    np.isfinite(metric)
                    and metric < float(run[threshold_name])
                )
                for run, metric in zip(members, control_metrics)
            ]
            for item, run, is_stable in zip(controls, members, control_stable):
                values = np.asarray(item[name], dtype=float)
                time = np.linspace(0.0, float(run["t_interval"]), len(values))
                control_axis.plot(
                    time,
                    values,
                    color="#555555",
                    alpha=0.30 if is_stable else 0.13,
                    linewidth=0.50,
                )
            best_values = np.asarray(controls[best_index][name], dtype=float)
            best_time = np.linspace(
                0.0,
                float(best["t_interval"]),
                len(best_values),
            )
            control_axis.plot(
                best_time,
                best_values,
                color="#b33f36",
                linewidth=1.0,
            )
            control_axis.set_ylabel(label, fontsize=8.0)
            control_axis.tick_params(labelsize=6.8)
            control_axis.grid(color="#d8d8d8", linewidth=0.4, alpha=0.35)
            control_axis.margins(x=0)
            final_time = max(float(run["t_interval"]) for run in members)
            _set_control_time_ticks(control_axis, final_time)
            if row_index == len(entries) - 1:
                control_axis.set_xlabel("Dimensionless time", fontsize=8.0)
            else:
                control_axis.tick_params(axis="x", labelbottom=False)
            best_metric = control_metrics[best_index]
            best_threshold = best[threshold_name]
            control_axis.text(
                0.012,
                0.975,
                (
                    f"Stable: {sum(control_stable)}/{len(control_stable)}\n"
                    f"Best stability: {_display_number(best_metric, 4)}; "
                    f"tol: {_display_number(best_threshold, 4) if best_threshold is not None else 'none'}"
                ),
                transform=control_axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#d2d2ce",
                    "alpha": 0.88,
                },
            )
            control_summary[name] = {
                "stable": int(sum(control_stable)),
                "total": len(control_stable),
                "best_metric": best_metric,
                "best_tolerance": best_threshold,
            }

        summary_records.append(
            {
                "sweep_value": display_value,
                "best_score": float(best["best_score"]),
                "median_best_score": float(np.nanmedian(best_scores)),
                "score_stable": int(sum(score_stable)),
                "total": len(members),
                "controls": control_summary,
            }
        )

    figure._sweep_parameter = sweep.name
    figure._summary_records = tuple(summary_records)
    return figure


def _summary_sample_steps(histories, history_points):
    maximum_step = max(len(history["score"]) - 1 for history in histories)
    steps = np.unique(
        np.linspace(
            0,
            maximum_step,
            min(history_points, maximum_step + 1),
        ).astype(int)
    )
    return maximum_step, steps


def _summary_best_by_sweep(runs, sweep):
    return _best_sweep_runs(runs, sweep)


def _summary_rectangle(
    figure,
    subplot_spec,
    runs,
    *,
    load_history,
    load_tolerances,
    load_controls,
    history_points,
    title="",
    colours=None,
    include_median_tick=True,
    show_stability_metadata=True,
    compact=False,
    show_step_axis=True,
    show_score_y=True,
    show_u_y=True,
    show_v_y=True,
    show_reference_ticks=True,
    colourbar_spec=None,
    show_colourbar_label=True,
):
    """Draw one flush score/u/v rectangle and return its axes plus metadata."""

    finite = [run for run in runs if np.isfinite(run.get("best_score", np.nan))]
    if not finite:
        return None
    best = max(finite, key=lambda run: run["best_score"])
    histories = [load_history(run["run_id"]) for run in finite]
    controls = [load_controls(run["run_id"]) for run in finite]
    tolerances = (
        [load_tolerances(run["run_id"]) for run in finite]
        if show_stability_metadata
        else [None] * len(finite)
    )
    maximum_step, sample_steps = _summary_sample_steps(histories, history_points)
    score_matrix = np.asarray(
        [
            _sample_summary_history(history["score"], sample_steps)
            for history in histories
        ],
        dtype=float,
    )

    has_colourbar = colourbar_spec is not None
    inner = subplot_spec.subgridspec(
        3 if has_colourbar else 2,
        2,
        height_ratios=(1.18, 1.0, 0.12) if has_colourbar else (1.18, 1.0),
        hspace=0.0,
        wspace=0.0,
    )
    score_axis = figure.add_subplot(inner[0, :])
    u_axis = figure.add_subplot(inner[1, 0])
    v_axis = figure.add_subplot(inner[1, 1])
    colourbar_axis = figure.add_subplot(inner[2, :]) if has_colourbar else None

    if colours is None:
        for values in score_matrix:
            score_axis.plot(
                sample_steps,
                values,
                color="#59636b",
                linewidth=0.52,
                alpha=0.24,
                zorder=1.8,
            )
        lower, median_history, upper = np.nanpercentile(
            score_matrix, [10.0, 50.0, 90.0], axis=0
        )
        score_axis.fill_between(
            sample_steps,
            lower,
            upper,
            color="#8ab8d8",
            alpha=0.28,
            linewidth=0.0,
        )
        score_axis.plot(
            sample_steps,
            median_history,
            color="#2474b5",
            linewidth=0.85,
        )
        best_index = finite.index(best)
        score_axis.plot(
            sample_steps,
            score_matrix[best_index],
            color="#b33f36",
            linewidth=1.05,
            zorder=3,
        )
    else:
        for run, values in zip(finite, score_matrix):
            score_axis.plot(
                sample_steps,
                values,
                color=colours[run["run_id"]],
                linewidth=1.0 if run is best else 0.8,
                alpha=1.0 if run is best else 0.82,
            )

    _draw_schedule_change_lines(
        score_axis,
        finite,
        histories=histories,
        maximum_step=maximum_step,
    )

    score_axis.grid(color="#d8d8d8", linewidth=0.4, alpha=0.35)
    score_axis.margins(x=0)
    score_axis.set_ylabel(
        r"$J_{\mathrm{reg}}$" if show_score_y else "",
        fontsize=7.6 if compact else 8.2,
    )
    score_axis.xaxis.set_label_position("top")
    score_axis.xaxis.tick_top()
    score_axis.tick_params(
        axis="x",
        which="both",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        labelsize=6.2 if compact else 6.8,
    )
    score_axis.tick_params(
        axis="y",
        labelleft=True,
        left=True,
        labelsize=6.2 if compact else 6.8,
    )
    score_axis.set_xlabel(
        "Optimisation step" if show_step_axis else "",
        fontsize=7.2 if compact else 8.0,
    )
    if title:
        score_axis.set_title(
            title,
            loc="left",
            fontsize=7.4 if compact else 8.4,
            pad=3.0,
        )
    if maximum_step > 1:
        _set_axis_data_values(score_axis, "x", [0.0, 1.0, float(maximum_step)])
        _apply_log_scales(
            score_axis,
            log_base_x=10,
            base_x=None,
            x_range=(0.0, float(maximum_step)),
        )
    _add_score_reference_ticks(
        score_axis,
        finite,
        include_median=include_median_tick,
        show_ticks=show_reference_ticks,
    )

    for axis, name, label, show_y in (
        (u_axis, "u", r"$u$", show_u_y),
        (v_axis, "v", r"$\nu$", show_v_y),
    ):
        for item, run in zip(controls, finite):
            values = np.asarray(item[name], dtype=float)
            control_position = np.linspace(0.0, 1.0, len(values))
            if colours is None:
                colour = "#b33f36" if run is best else "#555555"
                alpha = 1.0 if run is best else 0.20
                linewidth = 1.0 if run is best else 0.48
            else:
                colour = colours[run["run_id"]]
                alpha = 1.0 if run is best else 0.82
                linewidth = 1.0 if run is best else 0.72
            axis.plot(
                control_position,
                values,
                color=colour,
                alpha=alpha,
                linewidth=linewidth,
            )
        axis.grid(color="#d8d8d8", linewidth=0.4, alpha=0.35)
        axis.margins(x=0)
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([])
        axis.set_xlabel("")
        axis.set_ylabel(label if show_y else "", fontsize=7.6 if compact else 8.2)
        axis.tick_params(
            axis="y",
            labelsize=6.2 if compact else 6.8,
            labelleft=axis is u_axis,
            left=axis is u_axis,
            labelright=axis is v_axis,
            right=axis is v_axis,
        )
    u_cap = max(float(run["u_max"]) for run in finite)
    u_axis.set_ylim(0.0, u_cap)
    u_axis.yaxis.tick_left()
    u_axis.yaxis.set_label_position("left")
    v_axis.yaxis.tick_right()
    v_axis.yaxis.set_label_position("right")
    u_axis.spines["right"].set_visible(False)
    v_axis.spines["left"].set_visible(True)
    v_axis.spines["left"].set_color("#777777")
    v_axis.spines["left"].set_linewidth(0.6)

    summary_colourbar = None
    if colourbar_spec is not None:
        colour_sweep, mappable, coordinates = colourbar_spec
        summary_colourbar = _add_sweep_colourbar(
            figure,
            (score_axis, u_axis, v_axis),
            colour_sweep,
            mappable,
            coordinates,
            orientation="horizontal",
            cax=colourbar_axis,
            show_label=show_colourbar_label,
        )
        summary_colourbar.ax.tick_params(labelsize=6.0 if compact else 6.6, pad=1.0)
        summary_colourbar.ax.xaxis.label.set_size(7.0 if compact else 7.6)
        summary_colourbar.ax.xaxis.labelpad = 1.0
        score_axis._summary_colourbar = summary_colourbar

    best_scores = np.asarray([run["best_score"] for run in finite], dtype=float)
    record = {
        "title": title,
        "run_ids": tuple(run["run_id"] for run in finite),
        "best_score": float(np.max(best_scores)),
        "median_best_score": float(np.median(best_scores)),
        "u_cap": u_cap,
    }
    if show_stability_metadata:
        score_metrics = [
            _last_finite_tolerance(item, "score_tolerance") for item in tolerances
        ]
        record["score_stable"] = int(
            sum(
                run.get("J_tol") is None
                or (np.isfinite(metric) and metric < float(run["J_tol"]))
                for run, metric in zip(finite, score_metrics)
            )
        )
        control_records = {}
        for name in ("u", "v"):
            metrics = [
                _last_finite_tolerance(item, f"{name}_tolerance")
                for item in tolerances
            ]
            stable = [
                run[f"{name}_tol"] is None
                or (
                    np.isfinite(metric)
                    and metric < float(run[f"{name}_tol"])
                )
                for run, metric in zip(finite, metrics)
            ]
            control_records[name] = {"stable": int(sum(stable)), "total": len(stable)}
        record["controls"] = control_records
    return (score_axis, u_axis, v_axis), record


def _validate_summary_arguments(runs, history_points, sweeps):
    if isinstance(history_points, bool) or not isinstance(history_points, int) or history_points < 1:
        raise ValueError("history_points must be a positive integer.")
    runs = list(runs)
    if not runs:
        raise ValueError("At least one retrieved run is required for plotting.")
    names = [sweep.name for sweep in sweeps]
    if len(set(names)) != len(names):
        raise ValueError("Summary sweep parameters must be distinct.")
    return runs


def _summary_entries(runs, sweep):
    groups = _sweep_groups(runs, sweep)
    return [
        (display_value, groups[key])
        for key, display_value in zip(sweep.keys, sweep.display_values)
        if any(np.isfinite(run.get("best_score", np.nan)) for run in groups[key])
    ]


def plot_single_sweep_summary(
    runs,
    sweep,
    *,
    load_history,
    load_tolerances,
    load_controls,
    history_points=1200,
):
    """Plot one flush score/u/v rectangle for every value of one sweep."""

    runs = _validate_summary_arguments(runs, history_points, (sweep,))
    entries = _summary_entries(runs, sweep)
    if not entries:
        raise ValueError("The parameter sweep contains no finite best scores.")
    plt, _, _, _, _, _, _, _ = _plot_modules()
    figure = plt.figure(
        figsize=(11.2, 4.15 * len(entries)),
        dpi=FIGURE_DPI,
    )
    outer = figure.add_gridspec(len(entries), 1, hspace=0.20)
    figure.subplots_adjust(left=0.13, right=0.92, bottom=0.06, top=0.95)
    label = SWEEP_LABELS.get(sweep.name, sweep.name.replace("_", " "))
    records = []
    axes = []
    sweep_labels = []
    for index, (display_value, members) in enumerate(entries):
        result = _summary_rectangle(
            figure,
            outer[index, 0],
            members,
            load_history=load_history,
            load_tolerances=load_tolerances,
            load_controls=load_controls,
            history_points=history_points,
            show_step_axis=index == 0,
        )
        if result is not None:
            panel_axes, record = result
            axes.extend(panel_axes)
            record["sweep_value"] = display_value
            records.append(record)
        position = outer[index, 0].get_position(figure)
        sweep_labels.append(
            figure.text(
                position.x0 - 0.045,
                0.5 * (position.y0 + position.y1),
                f"{label} = {_sweep_value_label(display_value)}",
                rotation=90,
                ha="center",
                va="center",
                fontsize=8.4,
            )
        )
    figure._sweep_parameter = sweep.name
    figure._summary_records = tuple(records)
    figure._summary_axes = tuple(axes)
    figure._summary_sweep_labels = tuple(sweep_labels)
    return figure


def plot_double_sweep_summary(
    runs,
    *,
    separate_sweep,
    colour_sweep,
    load_history,
    load_tolerances,
    load_controls,
    history_points=1200,
):
    """Repeat summary rectangles by one sweep and colour best runs by another."""

    runs = _validate_summary_arguments(
        runs, history_points, (separate_sweep, colour_sweep)
    )
    entries = _summary_entries(runs, separate_sweep)
    if not entries:
        raise ValueError("The selected double sweep contains no finite best scores.")
    colours, mappable, coordinates = _sweep_visuals(colour_sweep)
    plt, _, _, _, _, _, _, _ = _plot_modules()
    figure = plt.figure(
        figsize=(11.6, 4.45 * len(entries)),
        dpi=FIGURE_DPI,
    )
    outer = figure.add_gridspec(len(entries), 1, hspace=0.18)
    figure.subplots_adjust(left=0.13, right=0.92, bottom=0.06, top=0.95)
    label = SWEEP_LABELS.get(
        separate_sweep.name, separate_sweep.name.replace("_", " ")
    )
    records = []
    axes = []
    colourbars = []
    sweep_labels = []
    for index, (display_value, members) in enumerate(entries):
        selected = _summary_best_by_sweep(members, colour_sweep)
        run_colours = {
            run["run_id"]: colours[_freeze_value(run[colour_sweep.name])]
            for run in selected
        }
        result = _summary_rectangle(
            figure,
            outer[index, 0],
            selected,
            load_history=load_history,
            load_tolerances=load_tolerances,
            load_controls=load_controls,
            history_points=history_points,
            colours=run_colours,
            show_step_axis=index == 0,
            colourbar_spec=(colour_sweep, mappable, coordinates),
        )
        if result is not None:
            panel_axes, record = result
            axes.extend(panel_axes)
            record["separate_sweep_value"] = display_value
            records.append(record)
            colourbars.append(panel_axes[0]._summary_colourbar)
        position = outer[index, 0].get_position(figure)
        sweep_labels.append(
            figure.text(
                position.x0 - 0.045,
                0.5 * (position.y0 + position.y1),
                f"{label} = {_sweep_value_label(display_value)}",
                rotation=90,
                ha="center",
                va="center",
                fontsize=8.4,
            )
        )
    figure._separate_sweep_parameter = separate_sweep.name
    figure._colour_sweep_parameter = colour_sweep.name
    figure._summary_records = tuple(records)
    figure._summary_axes = tuple(axes)
    figure._summary_colourbars = tuple(colourbars)
    figure._summary_colourbar = colourbars[0]
    figure._summary_sweep_labels = tuple(sweep_labels)
    return figure


def plot_triple_sweep_summary(
    runs,
    *,
    row_sweep,
    column_sweep,
    colour_sweep,
    load_history,
    load_tolerances,
    load_controls,
    history_points=1200,
):
    """Create a row/column matrix whose best combination runs use a colour sweep."""

    runs = _validate_summary_arguments(
        runs, history_points, (row_sweep, column_sweep, colour_sweep)
    )
    row_groups = _sweep_groups(runs, row_sweep)
    column_groups = _sweep_groups(runs, column_sweep)
    colours, mappable, coordinates = _sweep_visuals(colour_sweep)
    plt, _, _, _, _, _, _, _ = _plot_modules()
    row_count = len(row_sweep.keys)
    column_count = len(column_sweep.keys)
    figure = plt.figure(
        figsize=(7.0 * column_count, 4.0 * row_count),
        dpi=FIGURE_DPI,
    )
    outer = figure.add_gridspec(
        row_count,
        column_count,
        hspace=0.18,
        wspace=0.14,
    )
    figure.subplots_adjust(left=0.09, right=0.95, bottom=0.06, top=0.91)
    axes = []
    records = []
    colourbars = []
    for row_index, (row_key, row_value) in enumerate(
        zip(row_sweep.keys, row_sweep.display_values)
    ):
        row_ids = {run["run_id"] for run in row_groups[row_key]}
        for column_index, (column_key, column_value) in enumerate(
            zip(column_sweep.keys, column_sweep.display_values)
        ):
            members = [
                run
                for run in column_groups[column_key]
                if run["run_id"] in row_ids
            ]
            selected = _summary_best_by_sweep(members, colour_sweep)
            if not selected:
                continue
            run_colours = {
                run["run_id"]: colours[_freeze_value(run[colour_sweep.name])]
                for run in selected
            }
            result = _summary_rectangle(
                figure,
                outer[row_index, column_index],
                selected,
                load_history=load_history,
                load_tolerances=load_tolerances,
                load_controls=load_controls,
                history_points=history_points,
                colours=run_colours,
                include_median_tick=False,
                show_stability_metadata=False,
                compact=True,
                show_step_axis=row_index == 0,
                show_score_y=column_index == 0,
                show_u_y=column_index == 0,
                show_v_y=column_index == column_count - 1,
                show_reference_ticks=True,
                colourbar_spec=(colour_sweep, mappable, coordinates),
                show_colourbar_label=row_index == row_count - 1,
            )
            if result is not None:
                panel_axes, record = result
                axes.extend(panel_axes)
                record.update(
                    {
                        "row_sweep_value": row_value,
                        "column_sweep_value": column_value,
                    }
                )
                records.append(record)
                colourbars.append(panel_axes[0]._summary_colourbar)
    if not axes:
        raise ValueError("The selected triple sweep contains no finite best scores.")
    row_label = SWEEP_LABELS.get(row_sweep.name, row_sweep.name.replace("_", " "))
    column_label = SWEEP_LABELS.get(
        column_sweep.name, column_sweep.name.replace("_", " ")
    )
    row_labels = []
    for row_index, row_value in enumerate(row_sweep.display_values):
        position = outer[row_index, 0].get_position(figure)
        row_labels.append(
            figure.text(
                position.x0 - 0.035,
                0.5 * (position.y0 + position.y1),
                f"{row_label} = {_sweep_value_label(row_value)}",
                rotation=90,
                ha="center",
                va="center",
                fontsize=7.8,
            )
        )
    column_labels = []
    for column_index, column_value in enumerate(column_sweep.display_values):
        position = outer[0, column_index].get_position(figure)
        column_labels.append(
            figure.text(
                0.5 * (position.x0 + position.x1),
                position.y1 + 0.045,
                f"{column_label} = {_sweep_value_label(column_value)}",
                ha="center",
                va="center",
                fontsize=7.8,
            )
        )
    figure._row_sweep_parameter = row_sweep.name
    figure._column_sweep_parameter = column_sweep.name
    figure._colour_sweep_parameter = colour_sweep.name
    figure._summary_records = tuple(records)
    figure._summary_axes = tuple(axes)
    figure._summary_colourbars = tuple(colourbars)
    figure._summary_colourbar = colourbars[0]
    figure._summary_row_labels = tuple(row_labels)
    figure._summary_column_labels = tuple(column_labels)
    return figure


def plot_sweep_run_summaries(
    runs,
    sweep,
    *,
    load_history,
    load_tolerances,
    load_controls,
    history_points=1200,
):
    """Backward-compatible alias for :func:`plot_single_sweep_summary`."""

    return plot_single_sweep_summary(
        runs,
        sweep,
        load_history=load_history,
        load_tolerances=load_tolerances,
        load_controls=load_controls,
        history_points=history_points,
    )


def plot_convergence(
    runs,
    *,
    sweep=None,
    log_base_x=None,
    log_base_y=None,
    base_x="axis",
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
):
    """Plot convergence for a supplied sweep or a numbered initialization sweep."""

    runs = list(runs)
    sweep = _make_initialization_sweep(runs) if sweep is None else sweep
    return _plot_sweep_convergence(
        runs,
        sweep,
        log_base_x=log_base_x,
        log_base_y=log_base_y,
        base_x=base_x,
        base_y=base_y,
        x_multiplier=x_multiplier,
        y_multiplier=y_multiplier,
        x_range=x_range,
        y_range=y_range,
        x_label=x_label,
        y_label=y_label,
    )


def plot_yield_distribution(
    runs,
    *,
    sweep=None,
    log_base_x=None,
    log_base_y=None,
    base_x=None,
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    point_size=24.0,
    line_alpha=0.22,
    seed_sensitivity_log_base_y=10,
    seed_sensitivity_base_y=None,
    seed_sensitivity_y_multiplier=1.0,
    seed_sensitivity_y_range=None,
    seed_sensitivity_tolerance=None,
):
    """Plot yield for a supplied sweep or a numbered initialization sweep."""

    runs = list(runs)
    sweep = _make_initialization_sweep(runs) if sweep is None else sweep
    return _plot_sweep_yield_distribution(
        runs,
        sweep,
        log_base_x=log_base_x,
        log_base_y=log_base_y,
        base_x=base_x,
        base_y=base_y,
        x_multiplier=x_multiplier,
        y_multiplier=y_multiplier,
        x_range=x_range,
        y_range=y_range,
        x_label=x_label,
        y_label=y_label,
        point_size=point_size,
        line_alpha=line_alpha,
        seed_sensitivity_log_base_y=seed_sensitivity_log_base_y,
        seed_sensitivity_base_y=seed_sensitivity_base_y,
        seed_sensitivity_y_multiplier=seed_sensitivity_y_multiplier,
        seed_sensitivity_y_range=seed_sensitivity_y_range,
        seed_sensitivity_tolerance=seed_sensitivity_tolerance,
    )


def plot_controls(
    runs,
    *,
    sweep=None,
    log_base_x=None,
    log_base_y=None,
    base_x="axis",
    base_y="axis",
    x_multiplier=1.0,
    y_multiplier=1.0,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
):
    """Plot controls for a supplied sweep or a numbered initialization sweep."""

    runs = list(runs)
    sweep = _make_initialization_sweep(runs) if sweep is None else sweep
    return _plot_sweep_controls(
        runs,
        sweep,
        log_base_x=log_base_x,
        log_base_y=log_base_y,
        base_x=base_x,
        base_y=base_y,
        x_multiplier=x_multiplier,
        y_multiplier=y_multiplier,
        x_range=x_range,
        y_range=y_range,
        x_label=x_label,
        y_label=y_label,
    )


def plot_standard_figures(runs, *, sweep_parameter=None):
    """Return one universal three-figure view for a parameter or initialization sweep."""

    runs = list(runs)
    sweep = _detect_sweep(runs, sweep_parameter)
    return {
        "convergence": plot_convergence(
            runs,
            sweep=sweep,
            log_base_x=10,
            log_base_y=2,
            base_x=None,
            base_y=None,
        ),
        "distribution": plot_yield_distribution(
            runs,
            sweep=sweep,
            log_base_y=10,
            base_y=None,
            point_size=12.0,
            line_alpha=0.22,
        ),
        "controls": plot_controls(runs, sweep=sweep),
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
            options = {"dpi": PNG_DPI} if file_format == "png" else {}
            figure.savefig(path, bbox_inches="tight", **options)
            saved[name][file_format] = path
        if close:
            plt.close(figure)
    return saved
