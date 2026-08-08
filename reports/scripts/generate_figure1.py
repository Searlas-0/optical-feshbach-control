#!/usr/bin/env python3
"""Generate Figure 1, the optical-Feshbach control schematic.

The default output path is resolved relative to this file, so the script can
be run from either the repository root or the ``reports`` directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon


REPORT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPORT_DIR / "figures" / "figure1_ofr_schematic.pdf"


def scattering_ratio(u: float, v: np.ndarray) -> np.ndarray:
    """Return a_c/a_bg for dimensionless intensity u and detuning v."""

    return 1.0 + u / (-v - u + 0.5j)


def draw_level_diagram(ax: plt.Axes) -> None:
    """Draw the open/closed-channel optical coupling in panel (a)."""

    blue = "#2474A6"
    red = "#B6403A"
    gold = "#D49A22"
    grey = "#5B6573"

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # The slight slope and multiple levels indicate an open-channel continuum.
    continuum = Polygon(
        [(0.04, 0.13), (0.56, 0.19), (0.56, 0.32), (0.04, 0.26)],
        closed=True,
        facecolor="#DCECF4",
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(continuum)
    for offset in (0.02, 0.055, 0.09):
        ax.plot(
            [0.06, 0.53],
            [0.15 + offset, 0.21 + offset],
            color=blue,
            lw=0.75,
            alpha=0.8,
        )
    ax.text(
        0.05,
        0.08,
        r"open-channel continuum $|o\rangle$",
        color=blue,
        fontsize=6.2,
        ha="left",
        va="center",
    )

    # The laser photon reaches a virtual energy (dashed); nu is its detuning
    # from the excited molecular level.
    ax.plot([0.16, 0.63], [0.82, 0.82], color=red, lw=2.0, solid_capstyle="round")
    ax.text(
        0.16,
        0.88,
        r"excited molecule $|c\rangle$",
        color=red,
        fontsize=6.2,
        ha="left",
        va="center",
    )
    ax.plot([0.14, 0.64], [0.68, 0.68], color=grey, lw=0.8, ls=(0, (2, 2)))
    ax.text(0.15, 0.605, r"laser energy $\hbar\omega_L$", color=grey, fontsize=5.7)

    detuning = FancyArrowPatch(
        (0.58, 0.685),
        (0.58, 0.815),
        arrowstyle="<->",
        mutation_scale=6,
        color=grey,
        lw=0.8,
    )
    ax.add_patch(detuning)
    ax.text(0.605, 0.75, r"$\hbar\nu$", color=grey, fontsize=6.2, va="center")

    coupling = FancyArrowPatch(
        (0.34, 0.30),
        (0.34, 0.805),
        arrowstyle="-|>",
        mutation_scale=7,
        color=gold,
        lw=1.5,
    )
    ax.add_patch(coupling)
    ax.text(0.365, 0.48, r"$W(I)$", color="#966A10", fontsize=6.4, va="center")
    ax.text(
        0.365,
        0.405,
        r"$\Gamma(I)\!\propto\!|W|^2$",
        color="#966A10",
        fontsize=5.8,
        va="center",
    )

    decay = FancyArrowPatch(
        (0.62, 0.81),
        (0.84, 0.42),
        arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.16",
        mutation_scale=7,
        color=red,
        lw=1.2,
    )
    ax.add_patch(decay)
    ax.text(0.73, 0.66, r"decay $\gamma$", color=red, fontsize=6.2, rotation=-39)
    ax.text(0.80, 0.37, "loss channels", color=red, fontsize=5.5, ha="center")

    ax.text(0.0, 0.98, "(a)", fontsize=7.2, fontweight="bold", va="top")


def draw_accessible_loci(ax: plt.Axes) -> None:
    """Plot the linked real and imaginary scattering-length quadratures."""

    colours = ["#5AA6C8", "#2474A6", "#143D66"]
    intensities = [0.25, 0.55, 1.0]
    v = np.linspace(-20.0, 20.0, 2400)

    # For 0 <= u <= 1 and unrestricted v, the accessible set is the disk
    # bounded by the u=1 locus: (Re z - 1)^2 + (Im z + 1)^2 <= 1.
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    boundary_x = 1.0 + np.cos(theta)
    boundary_y = -1.0 + np.sin(theta)
    ax.fill(boundary_x, boundary_y, color="#DCECF4", alpha=0.58, lw=0)

    for colour, u in zip(colours, intensities, strict=True):
        ratio = scattering_ratio(u, v)
        ax.plot(ratio.real, ratio.imag, color=colour, lw=1.15, label=rf"$u={u:g}$")

    # Mark several detunings on one locus to show that v moves the operating
    # point while u selects its circle.
    selected_u = 0.55
    selected_v = np.array([-1.5, -0.55, 0.5])
    selected = scattering_ratio(selected_u, selected_v)
    ax.scatter(
        selected.real,
        selected.imag,
        s=9,
        facecolor="white",
        edgecolor=colours[1],
        linewidth=0.7,
        zorder=4,
    )
    ax.annotate(
        r"vary $v$",
        xy=(selected.real[1], selected.imag[1]),
        xytext=(1.55, -0.20),
        fontsize=5.8,
        color=colours[1],
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": colours[1]},
    )

    ax.axhline(0.0, color="#7B8490", lw=0.55, zorder=-1)
    ax.axvline(1.0, color="#7B8490", lw=0.55, ls=(0, (2, 2)), zorder=-1)
    ax.plot([1.0], [0.0], marker="o", ms=2.5, color="#27313D", zorder=5)
    ax.text(1.03, 0.07, r"off: $a_c/a_{\rm bg}=1$", fontsize=5.4, color="#27313D")

    ax.set_xlim(-0.12, 2.12)
    ax.set_ylim(-2.12, 0.20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\mathrm{Re}(a_c/a_{\rm bg})$", fontsize=6.2, labelpad=1.5)
    ax.set_ylabel(r"$\mathrm{Im}(a_c/a_{\rm bg})$", fontsize=6.2, labelpad=1.5)
    ax.tick_params(axis="both", which="major", labelsize=5.6, length=2.2, width=0.6, pad=1.5)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-2, -1, 0])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    legend = ax.legend(
        loc="lower left",
        frameon=True,
        fontsize=5.5,
        handlelength=1.6,
        borderaxespad=0.2,
        labelspacing=0.25,
        facecolor="white",
        framealpha=0.78,
        edgecolor="none",
    )
    legend.set_title(r"fixed intensity $u=\Gamma/\gamma$", prop={"size": 5.5})
    ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, fontsize=7.2, fontweight="bold", va="top")


def build_figure() -> plt.Figure:
    """Create the complete publication-sized two-panel figure."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(3.45, 1.82),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.24},
    )
    draw_level_diagram(axes[0])
    draw_accessible_loci(axes[1])
    figure.subplots_adjust(left=0.02, right=0.995, bottom=0.18, top=0.98)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output PDF or image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="resolution for raster outputs (default: 300)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = build_figure()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
