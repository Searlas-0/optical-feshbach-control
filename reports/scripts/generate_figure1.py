#!/usr/bin/env python3
"""Generate the two-channel optical-Feshbach schematic used in the report.

The drawing follows the energy-landscape logic of Fig. 1 in Chin et al., Rev.
Mod. Phys. 82, 1225 (2010), but adapts it to a photon-dressed closed channel.
It is a conceptual diagram rather than a fitted molecular potential.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


REPORT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPORT_DIR / "figures" / "figure1_ofr_schematic.pdf"

NAVY = "#143D66"
BLUE = "#2474A6"
RED = "#B6403A"
GOLD = "#D49A22"


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#7E8791")
    axis.spines[["left", "bottom"]].set_linewidth(0.65)
    axis.tick_params(labelsize=6.0, width=0.55, length=2.2, pad=1.6)


def draw_channel_potentials(axis: plt.Axes) -> None:
    """Draw a clear open/closed-channel energy landscape."""

    separation = np.linspace(0.18, 1.25, 900)

    # Smooth model curves chosen only to communicate two distinct thresholds
    # and a closed-channel molecular well. They are not fitted Sr2 potentials.
    sigma = 0.30
    open_potential = 0.26 * (
        (sigma / separation) ** 12 - 2.0 * (sigma / separation) ** 6
    )
    closed_potential = 0.62 + 0.52 * (
        (1.0 - np.exp(-7.0 * (separation - 0.43))) ** 2 - 1.0
    )

    axis.axhline(0.0, color="#8E969F", lw=0.65, ls=(0, (2.2, 2.2)), zorder=0)
    axis.plot(separation, open_potential, color=BLUE, lw=2.0, zorder=3)
    axis.plot(separation, closed_potential, color=RED, lw=2.0, zorder=3)

    collision_energy = 0.035
    closed_energy = 0.30
    axis.plot([0.58, 1.21], [collision_energy] * 2, color=BLUE, lw=1.35)
    # Centre the schematic bound-state segment on the minimum of the closed
    # potential (R ~= 0.43), rather than letting it drift onto the outer wing.
    axis.plot([0.25, 0.61], [closed_energy] * 2, color=RED, lw=1.55)

    axis.text(
        0.84,
        -0.115,
        r"open entrance potential $V_o(R)$",
        color=BLUE,
        fontsize=7.1,
        ha="center",
    )
    axis.text(
        0.76,
        0.765,
        r"photon-dressed closed potential  $V_c(R)-\hbar\omega_L$",
        color=RED,
        fontsize=6.8,
        ha="center",
    )
    axis.text(
        0.60,
        collision_energy - 0.045,
        r"$E\simeq0$",
        color=BLUE,
        fontsize=7.2,
        ha="left",
        va="top",
    )
    axis.text(
        0.43,
        closed_energy + 0.055,
        r"molecular level $E_c(\nu)$",
        color=RED,
        fontsize=6.8,
        ha="center",
    )

    coupling = FancyArrowPatch(
        (0.595, collision_energy + 0.012),
        (0.595, closed_energy - 0.012),
        arrowstyle="<->",
        mutation_scale=7.5,
        color=GOLD,
        lw=1.3,
        zorder=5,
    )
    axis.add_patch(coupling)
    axis.text(
        0.62,
        0.17,
        r"intensity coupling $W(I)$",
        color="#966A10",
        fontsize=6.6,
        va="center",
    )

    mismatch = FancyArrowPatch(
        (1.03, collision_energy),
        (1.03, closed_energy),
        arrowstyle="<->",
        mutation_scale=7.0,
        color=GOLD,
        lw=1.0,
    )
    axis.add_patch(mismatch)
    axis.text(
        1.055,
        0.17,
        "detuning $\\nu$ sets\n" r"gap $E_c-E$",
        color="#966A10",
        fontsize=6.4,
        ha="left",
        va="center",
    )

    axis.set_xlim(0.18, 1.25)
    axis.set_ylim(-0.36, 1.03)
    axis.set_xticks([])
    axis.set_yticks([0.0])
    axis.set_yticklabels(["0"])
    axis.set_xlabel(r"atomic separation $R$", fontsize=7.2, labelpad=2.0)
    axis.set_ylabel("energy", fontsize=7.2, labelpad=2.0)
    axis.set_title(
        "Open and photon-dressed closed channels",
        loc="left",
        color=NAVY,
        fontsize=8.5,
        fontweight="bold",
        pad=3.0,
    )
    _style_axis(axis)


def build_figure() -> plt.Figure:
    """Create the publication-sized single-panel schematic."""

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
    figure, axis = plt.subplots(figsize=(5.15, 2.35))
    draw_channel_potentials(axis)
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.90)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output PDF or image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
