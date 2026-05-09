"""Reusable plotting defaults for lattice-QCD analysis."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Color palette.
GREY = "#808080"
RED = "#FF6F6F"
PEACH = "#FF9E6F"
ORANGE = "#FFBC6F"
SUNKIST = "#FFDF6F"
YELLOW = "#FFEE6F"
LIME = "#CBF169"
GREEN = "#5CD25C"
TURQUOISE = "#4AAB89"
BLUE = "#508EAD"
GRAPE = "#635BB1"
VIOLET = "#7C5AB8"
FUCHSIA = "#C3559F"
BROWN = "#6B3F3F"

COLOR_CYCLE = [
    BLUE,
    ORANGE,
    GREEN,
    RED,
    VIOLET,
    FUCHSIA,
    TURQUOISE,
    GRAPE,
    LIME,
    PEACH,
    SUNKIST,
    YELLOW,
    BROWN,
]

MARKER_CYCLE = [
    ".",
    "o",
    "s",
    "P",
    "X",
    "*",
    "p",
    "D",
    "<",
    ">",
    "^",
    "v",
    "1",
    "2",
    "3",
    "4",
    "+",
    "x",
]

FONT_CONFIG = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
}

FIG_WIDTH = 6.75
GOLDEN_RATIO = 1.618034333
FIG_SIZE = (FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO)

PLOT_AXES = [0.15, 0.15, 0.8, 0.8]
FONT_SIZE = {"fontsize": 18}
LEGEND_SIZE = {"fontsize": 14}
LABEL_SIZE = {"labelsize": 18}

ERRORBAR_STYLE = {
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1,
}

ERRORBAR_CIRCLE_STYLE = {
    "marker": "o",
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1.5,
}

TSEP = r"$t_{\mathrm{sep}}$"

TMIN_LABEL = r"$t_{\mathrm{min}}~/~a$"
TMAX_LABEL = r"$t_{\mathrm{max}}~/~a$"
TAU_CENTER_LABEL = r"$(\tau - t_{\rm{sep}}/2)~/~a$"
TSEP_LABEL = r"${t_{\mathrm{sep}}~/~a}$"
Z_LABEL = r"${z~/~a}$"
LAMBDA_LABEL = r"$\lambda = z P^z$"
MEFF_LABEL = r"${m}_{\mathrm{eff}}$"

RATIO_REAL_LABEL = r"$\Re\left[\mathcal{R}(t_{\mathrm{sep}},\tau)\right]$"
RATIO_IMAG_LABEL = r"$\Im\left[\mathcal{R}(t_{\mathrm{sep}},\tau)\right]$"


def apply_plot_style() -> None:
    """Apply package default font settings to matplotlib rcParams."""
    rcParams.update(FONT_CONFIG)


def auto_ylim(
    y_data: Sequence[np.ndarray], yerr_data: Sequence[np.ndarray], y_range_ratio: float = 4.0
) -> tuple[float, float]:
    """Compute y-limits from data and uncertainties with symmetric margin."""
    all_y = np.concatenate(
        [y + yerr for y, yerr in zip(y_data, yerr_data)]
        + [y - yerr for y, yerr in zip(y_data, yerr_data)]
    )
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    y_range = y_max - y_min
    return y_min - y_range / y_range_ratio, y_max + y_range / y_range_ratio


def default_plot() -> tuple[Figure, Axes]:
    """Create a default single-panel plot."""
    apply_plot_style()
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.axes()
    ax.tick_params(direction="in", top=True, right=True, **LABEL_SIZE)
    ax.grid(linestyle=":")
    return fig, ax


def default_sub_plot(height_ratio: int = 3) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create default 2-row subplots with a shared x-axis."""
    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=FIG_SIZE,
        gridspec_kw={"height_ratios": [height_ratio, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)

    for ax in (ax1, ax2):
        ax.tick_params(direction="in", top=True, right=True, **LABEL_SIZE)
        ax.grid(linestyle=":")

    return fig, (ax1, ax2)

