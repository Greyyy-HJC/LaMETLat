
"""Reusable plotting defaults for lattice-QCD analysis."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Modern, publication-oriented palette (Nature-style inspired).
GREY = "#7F7F7F"
RED = "#D62728"
PEACH = "#FFBE7A"
ORANGE = "#E69F00"
SUNKIST = "#F2C12E"
YELLOW = "#FFD54F"
LIME = "#B2DF8A"
GREEN = "#2CA02C"
TURQUOISE = "#1B9E77"
BLUE = "#4E79A7"
GRAPE = "#6A3D9A"
VIOLET = "#7B6FD0"
FUCHSIA = "#CC79A7"
BROWN = "#8C564B"
EMERALD = "#009E73"
SKY = "#56B4E9"
GOLD = "#F0E442"
ROYAL_BLUE = "#0072B2"
VERMILION = "#D55E00"
SILVER = "#999999"
OCHRE = "#A6761D"
LEAF = "#66A61E"
AZURE = "#1F78B4"
CRIMSON = "#E31A1C"
ROSE = "#FB9A99"
LAVENDER = "#CAB2D6"
UMBER = "#B15928"

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
    EMERALD,
    SKY,
    GOLD,
    ROYAL_BLUE,
    VERMILION,
    SILVER,
    OCHRE,
    LEAF,
    AZURE,
    CRIMSON,
    ROSE,
    LAVENDER,
    UMBER,
]

def darken_color(color, factor=0.65):
    """
    Darken a hex color by multiplying RGB channels with factor.
    factor < 1 => darker
    """
    rgb = mcolors.to_rgb(color)
    dark_rgb = tuple(max(min(c * factor, 1), 0) for c in rgb)
    return mcolors.to_hex(dark_rgb)

EDGE_COLOR_CYCLE = [
    darken_color(c, factor=0.55)
    for c in COLOR_CYCLE
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
    "h",
    "H",
    "d",
    "|",
    "_",
    ",",
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
LABEL_SIZE = {"labelsize": 18}
LEGEND_SIZE = {"fontsize": 14}

FONT_SIZE_LG = {"fontsize": 20}
LABEL_SIZE_LG = {"labelsize": 20}
LEGEND_SIZE_LG = {"fontsize": 18}

ERRORBAR_STYLE = {
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1,
}



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


import numpy as np
import gvar as gv


def jk_ls_avg(jk_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average jackknife samples into gvar values."""
    jk_arr = np.asarray(jk_ls)
    assert np.isrealobj(jk_arr), "jk_ls must contain real-valued samples"
    if axis != 0:
        jk_arr = np.swapaxes(jk_arr, 0, axis)

    shape = jk_arr.shape
    jk_flat = jk_arr.reshape(shape[0], -1)
    n_sample = jk_flat.shape[0]
    mean = np.mean(jk_flat, axis=0)

    if jk_flat.shape[1] == 1:
        sdev = np.std(jk_flat, axis=0) * np.sqrt(n_sample - 1)
        return gv.gvar(mean, sdev)

    cov = np.cov(jk_flat, rowvar=False) * (n_sample - 1)
    out = gv.gvar(mean, cov)
    return out.reshape(shape[1:])


