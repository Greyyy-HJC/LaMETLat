"""Three-dimensional surface plots for 2D functions (e.g. integration kernels)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_surface_from_grid(
    x1: np.ndarray,
    x2: np.ndarray,
    z: np.ndarray,
    *,
    title: str | None = None,
    xlabel: str = r"$x_1$",
    ylabel: str = r"$x_2$",
    zlabel: str | None = None,
    cmap: str = "plasma",
    elev: float = 28.0,
    azim: float = -58.0,
    colorbar_label: str | None = None,
    figsize: tuple[float, float] | None = None,
    surface_alpha: float = 0.92,
    antialiased: bool = True,
    show: bool = True,
) -> tuple["Figure", Any]:
    """
    Plot a 3D surface from mesh grids ``x1``, ``x2``, ``z`` (identical shapes).

    Styling matches LaMET defaults: light panes, ``plasma``-style colormap, tilted view,
    and a compact colorbar suitable for notes or papers.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — register 3d projection

    import matplotlib.pyplot as plt
    from matplotlib import cm

    from lametlat.plotting.plot_settings import (
        FONT_SIZE,
        FIG_WIDTH,
        GOLDEN_RATIO,
        apply_plot_style,
    )

    apply_plot_style()

    x1_m = np.asarray(x1)
    x2_m = np.asarray(x2)
    z_m = np.asarray(z)
    if x1_m.shape != x2_m.shape or x1_m.shape != z_m.shape:
        raise ValueError("x1, x2, and z must have the same shape")

    if figsize is None:
        figsize = (FIG_WIDTH * 1.08, (FIG_WIDTH * 1.08) / GOLDEN_RATIO)

    fig = plt.figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    if zlabel is None:
        zlabel = r"$f(x_1, x_2)$"

    z_min = float(np.nanmin(z_m))
    z_max = float(np.nanmax(z_m))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max == z_min:
        norm = None
    else:
        norm = cm.colors.Normalize(vmin=z_min, vmax=z_max)

    surf = ax.plot_surface(
        x1_m,
        x2_m,
        z_m,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        antialiased=antialiased,
        alpha=surface_alpha,
        rstride=1,
        cstride=1,
        shade=True,
    )

    ax.set_xlabel(xlabel, **FONT_SIZE)
    ax.set_ylabel(ylabel, **FONT_SIZE)
    ax.set_zlabel(zlabel, **FONT_SIZE, labelpad=8)

    if title:
        ax.set_title(title, pad=14)

    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="-", linewidth=0.35, alpha=0.38)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.1)
        axis.pane.set_edgecolor("0.88")

    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.06, top=0.92)

    cbar = fig.colorbar(
        surf,
        ax=ax,
        shrink=0.58,
        aspect=12,
        pad=0.1,
        fraction=0.032,
    )
    cbar.ax.tick_params(labelsize=12)

    if show:
        plt.show()

    return fig, ax
