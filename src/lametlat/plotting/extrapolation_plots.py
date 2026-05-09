"""Plot helpers for long-distance extrapolation results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gvar as gv
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lametlat.correlators import gvar_ls_interpolate

from .plot_settings import (
    ERRORBAR_STYLE,
    FONT_SIZE,
    LAMBDA_LABEL,
    LEGEND_SIZE,
    RED,
    auto_ylim,
    default_plot,
)


def _plot_extrapolation_panel(
    lam_ls: Sequence[float] | np.ndarray,
    data_gv: Sequence[object] | np.ndarray,
    extrapolated_lam_ls: np.ndarray,
    extrapolated_mean: np.ndarray,
    extrapolated_sdev: np.ndarray,
    fit_idx_range: tuple[int, int] | Sequence[int],
    *,
    ylabel: str | None,
    xlim: tuple[float, float] | Sequence[float],
) -> tuple[Figure, Axes]:
    lam_array = np.asarray(lam_ls, dtype=float)
    data_mean = np.asarray(gv.mean(data_gv), dtype=float)
    data_sdev = np.asarray(gv.sdev(data_gv), dtype=float)
    fit_start, fit_stop = fit_idx_range

    fig, ax = default_plot()
    ax.errorbar(
        lam_array,
        data_mean,
        yerr=data_sdev,
        label="Data",
        **ERRORBAR_STYLE,
    )
    ax.fill_between(
        extrapolated_lam_ls,
        extrapolated_mean - extrapolated_sdev,
        extrapolated_mean + extrapolated_sdev,
        alpha=0.4,
        label="Extrapolated",
    )

    ax.axvline(lam_array[fit_start], ymin=0, ymax=0.5, color=RED, linestyle="--")
    ax.axvline(lam_array[fit_stop - 1], ymin=0, ymax=0.5, color=RED, linestyle="--")

    ax.set_xlabel(LAMBDA_LABEL, **FONT_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **FONT_SIZE)
    ax.set_ylim(
        auto_ylim(
            [data_mean, extrapolated_mean],
            [data_sdev, extrapolated_sdev],
        )
    )
    ax.set_xlim(xlim)
    ax.legend(**LEGEND_SIZE)
    return fig, ax


def extrapolation_comparison_plot(
    lam_ls,
    re_gv,
    im_gv,
    extrapolated_lam_ls,
    extrapolated_re_gv,
    extrapolated_im_gv,
    fit_idx_range,
    title,
    xlim=(-1, 25),
    ylabel_re=None,
    ylabel_im=None,
    save_path=None,
    show=False,
    interpolation_kind="cubic",
    interpolation_density=10,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot real and imaginary parts before and after lambda extrapolation."""
    del title

    extrapolated_lam_array = np.asarray(extrapolated_lam_ls, dtype=float)
    lam_interp = np.linspace(
        np.min(extrapolated_lam_array),
        np.max(extrapolated_lam_array),
        len(extrapolated_lam_array) * interpolation_density,
    )
    re_interp = gvar_ls_interpolate(
        extrapolated_lam_array,
        extrapolated_re_gv,
        lam_interp,
        kind=interpolation_kind,
    )
    im_interp = gvar_ls_interpolate(
        extrapolated_lam_array,
        extrapolated_im_gv,
        lam_interp,
        kind=interpolation_kind,
    )
    re_mean_interp = np.asarray(gv.mean(re_interp), dtype=float)
    re_sdev_interp = np.asarray(gv.sdev(re_interp), dtype=float)
    im_mean_interp = np.asarray(gv.mean(im_interp), dtype=float)
    im_sdev_interp = np.asarray(gv.sdev(im_interp), dtype=float)

    fig_real, ax_real = _plot_extrapolation_panel(
        lam_ls,
        re_gv,
        lam_interp,
        re_mean_interp,
        re_sdev_interp,
        fit_idx_range,
        ylabel=ylabel_re,
        xlim=xlim,
    )
    fig_imag, ax_imag = _plot_extrapolation_panel(
        lam_ls,
        im_gv,
        lam_interp,
        im_mean_interp,
        im_sdev_interp,
        fit_idx_range,
        ylabel=ylabel_im,
        xlim=xlim,
    )

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_real.savefig(
            path.with_name(f"{path.name}_real.pdf"),
            bbox_inches="tight",
            transparent=True,
        )
        fig_imag.savefig(
            path.with_name(f"{path.name}_imag.pdf"),
            bbox_inches="tight",
            transparent=True,
        )

    if show:
        fig_real.show()
        fig_imag.show()

    return (fig_real, ax_real), (fig_imag, ax_imag)
