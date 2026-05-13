"""Plot helpers for 2pt, ratio, and FH correlator observables."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import gvar as gv
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from lametlat.correlators import pt2_to_meff
from lametlat.ground_state import (
    ff_ratio_fcn,
    ff_sum_fcn,
    fh_im_fcn,
    fh_re_fcn,
    pt2_re_fcn,
    pt3_ratio_im_fcn,
    pt3_ratio_re_fcn,
    qda_im_fcn,
    qda_re_fcn,
)

from .plot_settings import (
    COLOR_CYCLE,
    ERRORBAR_CIRCLE_STYLE,
    ERRORBAR_STYLE,
    FONT_SIZE,
    LEGEND_SIZE,
    MEFF_LABEL,
    RATIO_IMAG_LABEL,
    RATIO_REAL_LABEL,
    TAU_CENTER_LABEL,
    TSEP,
    TSEP_LABEL,
    auto_ylim,
    default_plot,
)


def _trange_to_array(
    trange: tuple[int, int] | Sequence[int] | np.ndarray | None,
    *,
    default_stop: int | None = None,
    name: str = "trange",
) -> np.ndarray:
    if trange is None:
        if default_stop is None:
            raise ValueError(f"{name} is required")
        values = np.arange(default_stop, dtype=int)
    elif isinstance(trange, tuple):
        tmin, tmax = trange
        values = np.arange(tmin, tmax, dtype=int)
    else:
        values = np.asarray(trange, dtype=int)

    return values


def _meff_trange(trange: np.ndarray, boundary: str) -> np.ndarray:
    if boundary in {"periodic", "anti-periodic"}:
        return trange[1:-1]
    if boundary == "none":
        return trange[:-1]
    raise ValueError(f"unsupported boundary mode: {boundary!r}")


def _fit_result_list(fit_results: object | Sequence[object] | None) -> list[object]:
    if fit_results is None:
        return []
    if isinstance(fit_results, Sequence) and not isinstance(fit_results, (str, bytes)):
        return list(fit_results)
    return [fit_results]


def _fit_nstate(fit_result: object) -> int:
    nstate = getattr(fit_result, "nstate", None)
    if nstate is not None:
        return nstate

    params = getattr(fit_result, "p", {})
    z_keys = [
        key for key in params.keys() if isinstance(key, str) and key.startswith("z")
    ]
    if z_keys:
        return max(int(key[1:]) for key in z_keys) + 1
    return 2


def _fh_fit_nstate(fit_result: object) -> int:
    params = getattr(fit_result, "p", {})
    if "sum_den_exp_coeff" in params:
        return 2
    return 1


def pt2_plot(
    pt2_gv_ls: list[np.ndarray],
    boundary: str = "none",
    trange: tuple[int, int] | Sequence[int] | np.ndarray | None = None,
    *,
    fit_results: object | Sequence[object] | None = None,
    fit_tmin: int | None = None,
    fit_tmax: int | None = None,
    fit_label: str = "Fit",
    Lt: int | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot C2pt and effective mass from resampled pt2 data.

    When ``fit_results`` is given, pass ``fit_tmin`` and ``fit_tmax`` using the
    same half-open window as ``pt2_fit(..., tmin, tmax, Lt, ...)`` (times
    ``tmin <= t < tmax``). Optional ``Lt`` defaults to the correlator length;
    set it if that differs from the temporal extent used in the fit.
    """

    data_length = len(pt2_gv_ls[0])
    t = _trange_to_array(trange, default_stop=data_length)
    fits = _fit_result_list(fit_results)
    fit_lt = data_length if Lt is None else Lt

    if fits and (fit_tmin is None or fit_tmax is None):
        raise ValueError(
            "pt2_plot: fit_results requires fit_tmin and fit_tmax "
            "(same as pt2_fit pt2_avg, tmin, tmax, ...)."
        )

    fig_c2, ax_c2 = default_plot()
    for idx, pt2_gv in enumerate(pt2_gv_ls):
        label = "Data" if fits and idx == 0 else None
        ax_c2.errorbar(
            t,
            gv.mean(pt2_gv[t]),
            yerr=gv.sdev(pt2_gv[t]),
            label=label,
            **ERRORBAR_STYLE,
        )
    ax_c2.set_yscale("log")
    ax_c2.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_c2.set_ylabel(r"$C_{2\mathrm{pt}}(t_{\mathrm{sep}})$", **FONT_SIZE)

    fig_meff, ax_meff = default_plot()
    for pt2_gv in pt2_gv_ls:
        meff_gv = pt2_to_meff(pt2_gv[t], boundary=boundary)
        ax_meff.errorbar(
            _meff_trange(t, boundary),
            gv.mean(meff_gv),
            yerr=gv.sdev(meff_gv),
            **ERRORBAR_STYLE,
        )
    ax_meff.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_meff.set_ylabel(MEFF_LABEL, **FONT_SIZE)

    for idx, fit_result in enumerate(fits):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        fit_t = np.arange(fit_tmin, fit_tmax, dtype=int)
        fit_y = pt2_re_fcn(
            fit_t,
            fit_result.p,
            fit_lt,
            nstate=_fit_nstate(fit_result),
        )
        fit_mean = gv.mean(fit_y)
        fit_sdev = gv.sdev(fit_y)
        legend_label = fit_label if len(fits) == 1 else f"{fit_label} ({idx + 1})"

        positive_mean = fit_mean > 0
        if np.any(positive_mean):
            ax_c2.plot(
                fit_t[positive_mean],
                fit_mean[positive_mean],
                color=color,
                label=legend_label,
            )
        positive_band = (fit_mean - fit_sdev > 0) & (fit_mean + fit_sdev > 0)
        if np.any(positive_band):
            ax_c2.fill_between(
                fit_t[positive_band],
                (fit_mean - fit_sdev)[positive_band],
                (fit_mean + fit_sdev)[positive_band],
                color=color,
                alpha=0.35,
            )

        fit_meff = pt2_to_meff(fit_y, boundary=boundary)
        ax_meff.plot(
            _meff_trange(fit_t, boundary),
            gv.mean(fit_meff),
            color=color,
            label=legend_label,
        )
        ax_meff.fill_between(
            _meff_trange(fit_t, boundary),
            gv.mean(fit_meff) - gv.sdev(fit_meff),
            gv.mean(fit_meff) + gv.sdev(fit_meff),
            color=color,
            alpha=0.35,
        )

    if fits:
        ax_c2.legend(**LEGEND_SIZE)
        ax_meff.legend(**LEGEND_SIZE)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_c2.savefig(
            path.with_name(f"{path.name}_c2pt.pdf"),
            bbox_inches="tight",
            transparent=True,
        )
        fig_meff.savefig(
            path.with_name(f"{path.name}_meff.pdf"),
            bbox_inches="tight",
            transparent=True,
        )

    if show:
        fig_c2.show()
        fig_meff.show()

    return (fig_c2, ax_c2), (fig_meff, ax_meff)


def pt3_ratio_plot(
    tau_dict: dict[int, np.ndarray],
    ratio_real: dict[int, np.ndarray],
    ratio_imag: dict[int, np.ndarray] | None = None,
    *,
    fit_result: object | None = None,
    fit_tsep_ls: Sequence[int] | None = None,
    fit_tau_cut: int = 1,
    fit_label: str = "Fit",
    Lt: int | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot 3pt ratio vs tau for each tsep from precomputed arrays."""
    
    tsep_ls = sorted(ratio_real.keys())
    fit_lt = max(tsep_ls) + 1 if Lt is None else Lt
    nstate = 2 if fit_result is None else _fit_nstate(fit_result)
    
    fig_real, ax_real = default_plot()
    y_data_re: list[np.ndarray] = []
    yerr_re: list[np.ndarray] = []
    for tsep in tsep_ls:
        y_mean = gv.mean(ratio_real[tsep])
        y_sdev = gv.sdev(ratio_real[tsep])
        ax_real.errorbar(
            tau_dict[tsep] - tsep / 2,
            y_mean,
            yerr=y_sdev,
            label=f"{TSEP}={tsep} $a$",
            **ERRORBAR_CIRCLE_STYLE,
        )
        y_data_re.append(np.asarray(y_mean, dtype=float))
        yerr_re.append(np.asarray(y_sdev, dtype=float))

    if fit_result is not None:
        fit_tseps = list(tsep_ls if fit_tsep_ls is None else fit_tsep_ls)
        for tsep in fit_tseps:
            idx = tsep_ls.index(tsep)
            color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            fit_tau = np.linspace(fit_tau_cut - 0.5, tsep - fit_tau_cut + 0.5, 200)
            fit_t = np.full_like(fit_tau, float(tsep))
            fit_ratio = pt3_ratio_re_fcn(fit_t, fit_tau, fit_result.p, fit_lt, nstate=nstate)
            fit_mean = gv.mean(fit_ratio)
            fit_sdev = gv.sdev(fit_ratio)
            fit_x = fit_tau - tsep / 2
            ax_real.fill_between(
                fit_x,
                fit_mean - fit_sdev,
                fit_mean + fit_sdev,
                color=color,
                alpha=0.3,
            )
            y_data_re.append(np.asarray(fit_mean, dtype=float))
            yerr_re.append(np.asarray(fit_sdev, dtype=float))

        center_min = min(
            np.min(np.asarray(tau_dict[tsep], dtype=float) - tsep / 2)
            for tsep in tsep_ls
        )
        center_max = max(
            np.max(np.asarray(tau_dict[tsep], dtype=float) - tsep / 2)
            for tsep in tsep_ls
        )
        band_x = np.linspace(center_min, center_max, 2)
        matrix_element = fit_result.p["O00_re"] / (2 * fit_result.p["E0"])
        band_mean = np.full(2, gv.mean(matrix_element), dtype=float)
        band_sdev = np.full(2, gv.sdev(matrix_element), dtype=float)
        ax_real.fill_between(
            band_x,
            band_mean - band_sdev,
            band_mean + band_sdev,
            color="grey",
            alpha=0.35,
            label=fit_label,
        )
        y_data_re.append(band_mean)
        yerr_re.append(band_sdev)

    ax_real.set_xlabel(TAU_CENTER_LABEL, **FONT_SIZE)
    ax_real.set_ylabel(RATIO_REAL_LABEL, **FONT_SIZE)
    ax_real.legend(ncol=2, loc="upper right", **LEGEND_SIZE)
    ax_real.set_ylim(auto_ylim(y_data_re, yerr_re, 2))
    
    if ratio_imag is not None:
        fig_imag, ax_imag = default_plot()
        y_data_im: list[np.ndarray] = []
        yerr_im: list[np.ndarray] = []
        for tsep in tsep_ls:
            y_mean = gv.mean(ratio_imag[tsep])
            y_sdev = gv.sdev(ratio_imag[tsep])
            ax_imag.errorbar(
                tau_dict[tsep] - tsep / 2,
                y_mean,
                yerr=y_sdev,
                label=f"{TSEP}={tsep} $a$",
                **ERRORBAR_CIRCLE_STYLE,
            )
            y_data_im.append(np.asarray(y_mean, dtype=float))
            yerr_im.append(np.asarray(y_sdev, dtype=float))

        if fit_result is not None:
            fit_tseps = list(tsep_ls if fit_tsep_ls is None else fit_tsep_ls)
            for tsep in fit_tseps:
                idx = tsep_ls.index(tsep)
                color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
                fit_tau = np.linspace(fit_tau_cut - 0.5, tsep - fit_tau_cut + 0.5, 200)
                fit_t = np.full_like(fit_tau, float(tsep))
                fit_ratio = pt3_ratio_im_fcn(fit_t, fit_tau, fit_result.p, fit_lt, nstate=nstate)
                fit_mean = gv.mean(fit_ratio)
                fit_sdev = gv.sdev(fit_ratio)
                fit_x = fit_tau - tsep / 2
                ax_imag.fill_between(
                    fit_x,
                    fit_mean - fit_sdev,
                    fit_mean + fit_sdev,
                    color=color,
                    alpha=0.3,
                )
                y_data_im.append(np.asarray(fit_mean, dtype=float))
                yerr_im.append(np.asarray(fit_sdev, dtype=float))

            center_min = min(
                np.min(np.asarray(tau_dict[tsep], dtype=float) - tsep / 2)
                for tsep in tsep_ls
            )
            center_max = max(
                np.max(np.asarray(tau_dict[tsep], dtype=float) - tsep / 2)
                for tsep in tsep_ls
            )
            band_x = np.linspace(center_min, center_max, 2)
            matrix_element = fit_result.p["O00_im"] / (2 * fit_result.p["E0"])
            band_mean = np.full(2, gv.mean(matrix_element), dtype=float)
            band_sdev = np.full(2, gv.sdev(matrix_element), dtype=float)
            ax_imag.fill_between(
                band_x,
                band_mean - band_sdev,
                band_mean + band_sdev,
                color="grey",
                alpha=0.35,
                label=fit_label,
            )
            y_data_im.append(band_mean)
            yerr_im.append(band_sdev)

        ax_imag.set_xlabel(TAU_CENTER_LABEL, **FONT_SIZE)
        ax_imag.set_ylabel(RATIO_IMAG_LABEL, **FONT_SIZE)
        ax_imag.legend(ncol=2, loc="upper right", **LEGEND_SIZE)
        ax_imag.set_ylim(auto_ylim(y_data_im, yerr_im, 2))
        
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
    
    else:
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig_real.savefig(
                path.with_name(f"{path.name}_real.pdf"),
                bbox_inches="tight",
                transparent=True,
            )
        if show:
            fig_real.show()
        return (fig_real, ax_real)


def ff_ratio_plot(
    tau_dict: dict[int, np.ndarray],
    ratio_real: dict[int, np.ndarray],
    *,
    fit_result: object | None = None,
    fit_tsep_ls: Sequence[int] | None = None,
    fit_tau_cut: int = 1,
    fit_label: str = "Fit",
    ff_sign: float = 1.0,
    id_label: dict[str, object] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot form-factor ratio data with optional two-state fit bands."""

    tsep_ls = sorted(ratio_real.keys())
    title_prefix = ""
    if id_label is not None:
        title_prefix = ", ".join(f"{key} = {value}" for key, value in id_label.items())

    fig, ax = default_plot()
    y_data_ls: list[np.ndarray] = []
    yerr_data_ls: list[np.ndarray] = []

    for idx, tsep in enumerate(tsep_ls):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        ratio_row = np.asarray(ratio_real[tsep], dtype=object)
        tau_center = np.asarray(tau_dict[tsep], dtype=float) - tsep / 2
        y_mean = gv.mean(ratio_row)
        y_sdev = gv.sdev(ratio_row)
        ax.errorbar(
            tau_center,
            y_mean,
            yerr=y_sdev,
            label=f"{TSEP}={tsep} $a$",
            color=color,
            **ERRORBAR_CIRCLE_STYLE,
        )
        y_data_ls.append(np.asarray(y_mean, dtype=float))
        yerr_data_ls.append(np.asarray(y_sdev, dtype=float))

    if fit_result is not None:
        fit_tseps = list(tsep_ls if fit_tsep_ls is None else fit_tsep_ls)
        for tsep in fit_tseps:
            idx = tsep_ls.index(tsep)
            color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            fit_tau = np.linspace(fit_tau_cut - 0.5, tsep - fit_tau_cut + 0.5, 200)
            fit_t = np.full_like(fit_tau, float(tsep))
            fit_ratio = ff_ratio_fcn((fit_t, fit_tau), fit_result.p)
            fit_mean = gv.mean(fit_ratio)
            fit_sdev = gv.sdev(fit_ratio)
            fit_x = fit_tau - tsep / 2
            ax.fill_between(
                fit_x,
                fit_mean - fit_sdev,
                fit_mean + fit_sdev,
                color=color,
                alpha=0.3,
            )
            y_data_ls.append(np.asarray(fit_mean, dtype=float))
            yerr_data_ls.append(np.asarray(fit_sdev, dtype=float))

        ff_param = fit_result.p["ff"]

        center_min = min(np.min(np.asarray(tau_dict[tsep], dtype=float) - tsep / 2) for tsep in tsep_ls)
        center_max = max(np.max(np.asarray(tau_dict[tsep], dtype=float) - tsep / 2) for tsep in tsep_ls)
        band_x = np.linspace(center_min, center_max, 2)
        band_mean = np.full(2, gv.mean(ff_sign * ff_param), dtype=float)
        band_sdev = np.full(2, gv.sdev(ff_sign * ff_param), dtype=float)
        ax.fill_between(
            band_x,
            band_mean - band_sdev,
            band_mean + band_sdev,
            color="grey",
            alpha=0.35,
            label=fit_label,
        )
        y_data_ls.append(band_mean)
        yerr_data_ls.append(band_sdev)

    ax.set_xlabel(TAU_CENTER_LABEL, **FONT_SIZE)
    ax.set_ylabel(RATIO_REAL_LABEL, **FONT_SIZE)
    ax.set_ylim(auto_ylim(y_data_ls, yerr_data_ls, 2))
    ax.legend(ncol=2, loc="upper right", **LEGEND_SIZE)
    if title_prefix:
        ax.set_title(title_prefix, **FONT_SIZE)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_name(f"{path.name}_real.pdf"), bbox_inches="tight", transparent=True)
    if show:
        fig.show()
    return fig, ax


def ff_sum_plot(
    tsep_ls: Sequence[int],
    sum_real: np.ndarray | Sequence[object],
    *,
    fit_result: object | None = None,
    fit_tsep_ls: Sequence[int] | None = None,
    fit_tau_cut: int = 1,
    fit_label: str = "Fit",
    ff_sign: float = 1.0,
    id_label: dict[str, object] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot tau-averaged form-factor sum data with optional fit bands."""

    t = np.asarray(tsep_ls, dtype=int)
    sum_arr = np.asarray(sum_real, dtype=object)

    title_prefix = ""
    if id_label is not None:
        title_prefix = ", ".join(f"{key} = {value}" for key, value in id_label.items())

    fig, ax = default_plot()
    sum_mean = gv.mean(sum_arr)
    sum_sdev = gv.sdev(sum_arr)
    y_data_ls: list[np.ndarray] = [np.asarray(sum_mean, dtype=float)]
    yerr_data_ls: list[np.ndarray] = [np.asarray(sum_sdev, dtype=float)]

    ax.errorbar(
        t,
        sum_mean,
        yerr=sum_sdev,
        label="Data",
        **ERRORBAR_CIRCLE_STYLE,
    )

    if fit_result is not None:
        fit_t = np.asarray(list(t if fit_tsep_ls is None else fit_tsep_ls), dtype=float)

        fit_sum = ff_sum_fcn(fit_t, fit_tau_cut, fit_result.p)
        fit_mean = gv.mean(fit_sum)
        fit_sdev = gv.sdev(fit_sum)
        ax.fill_between(
            fit_t,
            fit_mean - fit_sdev,
            fit_mean + fit_sdev,
            alpha=0.35,
        )
        y_data_ls.append(np.asarray(fit_mean, dtype=float))
        yerr_data_ls.append(np.asarray(fit_sdev, dtype=float))

        ff_param = fit_result.p["ff"]

        band_x = np.linspace(float(np.min(t)), float(np.max(t)), 2)
        band_mean = np.full(2, gv.mean(ff_sign * ff_param), dtype=float)
        band_sdev = np.full(2, gv.sdev(ff_sign * ff_param), dtype=float)
        ax.fill_between(
            band_x,
            band_mean - band_sdev,
            band_mean + band_sdev,
            color="grey",
            alpha=0.35,
            label=fit_label,
        )
        y_data_ls.append(band_mean)
        yerr_data_ls.append(band_sdev)

    ax.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax.set_ylabel(r"$\mathcal{S}(t_{\mathrm{sep}})$", **FONT_SIZE)
    ax.set_ylim(auto_ylim(y_data_ls, yerr_data_ls, 2))
    ax.legend(ncol=2, loc="upper right", **LEGEND_SIZE)
    if title_prefix:
        ax.set_title(title_prefix, **FONT_SIZE)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_name(f"{path.name}_real.pdf"), bbox_inches="tight", transparent=True)
    if show:
        fig.show()
    return fig, ax


def ff_joint_plot(
    tau_dict: dict[int, np.ndarray],
    ratio_real: dict[int, np.ndarray],
    *,
    sum_tsep_ls: Sequence[int] | None = None,
    sum_real: np.ndarray | Sequence[object] | None = None,
    fit_result: object | None = None,
    fit_tsep_ls: Sequence[int] | None = None,
    fit_tau_cut: int = 1,
    fit_label: str = "Fit",
    ff_sign: float = 1.0,
    id_label: dict[str, object] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]]:
    """Plot ratio and tau-averaged sum panels for form-factor analyses."""

    if sum_real is None:
        inferred_tsep_ls = sorted(ratio_real.keys()) if sum_tsep_ls is None else list(sum_tsep_ls)
        inferred_sum = []
        for tsep in inferred_tsep_ls:
            inferred_sum.append(np.mean(np.asarray(ratio_real[tsep], dtype=object)))
        sum_tsep_arr = np.asarray(inferred_tsep_ls, dtype=int)
        sum_arr = np.asarray(inferred_sum, dtype=object)
    else:
        sum_tsep_arr = np.asarray(sum_tsep_ls, dtype=int)
        sum_arr = np.asarray(sum_real, dtype=object)

    ratio_save_path: str | Path | None = None
    sum_save_path: str | Path | None = None
    if save_path is not None:
        base = Path(save_path)
        ratio_save_path = base.with_name(f"{base.name}_ratio")
        sum_save_path = base.with_name(f"{base.name}_sum")

    fig_ratio, ax_ratio = ff_ratio_plot(
        tau_dict,
        ratio_real,
        fit_result=fit_result,
        fit_tsep_ls=fit_tsep_ls,
        fit_tau_cut=fit_tau_cut,
        fit_label=fit_label,
        ff_sign=ff_sign,
        id_label=id_label,
        save_path=ratio_save_path,
        show=show,
    )
    fig_sum, ax_sum = ff_sum_plot(
        sum_tsep_arr,
        sum_arr,
        fit_result=fit_result,
        fit_tsep_ls=fit_tsep_ls,
        fit_tau_cut=fit_tau_cut,
        fit_label=fit_label,
        ff_sign=ff_sign,
        id_label=id_label,
        save_path=sum_save_path,
        show=show,
    )
    return (fig_ratio, ax_ratio), (fig_sum, ax_sum)


def qda_ratio_plot(
    trange: tuple[int, int] | Sequence[int] | np.ndarray,
    qda_ratio_real: np.ndarray,
    qda_ratio_imag: np.ndarray | None = None,
    *,
    fit_result: object | None = None,
    pt2_fit_result: object | None = None,
    fit_trange: tuple[int, int] | Sequence[int] | np.ndarray | None = None,
    fit_label: str = "Fit",
    Lt: int | None = None,
    id_label: dict[str, object] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]] | tuple[Figure, Axes]:
    """Plot qDA/2pt ratio data with optional fit-result bands."""
    t = _trange_to_array(trange, name="trange")
    fit_t = (
        _trange_to_array(fit_trange, default_stop=len(t), name="fit_trange")
        if fit_trange is not None
        else t
    )
    fit_lt = max(np.max(t), np.max(fit_t)) + 1 if Lt is None else Lt

    title_prefix = ""
    if id_label is not None:
        title_prefix = ", ".join(f"{key} = {value}" for key, value in id_label.items())
        title_prefix = f"{title_prefix}, "

    fig_real, ax_real = default_plot()
    y_data_re = [np.asarray(gv.mean(qda_ratio_real), dtype=float)]
    yerr_re = [np.asarray(gv.sdev(qda_ratio_real), dtype=float)]
    ax_real.errorbar(
        t,
        gv.mean(qda_ratio_real),
        yerr=gv.sdev(qda_ratio_real),
        label="Data",
        **ERRORBAR_CIRCLE_STYLE,
    )
    if fit_result is not None:
        pt2_params = fit_result.p if pt2_fit_result is None else pt2_fit_result.p
        pt2_nstate = _fit_nstate(fit_result if pt2_fit_result is None else pt2_fit_result)
        qda_nstate = _fit_nstate(fit_result)
        fit_ratio = qda_re_fcn(
            fit_t, fit_result.p, fit_lt, nstate=qda_nstate
        ) / pt2_re_fcn(fit_t, pt2_params, fit_lt, nstate=pt2_nstate)
        fit_mean = gv.mean(fit_ratio)
        fit_sdev = gv.sdev(fit_ratio)
        ax_real.fill_between(
            fit_t,
            fit_mean - fit_sdev,
            fit_mean + fit_sdev,
            alpha=0.35,
            label=fit_label,
        )
        y_data_re.append(np.asarray(fit_mean, dtype=float))
        yerr_re.append(np.asarray(fit_sdev, dtype=float))

    ax_real.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_real.set_ylabel(r"$\Re[R_{\mathrm{qDA}}(t_{\mathrm{sep}})]$", **FONT_SIZE)
    ax_real.legend(**LEGEND_SIZE)
    ax_real.set_title(f"{title_prefix}R_qDA_real", **FONT_SIZE)
    ax_real.set_ylim(auto_ylim(y_data_re, yerr_re))

    if qda_ratio_imag is not None:
        fig_imag, ax_imag = default_plot()
        y_data_im = [np.asarray(gv.mean(qda_ratio_imag), dtype=float)]
        yerr_im = [np.asarray(gv.sdev(qda_ratio_imag), dtype=float)]
        ax_imag.errorbar(
            t,
            gv.mean(qda_ratio_imag),
            yerr=gv.sdev(qda_ratio_imag),
            label="Data",
            **ERRORBAR_CIRCLE_STYLE,
        )
        if fit_result is not None:
            pt2_params = fit_result.p if pt2_fit_result is None else pt2_fit_result.p
            pt2_nstate = _fit_nstate(fit_result if pt2_fit_result is None else pt2_fit_result)
            qda_nstate = _fit_nstate(fit_result)
            fit_ratio = qda_im_fcn(
                fit_t, fit_result.p, fit_lt, nstate=qda_nstate
            ) / pt2_re_fcn(fit_t, pt2_params, fit_lt, nstate=pt2_nstate)
            fit_mean = gv.mean(fit_ratio)
            fit_sdev = gv.sdev(fit_ratio)
            ax_imag.fill_between(
                fit_t,
                fit_mean - fit_sdev,
                fit_mean + fit_sdev,
                alpha=0.35,
                label=fit_label,
            )
            y_data_im.append(np.asarray(fit_mean, dtype=float))
            yerr_im.append(np.asarray(fit_sdev, dtype=float))

        ax_imag.set_xlabel(TSEP_LABEL, **FONT_SIZE)
        ax_imag.set_ylabel(r"$\Im[R_{\mathrm{qDA}}(t_{\mathrm{sep}})]$", **FONT_SIZE)
        ax_imag.legend(**LEGEND_SIZE)
        ax_imag.set_title(f"{title_prefix}R_qDA_imag", **FONT_SIZE)
        ax_imag.set_ylim(auto_ylim(y_data_im, yerr_im))

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig_real.savefig(path.with_name(f"{path.name}_real.pdf"), bbox_inches="tight", transparent=True)
            fig_imag.savefig(path.with_name(f"{path.name}_imag.pdf"), bbox_inches="tight", transparent=True)

        if show:
            fig_real.show()
            fig_imag.show()
        return (fig_real, ax_real), (fig_imag, ax_imag)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_real.savefig(path.with_name(f"{path.name}_real.pdf"), bbox_inches="tight", transparent=True)
    if show:
        fig_real.show()
    return fig_real, ax_real


def fh_plot(
    tsep_ls: Sequence[int],
    fh_real: np.ndarray,
    fh_imag: np.ndarray | None = None,
    *,
    fit_result: object | None = None,
    fit_tsep_ls: Sequence[int] | None = None,
    fit_tau_cut: int = 0,
    fit_label: str = "Fit",
    dt: int | float | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[tuple[Figure, Axes], tuple[Figure, Axes]] | tuple[Figure, Axes]:
    """Plot FH vs tsep from precomputed gvar arrays."""

    t = np.asarray(tsep_ls, dtype=float)
    fit_t = np.asarray(list(t if fit_tsep_ls is None else fit_tsep_ls), dtype=float)
    fit_dt = (t[1] - t[0]) if dt is None and len(t) > 1 else (1 if dt is None else dt)
    fit_nstate = 1 if fit_result is None else _fh_fit_nstate(fit_result)

    fig_real, ax_real = default_plot()
    y_data_re = [np.asarray(gv.mean(fh_real), dtype=float)]
    yerr_re = [np.asarray(gv.sdev(fh_real), dtype=float)]
    ax_real.errorbar(
        t,
        gv.mean(fh_real),
        yerr=gv.sdev(fh_real),
        label="Data",
        **ERRORBAR_CIRCLE_STYLE,
    )
    if fit_result is not None:
        fit_fh = fh_re_fcn(fit_t, fit_tau_cut, fit_result.p, nstate=fit_nstate, dt=fit_dt)
        fit_mean = gv.mean(fit_fh)
        fit_sdev = gv.sdev(fit_fh)
        ax_real.fill_between(
            fit_t,
            fit_mean - fit_sdev,
            fit_mean + fit_sdev,
            alpha=0.35,
            label=fit_label,
        )
        y_data_re.append(np.asarray(fit_mean, dtype=float))
        yerr_re.append(np.asarray(fit_sdev, dtype=float))
    ax_real.set_xlabel(TSEP_LABEL, **FONT_SIZE)
    ax_real.set_ylabel(r"$\Re[\mathrm{FH}(t_{\mathrm{sep}})]$", **FONT_SIZE)
    ax_real.set_ylim(auto_ylim(y_data_re, yerr_re, 2))
    ax_real.legend(**LEGEND_SIZE)

    if fh_imag is not None:
        fig_imag, ax_imag = default_plot()
        y_data_im = [np.asarray(gv.mean(fh_imag), dtype=float)]
        yerr_im = [np.asarray(gv.sdev(fh_imag), dtype=float)]
        ax_imag.errorbar(
            t,
            gv.mean(fh_imag),
            yerr=gv.sdev(fh_imag),
            label="Data",
            **ERRORBAR_CIRCLE_STYLE,
        )
        if fit_result is not None:
            fit_fh = fh_im_fcn(fit_t, fit_tau_cut, fit_result.p, nstate=fit_nstate, dt=fit_dt)
            fit_mean = gv.mean(fit_fh)
            fit_sdev = gv.sdev(fit_fh)
            ax_imag.fill_between(
                fit_t,
                fit_mean - fit_sdev,
                fit_mean + fit_sdev,
                alpha=0.35,
                label=fit_label,
            )
            y_data_im.append(np.asarray(fit_mean, dtype=float))
            yerr_im.append(np.asarray(fit_sdev, dtype=float))
        ax_imag.set_xlabel(TSEP_LABEL, **FONT_SIZE)
        ax_imag.set_ylabel(r"$\Im[\mathrm{FH}(t_{\mathrm{sep}})]$", **FONT_SIZE)
        ax_imag.set_ylim(auto_ylim(y_data_im, yerr_im, 2))
        ax_imag.legend(**LEGEND_SIZE)

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig_real.savefig(path.with_name(f"{path.name}_real.pdf"), bbox_inches="tight", transparent=True)
            fig_imag.savefig(path.with_name(f"{path.name}_imag.pdf"), bbox_inches="tight", transparent=True)

        if show:
            fig_real.show()
            fig_imag.show()
        return (fig_real, ax_real), (fig_imag, ax_imag)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_real.savefig(path.with_name(f"{path.name}_real.pdf"), bbox_inches="tight", transparent=True)
    if show:
        fig_real.show()
    return fig_real, ax_real
