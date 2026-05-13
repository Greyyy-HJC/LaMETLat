"""Joint three-point ratio and Feynman-Hellmann fits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import gvar as gv
import lsqfit as lsf
import numpy as np

from lametlat.utils.logger import log_nonlinear_fit_quality

from .fit_funcs import (
    fh_im_fcn,
    fh_re_fcn,
    fit_parts,
    pt3_ratio_im_fcn,
    pt3_ratio_re_fcn,
)
from .pt3_ratio_fit import _pt3_ratio_prior, _update_prior_from_pt2_fit


def pt3_ratio_fh_joint_fit(
    tsep_ls: Sequence[int],
    tau_cut: int,
    ratio_real: Mapping[int, np.ndarray],
    ratio_imag: Mapping[int, np.ndarray],
    fh_real: np.ndarray,
    fh_imag: np.ndarray,
    Lt: int,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Joint fit of two-state 3pt ratios and one-state FH data."""

    parts = fit_parts(part)
    priors = _pt3_ratio_prior(nstate=2) if prior is None else prior
    if pt2_fit_res is not None:
        _update_prior_from_pt2_fit(priors, pt2_fit_res, nstate=2)

    ratio_t: list[int] = []
    ratio_tau: list[int] = []
    fit_ratio_real: list = []
    fit_ratio_imag: list = []
    for tsep in tsep_ls:
        real_row = np.asarray(ratio_real[tsep], dtype=object)
        imag_row = np.asarray(ratio_imag[tsep], dtype=object)
        for tau in range(tau_cut, tsep + 1 - tau_cut):
            ratio_t.append(tsep)
            ratio_tau.append(tau)
            fit_ratio_real.append(real_row[tau])
            fit_ratio_imag.append(imag_row[tau])

    x_data = [
        np.array(ratio_t, dtype=float),
        np.array(ratio_tau, dtype=float),
        np.asarray(tsep_ls[:-1], dtype=float),
    ]
    all_y_data = {
        "ratio_re": fit_ratio_real,
        "ratio_im": fit_ratio_imag,
        "fh_re": np.asarray(fh_real, dtype=object),
        "fh_im": np.asarray(fh_imag, dtype=object),
    }
    y_data = {}
    for selected in parts:
        y_data[f"ratio_{selected}"] = all_y_data[f"ratio_{selected}"]
        y_data[f"fh_{selected}"] = all_y_data[f"fh_{selected}"]

    def fcn(x: list[np.ndarray], p: dict) -> dict[str, np.ndarray]:
        ratio_t_arr, ratio_tau_arr, fh_t = x
        values = {
            "ratio_re": pt3_ratio_re_fcn(ratio_t_arr, ratio_tau_arr, p, Lt, nstate=2),
            "ratio_im": pt3_ratio_im_fcn(ratio_t_arr, ratio_tau_arr, p, Lt, nstate=2),
            "fh_re": fh_re_fcn(fh_t, tau_cut=0, p=p, nstate=1),
            "fh_im": fh_im_fcn(fh_t, tau_cut=0, p=p, nstate=1),
        }
        out = {}
        for selected in parts:
            out[f"ratio_{selected}"] = values[f"ratio_{selected}"]
            out[f"fh_{selected}"] = values[f"fh_{selected}"]
        return out

    fit_res = lsf.nonlinear_fit(
        data=(x_data, y_data),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
    )

    log_nonlinear_fit_quality(fit_res, kind=f"3pt ratio + FH {part}", label=label)

    return fit_res


def pt3_ratio_two_state_fh_one_state_fit(
    tsep_ls: Sequence[int],
    tau_cut: int,
    ratio_real: Mapping[int, np.ndarray],
    ratio_imag: Mapping[int, np.ndarray],
    fh_real: np.ndarray,
    fh_imag: np.ndarray,
    Lt: int,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Fit two-state 3pt ratios jointly with one-state FH data."""

    return pt3_ratio_fh_joint_fit(
        tsep_ls,
        tau_cut,
        ratio_real,
        ratio_imag,
        fh_real,
        fh_imag,
        Lt,
        prior=prior,
        pt2_fit_res=pt2_fit_res,
        label=label,
        maxit=maxit,
        part=part,
    )
