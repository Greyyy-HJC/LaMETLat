"""Feynman-Hellmann ground-state fits."""

from __future__ import annotations

from collections.abc import Sequence

import gvar as gv
import lsqfit as lsf
import numpy as np

from lametlat.utils.logger import log_nonlinear_fit_quality

from .fit_funcs import fh_im_fcn, fh_re_fcn, fit_parts, general_prior


def _fh_prior(nstate: int) -> gv.BufferDict:
    prior = general_prior(nstate=nstate)
    fh_prior = gv.BufferDict()
    fh_prior["E0"] = prior["E0"]
    fh_prior["O00_re"] = prior["O00_re"]
    fh_prior["O00_im"] = prior["O00_im"]

    if nstate == 1:
        return fh_prior
    if nstate != 2:
        raise ValueError("FH fits currently support nstate <= 2")

    fh_prior["log(dE1)"] = prior["log(dE1)"]
    for part in ["re", "im"]:
        fh_prior[f"sum_{part}_excited_coeff"] = prior[
            f"sum_{part}_excited_coeff"
        ]
        fh_prior[f"sum_{part}_offset"] = prior[f"sum_{part}_offset"]
        fh_prior[f"sum_{part}_exp_offset"] = prior[f"sum_{part}_exp_offset"]
    fh_prior["sum_den_exp_coeff"] = prior["sum_den_exp_coeff"]
    return fh_prior


def _infer_dt(tsep_ls: Sequence[int], dt: int | float | None) -> int | float:
    if dt is not None:
        return dt

    tseps = np.asarray(tsep_ls)
    if len(tseps) < 2:
        raise ValueError("FH fit requires at least two tsep values")
    diffs = np.diff(tseps)
    return int(diffs[0])


def _update_prior_from_pt2_fit(
    prior: gv.BufferDict | dict[str, gv.GVar],
    pt2_fit_res: lsf.nonlinear_fit,
    nstate: int,
) -> None:
    prior["E0"] = pt2_fit_res.p["E0"]
    for state in range(1, nstate):
        prior[f"log(dE{state})"] = pt2_fit_res.p[f"log(dE{state})"]


def fh_fit(
    fh_real: np.ndarray,
    fh_imag: np.ndarray,
    tsep_ls: Sequence[int],
    tau_cut: int,
    *,
    nstate: int = 2,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    dt: int | float | None = None,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Fit real and imaginary FH data with a one- or two-state ansatz."""

    parts = fit_parts(part)
    fit_dt = _infer_dt(tsep_ls, dt)
    fit_t = np.asarray(tsep_ls[:-1], dtype=float)
    fit_real = np.asarray(fh_real, dtype=object)
    fit_imag = np.asarray(fh_imag, dtype=object)
    if fit_t.size == 0:
        raise ValueError("FH fit window must contain at least one point")
    if fit_real.ndim != 1:
        raise ValueError(f"fh_real must be one-dimensional, got shape {fit_real.shape}")
    if fit_imag.ndim != 1:
        raise ValueError(f"fh_imag must be one-dimensional, got shape {fit_imag.shape}")
    if len(fit_real) != len(fit_t):
        raise ValueError(
            f"fh_real length must match len(tsep_ls) - 1: {len(fit_real)} != {len(fit_t)}"
        )
    if len(fit_imag) != len(fit_t):
        raise ValueError(
            f"fh_imag length must match len(tsep_ls) - 1: {len(fit_imag)} != {len(fit_t)}"
        )

    priors = _fh_prior(nstate=nstate) if prior is None else prior
    if pt2_fit_res is not None:
        _update_prior_from_pt2_fit(priors, pt2_fit_res, nstate)

    all_x_data = {"re": fit_t, "im": fit_t}
    all_y_data = {"re": fit_real, "im": fit_imag}
    x_data = {key: all_x_data[key] for key in parts}
    y_data = {key: all_y_data[key] for key in parts}

    def fcn(x: dict[str, np.ndarray], p: dict) -> dict[str, np.ndarray]:
        t = x[parts[0]]
        values = {
            "re": fh_re_fcn(t, tau_cut, p, nstate=nstate, dt=fit_dt),
            "im": fh_im_fcn(t, tau_cut, p, nstate=nstate, dt=fit_dt),
        }
        return {key: values[key] for key in parts}

    fit_res = lsf.nonlinear_fit(
        data=(x_data, y_data),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
    )

    log_nonlinear_fit_quality(fit_res, kind=f"FH {part}", label=label)

    return fit_res


def fh_one_state_fit(
    fh_real: np.ndarray,
    fh_imag: np.ndarray,
    tsep_ls: Sequence[int],
    tau_cut: int = 0,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    dt: int | float | None = None,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Fit real and imaginary FH data with the one-state model."""

    return fh_fit(
        fh_real,
        fh_imag,
        tsep_ls,
        tau_cut,
        nstate=1,
        prior=prior,
        pt2_fit_res=pt2_fit_res,
        label=label,
        maxit=maxit,
        dt=dt,
        part=part,
    )


def fh_two_state_fit(
    fh_real: np.ndarray,
    fh_imag: np.ndarray,
    tsep_ls: Sequence[int],
    tau_cut: int,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    dt: int | float | None = None,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Fit real and imaginary FH data with the default two-state model."""

    return fh_fit(
        fh_real,
        fh_imag,
        tsep_ls,
        tau_cut,
        nstate=2,
        prior=prior,
        pt2_fit_res=pt2_fit_res,
        label=label,
        maxit=maxit,
        dt=dt,
        part=part,
    )
