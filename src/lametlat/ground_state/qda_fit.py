"""qDA/TMDWF ground-state fits."""

from __future__ import annotations

from collections.abc import Sequence

import gvar as gv
import lsqfit as lsf
import numpy as np

from lametlat.utils.logger import log_nonlinear_fit_quality

from .fit_funcs import fit_parts, general_prior, pt2_re_fcn, qda_im_fcn, qda_re_fcn


def _qda_prior(nstate: int) -> gv.BufferDict:
    prior = general_prior(nstate=nstate)
    qda_prior = gv.BufferDict()
    qda_prior["E0"] = prior["E0"]
    for state in range(1, nstate):
        qda_prior[f"log(dE{state})"] = prior[f"log(dE{state})"]
    for state in range(nstate):
        qda_prior[f"z{state}"] = prior[f"z{state}"]
        qda_prior[f"O0{state}_re"] = prior[f"O0{state}_re"]
        qda_prior[f"O0{state}_im"] = prior[f"O0{state}_im"]
    return qda_prior


def _update_prior_from_pt2_fit(
    prior: gv.BufferDict | dict[str, gv.GVar],
    pt2_fit_res: lsf.nonlinear_fit,
    nstate: int,
    *,
    width_scale: float = 5.0,
) -> None:
    for key in ["E0", *[f"log(dE{state})" for state in range(1, nstate)]]:
        prior[key] = gv.gvar(
            gv.mean(pt2_fit_res.p[key]),
            width_scale * gv.sdev(pt2_fit_res.p[key]),
        )
    for state in range(nstate):
        key = f"z{state}"
        prior[key] = gv.gvar(
            gv.mean(pt2_fit_res.p[key]),
            width_scale * gv.sdev(pt2_fit_res.p[key]),
        )


def qda_fit(
    qda_real: np.ndarray,
    qda_imag: np.ndarray,
    tmin: int,
    tmax: int,
    Lt: int,
    *,
    nstate: int = 2,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    p0: dict | None = None,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Fit real and imaginary qDA/TMDWF correlators with an n-state ansatz."""

    parts = fit_parts(part)
    priors = _qda_prior(nstate=nstate) if prior is None else prior
    if pt2_fit_res is not None:
        _update_prior_from_pt2_fit(priors, pt2_fit_res, nstate)

    fit_t = np.arange(tmin, tmax, dtype=int)
    all_fit_data = {
        "re": np.asarray(qda_real, dtype=object)[fit_t],
        "im": np.asarray(qda_imag, dtype=object)[fit_t],
    }
    fit_data = {key: all_fit_data[key] for key in parts}

    def fcn(t: np.ndarray, p: dict) -> dict[str, np.ndarray]:
        values = {
            "re": qda_re_fcn(t, p, Lt, nstate=nstate),
            "im": qda_im_fcn(t, p, Lt, nstate=nstate),
        }
        return {key: values[key] for key in parts}

    fit_res = lsf.nonlinear_fit(
        data=(fit_t, fit_data),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
        p0=p0,
    )

    log_nonlinear_fit_quality(fit_res, kind=f"qDA {part}", label=label)

    return fit_res


def qda_two_state_fit(
    qda_real: np.ndarray,
    qda_imag: np.ndarray,
    tmin: int,
    tmax: int,
    Lt: int,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    p0: dict | None = None,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Fit real and imaginary qDA/TMDWF correlators with the two-state model."""

    return qda_fit(
        qda_real,
        qda_imag,
        tmin,
        tmax,
        Lt,
        nstate=2,
        prior=prior,
        pt2_fit_res=pt2_fit_res,
        label=label,
        maxit=maxit,
        p0=p0,
        part=part,
    )


def qda_joint_fit(
    pt2_avg: np.ndarray,
    qda_real: np.ndarray,
    qda_imag: np.ndarray,
    pt2_trange: Sequence[int],
    qda_trange: Sequence[int],
    Lt: int,
    *,
    nstate: int = 2,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    label: str | None = None,
    maxit: int = 10000,
    p0: dict | None = None,
    svdcut: float | None = 1e-6,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Joint fit of two-point and qDA/TMDWF correlators."""

    parts = fit_parts(part)
    priors = _qda_prior(nstate=nstate) if prior is None else prior
    pt2_t = np.asarray(pt2_trange, dtype=int)
    qda_t = np.asarray(qda_trange, dtype=int)
    x_data = [pt2_t, qda_t]
    all_fit_data = {
        "pt2": np.asarray(pt2_avg, dtype=object)[pt2_t],
        "re": np.asarray(qda_real, dtype=object)[qda_t],
        "im": np.asarray(qda_imag, dtype=object)[qda_t],
    }
    fit_data = {"pt2": all_fit_data["pt2"]}
    fit_data.update({key: all_fit_data[key] for key in parts})

    def fcn(x: list[np.ndarray], p: dict) -> dict[str, np.ndarray]:
        pt2_x, qda_x = x
        values = {
            "pt2": pt2_re_fcn(pt2_x, p, Lt, nstate=nstate),
            "re": qda_re_fcn(qda_x, p, Lt, nstate=nstate),
            "im": qda_im_fcn(qda_x, p, Lt, nstate=nstate),
        }
        out = {"pt2": values["pt2"]}
        out.update({key: values[key] for key in parts})
        return out

    fit_res = lsf.nonlinear_fit(
        data=(x_data, fit_data),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
        p0=p0,
        svdcut=svdcut,
    )

    log_nonlinear_fit_quality(fit_res, kind=f"qDA joint {part}", label=label)

    return fit_res


def qda_two_state_joint_fit(
    pt2_avg: np.ndarray,
    qda_real: np.ndarray,
    qda_imag: np.ndarray,
    pt2_trange: Sequence[int],
    qda_trange: Sequence[int],
    Lt: int,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    label: str | None = None,
    maxit: int = 10000,
    p0: dict | None = None,
    svdcut: float | None = 1e-6,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Joint fit of two-point and qDA/TMDWF correlators with the two-state model."""

    return qda_joint_fit(
        pt2_avg,
        qda_real,
        qda_imag,
        pt2_trange,
        qda_trange,
        Lt,
        nstate=2,
        prior=prior,
        label=label,
        maxit=maxit,
        p0=p0,
        svdcut=svdcut,
        part=part,
    )
