"""Three-point ratio ground-state fits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import gvar as gv
import lsqfit as lsf
import numpy as np

from lametlat.utils.logger import log_nonlinear_fit_quality

from .fit_funcs import fit_parts, general_prior, pt3_ratio_im_fcn, pt3_ratio_re_fcn


def _pt3_ratio_prior(nstate: int) -> gv.BufferDict:
    prior = general_prior(nstate=nstate)
    ratio_prior = gv.BufferDict()
    ratio_prior["E0"] = prior["E0"]
    for state in range(1, nstate):
        ratio_prior[f"log(dE{state})"] = prior[f"log(dE{state})"]
    for state in range(nstate):
        ratio_prior[f"z{state}"] = prior[f"z{state}"]
    for row in range(nstate):
        for col in range(row, nstate):
            ratio_prior[f"O{row}{col}_re"] = prior[f"O{row}{col}_re"]
            ratio_prior[f"O{row}{col}_im"] = prior[f"O{row}{col}_im"]
    return ratio_prior


def _update_prior_from_pt2_fit(
    prior: gv.BufferDict | dict[str, gv.GVar],
    pt2_fit_res: lsf.nonlinear_fit,
    nstate: int,
) -> None:
    for key in ["E0", *[f"log(dE{state})" for state in range(1, nstate)]]:
        prior[key] = pt2_fit_res.p[key]
    for state in range(nstate):
        prior[f"z{state}"] = pt2_fit_res.p[f"z{state}"]


def pt3_ratio_fit(
    tsep_ls: Sequence[int],
    tau_cut: int,
    ratio_real: Mapping[int, np.ndarray],
    ratio_imag: Mapping[int, np.ndarray],
    Lt: int,
    *,
    nstate: int = 2,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Fit real and imaginary 3pt/2pt ratio data with an n-state ansatz."""

    parts = fit_parts(part)
    priors = _pt3_ratio_prior(nstate=nstate) if prior is None else prior
    if pt2_fit_res is not None:
        _update_prior_from_pt2_fit(priors, pt2_fit_res, nstate)

    ts: list[int] = []
    taus: list[int] = []
    fit_real: list = []
    fit_imag: list = []
    for tsep in tsep_ls:
        if tsep not in ratio_real:
            raise KeyError(f"ratio_real is missing tsep {tsep}")
        if tsep not in ratio_imag:
            raise KeyError(f"ratio_imag is missing tsep {tsep}")

        tau_range = range(tau_cut, tsep + 1 - tau_cut)
        if len(tau_range) == 0:
            raise ValueError(
                f"empty tau fit window for tsep {tsep} and tau_cut {tau_cut}"
            )

        real_row = np.asarray(ratio_real[tsep], dtype=object)
        imag_row = np.asarray(ratio_imag[tsep], dtype=object)

        for tau in tau_range:
            ts.append(tsep)
            taus.append(tau)
            fit_real.append(real_row[tau])
            fit_imag.append(imag_row[tau])

    x_vecs = [np.array(ts, dtype=float), np.array(taus, dtype=float)]
    all_y_data = {"re": fit_real, "im": fit_imag}
    y_data = {key: all_y_data[key] for key in parts}

    def fcn(x: list[np.ndarray], p: dict) -> dict[str, np.ndarray]:
        values = {
            "re": pt3_ratio_re_fcn(x[0], x[1], p, Lt, nstate=nstate),
            "im": pt3_ratio_im_fcn(x[0], x[1], p, Lt, nstate=nstate),
        }
        return {key: values[key] for key in parts}

    fit_res = lsf.nonlinear_fit(
        data=(x_vecs, y_data),
        prior=priors,
        fcn=fcn,
        maxit=maxit,
    )

    log_nonlinear_fit_quality(fit_res, kind=f"3pt ratio {part}", label=label)

    return fit_res


def pt3_ratio_two_state_fit(
    tsep_ls: Sequence[int],
    tau_cut: int,
    ratio_real: Mapping[int, np.ndarray],
    ratio_imag: Mapping[int, np.ndarray],
    Lt: int,
    *,
    prior: gv.BufferDict | dict[str, gv.GVar] | None = None,
    pt2_fit_res: lsf.nonlinear_fit | None = None,
    label: str | None = None,
    maxit: int = 10000,
    part: str = "both",
) -> lsf.nonlinear_fit:
    """Fit real and imaginary 3pt/2pt ratio data with the default two-state model."""

    return pt3_ratio_fit(
        tsep_ls,
        tau_cut,
        ratio_real,
        ratio_imag,
        Lt,
        nstate=2,
        prior=prior,
        pt2_fit_res=pt2_fit_res,
        label=label,
        maxit=maxit,
        part=part,
    )
