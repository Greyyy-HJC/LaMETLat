"""Helper functions for LaMETLat."""

from __future__ import annotations

import numpy as np
import gvar as gv
import lsqfit as lsf
from lametlat.utils.logger import log_nonlinear_fit_quality

def constant_fit(data, const_prior=gv.gvar(0, 100), label=None, log_quality=False):
    def fcn(x, p):
        return x * 0 + p["const"]

    priors = gv.BufferDict({"const": const_prior})
    x = np.arange(len(data))
    fit_res = lsf.nonlinear_fit(
        data=(x, data),
        prior=priors,
        fcn=fcn,
        maxit=10000,
        svdcut=1e-4,
        fitter="scipy_least_squares",
    )
    if log_quality:
        log_nonlinear_fit_quality(fit_res, kind="constant", label=label)

    return fit_res.p["const"]