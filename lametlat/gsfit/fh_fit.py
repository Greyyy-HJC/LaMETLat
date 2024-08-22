# %%
import logging
import numpy as np
import lsqfit as lsf

from lametlat.gsfit.fit_funcs import fh_re_fcn, fh_im_fcn
from lametlat.gsfit.prior_setting import fh_fit
from lametlat.utils.plot_settings import *


def fh_one_state_fit(fh_re_avg, fh_im_avg, tsep_ls, id_label):
    """
    Perform a fh fit with one state.

    Args:
        fh_re_avg (array-like): The real part of the fh to be fitted.
        fh_im_avg (array-like): The imaginary part of the fh to be fitted.
        tsep_ls (array-like): The list of time separations.
        id_label (dict): A dictionary containing the labels for px, py, pz, b, and z.

    Returns:
        fh_fit_res (object): The result of the fh fit.

    Raises:
        None

    """
    priors = fh_fit()

    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    # * fit function
    def fcn(x, p):
        t = x["re"]
        re = fh_re_fcn(t, p)
        im = fh_im_fcn(t, p)
        val = {"re": re, "im": im}
        return val

    x_dic = {"re": np.array(tsep_ls), "im": np.array(tsep_ls)}
    y_dic = {"re": fh_re_avg, "im": fh_im_avg}
    fh_fit_res = lsf.nonlinear_fit(
        data=(x_dic, y_dic), prior=priors, fcn=fcn, maxit=10000
    )

    if fh_fit_res.Q < 0.05:
        logging.warning(
            f">>> Bad fh fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fh_fit_res.Q:.3f}, Chi2/dof = {fh_fit_res.chi2/fh_fit_res.dof:.3f}"
        )
    else:
        logging.info(
            f">>> Good fh fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fh_fit_res.Q:.3f}, Chi2/dof = {fh_fit_res.chi2/fh_fit_res.dof:.3f}"
        )

    return fh_fit_res