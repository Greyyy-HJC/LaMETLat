# %%
import numpy as np
import lsqfit as lsf

from lametlat.utils.log import log_count_fit
from lametlat.gsfit.fit_funcs import sum_re_fcn, sum_im_fcn
from lametlat.gsfit.prior_setting import summation_fit


def single_sum_fit(sum_re_avg, sum_im_avg, tsep_ls, tau_cut, id_label):
    """
    Perform a single sum fit.

    Args:
        sum_re_avg (array-like): The real part of the sum to be fitted.
        sum_im_avg (array-like): The imaginary part of the sum to be fitted.
        tsep_ls (array-like): The list of time separations.
        tau_cut (float): The cutoff value for tau.
        id_label (dict): A dictionary containing the labels for px, py, pz, b, and z.

    Returns:
        sum_fit_res (object): The result of the sum fit.

    Raises:
        None

    """
    priors = summation_fit()

    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    # * fit function
    def fcn(x, p):
        t = x["re"]
        re = sum_re_fcn(t, tau_cut, p)
        im = sum_im_fcn(t, tau_cut, p)
        val = {"re": re, "im": im}
        return val

    x_dic = {"re": np.array(tsep_ls), "im": np.array(tsep_ls)}
    y_dic = {"re": sum_re_avg, "im": sum_im_avg}
    sum_fit_res = lsf.nonlinear_fit(
        data=(x_dic, y_dic), prior=priors, fcn=fcn, maxit=10000
    )

    if sum_fit_res.Q < 0.05:
        log_count_fit(
            f">>> Bad sum fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {sum_fit_res.Q}"
        )
    else:
        log_count_fit()

    return sum_fit_res
