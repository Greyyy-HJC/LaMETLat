import logging
import numpy as np
import lsqfit as lsf
from lametlat.gsfit.fit_funcs import pt2_re_fcn
from lametlat.gsfit.prior_setting import two_state_fit



def pt2_two_state_fit(pt2_avg, tmin, tmax, Lt, label=None):
    """
    Perform a 2-point fit with two states.

    Args:
        pt2_avg (gvar list): The averaged 2-point data.
        tmin (int): The minimum time value for the fit range.
        tmax (int): The maximum time value for the fit range.
        Lt (int): The temporal size of the lattice.
        label (str, optional): A label for the fit. Defaults to None.

    Returns:
        FitResult: The result of the fit.

    Raises:
        None

    """

    priors = two_state_fit()

    def fcn(t, p):
        return pt2_re_fcn(t, p, Lt)

    # Compute the range only once, outside of the loop
    t_range = np.arange(tmin, tmax)

    # Normalize the 2pt data only once for each dataset
    normalization_factor = pt2_avg[0]
    fit_pt2 = pt2_avg[tmin:tmax] / normalization_factor

    fit_res = lsf.nonlinear_fit(
        data=(t_range, fit_pt2), prior=priors, fcn=fcn, maxit=10000
    )

    if fit_res.Q < 0.05:
        logging.warning(f">>> Bad 2pt {label} fit with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")
    else:
        logging.info(f">>> Good 2pt {label} fit with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")

    return fit_res
