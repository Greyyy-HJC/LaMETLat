import numpy as np
import lsqfit as lsf

from liblattice.gs_fit.log import log_count_fit
from liblattice.gs_fit.prior_setting import two_state_fit
from liblattice.gs_fit.fit_funcs import ra_re_fcn, ra_im_fcn


def single_ra_fit(
    ra_re_avg_dic, ra_im_avg_dic, tsep_ls, tau_cut, Lt, id_label, pt2_fit_res=None
):
    """
    Perform a single ratio fit.

    Args:
        ra_re_avg_dic (dict of gvar list): Dictionary containing the real part of the ratio average, keys are tsep.
        ra_im_avg_dic (dict of gvar list): Dictionary containing the imaginary part of the ratio average, keys are tsep.
        tsep_ls (list): List of time separations.
        tau_cut (int): Cut-off value for tau.
        pt2_fit_res (object): Object containing the 2pt fit results.
        Lt (int): The temporal size of the lattice.
        id_label (dict): Dictionary containing the labels for px, py, pz, b, and z.

    Returns:
        object: Object containing the fit results.

    Raises:
        None

    """

    priors = two_state_fit()

    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    def ra_fcn(x, p):
        ra_t, ra_tau = x
        return {
            "re": ra_re_fcn(ra_t, ra_tau, p, Lt),
            "im": ra_im_fcn(ra_t, ra_tau, p, Lt),
        }

    # Set 2pt fit results as priors
    if pt2_fit_res is not None:
        priors.update(
            {key: pt2_fit_res.p[key] for key in ["E0", "log(dE1)", "re_z0", "re_z1"]}
        )

    # Prepare data for fit
    temp_t, temp_tau, ra_fit_re, ra_fit_im = [], [], [], []
    for tsep in tsep_ls:
        for tau in range(tau_cut, tsep + 1 - tau_cut):
            temp_t.append(tsep)
            temp_tau.append(tau)
            ra_fit_re.append(ra_re_avg_dic[f"tsep_{tsep}"][tau])
            ra_fit_im.append(ra_im_avg_dic[f"tsep_{tsep}"][tau])

    # Perform the fit
    tsep_tau_ls = [np.array(temp_t), np.array(temp_tau)]
    ra_fit = {"re": ra_fit_re, "im": ra_fit_im}
    ra_fit_res = lsf.nonlinear_fit(
        data=(tsep_tau_ls, ra_fit), prior=priors, fcn=ra_fcn, maxit=10000
    )

    # Check the quality of the fit
    if ra_fit_res.Q < 0.05:
        log_count_fit(
            f">>> Bad fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {ra_fit_res.Q}"
        )
    else:
        log_count_fit()

    return ra_fit_res
