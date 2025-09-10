"""
Do the coordinate extrapolation by fitting the large z data.
Fit a single list independently.
Only fit the z > 0 part, and then mirror the z < 0 part with symmetry.
"""

# %%
import logging
my_logger = logging.getLogger("my_logger")

import lsqfit as lsf
from lametlat.utils.plot_settings import *
from lametlat.extrapolation.fit_funcs import *
from lametlat.extrapolation.prior_setting import *

def extrapolate_no_fit(lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length=200, weight_ini=0):
    """
    Extrapolate the quasi distribution at large lambda with no fit, just do a weighted average between the data points and zero in the fit region, note here we need to concatenate the data points and extrapolated points together at the point fit_idx_range[0]

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        extrapolated_length (int): the maximum lambda value of the extrapolation
        weight_ini (float): the initial weight of the fit part in the fit region

    Returns:
        three lists after extrapolation: lambda list, real part of quasi distribution, imag part of quasi distribution
    """
    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    # *: extrapolate the data points to weighted average with zero
    re_gv_part2 = [gv.gvar(0, 0) for _ in range(len(lam_ls_part2))]
    im_gv_part2 = [gv.gvar(0, 0) for _ in range(len(lam_ls_part2))]
    
    # *: Smooth the connection point
    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    
    # Define the number of points for gradual weighting
    num_gradual_points = fit_idx_range[1] - fit_idx_range[0]
    
    # Calculate weights for gradual transition
    weights = np.linspace(weight_ini, 1, num_gradual_points) #todo
    
    # Prepare lists for the weighted averages
    weighted_re = []
    weighted_im = []
    
    for i in range(num_gradual_points):
        w = weights[i]
        weighted_re.append(w * re_gv_part2[i] + (1 - w) * re_gv[fit_idx_range[0] + i])
        weighted_im.append(w * im_gv_part2[i] + (1 - w) * im_gv[fit_idx_range[0] + i])
    
    # Combine the parts
    extrapolated_re_gv = list(re_gv_part1) + weighted_re + list(re_gv_part2[num_gradual_points:])
    extrapolated_im_gv = list(im_gv_part1) + weighted_im + list(im_gv_part2[num_gradual_points:])


    return (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        None,
        None,
    )
    
    

def extrapolate_exp_sin(lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length=200, weight_ini=0, m0=0):
    """
    Fit and extrapolate the quasi distribution at large lambda using simple exponential decay, note here we need to concatenate the data points and extrapolated points together at the point fit_idx_range[0]
    
    check https://arxiv.org/pdf/2505.14619, Eq. 7

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        extrapolated_length (int): the maximum lambda value of the extrapolation
        weight_ini (float): the initial weight of the fit part in the fit region
        m0 (float): the minimum value of meff in exp decay exp(-m0 * lam)

    Returns:
        three lists after extrapolation: lambda list, real part of quasi distribution, imag part of quasi distribution
    """
    exp_fcn = exp_sin_fcn(m0)
    
    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda
    
    lam_re_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    lam_im_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    re_fit = re_gv[fit_idx_range[0] : fit_idx_range[1]]
    im_fit = im_gv[fit_idx_range[0] : fit_idx_range[1]]

    priors_re = exp_decay_prior()
    priors_im = exp_decay_prior()

    fit_result_re = lsf.nonlinear_fit(
        data=(lam_re_fit, re_fit),
        prior=priors_re,
        fcn=exp_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    fit_result_im = lsf.nonlinear_fit(
        data=(lam_im_fit, im_fit),
        prior=priors_im,
        fcn=exp_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    if fit_result_re.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}, meff = {fit_result_re.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}, meff = {fit_result_re.p['m']:.3f}"
        )

    if fit_result_im.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}, meff = {fit_result_im.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}, meff = {fit_result_im.p['m']:.3f}"
        )

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    re_gv_part2 = exp_fcn(lam_ls_part2, fit_result_re.p)
    im_gv_part2 = exp_fcn(lam_ls_part2, fit_result_im.p)
    
    # *: standardize the way to do the extrapolation
    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    extrapolated_re_gv = list(re_gv_part1) + list(re_gv_part2)
    extrapolated_im_gv = list(im_gv_part1) + list(im_gv_part2)
    
    # *: Smooth the connection point
    # Define the number of points for gradual weighting
    num_gradual_points = fit_idx_range[1] - fit_idx_range[0]
    
    # Calculate weights for gradual transition
    weights = np.linspace(weight_ini, 1, num_gradual_points)
    
    # Prepare lists for the weighted averages
    weighted_re = []
    weighted_im = []
    
    for i in range(num_gradual_points):
        w = weights[i]
        weighted_re.append(w * re_gv_part2[i] + (1 - w) * re_gv[fit_idx_range[0] + i])
        weighted_im.append(w * im_gv_part2[i] + (1 - w) * im_gv[fit_idx_range[0] + i])
    
    # Combine the parts
    extrapolated_re_gv = list(re_gv_part1) + weighted_re + list(re_gv_part2[num_gradual_points:])
    extrapolated_im_gv = list(im_gv_part1) + weighted_im + list(im_gv_part2[num_gradual_points:])


    return (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result_re,
        fit_result_im,
    )
    


def extrapolate_Regge(
    lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length=200
):  #! do the extrapolation in the coordinate space till lambda = extrapolated_length
    """
    Fit and extrapolate the quasi distribution at large lambda using Regge behavior, note here we need to concatenate the data points and extrapolated points together at the point fit_idx_range[0]

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        extrapolated_length (int): the maximum lambda value of the extrapolation

    Returns:
        three lists after extrapolation: lambda list, real part of quasi distribution, imag part of quasi distribution
    """
    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda

    lam_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    lam_dic = {"re": np.array(lam_fit), "im": np.array(lam_fit)}
    pdf_dic = {
        "re": re_gv[fit_idx_range[0] : fit_idx_range[1]],
        "im": im_gv[fit_idx_range[0] : fit_idx_range[1]],
    }

    priors = Regge_prior()

    def fcn(x, p):
        val = {}
        val["re"] = Regge_exp_re_fcn(x["re"], p)
        val["im"] = Regge_exp_im_fcn(x["im"], p)

        return val

    fit_result = lsf.nonlinear_fit(
        data=(lam_dic, pdf_dic),
        prior=priors,
        fcn=fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    if fit_result.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit with Q = {fit_result.Q:.3f}, Chi2/dof = {fit_result.chi2/fit_result.dof:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit with Q = {fit_result.Q:.3f}, Chi2/dof = {fit_result.chi2/fit_result.dof:.3f}"
        )

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    lam_dic_read = {
        "re": lam_ls_part2,
        "im": lam_ls_part2,
    }  # used to read the extrapolated points from the fit results
    re_gv_part2 = fcn(lam_dic_read, fit_result.p)["re"]
    im_gv_part2 = fcn(lam_dic_read, fit_result.p)["im"]

    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    extrapolated_re_gv = list(re_gv_part1) + list(re_gv_part2)
    extrapolated_im_gv = list(im_gv_part1) + list(im_gv_part2)

    return extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_result


def extrapolate_exp(lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length=200, weight_ini=0):
    """
    Fit and extrapolate the quasi distribution at large lambda using simple exponential decay, note here we need to concatenate the data points and extrapolated points together at the point fit_idx_range[0]

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        extrapolated_length (int): the maximum lambda value of the extrapolation
        weight_ini (float): the initial weight of the fit part in the fit region

    Returns:
        three lists after extrapolation: lambda list, real part of quasi distribution, imag part of quasi distribution
    """
    exp_fcn = exp_power_fcn
    
    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda
    
    lam_re_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    lam_im_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    re_fit = re_gv[fit_idx_range[0] : fit_idx_range[1]]
    im_fit = im_gv[fit_idx_range[0] : fit_idx_range[1]]

    priors_re = exp_decay_prior()
    # priors_re["a"] = gv.gvar(re_fit[0].mean, re_fit[0].mean)
    priors_im = exp_decay_prior()
    # priors_im["a"] = gv.gvar(im_fit[0].mean, im_fit[0].mean)

    fit_result_re = lsf.nonlinear_fit(
        data=(lam_re_fit, re_fit),
        prior=priors_re,
        fcn=exp_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    fit_result_im = lsf.nonlinear_fit(
        data=(lam_im_fit, im_fit),
        prior=priors_im,
        fcn=exp_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    if fit_result_re.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}, meff = {fit_result_re.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}, meff = {fit_result_re.p['m']:.3f}"
        )

    if fit_result_im.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}, meff = {fit_result_im.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}, meff = {fit_result_im.p['m']:.3f}"
        )

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    re_gv_part2 = exp_fcn(lam_ls_part2, fit_result_re.p)
    im_gv_part2 = exp_fcn(lam_ls_part2, fit_result_im.p)
    
    # *: standardize the way to do the extrapolation
    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    extrapolated_re_gv = list(re_gv_part1) + list(re_gv_part2)
    extrapolated_im_gv = list(im_gv_part1) + list(im_gv_part2)
    
    # *: Smooth the connection point
    # Define the number of points for gradual weighting
    num_gradual_points = fit_idx_range[1] - fit_idx_range[0]
    
    # Calculate weights for gradual transition
    weights = np.linspace(weight_ini, 1, num_gradual_points)
    
    # Prepare lists for the weighted averages
    weighted_re = []
    weighted_im = []
    
    for i in range(num_gradual_points):
        w = weights[i]
        weighted_re.append(w * re_gv_part2[i] + (1 - w) * re_gv[fit_idx_range[0] + i])
        weighted_im.append(w * im_gv_part2[i] + (1 - w) * im_gv[fit_idx_range[0] + i])
    
    # Combine the parts
    extrapolated_re_gv = list(re_gv_part1) + weighted_re + list(re_gv_part2[num_gradual_points:])
    extrapolated_im_gv = list(im_gv_part1) + weighted_im + list(im_gv_part2[num_gradual_points:])


    return (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result_re,
        fit_result_im,
    )


def extrapolate_exp_poly(lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length=200, weight_ini=0):
    """
    Fit and extrapolate the quasi distribution at large lambda using simple exponential decay, note here we need to concatenate the data points and extrapolated points together at the point fit_idx_range[0]

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        extrapolated_length (int): the maximum lambda value of the extrapolation
        weight_ini (float): the initial weight of the fit part in the fit region

    Returns:
        three lists after extrapolation: lambda list, real part of quasi distribution, imag part of quasi distribution
    """
    exp_fcn = exp_poly_fcn
    
    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda
    
    lam_re_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    lam_im_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    re_fit = re_gv[fit_idx_range[0] : fit_idx_range[1]]
    im_fit = im_gv[fit_idx_range[0] : fit_idx_range[1]]

    priors_re = exp_decay_prior()
    # priors_re["b"] = gv.gvar(re_fit[0].mean, re_fit[0].mean / 2)
    priors_im = exp_decay_prior()
    # priors_im["b"] = gv.gvar(im_fit[0].mean, im_fit[0].mean / 2)

    fit_result_re = lsf.nonlinear_fit(
        data=(lam_re_fit, re_fit),
        prior=priors_re,
        fcn=exp_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    fit_result_im = lsf.nonlinear_fit(
        data=(lam_im_fit, im_fit),
        prior=priors_im,
        fcn=exp_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    if fit_result_re.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}, meff = {fit_result_re.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}, meff = {fit_result_re.p['m']:.3f}"
        )

    if fit_result_im.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}, meff = {fit_result_im.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}, meff = {fit_result_im.p['m']:.3f}"
        )

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    re_gv_part2 = exp_fcn(lam_ls_part2, fit_result_re.p)
    im_gv_part2 = exp_fcn(lam_ls_part2, fit_result_im.p)
    
    # *: standardize the way to do the extrapolation
    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    extrapolated_re_gv = list(re_gv_part1) + list(re_gv_part2)
    extrapolated_im_gv = list(im_gv_part1) + list(im_gv_part2)
    
    # *: Smooth the connection point
    # Define the number of points for gradual weighting
    num_gradual_points = fit_idx_range[1] - fit_idx_range[0]
    
    # Calculate weights for gradual transition
    weights = np.linspace(weight_ini, 1, num_gradual_points)
    
    # Prepare lists for the weighted averages
    weighted_re = []
    weighted_im = []
    
    for i in range(num_gradual_points):
        w = weights[i]
        weighted_re.append(w * re_gv_part2[i] + (1 - w) * re_gv[fit_idx_range[0] + i])
        weighted_im.append(w * im_gv_part2[i] + (1 - w) * im_gv[fit_idx_range[0] + i])
    
    # Combine the parts
    extrapolated_re_gv = list(re_gv_part1) + weighted_re + list(re_gv_part2[num_gradual_points:])
    extrapolated_im_gv = list(im_gv_part1) + weighted_im + list(im_gv_part2[num_gradual_points:])


    return (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result_re,
        fit_result_im,
    )



def extrapolate_poly(lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length=200, weight_ini=0):
    """
    Fit and extrapolate the quasi distribution at large lambda using simple exponential decay, note here we need to concatenate the data points and extrapolated points together at the point fit_idx_range[0]

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        extrapolated_length (int): the maximum lambda value of the extrapolation
        weight_ini (float): the initial weight of the fit part in the fit region

    Returns:
        three lists after extrapolation: lambda list, real part of quasi distribution, imag part of quasi distribution
    """
    exp_fcn = poly_fcn
    
    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda
    
    lam_re_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    lam_im_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    re_fit = re_gv[fit_idx_range[0] : fit_idx_range[1]]
    im_fit = im_gv[fit_idx_range[0] : fit_idx_range[1]]

    priors_re = exp_decay_prior()
    # priors_re["a"] = gv.gvar(re_fit[0].mean, re_fit[0].mean / 2)
    priors_im = exp_decay_prior()
    # priors_im["a"] = gv.gvar(im_fit[0].mean, im_fit[0].mean / 2)

    fit_result_re = lsf.nonlinear_fit(
        data=(lam_re_fit, re_fit),
        prior=priors_re,
        fcn=exp_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    fit_result_im = lsf.nonlinear_fit(
        data=(lam_im_fit, im_fit),
        prior=priors_im,
        fcn=exp_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    if fit_result_re.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}, meff = {fit_result_re.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}, meff = {fit_result_re.p['m']:.3f}"
        )

    if fit_result_im.Q < 0.05:
        my_logger.warning(
            f">>> Bad extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}, meff = {fit_result_im.p['m']:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}, meff = {fit_result_im.p['m']:.3f}"
        )

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    re_gv_part2 = exp_fcn(lam_ls_part2, fit_result_re.p)
    im_gv_part2 = exp_fcn(lam_ls_part2, fit_result_im.p)
    
    # *: standardize the way to do the extrapolation
    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    extrapolated_re_gv = list(re_gv_part1) + list(re_gv_part2)
    extrapolated_im_gv = list(im_gv_part1) + list(im_gv_part2)
    
    # *: Smooth the connection point
    # Define the number of points for gradual weighting
    num_gradual_points = fit_idx_range[1] - fit_idx_range[0]
    
    # Calculate weights for gradual transition
    weights = np.linspace(weight_ini, 1, num_gradual_points)
    
    # Prepare lists for the weighted averages
    weighted_re = []
    weighted_im = []
    
    for i in range(num_gradual_points):
        w = weights[i]
        weighted_re.append(w * re_gv_part2[i] + (1 - w) * re_gv[fit_idx_range[0] + i])
        weighted_im.append(w * im_gv_part2[i] + (1 - w) * im_gv[fit_idx_range[0] + i])
    
    # Combine the parts
    extrapolated_re_gv = list(re_gv_part1) + weighted_re + list(re_gv_part2[num_gradual_points:])
    extrapolated_im_gv = list(im_gv_part1) + weighted_im + list(im_gv_part2[num_gradual_points:])


    return (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result_re,
        fit_result_im,
    )



def bf_aft_extrapolation_plot(lam_ls, re_gv, im_gv, extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_idx_range, title, xlim=[-1, 25], ylabel_re=None, ylabel_im=None, save_path=None):      
    """Make a comparison plot of the coordinate distribution in lambda dependence before and after extrapolation, default not save.

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        save (bool, optional): whether save it. Defaults to True.
    """
    
    # Interpolate the extrapolated data
    from scipy import interpolate
    import gvar as gv
    import numpy as np

    # Define a finer lambda grid for interpolation
    lam_interp = np.linspace(min(extrapolated_lam_ls), max(extrapolated_lam_ls), len(extrapolated_lam_ls) * 10)

    # Interpolate real part
    re_mean = [v.mean for v in extrapolated_re_gv]
    re_sdev = [v.sdev for v in extrapolated_re_gv]
    f_re_mean = interpolate.interp1d(extrapolated_lam_ls, re_mean, kind='cubic')
    f_re_sdev = interpolate.interp1d(extrapolated_lam_ls, re_sdev, kind='cubic')
    re_mean_interp = f_re_mean(lam_interp)
    re_sdev_interp = f_re_sdev(lam_interp)

    # Interpolate imaginary part
    im_mean = [v.mean for v in extrapolated_im_gv]
    im_sdev = [v.sdev for v in extrapolated_im_gv]
    f_im_mean = interpolate.interp1d(extrapolated_lam_ls, im_mean, kind='cubic')
    f_im_sdev = interpolate.interp1d(extrapolated_lam_ls, im_sdev, kind='cubic')
    im_mean_interp = f_im_mean(lam_interp)
    im_sdev_interp = f_im_sdev(lam_interp)

    # Combine interpolated mean and sdev into gvar
    extrapolated_re_gv_interp = [gv.gvar(m, s) for m, s in zip(re_mean_interp, re_sdev_interp)]
    extrapolated_im_gv_interp = [gv.gvar(m, s) for m, s in zip(im_mean_interp, im_sdev_interp)]

    # Update variables with interpolated values
    extrapolated_lam_ls = lam_interp
    extrapolated_re_gv = extrapolated_re_gv_interp
    extrapolated_im_gv = extrapolated_im_gv_interp
    
    

    # * plot the real part
    fig, ax = default_plot()
    ax.errorbar(
        lam_ls, [v.mean for v in re_gv], [v.sdev for v in re_gv], label="Data", **errorb
    )
    ax.fill_between(
        extrapolated_lam_ls,
        [v.mean - v.sdev for v in extrapolated_re_gv],
        [v.mean + v.sdev for v in extrapolated_re_gv],
        alpha=0.4,
        label="Extrapolated",
    )

    ax.axvline(lam_ls[fit_idx_range[0]], ymin=0, ymax=0.5, color=red, linestyle="--")
    ax.axvline(
        lam_ls[fit_idx_range[1] - 1], ymin=0, ymax=0.5, color=red, linestyle="--"
    )

    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_xlabel(lambda_label, **fs_p)
    if ylabel_re is not None:
        ax.set_ylabel(ylabel_re, **fs_p)
    ax.set_ylim(
        auto_ylim(
            [gv.mean(re_gv), gv.mean(extrapolated_re_gv)],
            [gv.sdev(re_gv), gv.sdev(extrapolated_re_gv)],
        )
    )
    ax.set_xlim(xlim)
    # plt.title(title + " Real", **fs_p)
    plt.legend(**fs_p)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path + title + "_real.pdf", transparent=True)

    # * plot the imag part
    fig, ax = default_plot()
    ax.errorbar(
        lam_ls, [v.mean for v in im_gv], [v.sdev for v in im_gv], label="Data", **errorb
    )
    ax.fill_between(
        extrapolated_lam_ls,
        [v.mean - v.sdev for v in extrapolated_im_gv],
        [v.mean + v.sdev for v in extrapolated_im_gv],
        alpha=0.4,
        label="Extrapolated",
    )

    ax.axvline(lam_ls[fit_idx_range[0]], ymin=0, ymax=0.5, color=red, linestyle="--")
    ax.axvline(
        lam_ls[fit_idx_range[1] - 1], ymin=0, ymax=0.5, color=red, linestyle="--"
    )

    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_xlabel(lambda_label, **fs_p)
    if ylabel_im is not None:
        ax.set_ylabel(ylabel_im, **fs_p)
    ax.set_ylim(
        auto_ylim(
            [gv.mean(im_gv), gv.mean(extrapolated_im_gv)],
            [gv.sdev(im_gv), gv.sdev(extrapolated_im_gv)],
        )
    )
    ax.set_xlim(xlim)
    # plt.title(title + "_imag", **fs_p)
    plt.legend(**fs_p)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path + title + "_imag.pdf", transparent=True)

    return


if __name__ == "__main__":
    import numpy as np
    import gvar as gv
    import matplotlib.pyplot as plt
    
    from lametlat.utils.log import set_my_logger
    set_my_logger("lambda_extrapolation.log")

    # Example lambda values
    lam_ls = np.linspace(0, 10, 30)

    # Generate some example data (replace with your actual data)
    def example_data(lam):
        return np.exp(-lam) * np.sin(lam), np.exp(-lam) * np.cos(lam)

    re_gv = [gv.gvar(example_data(lam)[0], 0.1) for lam in lam_ls]
    im_gv = [gv.gvar(example_data(lam)[1], 0.1) for lam in lam_ls]

    # Define the fit range
    fit_idx_range = [15, 25]  # Fit from index 15 to 25

    # Perform the extrapolation
    # extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_result = extrapolate_Regge(lam_ls, re_gv, im_gv, fit_idx_range)

    extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_result, fit_result_im = extrapolate_exp(
        lam_ls, re_gv, im_gv, fit_idx_range
    )  # note here fit_result is fit_result_re  

    # Print some results
    print("Original lambda range:", lam_ls[0], "to", lam_ls[-1])
    print(
        "Extrapolated lambda range:",
        extrapolated_lam_ls[0],
        "to",
        extrapolated_lam_ls[-1],
    )
    print("Fit result:", fit_result)

    bf_aft_extrapolation_plot(
        lam_ls,
        re_gv,
        im_gv,
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_idx_range=fit_idx_range,
        title="Test extrapolation",
        xlim=[-1, 15],
        save_path=None,
    )


# %%
