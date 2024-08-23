"""
Do the coordinate extrapolation by fitting the large z data.
Fit a single list independently.
Only fit the z > 0 part, and then mirror the z < 0 part with symmetry.
"""

# %%
import logging
import lsqfit as lsf
from lametlat.utils.plot_settings import *
from lametlat.extrapolation.fit_funcs import *
from lametlat.extrapolation.prior_setting import *


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
        logging.warning(
            f">>> Bad extrapolation fit with Q = {fit_result.Q:.3f}, Chi2/dof = {fit_result.chi2/fit_result.dof:.3f}"
        )
    else:
        logging.info(
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


def extrapolate_exp(lam_ls, re_gv, im_gv, fit_idx_range, extrapolated_length=200):
    """
    Fit and extrapolate the quasi distribution at large lambda using simple exponential decay, note here we need to concatenate the data points and extrapolated points together at the point fit_idx_range[0]

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit

    Returns:
        three lists after extrapolation: lambda list, real part of quasi distribution, imag part of quasi distribution
    """
    lam_gap = abs(lam_ls[1] - lam_ls[0])  # the gap between two discrete lambda

    lam_fit = lam_ls[fit_idx_range[0] : fit_idx_range[1]]
    re_fit = re_gv[fit_idx_range[0] : fit_idx_range[1]]
    im_fit = im_gv[fit_idx_range[0] : fit_idx_range[1]]

    priors = exp_decay_prior()

    fit_result_re = lsf.nonlinear_fit(
        data=(lam_fit, re_fit),
        prior=priors,
        fcn=exp_poly_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    fit_result_im = lsf.nonlinear_fit(
        data=(lam_fit, im_fit),
        prior=priors,
        fcn=exp_poly_fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    if fit_result_re.Q < 0.05:
        logging.warning(
            f">>> Bad extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}"
        )
    else:
        logging.info(
            f">>> Good extrapolation fit for real part with Q = {fit_result_re.Q:.3f}, Chi2/dof = {fit_result_re.chi2/fit_result_re.dof:.3f}"
        )

    if fit_result_im.Q < 0.05:
        logging.warning(
            f">>> Bad extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}"
        )
    else:
        logging.info(
            f">>> Good extrapolation fit for imag part with Q = {fit_result_im.Q:.3f}, Chi2/dof = {fit_result_im.chi2/fit_result_im.dof:.3f}"
        )

    # * two parts: data points and extrapolated points
    lam_ls_part1 = lam_ls[: fit_idx_range[0]]
    re_gv_part1 = re_gv[: fit_idx_range[0]]
    im_gv_part1 = im_gv[: fit_idx_range[0]]

    lam_ls_part2 = np.arange(lam_ls[fit_idx_range[0]], extrapolated_length, lam_gap)

    re_gv_part2 = exp_poly_fcn(lam_ls_part2, fit_result_re.p)
    im_gv_part2 = exp_poly_fcn(lam_ls_part2, fit_result_im.p)

    extrapolated_lam_ls = list(lam_ls_part1) + list(lam_ls_part2)
    extrapolated_re_gv = list(re_gv_part1) + list(re_gv_part2)
    extrapolated_im_gv = list(im_gv_part1) + list(im_gv_part2)

    return (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result_re,
        fit_result_im,
    )


def bf_aft_extrapolation_plot(lam_ls, re_gv, im_gv, extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_idx_range, title, xlim=[-1, 25], save_path=None):      
    """Make a comparison plot of the coordinate distribution in lambda dependence before and after extrapolation, default not save.

    Args:
        lam_ls (list): lambda list
        re_gv (list): gvar list of real part of quasi distribution
        im_gv (list): gvar list of imag part of quasi distribution
        fit_idx_range (list): two int numbers, [0] is the start index, [1] is the end index, indicating the lambda range included in the fit
        save (bool, optional): whether save it. Defaults to True.
    """

    # * plot the real part
    fig, ax = default_plot()
    ax.errorbar(
        lam_ls, [v.mean for v in re_gv], [v.sdev for v in re_gv], label="data", **errorb
    )
    ax.fill_between(
        extrapolated_lam_ls,
        [v.mean - v.sdev for v in extrapolated_re_gv],
        [v.mean + v.sdev for v in extrapolated_re_gv],
        alpha=0.4,
        label="extrapolated",
    )

    ax.axvline(lam_ls[fit_idx_range[0]], ymin=0, ymax=0.5, color=red, linestyle="--")
    ax.axvline(
        lam_ls[fit_idx_range[1] - 1], ymin=0, ymax=0.5, color=red, linestyle="--"
    )

    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_xlabel(lambda_label, **fs_p)
    ax.set_ylim(
        auto_ylim(
            [gv.mean(re_gv), gv.mean(extrapolated_re_gv)],
            [gv.sdev(re_gv), gv.sdev(extrapolated_re_gv)],
        )
    )
    ax.set_xlim(xlim)
    plt.title(title + " Real", **fs_p)
    plt.legend(**fs_p)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path + title + "_real.pdf", transparent=True)

    # * plot the imag part
    fig, ax = default_plot()
    ax.errorbar(
        lam_ls, [v.mean for v in im_gv], [v.sdev for v in im_gv], label="data", **errorb
    )
    ax.fill_between(
        extrapolated_lam_ls,
        [v.mean - v.sdev for v in extrapolated_im_gv],
        [v.mean + v.sdev for v in extrapolated_im_gv],
        alpha=0.4,
        label="extrapolated",
    )

    ax.axvline(lam_ls[fit_idx_range[0]], ymin=0, ymax=0.5, color=red, linestyle="--")
    ax.axvline(
        lam_ls[fit_idx_range[1] - 1], ymin=0, ymax=0.5, color=red, linestyle="--"
    )

    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_xlabel(lambda_label, **fs_p)
    ax.set_ylim(
        auto_ylim(
            [gv.mean(im_gv), gv.mean(extrapolated_im_gv)],
            [gv.sdev(im_gv), gv.sdev(extrapolated_im_gv)],
        )
    )
    ax.set_xlim(xlim)
    plt.title(title + "_imag", **fs_p)
    plt.legend(**fs_p)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path + title + "_imag.pdf", transparent=True)

    return


if __name__ == "__main__":
    import numpy as np
    import gvar as gv
    import matplotlib.pyplot as plt

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

    (
        extrapolated_lam_ls,
        extrapolated_re_gv,
        extrapolated_im_gv,
        fit_result,
        fit_result_im,
    ) = extrapolate_exp(
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
