"""
Do the coordinate extrapolation by fitting the large z data.
Fit a single list independently.
Only fit the z > 0 part, and then mirror the z < 0 part with symmetry.
"""

import lsqfit as lsf
from lametlat.utils.plot_settings import *
from lametlat.extrapolation.fit_funcs import *
from lametlat.extrapolation.prior_setting import *

extrapolated_length = 200  #! do the extrapolation in the coordinate space till lambda = extrapolated_length


def extrapolate(lam_ls, re_gv, im_gv, fit_idx_range):
    """fit and extrapolate the quasi distribution at large lambda, note here we need to concatenate the data points and extrapolated points together

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

    def fcn(x, p):
        val = {}
        val["re"] = Regge_exp_re_fcn(x["re"], p)
        val["im"] = Regge_exp_im_fcn(x["im"], p)

        return val

    fit_result = lsf.nonlinear_fit(
        data=(lam_dic, pdf_dic),
        prior=Regge_prior(),
        fcn=fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
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
    extrapolated_lam_ls, extrapolated_re_gv, extrapolated_im_gv, fit_result = extrapolate(
        lam_ls, re_gv, im_gv, fit_idx_range
    )
    
    # Print some results
    print("Original lambda range:", lam_ls[0], "to", lam_ls[-1])
    print("Extrapolated lambda range:", extrapolated_lam_ls[0], "to", extrapolated_lam_ls[-1])
    print("Fit result:", fit_result)
    
    # Plot the results
    fig, ax = default_plot()
    
    # Plot original data with errorbars
    ax.errorbar(lam_ls, [v.mean for v in re_gv], [v.sdev for v in re_gv], fmt='.', label='Original Real', capsize=3, linestyle='')
    ax.errorbar(lam_ls, [v.mean for v in im_gv], [v.sdev for v in im_gv], fmt='.', label='Original Imag', capsize=3, linestyle='')
    
    # Plot extrapolated data with fill_between
    ax.fill_between(extrapolated_lam_ls, 
                    [v.mean - v.sdev for v in extrapolated_re_gv],
                    [v.mean + v.sdev for v in extrapolated_re_gv],
                    alpha=0.3, label='Extrapolated Real')
    ax.fill_between(extrapolated_lam_ls, 
                    [v.mean - v.sdev for v in extrapolated_im_gv],
                    [v.mean + v.sdev for v in extrapolated_im_gv],
                    alpha=0.3, label='Extrapolated Imag')
    
    # Add vertical dashed line at the point of extrapolation
    extrapolation_point = lam_ls[fit_idx_range[0]]
    ax.axvline(extrapolation_point, color='k', ls='--', ymax=0.5, label='Extrapolation')
    
    ax.set_xlabel('lambda', **fs_p)
    ax.set_ylabel('PDF', **fs_p)
    ax.set_xlim(0, 20)
    ax.legend(**fs_p)
    plt.tight_layout()
    plt.show()

