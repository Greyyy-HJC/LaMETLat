# %%
import logging
import numpy as np
import lsqfit as lsf
import gvar as gv

from lametlat.gsfit.fit_funcs import sum_re_fcn, sum_im_fcn
from lametlat.gsfit.prior_setting import summation_fit
from lametlat.utils.plot_settings import *


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
        logging.warning(
            f">>> Bad sum fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {sum_fit_res.Q:.3f}, Chi2/dof = {sum_fit_res.chi2/sum_fit_res.dof:.3f}"
        )
    else:
        logging.info(
            f">>> Good sum fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {sum_fit_res.Q:.3f}, Chi2/dof = {sum_fit_res.chi2/sum_fit_res.dof:.3f}"
        )

    return sum_fit_res

def plot_sum_fit_on_data(sum_re_avg, sum_im_avg, sum_fit_res, err_tsep_ls, fill_tsep_ls, id_label, tau_cut):
    
    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    def plot_part(part, sum_avg, sum_fcn):
        
        fig, ax = default_plot()
        
        ax.errorbar(err_tsep_ls, gv.mean(sum_avg), yerr=gv.sdev(sum_avg), label='Data', color=blue, **errorb)
        
        fit_t = np.linspace(fill_tsep_ls[0]-0.5, fill_tsep_ls[-1]+0.5, 100)
        fit_sum = sum_fcn(fit_t, tau_cut, sum_fit_res.p)
        
        ax.fill_between(fit_t, [v.mean + v.sdev for v in fit_sum], [v.mean - v.sdev for v in fit_sum], label='Fit', color=blue, alpha=0.4)
        
        ax.set_xlabel(r'$t_{\mathrm{sep}}$', **fs_p)
        ax.set_ylabel(r'$\mathcal{S}$', **fs_p)
        ax.legend(**fs_p)
        ax.set_title(f'PX = {px}, PY={py}, PZ={pz}, z={z}, b={b}, {part}', **fs_p)
        ax.set_ylim(auto_ylim([gv.mean(sum_avg), gv.mean(fit_sum)], [gv.sdev(sum_avg), gv.sdev(fit_sum)]))
        plt.tight_layout()
        
        return ax
        
    ax_real = plot_part('real', sum_re_avg, sum_re_fcn)
    ax_imag = plot_part('imag', sum_im_avg, sum_im_fcn)

    return ax_real, ax_imag
