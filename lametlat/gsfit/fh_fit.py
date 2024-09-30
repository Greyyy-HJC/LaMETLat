# %%
import logging
import numpy as np
import lsqfit as lsf
import gvar as gv

from lametlat.gsfit.fit_funcs import fh_re_fcn, fh_im_fcn, fh_re_2state_fcn, fh_im_2state_fcn
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
        re = fh_re_fcn(x["re"], p)
        im = fh_im_fcn(x["im"], p)
        val = {"re": re, "im": im}
        return val

    x_dic = {"re": np.array(tsep_ls)[:-1], "im": np.array(tsep_ls)[:-1]} # * Note here because we are fitting FH = sum(t + 1) - sum(t)
    y_dic = {"re": fh_re_avg, "im": fh_im_avg}
    #todo
    # x_dic = {"re": np.array(tsep_ls)[1:-1], "im": np.array(tsep_ls)[:-1]} # * Note here because we are fitting FH = sum(t + 1) - sum(t)
    # y_dic = {"re": fh_re_avg[1:], "im": fh_im_avg}
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


def fh_two_state_fit(fh_re_avg, fh_im_avg, tsep_ls, tau_cut, id_label, pt2_fit_res=None):
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
    # Set 2pt fit results as priors
    if pt2_fit_res is not None:
        priors.update( {key: pt2_fit_res.p[key] for key in ["log(dE1)"]} )

    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    # * fit function
    def fcn(x, p):
        t = x["re"]
        re = fh_re_2state_fcn(t, tau_cut, p)
        im = fh_im_2state_fcn(t, tau_cut, p)
        val = {"re": re, "im": im}
        return val

    x_dic = {"re": np.array(tsep_ls)[:-1], "im": np.array(tsep_ls)[:-1]} # * Note here because we are fitting FH = sum(t + 1) - sum(t)
    y_dic = {"re": fh_re_avg, "im": fh_im_avg}
    fh_fit_res = lsf.nonlinear_fit(
        data=(x_dic, y_dic), prior=priors, fcn=fcn, maxit=10000
    )

    if fh_fit_res.Q < 0.05:
        logging.warning(
            f">>> Bad fh two state fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fh_fit_res.Q:.3f}, Chi2/dof = {fh_fit_res.chi2/fh_fit_res.dof:.3f}"
        )
    else:
        logging.info(
            f">>> Good fh two state fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fh_fit_res.Q:.3f}, Chi2/dof = {fh_fit_res.chi2/fh_fit_res.dof:.3f}"
        )

    return fh_fit_res


def plot_fh_fit_on_data(fh_re_avg, fh_im_avg, fh_fit_res, err_tsep_ls, fill_tsep_ls, id_label):
    
    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    def plot_part(part, fh_avg, fh_fcn):
        
        fig, ax = default_plot()
        
        ax.errorbar(err_tsep_ls[:-1], gv.mean(fh_avg), yerr=gv.sdev(fh_avg), label='Data', color=blue, **errorb) # * Note here because we are fitting FH = sum(t + 1) - sum(t)
        
        fit_t = np.linspace(fill_tsep_ls[0]-0.5, fill_tsep_ls[-2]+0.5, 100) # * Note here because we are fitting FH = sum(t + 1) - sum(t)
        fit_fh = fh_fcn(fit_t, fh_fit_res.p)
        
        ax.fill_between(fit_t, [v.mean + v.sdev for v in fit_fh], [v.mean - v.sdev for v in fit_fh], label='Fit', color=blue, alpha=0.4)
        
        ax.set_xlabel(r'$t_{\mathrm{sep}}$', **fs_p)
        ax.set_ylabel(r'$FH$', **fs_p)
        ax.legend(**fs_p)
        ax.set_title(f'PX = {px}, PY={py}, PZ={pz}, z={z}, b={b}, {part}', **fs_p)
        ax.set_ylim(auto_ylim([gv.mean(fh_avg), gv.mean(fit_fh)], [gv.sdev(fh_avg), gv.sdev(fit_fh)]))
        plt.tight_layout()
        
        return ax
        
    ax_real = plot_part('real', fh_re_avg, fh_re_fcn)
    ax_imag = plot_part('imag', fh_im_avg, fh_im_fcn)

    return ax_real, ax_imag

def plot_fh_2state_fit_on_data(fh_re_avg, fh_im_avg, fh_fit_res, err_tsep_ls, fill_tsep_ls, tau_cut, id_label):
    
    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    def plot_part(part, fh_avg, fh_fcn):
        
        fig, ax = default_plot()
        
        ax.errorbar(err_tsep_ls[:-1], gv.mean(fh_avg), yerr=gv.sdev(fh_avg), label='Data', color=blue, **errorb) # * Note here because we are fitting FH = sum(t + 1) - sum(t)
        
        fit_t = np.linspace(fill_tsep_ls[0]-0.5, fill_tsep_ls[-2]+0.5, 100) # * Note here because we are fitting FH = sum(t + 1) - sum(t)
        fit_fh = fh_fcn(fit_t, tau_cut, fh_fit_res.p)
        
        ax.fill_between(fit_t, [v.mean + v.sdev for v in fit_fh], [v.mean - v.sdev for v in fit_fh], label='Fit', color=blue, alpha=0.4)
        
        ax.set_xlabel(r'$t_{\mathrm{sep}}$', **fs_p)
        ax.set_ylabel(r'$FH$', **fs_p)
        ax.legend(**fs_p)
        ax.set_title(f'PX = {px}, PY={py}, PZ={pz}, z={z}, b={b}, {part}', **fs_p)
        ax.set_ylim(auto_ylim([gv.mean(fh_avg), gv.mean(fit_fh)], [gv.sdev(fh_avg), gv.sdev(fit_fh)]))
        plt.tight_layout()
        
        return ax
        
    ax_real = plot_part('real', fh_re_avg, fh_re_2state_fcn)
    ax_imag = plot_part('imag', fh_im_avg, fh_im_2state_fcn)

    return ax_real, ax_imag