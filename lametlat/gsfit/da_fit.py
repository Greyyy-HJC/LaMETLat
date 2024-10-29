# %%
import logging
import numpy as np
import gvar as gv
import lsqfit as lsf
from lametlat.gsfit.fit_funcs import pt2_re_fcn, da_re_fcn, da_im_fcn
from lametlat.gsfit.prior_setting import two_state_fit
from lametlat.utils.plot_settings import *


def da_two_state_fit(da_re_avg, da_im_avg, tmin, tmax, Lt, id_label, pt2_fit_res=None):

    priors = two_state_fit()
    # Set 2pt fit results as priors
    if pt2_fit_res is not None:
        priors.update(
            {key: pt2_fit_res.p[key] for key in ["E0", "log(dE1)", "re_z0", "re_z1"]}
        )
        
        # Make prior width 10x larger for updated parameters
        for key in ["E0", "log(dE1)", "re_z0", "re_z1"]:
            priors[key] = gv.gvar(gv.mean(priors[key]), 5*gv.sdev(priors[key]))

        
    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    def da_fcn(x, p):
        da_t = x
        return {
            "re": da_re_fcn(da_t, p, Lt),
            "im": da_im_fcn(da_t, p, Lt),
        }

    # Compute the range only once, outside of the loop
    t_range = np.arange(tmin, tmax)

    fit_da_re = da_re_avg[tmin:tmax]
    fit_da_im = da_im_avg[tmin:tmax]
    
    fit_data = {"re": fit_da_re, "im": fit_da_im}

    fit_res = lsf.nonlinear_fit(
        data=(t_range, fit_data), prior=priors, fcn=da_fcn, maxit=10000
    )

    if fit_res.Q < 0.05:
        logging.warning(f">>> Bad DA fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")
    else:
        logging.info(f">>> Good DA fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")

    return fit_res


def plot_da_fit_on_ratio(pt2_avg, pt2_fit_res, da_re_avg, da_im_avg, da_fit_res, err_t_ls, fill_t_ls, id_label, Lt):

    id_label_str = ""
    for key in id_label:
        id_label_str += f"{key} = {id_label[key]}, "
        
    data_ratio_re = -1j * da_re_avg / pt2_avg
    data_ratio_im = -1j * da_im_avg / pt2_avg

    fit_pt2 = pt2_re_fcn(fill_t_ls, pt2_fit_res.p, Lt)
    fit_da_re = da_re_fcn(fill_t_ls, da_fit_res.p, Lt)
    fit_da_im = da_im_fcn(fill_t_ls, da_fit_res.p, Lt)
    
    fit_ratio_re = -1j * fit_da_re / fit_pt2
    fit_ratio_im = -1j * fit_da_im / fit_pt2
    
    fig, ax = default_plot()
    ax.errorbar(err_t_ls, gv.mean(data_ratio_re), yerr=gv.sdev(data_ratio_re), label='Data', color=blue, **errorb)
    ax.fill_between(fill_t_ls, [v.mean + v.sdev for v in fit_ratio_re], [v.mean - v.sdev for v in fit_ratio_re], color=red, alpha=0.4, label='Fit')
    ax.set_xlabel(r'$t_{\mathrm{sep}}$', **fs_p)
    ax.set_ylabel(r'$R_{\mathrm{DA}}$', **fs_p)
    ax.legend(**fs_p)
    ax.set_title(f'{id_label_str}R_DA_real', **fs_p)
    plt.tight_layout()
    plt.show()
    
    fig, ax = default_plot()
    ax.errorbar(err_t_ls, gv.mean(data_ratio_im), yerr=gv.sdev(data_ratio_im), label='Data', color=blue, **errorb)
    ax.fill_between(fill_t_ls, [v.mean + v.sdev for v in fit_ratio_im], [v.mean - v.sdev for v in fit_ratio_im], color=red, alpha=0.4, label='Fit')
    ax.set_xlabel(r'$t_{\mathrm{sep}}$', **fs_p)
    ax.set_ylabel(r'$R_{\mathrm{DA}}$', **fs_p)
    ax.legend(**fs_p)
    ax.set_title(f'{id_label_str}R_DA_imag', **fs_p)
    plt.tight_layout()
    plt.show()
    
    return ax

# %%
