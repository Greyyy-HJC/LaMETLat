# %%
import logging
my_logger = logging.getLogger("my_logger")

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
            {key: pt2_fit_res.p[key] for key in ["E0", "log(dE1)", "z0", "z1"]}
        )
        
        # Make prior width 10x larger for updated parameters
        for key in ["E0", "log(dE1)", "z0", "z1"]:
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
        my_logger.warning(f">>> Bad DA fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")
    else:
        my_logger.info(f">>> Good DA fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")

    return fit_res


def da_two_state_joint_fit(pt2_avg, da_re_avg, da_im_avg, pt2_trange, da_trange, Lt, id_label, p0=None):
    priors = two_state_fit()
    
    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    def da_fcn(x, p):
        pt2_t = x[0]
        da_t = x[1]
        return {
            "pt2": pt2_re_fcn(pt2_t, p, Lt),
            "re": da_re_fcn(da_t, p, Lt),
            "im": da_im_fcn(da_t, p, Lt),
        }

    # Compute the range only once, outside of the loop
    fit_pt2 = pt2_avg[pt2_trange]
    fit_da_re = da_re_avg[da_trange]
    fit_da_im = da_im_avg[da_trange]
    
    data_x = [pt2_trange, da_trange]
    fit_data = {"pt2": fit_pt2, "re": fit_da_re, "im": fit_da_im}

    fit_res = lsf.nonlinear_fit(
        data=(data_x, fit_data), prior=priors, fcn=da_fcn, maxit=10000,
        svdcut=1e-6, p0=p0,
    )

    if fit_res.Q < 0.05:
        my_logger.warning(f"\n>>> Bad DA joint fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")
    else:
        my_logger.info(f"\n>>> Good DA joint fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")

    return fit_res


def plot_da_fit_on_ratio(pt2_avg, pt2_fit_res, da_re_avg, da_im_avg, da_fit_res, err_t_ls, fill_t_ls, id_label, Lt):

    id_label_str = ""
    for key in id_label:
        id_label_str += f"{key} = {id_label[key]}, "
        
    data_ratio_re = da_re_avg / pt2_avg
    data_ratio_im = da_im_avg / pt2_avg

    fit_pt2 = pt2_re_fcn(fill_t_ls, pt2_fit_res.p, Lt)
    fit_da_re = da_re_fcn(fill_t_ls, da_fit_res.p, Lt)
    fit_da_im = da_im_fcn(fill_t_ls, da_fit_res.p, Lt)
    
    fit_ratio_re = fit_da_re / fit_pt2
    fit_ratio_im = fit_da_im / fit_pt2
    
    fig_re, ax_re = default_plot()
    ax_re.errorbar(err_t_ls, gv.mean(data_ratio_re), yerr=gv.sdev(data_ratio_re), label='Data', color=blue, **errorb)
    ax_re.fill_between(fill_t_ls, [v.mean + v.sdev for v in fit_ratio_re], [v.mean - v.sdev for v in fit_ratio_re], color=red, alpha=0.4, label='Fit')
    ax_re.set_xlabel(r'$t_{\mathrm{sep}}$', **fs_p)
    ax_re.set_ylabel(r'$R_{\mathrm{DA}}$', **fs_p)
    ax_re.legend(**fs_p)
    ax_re.set_title(f'{id_label_str}R_DA_real', **fs_p)
    ax_re.set_ylim(auto_ylim([gv.mean(data_ratio_re), gv.mean(fit_ratio_re)], [gv.sdev(data_ratio_re), gv.sdev(fit_ratio_re)]))
    plt.tight_layout()
    plt.show()
    
    fig_im, ax_im = default_plot()
    ax_im.errorbar(err_t_ls, gv.mean(data_ratio_im), yerr=gv.sdev(data_ratio_im), label='Data', color=blue, **errorb)
    ax_im.fill_between(fill_t_ls, [v.mean + v.sdev for v in fit_ratio_im], [v.mean - v.sdev for v in fit_ratio_im], color=red, alpha=0.4, label='Fit')
    ax_im.set_xlabel(r'$t_{\mathrm{sep}}$', **fs_p)
    ax_im.set_ylabel(r'$R_{\mathrm{DA}}$', **fs_p)
    ax_im.legend(**fs_p)
    ax_im.set_title(f'{id_label_str}R_DA_imag', **fs_p)
    ax_im.set_ylim(auto_ylim([gv.mean(data_ratio_im), gv.mean(fit_ratio_im)], [gv.sdev(data_ratio_im), gv.sdev(fit_ratio_im)]))
    plt.tight_layout()
    plt.show()
    
    return fig_re, fig_im, ax_re, ax_im

# %%

if __name__ == "__main__":
    import numpy as np
    import gvar as gv
    from lametlat.utils.plot_settings import default_plot
    
    # Test data parameters
    Lt = 64
    tmin, tmax = 2, 18
    t_range = np.arange(0, Lt)
    
    # Generate mock 2pt function data
    E0, E1 = 0.5, 1.2
    z0, z1 = 1.0, 1.5
    noise_level = 0.0001
    
    pt2_data = z0**2 / (2 * E0) * np.exp(-E0 * t_range) + z1**2 / (2 * E1) * np.exp(-E1 * t_range)
    pt2_data += np.random.normal(0, noise_level, Lt)
    pt2_avg = gv.gvar(pt2_data, np.full_like(pt2_data, noise_level))
    
    # Generate mock DA data
    # Using similar parameters but with different amplitudes
    da_re_data = 0.8 * z0**2 / (2 * E0) * np.exp(-E0 * t_range) + 0.6 * z1**2 / (2 * E1) * np.exp(-E1 * t_range)
    da_im_data = 0.3 * z0**2 / (2 * E0) * np.exp(-E0 * t_range) + 0.2 * z1**2 / (2 * E1) * np.exp(-E1 * t_range)
    
    da_re_data += np.random.normal(0, noise_level, Lt)
    da_im_data += np.random.normal(0, noise_level, Lt)
    
    da_re_avg = gv.gvar(da_re_data, np.full_like(da_re_data, noise_level))
    da_im_avg = gv.gvar(da_im_data, np.full_like(da_im_data, noise_level))
    
    # Create id_label dictionary
    id_label = {
        "px": 0,
        "py": 0,
        "pz": 0,
        "b": 0,
        "z": 0
    }
    
    # First perform 2pt fit
    from lametlat.gsfit.pt2_fit import pt2_two_state_fit
    pt2_fit_res = pt2_two_state_fit(pt2_avg, tmin, tmax, Lt, normalize=False, label="test")
    
    # Perform DA fit
    da_fit_res = da_two_state_fit(da_re_avg, da_im_avg, tmin, tmax, Lt, id_label, pt2_fit_res)
    
    # Or just do the joint fit
    pt2_trange = np.arange(tmin, tmax)
    da_trange = np.arange(tmin, tmax)
    joint_fit_res = da_two_state_joint_fit(pt2_avg, da_re_avg, da_im_avg, pt2_trange, da_trange, Lt, id_label)
    
    print("Chained DA Fit results:")
    print(da_fit_res.format(maxline=True))
    
    print("Joint DA Fit results:")
    print(joint_fit_res.format(maxline=True))
    
    # Plot the results
    err_t_ls = np.arange(tmax)
    fill_t_ls = np.arange(tmin, tmax)
    
    _ , _ , _ , _ = plot_da_fit_on_ratio(pt2_avg[err_t_ls], pt2_fit_res, da_re_avg[err_t_ls], da_im_avg[err_t_ls], da_fit_res, err_t_ls, fill_t_ls, id_label, Lt)
    
    _ , _ , _ , _ = plot_da_fit_on_ratio(pt2_avg[err_t_ls], joint_fit_res, da_re_avg[err_t_ls], da_im_avg[err_t_ls], joint_fit_res, err_t_ls, fill_t_ls, id_label, Lt)
    
    # Additional checks
    fitted_E0 = da_fit_res.p["E0"].mean
    fitted_E1 = (da_fit_res.p["E0"] + gv.exp(da_fit_res.p["log(dE1)"])).mean
    print(">>> Chained DA fit: ")
    print(f"Input E0: {E0:.4f}, Fitted E0: {fitted_E0:.4f}")
    print(f"Input E1: {E1:.4f}, Fitted E1: {fitted_E1:.4f}")
    
    fitted_E0 = joint_fit_res.p["E0"].mean
    fitted_E1 = (joint_fit_res.p["E0"] + gv.exp(joint_fit_res.p["log(dE1)"])).mean
    print(">>> Joint DA fit: ")
    print(f"Input E0: {E0:.4f}, Fitted E0: {fitted_E0:.4f}")
    print(f"Input E1: {E1:.4f}, Fitted E1: {fitted_E1:.4f}")

# %%
