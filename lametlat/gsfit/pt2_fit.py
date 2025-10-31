# %%
import logging
my_logger = logging.getLogger("my_logger")

import numpy as np
import gvar as gv
import lsqfit as lsf
from lametlat.gsfit.fit_funcs import pt2_re_fcn
from lametlat.gsfit.prior_setting import two_state_fit
from lametlat.utils.plot_settings import *


def pt2_two_state_fit(pt2_avg, tmin, tmax, Lt, normalize=True, label=None):
    """
    Perform a 2-point fit with two states.

    Args:
        pt2_avg (gvar list): The averaged 2-point data.
        tmin (int): The minimum time value for the fit range.
        tmax (int): The maximum time value for the fit range.
        Lt (int): The temporal size of the lattice.
        normalize (bool, optional): Whether to normalize the 2pt data. Defaults to True.
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
    if normalize:
        normalization_factor = abs(pt2_avg[0].mean)
        fit_pt2 = pt2_avg[tmin:tmax] / normalization_factor
    else:
        fit_pt2 = pt2_avg[tmin:tmax]
    fit_res = lsf.nonlinear_fit(
        data=(t_range, fit_pt2), prior=priors, fcn=fcn, maxit=10000
    )

    if fit_res.Q < 0.05:
        my_logger.warning(f">>> Bad 2pt {label} fit with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")
    else:
        my_logger.info(f">>> Good 2pt {label} fit with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")

    return fit_res


def plot_2pt_fit_on_meff(pt2_avg, pt2_fit_res, err_t_ls, fill_t_ls, id_label, Lt):
    """
    Plot the 2pt fit on meff.
    
    Args:
        pt2_avg (gvar list): The averaged 2-point data, same shape as err_t_ls.
        pt2_fit_res (FitResult): The result of the fit.
        err_t_ls (list): The error time list.
        fill_t_ls (list): The fill time list.
    """
    
    from lametlat.preprocess.read_raw import pt2_to_meff
    
    id_label_str = ""
    for key in id_label:
        id_label_str += f"{key} = {id_label[key]}, "

    fit_pt2 = pt2_re_fcn(fill_t_ls, pt2_fit_res.p, Lt)
    
    meff_avg = pt2_to_meff(pt2_avg)
    meff_fit = pt2_to_meff(fit_pt2)
    
    len_diff = abs(len(err_t_ls) - len(meff_avg))
    
    fig, ax = default_plot()
    ax.errorbar(err_t_ls[:-len_diff], gv.mean(meff_avg), yerr=gv.sdev(meff_avg), label='Data', color=blue, **errorb)
    ax.fill_between(fill_t_ls[:-len_diff], [v.mean + v.sdev for v in meff_fit], [v.mean - v.sdev for v in meff_fit], color=red, alpha=0.4, label='Fit')
    ax.set_xlabel(r'$t_{\mathrm{sep}}$', **fs_p)
    ax.set_ylabel(r'$m_{\mathrm{eff}}$', **fs_p)
    ax.legend(**fs_p)
    ax.set_title(f'{id_label_str}meff', **fs_p)
    plt.tight_layout()
    plt.show()
    
    return fig, ax
    
if __name__ == "__main__":
    import numpy as np
    import gvar as gv
    from lametlat.utils.plot_settings import default_plot
    from lametlat.preprocess.read_raw import pt2_to_meff
    
    # Test data
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
    
    # Perform the fit
    fit_res = pt2_two_state_fit(pt2_avg, tmin, tmax, Lt, "test")
    
    print("Fit results:")
    print(fit_res.format(maxline=True))
    
    # Plot the results
    err_t_ls = t_range  # Exclude the last point for meff calculation
    fill_t_ls = np.arange(tmin, tmax)
    
    id_label = {"test": "value"}
    ax = plot_2pt_fit_on_meff(pt2_avg[:20], fit_res, err_t_ls[:20], fill_t_ls, id_label, Lt)
    
    # Additional checks
    fitted_E0 = fit_res.p["E0"].mean
    fitted_E1 = (fit_res.p["E0"] + gv.exp(fit_res.p["log(dE1)"])).mean
    
    print(f"Input E0: {E0:.4f}, Fitted E0: {fitted_E0:.4f}")
    print(f"Input E1: {E1:.4f}, Fitted E1: {fitted_E1:.4f}")
    
# %%
