import logging
my_logger = logging.getLogger("my_logger")

import numpy as np
import lsqfit as lsf
import gvar as gv

from lametlat.gsfit.prior_setting import two_state_fit
from lametlat.gsfit.fit_funcs import ra_re_fcn, ra_im_fcn, fh_re_fcn, fh_im_fcn
from lametlat.utils.plot_settings import *


def ra_two_state_fh_one_state_fit(
    ra_re_avg_dic, ra_im_avg_dic, fh_re_avg, fh_im_avg, tsep_ls, tau_cut, Lt, id_label, pt2_fit_res=None
):
    """
    Perform a joint fit with two states for ratio and one state for fh.

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
    # Set 2pt fit results as priors
    if pt2_fit_res is not None:
        priors.update(
            {key: pt2_fit_res.p[key] for key in ["E0", "log(dE1)", "re_z0", "re_z1"]}
        )

    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]

    def joint_fcn(x, p):
        ra_t, ra_tau, fh_t = x
        return {
            "ratio_re": ra_re_fcn(ra_t, ra_tau, p, Lt),
            "ratio_im": ra_im_fcn(ra_t, ra_tau, p, Lt),
            "fh_re": fh_re_fcn(fh_t, p),
            "fh_im": fh_im_fcn(fh_t, p),
        }

    # Prepare data for fit
    temp_t, temp_tau, temp_fh_t, ra_fit_re, ra_fit_im = [], [], [], [], []
    for tsep in tsep_ls:
        for tau in range(tau_cut, tsep + 1 - tau_cut):
            temp_t.append(tsep)
            temp_tau.append(tau)
            ra_fit_re.append(ra_re_avg_dic[f"tsep_{tsep}"][tau])
            ra_fit_im.append(ra_im_avg_dic[f"tsep_{tsep}"][tau])
            
        temp_fh_t.append(tsep)

    fh_fit_re = fh_re_avg
    fh_fit_im = fh_im_avg

    # Perform the fit
    tsep_tau_ls = [np.array(temp_t), np.array(temp_tau), np.array(temp_fh_t)[:-1]] # * Note here because we are fitting FH = sum(t + 1) - sum(t)
    ra_fit = {"ratio_re": ra_fit_re, "ratio_im": ra_fit_im, "fh_re": fh_fit_re, "fh_im": fh_fit_im}
    
    joint_fit_res = lsf.nonlinear_fit(
        data=(tsep_tau_ls, ra_fit), prior=priors, fcn=joint_fcn, maxit=10000
    )

    # Check the quality of the fit
    if joint_fit_res.Q < 0.05:
        my_logger.warning(
            f">>> Bad ratio + fh fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {joint_fit_res.Q:.3f}, Chi2/dof = {joint_fit_res.chi2/joint_fit_res.dof:.3f}"
        )
    else:
        my_logger.info(
            f">>> Good ratio + fh fit for PX = {px}, PY = {py}, PZ = {pz}, z = {z}, b = {b} with Q = {joint_fit_res.Q:.3f}, Chi2/dof = {joint_fit_res.chi2/joint_fit_res.dof:.3f}"
        )

    return joint_fit_res


def plot_ra_fh_fit_on_data(ra_re_avg, ra_im_avg, joint_fit_res, err_tsep_ls, fill_tsep_ls, Lt, id_label, err_tau_cut=1, fill_tau_cut=1):

    id_label_str = ""
    for key in id_label:
        id_label_str += f"{key} = {id_label[key]}, "
 
    def plot_part(part, ra_avg, ra_fcn, pdf_key):
        
        # Combine and deduplicate tsep lists
        all_tsep_ls = list(set(err_tsep_ls + fill_tsep_ls))
        all_tsep_ls.sort()
        all_tsep_array = np.array(all_tsep_ls)
        
        fig, ax = default_plot()
        
        y_data_ls = []
        yerr_data_ls = []
        for id, tsep in enumerate(err_tsep_ls):
            
            id_all = np.where(all_tsep_array == tsep)[0][0]
            
            tau_range = np.arange(err_tau_cut, tsep + 1 - err_tau_cut)
            err_x_ls = tau_range - tsep / 2
            err_y_ls = gv.mean(ra_avg[id, err_tau_cut:tsep + 1 - err_tau_cut])
            err_yerr_ls = gv.sdev(ra_avg[id, err_tau_cut:tsep + 1 - err_tau_cut])
            
            ax.errorbar(err_x_ls, err_y_ls, err_yerr_ls, label=r'$t_{\mathrm{sep}}$' + f' = {tsep} a', color=color_ls[id_all], **errorb)
            
            y_data_ls.append(err_y_ls)
            yerr_data_ls.append(err_yerr_ls)

        for id, tsep in enumerate(fill_tsep_ls):
            id_all = np.where(all_tsep_array == tsep)[0][0]
            
            fit_tau = np.linspace(fill_tau_cut - 0.5, tsep - fill_tau_cut + 0.5, 100)
            fit_t = np.ones_like(fit_tau) * tsep
            fit_ratio = ra_fcn(fit_t, fit_tau, joint_fit_res.p, Lt)

            fill_x_ls = fit_tau - tsep / 2
            fill_y_ls = gv.mean(fit_ratio)
            fill_yerr_ls = gv.sdev(fit_ratio)

            ax.fill_between(fill_x_ls, [v.mean + v.sdev for v in fit_ratio], [v.mean - v.sdev for v in fit_ratio], color=color_ls[id_all], alpha=0.4)

            y_data_ls.append(fill_y_ls)
            yerr_data_ls.append(fill_yerr_ls)

        band_x = np.arange(-6, 7)
        band_y_ls = np.ones_like(band_x) * gv.mean(joint_fit_res.p[pdf_key])
        band_yerr_ls = np.ones_like(band_x) * gv.sdev(joint_fit_res.p[pdf_key])
        ax.fill_between(band_x, band_y_ls+band_yerr_ls, band_y_ls-band_yerr_ls, color=grey, alpha=0.5, label='Ratio + FH Fit')
        
        y_data_ls.append(band_y_ls)
        yerr_data_ls.append(band_yerr_ls)
        
        ax.set_xlabel(r'($\tau - t_{\mathrm{sep}}/2$) / a', **fs_p)
        ax.set_ylabel(r'$\mathcal{R} (t_{\mathrm{sep}}, \tau)$', **fs_p)
        ax.set_ylim(auto_ylim(y_data_ls, yerr_data_ls, y_range_ratio=2))
        ax.legend(ncol=3, loc='upper right', **fs_small_p)
        ax.set_title(f'{id_label_str}{part}', **fs_p)
        plt.tight_layout()
        
        return fig, ax

    # Plot real part
    fig_real, ax_real = plot_part('real', ra_re_avg, ra_re_fcn, 'pdf_re')

    # Plot imaginary part
    fig_imag, ax_imag = plot_part('imag', ra_im_avg, ra_im_fcn, 'pdf_im')

    return fig_real, fig_imag, ax_real, ax_imag