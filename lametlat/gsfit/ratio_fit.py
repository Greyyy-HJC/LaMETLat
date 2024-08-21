import numpy as np
import lsqfit as lsf
import gvar as gv

from lametlat.utils.log import log_count_fit
from lametlat.gsfit.prior_setting import two_state_fit
from lametlat.gsfit.fit_funcs import ra_re_fcn, ra_im_fcn
from lametlat.utils.plot_settings import *
from lametlat.utils.resampling import bs_ls_avg


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


def plot_ra_fit_on_data(ra_re_avg, ra_im_avg, ra_fit_res, err_tsep_ls, fill_tsep_ls, Lt, id_label, err_tau_cut=1, fill_tau_cut=1):

    px = id_label["px"]
    py = id_label["py"]
    pz = id_label["pz"]
    b = id_label["b"]
    z = id_label["z"]
 
    def plot_part(part, ra_avg, ra_fcn, pdf_key):
        
        fig, ax = default_plot()
        
        for id, tsep in enumerate(err_tsep_ls):
            tau_range = np.arange(err_tau_cut, tsep + 1 - err_tau_cut)
            err_x_ls = tau_range - tsep / 2
            err_y_ls = gv.mean(ra_avg[id, err_tau_cut:tsep + 1 - err_tau_cut])
            err_yerr_ls = gv.sdev(ra_avg[id, err_tau_cut:tsep + 1 - err_tau_cut])
            
            ax.errorbar(err_x_ls, err_y_ls, err_yerr_ls, label=r'$t_{\mathrm{sep}}$' + f' = {tsep}', color=color_ls[id], **errorb)

        for id, tsep in enumerate(fill_tsep_ls):
            fit_tau = np.linspace(fill_tau_cut - 0.5, tsep - fill_tau_cut + 0.5, 100)
            fit_t = np.ones_like(fit_tau) * tsep
            fit_ratio = ra_fcn(fit_t, fit_tau, ra_fit_res.p, Lt)

            fill_x_ls = fit_tau - tsep / 2
            fill_y_ls = gv.mean(fit_ratio)
            fill_yerr_ls = gv.sdev(fit_ratio)

            ax.fill_between(fill_x_ls, [v.mean + v.sdev for v in fit_ratio], [v.mean - v.sdev for v in fit_ratio], color=color_ls[id], alpha=0.4)

        band_x = np.arange(-6, 7)
        band_y_ls = np.ones_like(band_x) * gv.mean(ra_fit_res.p[pdf_key])
        band_yerr_ls = np.ones_like(band_x) * gv.sdev(ra_fit_res.p[pdf_key])
        ax.fill_between(band_x, band_y_ls+band_yerr_ls, band_y_ls-band_yerr_ls, color=grey, alpha=0.5, label='Fit')
        
        ax.set_xlabel(r'$\tau - t_{\mathrm{sep}}/2$', **fs_p)
        ax.set_ylabel(r'$\mathcal{R}$', **fs_p)
        ax.set_ylim(auto_ylim([err_y_ls, fill_y_ls, band_y_ls], [err_yerr_ls, fill_yerr_ls, band_yerr_ls]))
        ax.legend(ncol=2, **fs_p)
        ax.set_title(f'PX = {px}, PY={py}, PZ={pz}, z={z}, b={b}, {part}', **fs_p)
        plt.tight_layout()
        
        return ax

    # Plot real part
    ax_real = plot_part('real', ra_re_avg, ra_re_fcn, 'pdf_re')

    # Plot imaginary part
    ax_imag = plot_part('imag', ra_im_avg, ra_im_fcn, 'pdf_im')

    return ax_real, ax_imag 