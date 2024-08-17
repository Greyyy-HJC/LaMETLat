"""
Functions used to make the dispersion relation plot.
You can find an example usage at the end of this file.
"""

import lsqfit as lsf
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

from liblattice.general.constants import *
from liblattice.general.plot_settings import *


plt_axes = [0.15, 0.15, 0.8, 0.8]  # left, bottom, width, height
fs_p = {"fontsize": 13}  # font size of text, label, ticks
ls_p = {"labelsize": 13}


def meff_fit(t_ls, meff_ls):
    """
    constant fit of meff

    Args:
        t_ls :
        meff_ls :

    Returns:
        the gvar fit result of meff
    """

    def fcn(x, p):
        return p["meff"] + x * 0

    priors = gv.BufferDict()
    priors["meff"] = gv.gvar(1, 10)

    fit_res = lsf.nonlinear_fit(
        data=(t_ls, meff_ls),
        prior=priors,
        fcn=fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    return fit_res.p["meff"]


def disp_relation_plot(a, Ls, mom_ls, meff_ls, title, save=True, m0=None, ylim=None):
    """make a dispersion relation plot from gvar meff list and momentum list

    Args:
        a (float): lattice spacing
        Ls (int): lattice length in the space direction
        mom_ls (list): list of momentum numbers, like [0, 2, 4, 6, 8, 10, 12]
        meff_ls (list): gvar list of effective mass
        title (str): title of the plot, also used as the file name to save the plot
        m0 (float, optional): the static mass of the particle. Defaults to None, if not None, will plot the dispersion relation line with the static mass.

    Returns:
        dict: fit result of the dispersion relation fit
    """
    # * convert the momentum list to physical unit GeV
    mom_ls = np.array(mom_ls)  # for the following unit convert
    p_ls = lat_unit_convert(mom_ls, a, Ls, dimension="P")
    # * convert the effective mass list to physical unit GeV
    meff_ls = np.array(meff_ls)  # for the following unit convert
    E_ls = lat_unit_convert(meff_ls, a, Ls, dimension="M")

    # * fit the dispersion relation
    def fcn(x, p):
        return np.sqrt(
            p["m"] ** 2 + p["c1"] * x**2 + p["c2"] * x**4 * a**2 / (GEV_FM**2)
        )

    priors = gv.BufferDict()
    priors["m"] = gv.gvar(0.1, 10)
    priors["c1"] = gv.gvar(1, 10)
    priors["c2"] = gv.gvar(0, 10)

    fit_res = lsf.nonlinear_fit(
        data=(p_ls, E_ls),
        prior=priors,
        fcn=fcn,
        maxit=10000,
        svdcut=1e-100,
        fitter="scipy_least_squares",
    )

    print(fit_res.format(100))

    fit_x = np.arange(p_ls[0], p_ls[-1], 0.01)
    fit_y = fcn(fit_x, fit_res.p)

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(plt_axes)
    ax.errorbar(
        p_ls,
        [v.mean for v in E_ls],
        [v.sdev for v in E_ls],
        color=blue,
        label="disp",
        **errorb
    )
    ax.fill_between(
        fit_x,
        [v.mean + v.sdev for v in fit_y],
        [v.mean - v.sdev for v in fit_y],
        color=blue,
        alpha=0.5,
    )

    if m0 != None:
        ax.plot(
            p_ls,
            np.sqrt(p_ls**2 + m0**2),
            color=red,
            label="ref"
        )

    xmajor = MultipleLocator(0.5)
    xminor = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(xmajor)
    ax.xaxis.set_minor_locator(xminor)
    
    ymajor = MultipleLocator(0.5)
    yminor = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(ymajor)
    ax.yaxis.set_minor_locator(yminor)

    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_xlabel(r"$P$ / GeV", **fs_p)
    ax.set_ylabel(r"$E$ / GeV", **fs_p)
    if ylim != None:
        ax.set_ylim(ylim)
    plt.legend()
    plt.title(title, **fs_p)
    if save == True:
        plt.savefig("../output/plots/" + title + ".pdf", transparent=True)
    plt.show()

    return fit_res


if __name__ == "__main__":
    """
    Example usage of this module
    """
    import h5py as h5

    from liblattice.preprocess.resampling import *
    from liblattice.preprocess.read_raw import pt2_to_meff

    file = h5.File("test_data/a12m130p_tmdpdf_m220_2pt.h5", "r")["hadron_121050"]

    meff_ls = []
    for mom in range(0, 14, 2):
        # * read the test data with different momentum
        data_real = file["mom_{}".format(mom)][:, 1:, 1]

        # * bootstrap the data and average
        data_real_bs = bootstrap(data_real, 500, axis=0)
        data_real_bs_avg = bs_ls_avg(data_real_bs)

        # * calculate the effective mass
        meff_avg = pt2_to_meff(data_real_bs_avg)

        # * fit the effective mass with constant fit
        meff_fit_res = meff_fit(np.arange(4, 8), meff_avg[4:8])
        meff_ls.append(meff_fit_res)

    print(meff_ls)

    fit_res = disp_relation_plot(
        a=0.12,
        Ls=48,
        mom_ls=np.arange(0, 14, 2),
        meff_ls=meff_ls,
        title="disp_test",
        save=False,
    )
