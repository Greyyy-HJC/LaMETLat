import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import lsqfit as lsf

from lametlat.utils.plot_settings import *
from lametlat.utils.constants import *


def disp_relation_plot(a, Ls, mom_ls, meff_ls, m0=None):
    """
    Plot the dispersion relation.

    Args:
        a (float): Lattice spacing in fm.
        Ls (int): Spatial lattice size.
        mom_ls (list): List of momenta.
        meff_ls (list): List of effective masses corresponding to momenta.
        m0 (float, optional): Rest mass in GeV. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The plot axes object.
    """
    # Convert lattice momenta to physical momenta in GeV
    p_array = lat_unit_convert(np.array(mom_ls), a, Ls, dimension="P")
    
    # Convert effective masses to physical energies in GeV
    E_array = lat_unit_convert(np.array(meff_ls), a, Ls, dimension="M")

    # Define the fit function
    def fcn(x, p):
        return np.sqrt(p["m"]**2 + p["c1"]*x**2 + p["c2"]*x**4*a**2/(GEV_FM**2))

    # Perform the fit
    priors = gv.BufferDict()
    if m0 is not None:
        priors["m"] = gv.gvar(m0, 0.5)
    else:
        priors["m"] = gv.gvar(1, 1)
    priors["c1"] = gv.gvar(1, 0.5)
    priors["c2"] = gv.gvar(0, 0.5)   
    fit_res = lsf.nonlinear_fit(data=(p_array, E_array), prior=priors, fcn=fcn)
    
    print(fit_res.format(100))

    # Generate points for the fit curve
    fit_x = np.linspace(p_array[0], p_array[-1], 100)
    fit_y = fcn(fit_x, fit_res.p)

    # Create the plot
    fig, ax = default_plot()
    
    # Plot data points
    ax.errorbar(
        p_array,
        [v.mean for v in E_array],
        [v.sdev for v in E_array],
        color=blue,
        marker="x",
        **errorb
    )
    
    # Plot fit curve with error band
    ax.fill_between(
        fit_x,
        [v.mean + v.sdev for v in fit_y],
        [v.mean - v.sdev for v in fit_y],
        color=blue,
        alpha=0.5,
    )

    # Plot m0 curve if provided
    if m0 is not None:
        ax.plot(
            p_array,
            np.sqrt(p_array**2 + m0**2),
            color=red,
            label=r"$m_0$ = {} GeV".format(m0)
        )

    # Set plot properties
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    ax.set_xlabel(r"$P$ / GeV", **fs_p)
    ax.set_ylabel(r"$E$ / GeV", **fs_p)
    ax.set_ylim(auto_ylim([gv.mean(fit_y)], [gv.sdev(fit_y)]))
    ax.text(0.7*p_array[-1], 0.3*(ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0], r"$c_1$: {:.3f}".format(fit_res.p['c1']), **fs_p)
    ax.text(0.7*p_array[-1], 0.2*(ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0], r"$c_2$: {:.3f}".format(fit_res.p['c2']), **fs_p)
    
    if m0 is not None:
        ax.legend(**fs_p)
    
    plt.tight_layout()
    plt.show()
    return fig
