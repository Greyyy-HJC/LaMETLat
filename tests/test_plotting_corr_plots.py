import gvar as gv
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lametlat.ground_state import pt2_fit, pt2_re_fcn
from lametlat.plotting import pt2_plot, qda_ratio_plot


def test_qda_ratio_plot_draws_real_and_imag_data():
    trange = np.arange(3)
    qda_ratio_real = gv.gvar([1.0, 1.2, 1.5], [0.1, 0.1, 0.2])
    qda_ratio_imag = gv.gvar([0.2, 0.1, -0.1], [0.05, 0.05, 0.06])

    (fig_real, ax_real), (fig_imag, ax_imag) = qda_ratio_plot(
        trange,
        qda_ratio_real,
        qda_ratio_imag,
        id_label={"pz": 0, "z": 1},
    )

    assert ax_real.get_xlabel()
    assert ax_imag.get_xlabel()
    assert "qDA" in ax_real.get_ylabel()
    assert "qDA" in ax_imag.get_ylabel()

    plt.close(fig_real)
    plt.close(fig_imag)


def test_qda_ratio_plot_validates_trange_length():
    qda_ratio_real = gv.gvar([1.0, 1.2, 1.5], [0.1, 0.1, 0.2])

    with pytest.raises(ValueError, match="trange length"):
        qda_ratio_plot(np.arange(2), qda_ratio_real)


def test_pt2_plot_draws_fit_overlay():
    Lt = 32
    t = np.arange(Lt)
    params = {"E0": 0.45, "dE1": 0.55, "z0": 1.1, "z1": 0.45}
    mean = pt2_re_fcn(t, params, Lt, nstate=2)
    pt2_avg = gv.gvar(mean, np.maximum(0.02 * mean, 1e-6))

    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(0.45, 0.2)
    prior["log(dE1)"] = gv.gvar(np.log(0.55), 0.4)
    prior["z0"] = gv.gvar(1.1, 0.4)
    prior["z1"] = gv.gvar(0.45, 0.4)
    fit = pt2_fit(pt2_avg, 3, 12, Lt, prior=prior, label="Fit")

    (fig_c2, ax_c2), (fig_meff, ax_meff) = pt2_plot(
        [pt2_avg],
        trange=(2, 14),
        fit_results=fit,
        fit_tmin=3,
        fit_tmax=12,
        fit_label="Fit",
    )

    assert ax_c2.get_yscale() == "log"
    assert ax_c2.get_legend() is not None
    assert ax_meff.get_legend() is not None
    assert len(ax_c2.lines) >= 2
    assert len(ax_meff.lines) >= 2

    plt.close(fig_c2)
    plt.close(fig_meff)
