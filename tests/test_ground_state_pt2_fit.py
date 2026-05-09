import gvar as gv
import numpy as np

from lametlat.ground_state import pt2_fit, pt2_re_fcn, pt2_two_state_fit


def _two_state_mock_data(Lt=32):
    t = np.arange(Lt)
    params = {"E0": 0.45, "dE1": 0.55, "z0": 1.1, "z1": 0.45}
    mean = pt2_re_fcn(t, params, Lt, nstate=2)
    sdev = np.maximum(0.02 * mean, 1e-6)
    return gv.gvar(mean, sdev)


def _two_state_prior():
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(0.45, 0.2)
    prior["log(dE1)"] = gv.gvar(np.log(0.55), 0.4)
    prior["z0"] = gv.gvar(1.1, 0.4)
    prior["z1"] = gv.gvar(0.45, 0.4)
    return prior


def test_pt2_fit_returns_lsqfit_result():
    Lt = 32
    pt2_avg = _two_state_mock_data(Lt=Lt)

    fit = pt2_fit(
        pt2_avg,
        3,
        12,
        Lt,
        prior=_two_state_prior(),
        label="mock",
    )

    assert fit.dof > 0
    assert 0 <= fit.Q <= 1


def test_pt2_two_state_fit_uses_log_gap_prior_directly():
    fit = pt2_two_state_fit(
        _two_state_mock_data(),
        3,
        12,
        32,
        prior=_two_state_prior(),
    )

    assert fit.p["dE1"].mean > 0
    assert fit.p["E0"].mean > 0
