import gvar as gv
import numpy as np

from lametlat.ground_state import (
    ff_joint_two_state_fit,
    ff_ratio_fcn,
    ff_ratio_two_state_fit,
    ff_sum_fcn,
    ff_sum_two_state_fit,
)


def _mock_ratio_table(
    p_true: dict,
    tsep_ls: list[int],
    *,
    noise: float,
) -> dict[int, np.ndarray]:
    ratio_by_tsep: dict[int, np.ndarray] = {}
    for tsep in tsep_ls:
        row = np.empty(tsep + 1, dtype=object)
        for tau in range(tsep + 1):
            m = ff_ratio_fcn((float(tsep), float(tau)), p_true)
            row[tau] = gv.gvar(float(np.asarray(m).reshape(())),
                               noise)
        ratio_by_tsep[tsep] = row
    return ratio_by_tsep


def _tight_prior(p_true: dict) -> gv.BufferDict:
    pr = gv.BufferDict()
    for k, v in p_true.items():
        pr[k] = gv.gvar(v, abs(v) * 0.25 + 0.08)
    return pr


def test_ff_ratio_two_state_fit_mock():
    p_true = {
        "dE1": 0.3,
        "ff": 0.7,
        "ff_excited_coeff": 0.2,
        "ff_den_exp_coeff": 0.1,
    }
    tsep_ls = [8, 10]
    tau_cut = 2
    ratio_by_tsep = _mock_ratio_table(p_true, tsep_ls, noise=5e-4)
    fit = ff_ratio_two_state_fit(
        tsep_ls,
        tau_cut,
        ratio_by_tsep,
        prior=_tight_prior(p_true),
        label="mock",
    )
    assert fit.dof > 0
    assert 0 <= fit.Q <= 1
    assert abs(fit.p["ff"].mean - p_true["ff"]) < 0.08


def test_ff_sum_two_state_fit_and_joint_agree():
    p_true = {
        "dE1": 0.25,
        "ff": 0.6,
        "ff_excited_coeff": 0.15,
        "ff_den_exp_coeff": 0.05,
    }
    tsep_ls = [8, 10]
    tau_cut = 2
    ratio_by_tsep = _mock_ratio_table(p_true, tsep_ls, noise=5e-4)

    sum_by_tsep = {
        t: gv.gvar(float(np.asarray(ff_sum_fcn(t, tau_cut, p_true)).reshape(())), 5e-4)
        for t in tsep_ls
    }
    prior = _tight_prior(p_true)

    fit_sum = ff_sum_two_state_fit(tsep_ls, tau_cut, sum_by_tsep, prior=prior)
    joint = ff_joint_two_state_fit(tsep_ls, tau_cut, ratio_by_tsep, prior=prior)

    assert fit_sum.dof > 0
    assert joint.dof > 0
    np.testing.assert_allclose(
        joint.p["ff"].mean,
        fit_sum.p["ff"].mean,
        rtol=0.2,
    )
