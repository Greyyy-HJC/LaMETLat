import numpy as np
import pytest

from lametlat.ground_state import (
    ff_ratio_fcn,
    ff_sum_fcn,
    fh_im_fcn,
    fh_re_fcn,
    general_prior,
    pt2_re_fcn,
    pt3_ratio_im_fcn,
    pt3_ratio_re_fcn,
    qda_im_fcn,
    qda_re_fcn,
    sum_im_fcn,
    sum_re_fcn,
)


def test_general_prior_default_two_state_keys():
    priors = general_prior()

    for key in [
        "E0",
        "log(dE1)",
        "z0",
        "z1",
        "O00_re",
        "O00_im",
        "O01_re",
        "O01_im",
        "O11_re",
        "O11_im",
        "sum_re_excited_coeff",
        "sum_re_offset",
        "sum_re_exp_offset",
        "sum_im_excited_coeff",
        "sum_im_offset",
        "sum_im_exp_offset",
        "sum_den_exp_coeff",
        "log(ff)",
        "ff_excited_coeff",
        "ff_den_exp_coeff",
    ]:
        assert key in priors

    assert "O10_re" not in priors
    assert "O10_im" not in priors
    assert "re_b1" not in priors
    assert "im_b1" not in priors


def test_general_prior_one_state_keys():
    priors = general_prior(nstate=1)

    assert "E0" in priors
    assert "z0" in priors
    assert "O00_re" in priors
    assert "O00_im" in priors
    assert "sum_re_const" in priors
    assert "sum_im_const" in priors
    assert "log(dE1)" not in priors
    assert "z1" not in priors
    assert "O01_re" not in priors
    assert "sum_re_excited_coeff" not in priors
    assert "log(ff)" not in priors


def test_general_prior_three_state_keys():
    priors = general_prior(nstate=3)

    for key in [
        "log(dE1)",
        "log(dE2)",
        "z0",
        "z1",
        "z2",
        "O00_re",
        "O01_re",
        "O02_re",
        "O11_re",
        "O12_re",
        "O22_re",
        "O00_im",
        "O01_im",
        "O02_im",
        "O11_im",
        "O12_im",
        "O22_im",
    ]:
        assert key in priors

    assert "O10_re" not in priors
    assert "O21_im" not in priors
    assert "sum_re_const" not in priors
    assert "sum_re_excited_coeff" not in priors


def test_pt2_re_fcn_two_state_matches_lametlat_formula():
    t = np.array([2.0, 4.0, 6.0])
    Lt = 64
    p = {"E0": 0.5, "dE1": 0.3, "z0": 1.2, "z1": 0.7}

    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    expected = p["z0"] ** 2 / (2 * e0) * (
        np.exp(-e0 * t) + np.exp(-e0 * (Lt - t))
    ) + p["z1"] ** 2 / (2 * e1) * (
        np.exp(-e1 * t) + np.exp(-e1 * (Lt - t))
    )

    np.testing.assert_allclose(pt2_re_fcn(t, p, Lt, nstate=2), expected)


def test_pt2_re_fcn_one_state_and_scalar_input():
    t = 3.0
    Lt = 32
    p = {"E0": 0.4, "z0": 1.5}

    expected = p["z0"] ** 2 / (2 * p["E0"]) * (
        np.exp(-p["E0"] * t) + np.exp(-p["E0"] * (Lt - t))
    )

    assert np.allclose(pt2_re_fcn(t, p, Lt, nstate=1), expected)


def test_pt2_re_fcn_three_state_uses_cumulative_energy_gaps():
    t = np.array([1.0, 2.0])
    Lt = 24
    p = {
        "E0": 0.2,
        "dE1": 0.3,
        "dE2": 0.4,
        "z0": 1.0,
        "z1": 1.5,
        "z2": 0.8,
    }

    expected = 0.0
    for energy, z in [(0.2, 1.0), (0.5, 1.5), (0.9, 0.8)]:
        expected += z**2 / (2 * energy) * (
            np.exp(-energy * t) + np.exp(-energy * (Lt - t))
        )

    np.testing.assert_allclose(pt2_re_fcn(t, p, Lt, nstate=3), expected)


def test_pt3_ratio_re_fcn_two_state_matches_lametlat_formula():
    ra_t = np.array([4.0, 6.0])
    ra_tau = np.array([1.0, 2.0])
    Lt = 64
    p = {
        "E0": 0.5,
        "dE1": 0.3,
        "z0": 1.2,
        "z1": 0.7,
        "O00_re": 0.8,
        "O01_re": 0.4,
        "O11_re": 0.2,
    }

    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    numerator = (
        p["O00_re"] * p["z0"] ** 2 * np.exp(-e0 * ra_t) / (2 * e0) / (2 * e0)
        + p["O01_re"]
        * p["z0"]
        * p["z1"]
        * np.exp(-e0 * (ra_t - ra_tau))
        * np.exp(-e1 * ra_tau)
        / (2 * e0)
        / (2 * e1)
        + p["O01_re"]
        * p["z1"]
        * p["z0"]
        * np.exp(-e1 * (ra_t - ra_tau))
        * np.exp(-e0 * ra_tau)
        / (2 * e1)
        / (2 * e0)
        + p["O11_re"] * p["z1"] ** 2 * np.exp(-e1 * ra_t) / (2 * e1) / (2 * e1)
    )
    denominator = pt2_re_fcn(ra_t, p, Lt, nstate=2)
    expected = numerator / denominator

    np.testing.assert_allclose(
        pt3_ratio_re_fcn(ra_t, ra_tau, p, Lt, nstate=2), expected
    )


def test_pt3_ratio_im_fcn_two_state_matches_lametlat_formula():
    ra_t = np.array([4.0, 6.0])
    ra_tau = np.array([1.0, 2.0])
    Lt = 64
    p = {
        "E0": 0.5,
        "dE1": 0.3,
        "z0": 1.2,
        "z1": 0.7,
        "O00_im": 0.7,
        "O01_im": 0.3,
        "O11_im": 0.1,
    }

    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    numerator = (
        p["O00_im"] * p["z0"] ** 2 * np.exp(-e0 * ra_t) / (2 * e0) / (2 * e0)
        + p["O01_im"]
        * p["z0"]
        * p["z1"]
        * np.exp(-e0 * (ra_t - ra_tau))
        * np.exp(-e1 * ra_tau)
        / (2 * e0)
        / (2 * e1)
        + p["O01_im"]
        * p["z1"]
        * p["z0"]
        * np.exp(-e1 * (ra_t - ra_tau))
        * np.exp(-e0 * ra_tau)
        / (2 * e1)
        / (2 * e0)
        + p["O11_im"] * p["z1"] ** 2 * np.exp(-e1 * ra_t) / (2 * e1) / (2 * e1)
    )
    denominator = pt2_re_fcn(ra_t, p, Lt, nstate=2)
    expected = numerator / denominator

    np.testing.assert_allclose(
        pt3_ratio_im_fcn(ra_t, ra_tau, p, Lt, nstate=2), expected
    )


def test_pt3_ratio_re_fcn_one_state_and_scalar_input():
    ra_t = 5.0
    ra_tau = 2.0
    Lt = 32
    p = {"E0": 0.4, "z0": 1.5, "O00_re": 0.8}

    numerator = (
        p["O00_re"]
        * p["z0"] ** 2
        * np.exp(-p["E0"] * ra_t)
        / (2 * p["E0"])
        / (2 * p["E0"])
    )
    expected = numerator / pt2_re_fcn(ra_t, p, Lt, nstate=1)

    assert np.allclose(pt3_ratio_re_fcn(ra_t, ra_tau, p, Lt, nstate=1), expected)


def test_pt3_ratio_re_fcn_three_state_uses_upper_triangle_matrix_elements():
    ra_t = np.array([4.0, 5.0])
    ra_tau = np.array([1.0, 2.0])
    Lt = 48
    p = {
        "E0": 0.2,
        "dE1": 0.3,
        "dE2": 0.4,
        "z0": 1.0,
        "z1": 1.2,
        "z2": 0.8,
        "O00_re": 0.1,
        "O01_re": 0.2,
        "O02_re": 0.3,
        "O11_re": 0.4,
        "O12_re": 0.5,
        "O22_re": 0.6,
    }

    energies = [0.2, 0.5, 0.9]
    numerator = 0.0
    for source_state, source_energy in enumerate(energies):
        for sink_state, sink_energy in enumerate(energies):
            row = min(source_state, sink_state)
            col = max(source_state, sink_state)
            numerator += (
                p[f"O{row}{col}_re"]
                * p[f"z{source_state}"]
                * p[f"z{sink_state}"]
                * np.exp(-source_energy * (ra_t - ra_tau))
                * np.exp(-sink_energy * ra_tau)
                / (2 * source_energy)
                / (2 * sink_energy)
            )
    expected = numerator / pt2_re_fcn(ra_t, p, Lt, nstate=3)

    np.testing.assert_allclose(
        pt3_ratio_re_fcn(ra_t, ra_tau, p, Lt, nstate=3), expected
    )


def test_sum_re_fcn_one_state_matches_lametlat_formula():
    t = np.array([4.0, 6.0])
    tau_cut = 1
    p = {"E0": 0.5, "O00_re": 0.8, "sum_re_const": 0.2}

    expected = p["O00_re"] * (t - 2 * tau_cut + 1) / (2 * p["E0"]) + p[
        "sum_re_const"
    ]

    np.testing.assert_allclose(sum_re_fcn(t, tau_cut, p, nstate=1), expected)


def test_sum_im_fcn_two_state_matches_lametlat_formula_with_clear_names():
    t = np.array([4.0, 6.0])
    tau_cut = 1
    p = {
        "E0": 0.5,
        "dE1": 0.3,
        "O00_im": 0.8,
        "sum_im_excited_coeff": 0.2,
        "sum_im_offset": 0.4,
        "sum_im_exp_offset": 0.6,
        "sum_den_exp_coeff": 0.1,
    }

    exp_term = np.exp(-p["dE1"] * t)
    numerator = (
        p["O00_im"]
        * (t - 2 * tau_cut + 1)
        * (1 + p["sum_im_excited_coeff"] * exp_term)
        + p["sum_im_offset"]
        + p["sum_im_exp_offset"] * exp_term
    )
    denominator = 2 * p["E0"] * (1 + p["sum_den_exp_coeff"] * exp_term)
    expected = numerator / denominator

    np.testing.assert_allclose(sum_im_fcn(t, tau_cut, p, nstate=2), expected)


def test_sum_re_fcn_two_state_accepts_explicit_energy_gap():
    t = np.array([4.0, 6.0])
    tau_cut = 1
    dE1 = 0.25
    p = {
        "E0": 0.5,
        "dE1": 0.3,
        "O00_re": 0.8,
        "sum_re_excited_coeff": 0.2,
        "sum_re_offset": 0.4,
        "sum_re_exp_offset": 0.6,
        "sum_den_exp_coeff": 0.1,
    }

    exp_term = np.exp(-dE1 * t)
    expected = (
        p["O00_re"]
        * (t - 2 * tau_cut + 1)
        * (1 + p["sum_re_excited_coeff"] * exp_term)
        + p["sum_re_offset"]
        + p["sum_re_exp_offset"] * exp_term
    ) / (2 * p["E0"] * (1 + p["sum_den_exp_coeff"] * exp_term))

    np.testing.assert_allclose(sum_re_fcn(t, tau_cut, p, nstate=2, dE1=dE1), expected)


def test_fh_re_fcn_one_state_matches_lametlat_formula():
    t = np.array([4.0, 6.0])
    p = {"E0": 0.5, "O00_re": 0.8}

    expected = p["O00_re"] / (2 * p["E0"]) + t * 0

    np.testing.assert_allclose(fh_re_fcn(t, tau_cut=1, p=p, nstate=1), expected)


def test_fh_im_fcn_two_state_is_sum_difference():
    t = np.array([4.0, 6.0])
    tau_cut = 1
    dt = 2
    p = {
        "E0": 0.5,
        "dE1": 0.3,
        "O00_im": 0.8,
        "sum_im_excited_coeff": 0.2,
        "sum_im_offset": 0.4,
        "sum_im_exp_offset": 0.6,
        "sum_den_exp_coeff": 0.1,
    }

    expected = (
        sum_im_fcn(t + dt, tau_cut, p, nstate=2)
        - sum_im_fcn(t, tau_cut, p, nstate=2)
    ) / dt

    np.testing.assert_allclose(fh_im_fcn(t, tau_cut, p, nstate=2, dt=dt), expected)


def test_qda_re_fcn_two_state_matches_lametlat_formula():
    qda_t = np.array([4.0, 6.0])
    Lt = 64
    p = {
        "E0": 0.5,
        "dE1": 0.3,
        "z0": 1.2,
        "z1": 0.7,
        "O00_re": 0.8,
        "O01_re": 0.4,
    }

    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    expected = p["z0"] / (2 * e0) * p["O00_re"] * (
        np.exp(-e0 * qda_t) + np.exp(-e0 * (Lt - qda_t))
    ) + p["z1"] / (2 * e1) * p["O01_re"] * (
        np.exp(-e1 * qda_t) + np.exp(-e1 * (Lt - qda_t))
    )

    np.testing.assert_allclose(qda_re_fcn(qda_t, p, Lt, nstate=2), expected)


def test_qda_im_fcn_three_state_uses_o0i_matrix_elements():
    qda_t = np.array([3.0, 5.0])
    Lt = 48
    p = {
        "E0": 0.2,
        "dE1": 0.3,
        "dE2": 0.4,
        "z0": 1.0,
        "z1": 1.2,
        "z2": 0.8,
        "O00_im": 0.1,
        "O01_im": 0.2,
        "O02_im": 0.3,
    }

    expected = 0.0
    for state, energy in enumerate([0.2, 0.5, 0.9]):
        expected += (
            p[f"z{state}"]
            / (2 * energy)
            * p[f"O0{state}_im"]
            * (np.exp(-energy * qda_t) + np.exp(-energy * (Lt - qda_t)))
        )

    np.testing.assert_allclose(qda_im_fcn(qda_t, p, Lt, nstate=3), expected)


def test_ff_ratio_fcn_matches_lametlat_formula_with_clear_names():
    ra_t = np.array([4.0, 6.0])
    ra_tau = np.array([1.0, 2.0])
    p = {
        "dE1": 0.3,
        "ff": 0.7,
        "ff_excited_coeff": 0.2,
        "ff_den_exp_coeff": 0.1,
    }

    expected = (
        -p["ff"]
        * (
            1
            + p["ff_excited_coeff"]
            * (
                np.exp(-p["dE1"] * ra_tau)
                + np.exp(-p["dE1"] * (ra_t - ra_tau))
            )
        )
        / (1 + p["ff_den_exp_coeff"] * np.exp(-p["dE1"] * ra_t / 2))
    )

    np.testing.assert_allclose(ff_ratio_fcn((ra_t, ra_tau), p), expected)


def test_ff_sum_fcn_matches_explicit_tau_average():
    tau_cut = 2
    t = np.array([8, 10])
    p = {
        "dE1": 0.25,
        "ff": 0.6,
        "ff_excited_coeff": 0.15,
        "ff_den_exp_coeff": 0.05,
    }
    expected = []
    for t_val in t:
        acc = 0.0
        for tau in range(tau_cut, int(t_val) + 1 - tau_cut):
            acc += float(
                ff_ratio_fcn((float(t_val), float(tau)), p),
            )
        expected.append(acc / (t_val - 2 * tau_cut + 1))
    np.testing.assert_allclose(ff_sum_fcn(t, tau_cut, p), expected)
    np.testing.assert_allclose(ff_sum_fcn(8, tau_cut, p), expected[0])


@pytest.mark.parametrize("fcn", [sum_re_fcn, sum_im_fcn, fh_re_fcn, fh_im_fcn])
def test_sum_and_fh_functions_currently_reject_more_than_two_states(fcn):
    p = {
        "E0": 0.5,
        "dE1": 0.3,
        "dE2": 0.4,
        "O00_re": 0.8,
        "O00_im": 0.7,
    }

    with pytest.raises(ValueError, match="nstate <= 2"):
        fcn(4.0, 1, p, nstate=3)


@pytest.mark.parametrize("nstate", [0, -1, 1.5, True])
def test_nstate_must_be_positive_integer(nstate):
    with pytest.raises(ValueError, match="nstate must be a positive integer"):
        general_prior(nstate=nstate)

    with pytest.raises(ValueError, match="nstate must be a positive integer"):
        pt2_re_fcn(1.0, {"E0": 1.0, "z0": 1.0}, Lt=8, nstate=nstate)

    with pytest.raises(ValueError, match="nstate must be a positive integer"):
        pt3_ratio_re_fcn(
            1.0,
            0.5,
            {"E0": 1.0, "z0": 1.0, "O00_re": 1.0},
            Lt=8,
            nstate=nstate,
        )

    with pytest.raises(ValueError, match="nstate must be a positive integer"):
        pt3_ratio_im_fcn(
            1.0,
            0.5,
            {"E0": 1.0, "z0": 1.0, "O00_im": 1.0},
            Lt=8,
            nstate=nstate,
        )

    with pytest.raises(ValueError, match="nstate must be a positive integer"):
        sum_re_fcn(1.0, 0, {"E0": 1.0, "O00_re": 1.0}, nstate=nstate)

    with pytest.raises(ValueError, match="nstate must be a positive integer"):
        fh_im_fcn(1.0, 0, {"E0": 1.0, "O00_im": 1.0}, nstate=nstate)

    with pytest.raises(ValueError, match="nstate must be a positive integer"):
        qda_re_fcn(1.0, {"E0": 1.0, "z0": 1.0, "O00_re": 1.0}, Lt=8, nstate=nstate)
