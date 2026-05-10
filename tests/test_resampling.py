import gvar as gv
import numpy as np
import pytest
from scipy.interpolate import interp1d

from lametlat.correlators import (
    bad_point_filter,
    bin_data,
    bootstrap,
    gvar_ls_interpolate,
    jackknife,
    sample_ls_interpolate,
)


def test_bad_point_filter_replaces_large_entries():
    data = np.array([[0.5, 2.0], [-3.0, 0.2]])

    out = bad_point_filter(data, threshold=1)

    np.testing.assert_allclose(data, np.array([[0.5, 2.0], [-3.0, 0.2]]))
    assert out[0, 0] == 0.5
    assert out[1, 1] == 0.2
    assert out[0, 1] in {-1, 1}
    assert out[1, 0] in {-1, 1}


def test_bin_data_axis0():
    data = np.array([1.0, 3.0, 5.0, 7.0])

    out = bin_data(data, 2)

    np.testing.assert_allclose(out, np.array([2.0, 6.0]))


def test_bin_data_axis1():
    data = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]])

    out = bin_data(data, 2, axis=1)

    np.testing.assert_allclose(out, np.array([[2.0, 6.0], [3.0, 7.0]]))


def test_bootstrap_returns_samples_and_indices():
    data = np.array([1.0, 2.0, 3.0])

    samples, indices = bootstrap(data, 4, seed=123, return_indices=True)

    assert samples.shape == (4,)
    assert indices.shape == (4, 3)
    np.testing.assert_allclose(samples, data[indices].mean(axis=1))


def test_bootstrap_axis1():
    data = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])

    samples, indices = bootstrap(data, 5, axis=1, seed=123, return_indices=True)

    assert samples.shape == (2, 5)
    np.testing.assert_allclose(samples[:, 0], data[:, indices[0]].mean(axis=1))


def test_jackknife_axis0():
    data = np.array([1.0, 2.0, 4.0])

    out = jackknife(data)

    np.testing.assert_allclose(out, np.array([3.0, 2.5, 1.5]))


def test_jackknife_axis1():
    data = np.array([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])

    out = jackknife(data, axis=1)

    np.testing.assert_allclose(out, np.array([[3.0, 2.5, 1.5], [6.0, 5.0, 3.0]]))


def test_jackknife_needs_two_samples():
    with pytest.raises(ValueError):
        jackknife(np.array([1.0]))


def test_gvar_ls_interpolate_uses_mean_and_sdev_envelopes():
    x = np.arange(5, dtype=float)
    x_new = np.linspace(0, 4, 9)
    mean = np.array([0.0, 1.0, 0.5, 1.5, 1.0])
    sdev = np.array([0.10, 0.15, 0.12, 0.20, 0.18])
    values = gv.gvar(mean, sdev)

    interpolated = gvar_ls_interpolate(x, values, x_new, n_samp=7, kind="linear")

    expected_mean = interp1d(x, mean, kind="linear")(x_new)
    expected_sdev = interp1d(x, sdev, kind="linear")(x_new)
    np.testing.assert_allclose(gv.mean(interpolated), expected_mean)
    np.testing.assert_allclose(gv.sdev(interpolated), expected_sdev)


def test_sample_ls_interpolate_applies_interpolation_per_sample():
    x = np.arange(5, dtype=float)
    x_new = np.linspace(0, 4, 9)
    sample_ls = np.array(
        [
            [0.0, 1.0, 0.5, 1.5, 1.0],
            [1.0, 1.5, 2.0, 2.5, 3.0],
        ]
    )

    interpolated = sample_ls_interpolate(x, sample_ls, x_new, kind="linear")
    expected = interp1d(x, sample_ls, kind="linear", axis=1)(x_new)

    np.testing.assert_allclose(interpolated, expected)
