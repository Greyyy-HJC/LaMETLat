import numpy as np
import pytest

from lametlat.correlators import get_qda_ratio_data


def test_get_qda_ratio_data_shape_and_values():
    pt2 = np.array(
        [
            [2.0 + 0.0j, 4.0 + 0.0j, 8.0 + 0.0j],
            [1.0 + 1.0j, 2.0 + 2.0j, 4.0 + 4.0j],
        ]
    )
    qda = 2.0 * pt2

    ratio_real, ratio_imag = get_qda_ratio_data(
        pt2_real=np.real(pt2),
        pt2_imag=np.imag(pt2),
        qda_real=np.real(qda),
        qda_imag=np.imag(qda),
    )

    assert ratio_real.shape == pt2.shape
    assert ratio_imag.shape == pt2.shape
    np.testing.assert_allclose(ratio_real, np.full(pt2.shape, 2.0))
    np.testing.assert_allclose(ratio_imag, np.zeros(pt2.shape))


def test_get_qda_ratio_data_sample_axis():
    pt2 = np.array(
        [
            [2.0 + 0.0j, 1.0 + 1.0j],
            [4.0 + 0.0j, 2.0 + 2.0j],
            [8.0 + 0.0j, 4.0 + 4.0j],
        ]
    )
    qda = 0.5 * pt2

    ratio_real, ratio_imag = get_qda_ratio_data(
        pt2_real=np.real(pt2),
        pt2_imag=np.imag(pt2),
        qda_real=np.real(qda),
        qda_imag=np.imag(qda),
        sample_axis=1,
    )

    assert ratio_real.shape == pt2.T.shape
    assert ratio_imag.shape == pt2.T.shape
    np.testing.assert_allclose(ratio_real, np.full(pt2.T.shape, 0.5))
    np.testing.assert_allclose(ratio_imag, np.zeros(pt2.T.shape))


def test_get_qda_ratio_data_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape must match"):
        get_qda_ratio_data(
            pt2_real=np.ones((2, 3)),
            pt2_imag=np.zeros((2, 3)),
            qda_real=np.ones((2, 2)),
            qda_imag=np.zeros((2, 2)),
        )
