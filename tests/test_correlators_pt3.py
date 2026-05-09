from pathlib import Path

import numpy as np
import pytest

from lametlat.correlators import read_pt3_h5


def _pt3_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "cache"
        / "qTMD_src5_D_1HYP_GSRC_W90_k0_PX0PY0PZ0dt4_PpUnpol_T.h5"
    )


def test_read_pt3_h5_single_dataset():
    path = _pt3_path()
    if not path.exists():
        pytest.skip("pt3 h5 file not present")

    data = read_pt3_h5(path, bT="bT0", bz="bz0")

    assert isinstance(data, np.ndarray)
    assert data.shape == (700, 6)
    assert np.iscomplexobj(data)


def test_read_pt3_h5_bt_mapping():
    path = _pt3_path()
    if not path.exists():
        pytest.skip("pt3 h5 file not present")

    bt_data = read_pt3_h5(path, bT="bT0")

    assert isinstance(bt_data, dict)
    assert "bz0" in bt_data
    assert bt_data["bz0"].shape == (700, 6)


def test_read_pt3_h5_requires_bt_for_bz():
    path = _pt3_path()
    if not path.exists():
        pytest.skip("pt3 h5 file not present")

    with pytest.raises(ValueError):
        read_pt3_h5(path, bz="bz0")


def test_read_pt3_h5_with_resampling_bs():
    path = _pt3_path()
    if not path.exists():
        pytest.skip("pt3 h5 file not present")

    data = read_pt3_h5(path, bT="bT0", bz="bz0", resampling="bs", n_samples=8)
    assert data.shape == (8, 6)


def test_read_pt3_h5_with_threshold_filter():
    path = _pt3_path()
    if not path.exists():
        pytest.skip("pt3 h5 file not present")

    data = read_pt3_h5(path, bT="bT0", bz="bz0", threshold=0)

    assert np.all(np.abs(data) <= 1)
