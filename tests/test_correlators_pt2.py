from pathlib import Path

import numpy as np
import pytest

from lametlat.correlators import read_pt2_h5


def _k0_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "cache"
        / "comb_c2pt"
        / "c2pt_merged_src5_1HYP_M140_GSRC_W45_k0.h5"
    )


def test_read_pt2_h5_single_dataset():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    data = read_pt2_h5(path, source_sink="SS", gamma="5", momentum="PX0PY0PZ0")

    assert isinstance(data, np.ndarray)
    assert data.shape == (700, 64)
    assert np.iscomplexobj(data)


def test_read_pt2_h5_gamma_mapping():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    gamma_data = read_pt2_h5(path, source_sink="SS", gamma="5")

    assert isinstance(gamma_data, dict)
    assert "PX0PY0PZ0" in gamma_data
    assert gamma_data["PX0PY0PZ0"].shape == (700, 64)


def test_read_pt2_h5_requires_gamma_for_momentum():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    with pytest.raises(ValueError):
        read_pt2_h5(path, source_sink="SS", momentum="PX0PY0PZ0")


def test_read_pt2_h5_with_resampling_bs():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    data = read_pt2_h5(
        path,
        source_sink="SS",
        gamma="5",
        momentum="PX0PY0PZ0",
        resampling="bs",
        n_samples=8,
    )
    assert data.shape == (8, 64)


def test_read_pt2_h5_with_normalization():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    data = read_pt2_h5(
        path,
        source_sink="SS",
        gamma="5",
        momentum="PX0PY0PZ0",
        normalization=True,
    )

    np.testing.assert_allclose(np.mean(data[:, 0]), 1.0)


def test_read_pt2_h5_gamma_mapping_with_normalization():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    gamma_data = read_pt2_h5(path, source_sink="SS", gamma="5", normalization=True)

    np.testing.assert_allclose(np.mean(gamma_data["PX0PY0PZ0"][:, 0]), 1.0)


def test_read_pt2_h5_with_threshold_filter():
    path = _k0_path()
    if not path.exists():
        pytest.skip("k0 h5 file not present")

    data = read_pt2_h5(
        path,
        source_sink="SS",
        gamma="5",
        momentum="PX0PY0PZ0",
        threshold=0,
    )

    assert np.all(np.abs(data) <= 1)
