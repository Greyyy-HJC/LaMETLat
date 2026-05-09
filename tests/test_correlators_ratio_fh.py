from pathlib import Path

import numpy as np
import pytest

from lametlat.correlators import get_fh_data, get_pt3_ratio_data, get_sum_data, read_pt2_h5, read_pt3_h5


def _pt2_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "cache"
        / "comb_c2pt_src32"
        / "c2pt_src5_1HYP_GSRC_W90_k0_5_merged.h5"
    )


def _pt3_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "cache"
        / "qTMD_src5_D_1HYP_GSRC_W90_k0_PX0PY0PZ0dt4_PpUnpol_T.h5"
    )


def test_get_pt3_ratio_data_shape_and_dtype():
    pt2_path = _pt2_path()
    pt3_path = _pt3_path()
    if not pt2_path.exists() or not pt3_path.exists():
        pytest.skip("pt2/pt3 h5 files not present")

    pt2 = read_pt2_h5(pt2_path, source_sink="SS", gamma="5", momentum="PX0PY0PZ0")
    pt3 = read_pt3_h5(pt3_path, bT="bT0", bz="bz0")
    pt3_by_tsep = {4: pt3, 6: 0.9 * pt3}

    ratio_real, ratio_imag = get_pt3_ratio_data(
        pt2_real=np.real(pt2),
        pt2_imag=np.imag(pt2),
        pt3_real={tsep: np.real(data) for tsep, data in pt3_by_tsep.items()},
        pt3_imag={tsep: np.imag(data) for tsep, data in pt3_by_tsep.items()},
    )

    assert set(ratio_real.keys()) == {4, 6}
    assert ratio_real[4].shape == (700, 6)
    assert ratio_imag[6].shape == (700, 6)
    assert np.issubdtype(ratio_real[4].dtype, np.floating)
    assert np.issubdtype(ratio_imag[4].dtype, np.floating)
    assert np.all(np.isfinite(ratio_real[4]))
    assert np.all(np.isfinite(ratio_imag[4]))


def test_get_sum_and_fh_data_shape_and_dtype():
    pt2_path = _pt2_path()
    pt3_path = _pt3_path()
    if not pt2_path.exists() or not pt3_path.exists():
        pytest.skip("pt2/pt3 h5 files not present")

    pt2 = read_pt2_h5(pt2_path, source_sink="SS", gamma="5", momentum="PX0PY0PZ0")
    pt3 = read_pt3_h5(pt3_path, bT="bT0", bz="bz0")
    pt3_by_tsep = {4: pt3, 6: 0.9 * pt3}

    pt3_real = {tsep: np.real(data) for tsep, data in pt3_by_tsep.items()}
    pt3_imag = {tsep: np.imag(data) for tsep, data in pt3_by_tsep.items()}

    sum_real, sum_imag = get_sum_data(
        pt2_real=np.real(pt2),
        pt2_imag=np.imag(pt2),
        pt3_real=pt3_real,
        pt3_imag=pt3_imag,
        tau_cut=1,
    )
    fh_real, fh_imag = get_fh_data(
        pt2_real=np.real(pt2),
        pt2_imag=np.imag(pt2),
        pt3_real=pt3_real,
        pt3_imag=pt3_imag,
        tau_cut=1,
    )

    assert sum_real[4].shape == (700,)
    assert sum_imag[6].shape == (700,)
    assert fh_real.shape == (700, 1)
    assert fh_imag.shape == (700, 1)
    assert np.issubdtype(fh_real.dtype, np.floating)
    assert np.issubdtype(fh_imag.dtype, np.floating)
    assert np.all(np.isfinite(fh_real))
    assert np.all(np.isfinite(fh_imag))
