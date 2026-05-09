"""3pt ratio helpers from already-loaded 2pt/3pt correlator arrays."""

from __future__ import annotations

import numpy as np


def get_pt3_ratio_data(
    pt2_real: np.ndarray,
    pt2_imag: np.ndarray,
    pt3_real: dict[int, np.ndarray],
    pt3_imag: dict[int, np.ndarray],
    *,
    sample_axis: int = 0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute sample-wise 3pt/2pt ratio for each tsep key in pt3 dict."""
    pt2_real = np.moveaxis(np.asarray(pt2_real), sample_axis, 0)
    pt2_imag = np.moveaxis(np.asarray(pt2_imag), sample_axis, 0)
    if pt2_real.shape != pt2_imag.shape or pt2_real.ndim != 2:
        raise ValueError("pt2_real and pt2_imag must be 2D arrays with same shape")
    if not isinstance(pt3_real, dict) or not pt3_real:
        raise ValueError("pt3_real must be a non-empty dict keyed by tsep")
    if not isinstance(pt3_imag, dict):
        raise ValueError("pt3_imag must be a dict keyed by tsep")
    if set(pt3_real) != set(pt3_imag):
        raise ValueError("pt3_real and pt3_imag must have identical tsep keys")

    ratio_real: dict[int, np.ndarray] = {}
    ratio_imag: dict[int, np.ndarray] = {}
    pt2_complex = pt2_real + 1j * pt2_imag

    for tsep in pt3_real:
        if not isinstance(tsep, int):
            raise ValueError("pt3 keys must be integer tsep values")
        if not (0 <= tsep < pt2_complex.shape[1]):
            raise ValueError(
                f"tsep index out of range for pt2 arrays shape {pt2_complex.shape}"
            )

        pt3_complex = np.moveaxis(
            np.asarray(pt3_real[tsep]), sample_axis, 0
        ) + 1j * np.moveaxis(np.asarray(pt3_imag[tsep]), sample_axis, 0)
        if pt3_complex.ndim != 2:
            raise ValueError(f"pt3[{tsep}] must be a 2D array")
        if pt3_complex.shape[0] != pt2_complex.shape[0]:
            raise ValueError(
                f"pt3[{tsep}] sample size mismatch with pt2: "
                f"{pt3_complex.shape[0]} != {pt2_complex.shape[0]}"
            )

        ratio = pt3_complex / pt2_complex[:, tsep][:, None]
        ratio_real[tsep] = np.real(ratio)
        ratio_imag[tsep] = np.imag(ratio)

    return ratio_real, ratio_imag
