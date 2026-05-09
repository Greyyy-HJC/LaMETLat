"""qDA ratio helpers from already-loaded 2pt/qDA correlator arrays."""

from __future__ import annotations

import numpy as np


def get_qda_ratio_data(
    pt2_real: np.ndarray,
    pt2_imag: np.ndarray,
    qda_real: np.ndarray,
    qda_imag: np.ndarray,
    *,
    sample_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sample-wise qDA/2pt ratio with aligned time separations."""
    pt2_real = np.moveaxis(np.asarray(pt2_real), sample_axis, 0)
    pt2_imag = np.moveaxis(np.asarray(pt2_imag), sample_axis, 0)
    qda_real = np.moveaxis(np.asarray(qda_real), sample_axis, 0)
    qda_imag = np.moveaxis(np.asarray(qda_imag), sample_axis, 0)

    if pt2_real.shape != pt2_imag.shape or pt2_real.ndim != 2:
        raise ValueError("pt2_real and pt2_imag must be 2D arrays with same shape")
    if qda_real.shape != qda_imag.shape or qda_real.ndim != 2:
        raise ValueError("qda_real and qda_imag must be 2D arrays with same shape")
    if qda_real.shape != pt2_real.shape:
        raise ValueError(
            "qda_real/qda_imag shape must match pt2_real/pt2_imag shape: "
            f"{qda_real.shape} != {pt2_real.shape}"
        )

    pt2_complex = pt2_real + 1j * pt2_imag
    qda_complex = qda_real + 1j * qda_imag
    ratio = qda_complex / pt2_complex

    return np.real(ratio), np.imag(ratio)
