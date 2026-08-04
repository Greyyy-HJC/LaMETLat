"""CG PDF matching utilities in the MSbar scheme."""

from __future__ import annotations

import numpy as np


def arctan_term(xi: float | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Analytic arctan/arctanh term in the CG quasi-PDF kernels."""
    xi_arr = np.asarray(xi, dtype=np.float64)
    result = np.zeros_like(xi_arr, dtype=np.float64)

    below_half = xi_arr < 0.5 - eps
    above_half = xi_arr > 0.5 + eps
    near_half = ~(below_half | above_half)

    if np.any(below_half):
        x_val = xi_arr[below_half]
        sqrt_term = np.sqrt(1.0 - 2.0 * x_val)
        atan_piece = np.arctan(sqrt_term / (np.abs(x_val) + eps)) / (sqrt_term + eps)
        prefactor = (3.0 * x_val - 1.0) / (x_val - 1.0 + eps)
        result[below_half] = prefactor * atan_piece

    if np.any(above_half):
        x_val = xi_arr[above_half]
        sqrt_term = np.sqrt(2.0 * x_val - 1.0)
        atanh_piece = np.arctanh(sqrt_term / (np.abs(x_val) + eps)) / (sqrt_term + eps)
        prefactor = (3.0 * x_val - 1.0) / (x_val - 1.0 + eps)
        result[above_half] = prefactor * atanh_piece

    if np.any(near_half):
        x_val = xi_arr[near_half]
        result[near_half] = (3.0 * x_val - 1.0) / (x_val - 1.0)

    return result
