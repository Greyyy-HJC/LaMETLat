"""CG PDF matching utilities in the MSbar scheme.

Explicit MS-bar ``gamma^t`` Coulomb-gauge PDF kernel from Eq. (2.14) of
arXiv:2602.11283. The returned matrix is the NLO correction only; it does
not include the LO identity.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from lametlat.utils.constants import CF, alphas_nloop


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
        result[near_half] = (3.0 * x_val - 1.0) / (x_val - 1.0) / (np.abs(x_val) + eps)

    return result


def C_msbar_plus_01(ksi: float, log_scale: float) -> float:
    """Return ``A(ksi)`` in the ``[0, 1]`` plus term of Eq. (2.16)."""
    return float((1.0 + ksi**2) / (1.0 - ksi) * log_scale + ksi - 1.0)


def C_msbar_plus_all(ksi: float, eps: float = 1e-12) -> float:
    """Return ``B(ksi)`` in the ``(-inf, inf)`` plus term of Eq. (2.16)."""
    one_minus_ksi = 1.0 - ksi
    sign_safe_denominator = one_minus_ksi + np.sign(one_minus_ksi) * eps
    signed_logs = (
        np.sign(ksi) * np.log(np.abs(ksi) + eps)
        + np.sign(one_minus_ksi) * np.log(np.abs(one_minus_ksi) + eps)
    )
    entry = (1.0 + ksi**2) / sign_safe_denominator * signed_logs
    entry += (
        np.sign(ksi)
        + float(arctan_term(ksi, eps))
        - 1.5 / (np.abs(one_minus_ksi) + eps)
    )
    return float(entry)


CoeffFn = Callable[[float, float], float]
Domain = tuple[float, float]


def _build_plus_matrix(
    lc_x_ls: np.ndarray,
    pz_gev: float,
    mu: float,
    quasi_x_ls: np.ndarray | None,
    eps: float,
    *,
    coeff: CoeffFn,
    subtraction_domain: Domain,
) -> np.ndarray:
    """Discretize one ``[coeff(ksi)]_+`` term into a convolution matrix.

    The regular coefficient is evaluated at
    ``log(4*quasi_x**2*pz_gev**2/mu**2)``. Its delta subtraction is evaluated
    at ``quasi_x=lc_x`` and integrated only over ``subtraction_domain``.
    The returned matrix already includes the uniform ``dquasi_x`` measure.
    """
    lc_x_ls = np.asarray(lc_x_ls, dtype=float)
    quasi_x_ls = np.asarray(
        lc_x_ls if quasi_x_ls is None else quasi_x_ls,
        dtype=float,
    )
    dquasi_x = float(quasi_x_ls[1] - quasi_x_ls[0])
    matrix = np.zeros((len(lc_x_ls), len(quasi_x_ls)))

    # Ordinary part g(lc_x/quasi_x)/|quasi_x|.
    domain_lo, domain_hi = subtraction_domain
    for lc_idx, lc_x in enumerate(lc_x_ls):
        for quasi_idx, quasi_x in enumerate(quasi_x_ls):
            ksi = lc_x / quasi_x
            if np.abs(1.0 - ksi) <= eps or ksi < domain_lo or ksi > domain_hi:
                continue
            log_scale = np.log(4.0 * quasi_x**2 * pz_gev**2 / mu**2)
            matrix[lc_idx, quasi_idx] = coeff(ksi, log_scale) / np.abs(quasi_x)

    # -delta(1-ksi) * integral_D dksi' g(ksi'), frozen at quasi_x=lc_x.
    for lc_idx, lc_x in enumerate(lc_x_ls):
        log_scale = np.log(4.0 * lc_x**2 * pz_gev**2 / mu**2)
        subtraction = 0.0

        for quasi_x in quasi_x_ls:
            ksi = lc_x / quasi_x
            if np.abs(1.0 - ksi) <= eps or ksi < domain_lo or ksi > domain_hi:
                continue
            subtraction += (
                coeff(ksi, log_scale)
                * np.abs(lc_x)
                * dquasi_x
                / quasi_x**2
            )

        pos = int(np.searchsorted(quasi_x_ls, lc_x))
        w_hi = (
            (lc_x - quasi_x_ls[pos - 1])
            / (quasi_x_ls[pos] - quasi_x_ls[pos - 1])
        )
        matrix[lc_idx, pos - 1] -= (1.0 - w_hi) * subtraction / dquasi_x
        matrix[lc_idx, pos] -= w_hi * subtraction / dquasi_x

    return matrix * dquasi_x


def CG_pdf_kernel_msbar(
    lc_x_ls: np.ndarray,
    pz_gev: float,
    mu: float = 2.0,
    quasi_x_ls: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the NLO MS-bar ``gamma^t`` PDF kernel from Eq. (2.14).

    Named matrices follow the paper: two plus pieces from ``C_r``, the plain
    ``1/(2|1-ksi|)`` fraction, and the finite ``delta(1-ksi)`` that absorbs
    ``-1/2 int_0^2 dksi'/|1-ksi'|``.
    """
    lc_x_ls = np.asarray(lc_x_ls, dtype=float)
    quasi_x_ls = np.asarray(
        lc_x_ls if quasi_x_ls is None else quasi_x_ls,
        dtype=float,
    )
    dquasi_x = float(quasi_x_ls[1] - quasi_x_ls[0])

    plus_01 = _build_plus_matrix(
        lc_x_ls,
        pz_gev,
        mu,
        quasi_x_ls,
        eps,
        coeff=lambda ksi, log_scale: C_msbar_plus_01(ksi, log_scale),
        subtraction_domain=(0.0, 1.0),
    )
    plus_all = _build_plus_matrix(
        lc_x_ls,
        pz_gev,
        mu,
        quasi_x_ls,
        eps,
        coeff=lambda ksi, log_scale: C_msbar_plus_all(ksi, eps),
        subtraction_domain=(-np.inf, np.inf),
    )

    # Plain 1/(2|1-ksi|) on the grid (ksi != 1); no plus subtraction here.
    frac_02 = np.zeros_like(plus_01)
    for lc_idx, lc_x in enumerate(lc_x_ls):
        for quasi_idx, quasi_x in enumerate(quasi_x_ls):
            ksi = lc_x / quasi_x
            if np.abs(1.0 - ksi) <= eps:
                continue
            frac_02[lc_idx, quasi_idx] = (
                0.5 / (np.abs(1.0 - ksi) + eps) / np.abs(quasi_x)
            )
    frac_02 *= dquasi_x

    # (1/2) delta(1-ksi) * [1 + log - int_0^2 dksi'/|1-ksi'|].
    finite_delta = np.zeros_like(plus_01)
    for lc_idx, lc_x in enumerate(lc_x_ls):
        log_scale = np.log(4.0 * lc_x**2 * pz_gev**2 / mu**2)
        integral = 0.0
        for quasi_x in quasi_x_ls:
            ksi = lc_x / quasi_x
            if np.abs(1.0 - ksi) <= eps or ksi < 0.0 or ksi > 2.0:
                continue
            integral += (
                1.0
                / (np.abs(1.0 - ksi) + eps)
                * np.abs(lc_x)
                * dquasi_x
                / quasi_x**2
            )
        coefficient = 0.5 * (1.0 + log_scale - integral)
        pos = int(np.searchsorted(quasi_x_ls, lc_x))
        w_hi = (
            (lc_x - quasi_x_ls[pos - 1])
            / (quasi_x_ls[pos] - quasi_x_ls[pos - 1])
        )
        finite_delta[lc_idx, pos - 1] = (1.0 - w_hi) * coefficient
        finite_delta[lc_idx, pos] = w_hi * coefficient

    alpha_s = alphas_nloop(mu, order=1, Nf=3)
    return (
        -alpha_s
        * CF
        / (2.0 * np.pi)
        * (plus_01 + plus_all + frac_02 + finite_delta)
    )
