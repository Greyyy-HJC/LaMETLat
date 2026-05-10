"""Discrete Fourier transforms in the Peskin convention (explicit sums).

The forward transform uses ``Δx / (2π) Σ_x e^{i x k} f(x)`` and the inverse uses
``Δk Σ_k e^{-i k x} f(k)``. Two-dimensional impact-parameter transforms use the
Hankel kernel ``J₀(k_T b_T)`` as in the legacy lattice utilities.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0


def sum_ft(x_ls, fx_ls, output_k):
    """Forward transform ``f(x) → f(k)`` returning a complex value or array.

    Computes ``(Δx / 2π) Σ_x exp(i x k) f(x)``.

    Parameters
    ----------
    x_ls, fx_ls:
        Sample positions and values on a uniform grid (same length).
    output_k:
        Scalar or 1D array of momentum values.

    Returns
    -------
    complex or numpy.ndarray
        ``f(k)`` with dtype ``complex128``. Scalar ``k`` yields a scalar numpy complex.
    """
    x = np.asarray(x_ls, dtype=float)
    fx = np.asarray(fx_ls)
    fx = fx.astype(complex, copy=False)
    delta_x = abs(x[1] - x[0])
    k = np.asarray(output_k, dtype=float)
    pref = delta_x / (2 * np.pi)

    if k.ndim == 0:
        phase = np.exp(1j * x * k)
        val = pref * np.dot(phase, fx)
        return val.astype(complex, copy=False)

    phase = np.exp(1j * np.multiply.outer(x, k))
    val = pref * (phase.T @ fx)
    return val.astype(complex, copy=False)


def sum_ft_re_im(x_ls, fx_re_ls, fx_im_ls, output_k):
    """Same forward transform as :func:`sum_ft`, but real and imaginary parts split.

    Matches the legacy ``cos`` / ``sin`` decomposition so inputs can stay real (e.g. ``gvar`` means).

    Parameters
    ----------
    x_ls:
        Sample positions.
    fx_re_ls, fx_im_ls:
        Real and imaginary parts of ``f(x)``.
    output_k:
        Scalar or 1D ``k`` values.

    Returns
    -------
    tuple
        ``(Re f(k), Im f(k))`` as NumPy scalars if ``output_k`` is scalar, else as
        ``numpy.ndarray`` vectors.
    """
    x = np.asarray(x_ls, dtype=float)
    fx_re = np.asarray(fx_re_ls, dtype=float)
    fx_im = np.asarray(fx_im_ls, dtype=float)
    delta_x = abs(x[1] - x[0])
    k = np.asarray(output_k, dtype=float)
    pref = delta_x / (2 * np.pi)
    fx = fx_re + 1j * fx_im

    if k.ndim == 0:
        val = pref * np.dot(np.exp(1j * x * k), fx)
        return val.real, val.imag

    phase = np.exp(1j * np.multiply.outer(x, k))
    val = pref * (phase.T @ fx)
    return val.real, val.imag


def complete_z_negative(lam_ls, re_ls, im_ls, *, im_flip_for_ft=False):
    """Complete the ``z < 0`` branch with symmetry constraints.

    Builds a symmetric grid from non-negative ``lam_ls`` (assumed to include
    zero): real part is extended as even and imaginary part as odd. Optionally,
    the imaginary part can be flipped first to match Fourier-transform
    convention choices used in legacy scripts.

    Parameters
    ----------
    lam_ls:
        1D non-negative coordinate grid (e.g. Ioffe time ``lambda``).
    re_ls, im_ls:
        Real and imaginary samples on ``lam_ls``.
    im_flip_for_ft:
        If ``True``, apply ``im -> -im`` before odd extension.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(lam_full, re_full, im_full)`` including the mirrored ``z < 0`` part.
    """
    lam = np.asarray(lam_ls, dtype=float)
    re = np.asarray(re_ls, dtype=float)
    im = np.asarray(im_ls, dtype=float)

    if im_flip_for_ft:
        im = -im

    lam_full = np.concatenate([-lam[::-1][:-1], lam])
    re_full = np.concatenate([re[::-1][:-1], re])
    im_full = np.concatenate([-im[::-1][:-1], im])
    return lam_full, re_full, im_full


def sum_inv_ft(k_ls, fk_ls, output_x):
    """Inverse transform ``f(k) → f(x)``: ``Δk Σ_k exp(-i k x) f(k)``."""
    k = np.asarray(k_ls, dtype=float)
    fk = np.asarray(fk_ls)
    fk = fk.astype(complex, copy=False)
    delta_k = abs(k[1] - k[0])
    x = np.asarray(output_x, dtype=float)

    if x.ndim == 0:
        phase = np.exp(-1j * k * x)
        val = delta_k * np.dot(phase, fk)
        return val.astype(complex, copy=False)

    phase = np.exp(-1j * np.multiply.outer(k, x))
    val = delta_k * (phase.T @ fk)
    return val.astype(complex, copy=False)


def sum_inv_ft_re_im(k_ls, fk_re_ls, fk_im_ls, output_x):
    """Inverse transform with real and imaginary parts of ``f(k)`` separated."""
    k = np.asarray(k_ls, dtype=float)
    fk_re = np.asarray(fk_re_ls, dtype=float)
    fk_im = np.asarray(fk_im_ls, dtype=float)
    delta_k = abs(k[1] - k[0])
    x = np.asarray(output_x, dtype=float)
    fk = fk_re + 1j * fk_im

    if x.ndim == 0:
        val = delta_k * np.dot(np.exp(-1j * k * x), fk)
        return val.real, val.imag

    phase = np.exp(-1j * np.multiply.outer(k, x))
    val = delta_k * (phase.T @ fk)
    return val.real, val.imag


def two_dim_ft(bT_gev, kT, f_bdep):
    """Hankel-type transform ``b_T → k_T`` with kernel ``J₀(k_T b_T)``.

    Computes ``(Δ b_T / 2π) Σ_{b_T} b_T f(b_T) J₀(k_T b_T)``.

    Parameters
    ----------
    bT_gev:
        1D impact-parameter grid (uniform spacing).
    kT:
        Scalar or 1D array of transverse momenta.
    f_bdep:
        Values ``f(b_T)`` on the same grid as ``bT_gev``.

    Returns
    -------
    numpy.floating or numpy.ndarray
        Transform value(s) at ``kT``.
    """
    b = np.asarray(bT_gev)
    f = np.asarray(f_bdep)
    delta_bT = abs(b[1] - b[0])
    k = np.asarray(kT)
    pref = delta_bT / (2 * np.pi)
    weight = b * f

    if k.ndim == 0:
        out = pref * np.dot(weight, j0(k * b))
        return out

    ker = j0(np.multiply.outer(k, b))
    out = pref * (ker @ weight)
    return out


def two_dim_inv_ft(kT_gev, bT, f_kdep):
    """Inverse Hankel-type transform ``k_T → b_T``.

    Computes ``Δ k_T (2π) Σ_{k_T} k_T f(k_T) J₀(k_T b_T)``.
    """
    kt = np.asarray(kT_gev, dtype=float)
    fk = np.asarray(f_kdep, dtype=float)
    delta_kT = abs(kt[1] - kt[0])
    b = np.asarray(bT, dtype=float)
    pref = delta_kT * (2 * np.pi)
    weight = kt * fk

    if b.ndim == 0:
        out = pref * np.dot(weight, j0(kt * b))
        return out

    ker = j0(np.multiply.outer(kt, b))
    out = pref * (weight @ ker)
    return out
