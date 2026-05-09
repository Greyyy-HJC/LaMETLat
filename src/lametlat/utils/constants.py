"""Physical constants and perturbative helper functions for LaMETLat."""

from __future__ import annotations

from typing import Final

import numpy as np

# Conversion factor: 1 fm^{-1} = 0.1973269631 GeV
GEV_FM: Final[float] = 0.1973269631
CF: Final[float] = 4.0 / 3.0
NF: Final[int] = 3
CA: Final[float] = 3.0
TF: Final[float] = 1.0 / 2.0

def lat_unit_convert(val: float, a: float, Ls: int, dimension: str) -> float:
    """Convert lattice units to GeV.

    Parameters
    ----------
    val:
        Input value in lattice units.
    a:
        Lattice spacing in fm.
    Ls:
        Spatial lattice size.
    dimension:
        ``"P"`` for momentum-like quantities, ``"M"`` for mass-like quantities.
    """
    if dimension == "P":
        return val * 2.0 * np.pi * GEV_FM / Ls / a
    if dimension == "M":
        return val * GEV_FM / a

    raise ValueError(f"Unsupported dimension {dimension!r}. Expected 'P' or 'M'.")


def beta(order: int = 0, Nf: int = 3) -> float:
    """Return QCD beta-function coefficient at LO/NLO/NNLO."""
    if order == 0:
        return 11.0 / 3.0 * CA - 4.0 / 3.0 * TF * Nf
    if order == 1:
        return 34.0 / 3.0 * CA**2 - (20.0 / 3.0 * CA + 4.0 * CF) * TF * Nf
    if order == 2:
        return (
            2857.0 / 54.0 * CA**3
            + (2.0 * CF**2 - 205.0 / 9.0 * CF * CA - 1415.0 / 27.0 * CA**2) * TF * Nf
            + (44.0 / 9.0 * CF + 158.0 / 27.0 * CA) * TF**2 * Nf**2
        )

    raise NotImplementedError(f"beta coefficient at order={order} is not implemented.")


def alphas_nloop(mu: float, order: int = 0, Nf: int = 3) -> float:
    """Compute running coupling :math:`alpha_s(mu)` up to NNLO."""
    a_s_ref = 0.293 / (4.0 * np.pi)
    b0 = beta(0, Nf)
    temp = 1.0 + a_s_ref * b0 * np.log((mu / 2.0) ** 2)

    if order == 0:
        return a_s_ref * 4.0 * np.pi / temp
    if order == 1:
        b1 = beta(1, Nf)
        return a_s_ref * 4.0 * np.pi / (temp + a_s_ref * b1 / b0 * np.log(temp))
    if order == 2:
        b1 = beta(1, Nf)
        b2 = beta(2, Nf)
        correction = (
            temp
            + a_s_ref * b1 / b0 * np.log(temp)
            + a_s_ref**2
            * (b2 / b0 * (1.0 - 1.0 / temp) + b1**2 / b0**2 * (np.log(temp) / temp + 1.0 / temp - 1.0))
        )
        return a_s_ref * 4.0 * np.pi / correction

    raise NotImplementedError(f"alpha_s at order={order} is not implemented.")


def Lz_func(z_fm: float, mu: float = 2.0) -> float:
    """Log term commonly appearing in perturbative matching kernels."""
    z2mu2 = z_fm**2 * mu**2 / GEV_FM**2
    val = z2mu2 * np.exp(2.0 * np.euler_gamma) / 4.0
    return np.log(val)


def CG_c0_func_nlo(z_fm: float, mu: float = 2.0, pol: str = "unpolarized", asorder: int = 0) -> float:
    """c0 coefficient in the SDF without LL resummation."""
    alpha_s = alphas_nloop(mu=mu, order=asorder, Nf=3)
    if pol == "unpolarized":
        const_term = 1.0  # gamma_t / gamma_t gamma_5
    elif pol == "helicity":
        const_term = 3.0  # gamma_z / gamma_z gamma_5
    elif pol == "transversity":
        const_term = 0.0  # gamma_z gamma_y
    else:
        raise ValueError(
            f"Unsupported polarization {pol!r}. "
            "Choose from: unpolarized, helicity, transversity."
        )
    return 1.0 + alpha_s * CF / (4.0 * np.pi) * (const_term - Lz_func(z_fm, mu))


def CG_c0_func_ll(
    z_fm: float,
    mu: float = 2.0,
    coeff: float = 1.0,
    Nf: int = 3,
    pol: str = "unpolarized",
    asorder: int = 0,
) -> float:
    """c0 coefficient after LL RGE resummation.

    See Eq. (B6) in arXiv:2504.04625.
    """
    mu0 = 2.0 * coeff * np.exp(-np.euler_gamma) / z_fm * GEV_FM
    alpha_0 = alphas_nloop(mu=mu0, order=asorder, Nf=Nf)
    alpha_1 = alphas_nloop(mu=mu, order=asorder, Nf=Nf)
    log_term = CF / beta(0, Nf) * np.log(alpha_1 / alpha_0)

    if pol == "transversity":
        # Transversity has zero anomalous dimension at NLO.
        return 1.0
    if pol not in ("unpolarized", "helicity"):
        raise ValueError(
            f"Unsupported polarization {pol!r}. "
            "Choose from: unpolarized, helicity, transversity."
        )
    return np.exp(log_term)


def CG_z_psi_ll(mu0: float, mu: float = 2.0, coeff: float = 1.0, Nf: int = 3, asorder: int = 0) -> float:
    """Renormalization factor of quark bilinears up to LL."""
    alpha_0 = alphas_nloop(mu=mu0 * coeff, order=asorder, Nf=Nf)
    alpha_1 = alphas_nloop(mu=mu, order=asorder, Nf=Nf)
    log_term = CF / beta(0, Nf) * np.log(alpha_1 / alpha_0)
    return np.exp(-log_term)


def soft_f_fix_msbar(b_fm: float, mu: float = 2.0) -> float:
    """Fixed-order MS-bar soft factor (one-loop correction in :math:`\\alpha_s`)."""
    alpha_s = alphas_nloop(mu=mu, order=2, Nf=3)
    return 1.0 - alpha_s * CF / np.pi * Lz_func(b_fm, mu)


def soft_f_rgr_msbar(b_fm: float, mu: float = 2.0) -> float:
    """LL-resummed MS-bar soft factor via running coupling."""
    alpha_s = alphas_nloop(mu=mu, order=2, Nf=3)
    mu0 = 2.0 * np.exp(-np.euler_gamma) * GEV_FM / b_fm
    alpha_s0 = alphas_nloop(mu=mu0, order=2, Nf=3)
    log_ratio = 4.0 * CF / beta(0, Nf=3) * np.log(alpha_s / alpha_s0)
    return np.exp(log_ratio)


def soft_f_fix_ratio(b_fm: float, mu: float = 2.0) -> float:
    """Product of ``soft_f_fix_msbar`` and squared unpolarized NLO :math:`c_0`."""
    c0 = CG_c0_func_nlo(b_fm, mu, pol="unpolarized", asorder=0)
    return soft_f_fix_msbar(b_fm, mu) * c0**2


def soft_f_rgr_ratio(b_fm: float, mu: float = 2.0) -> float:
    """Product of ``soft_f_rgr_msbar`` and squared unpolarized LL :math:`c_0`."""
    c0 = CG_c0_func_ll(b_fm, mu, asorder=0)
    return soft_f_rgr_msbar(b_fm, mu) * c0**2