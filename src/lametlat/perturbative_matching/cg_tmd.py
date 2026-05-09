"""CG TMD matching kernels and Sudakov factors (continuum QCD).

References use arXiv identifiers in docstrings where applicable.
"""
# %%
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import zeta

from lametlat.utils.constants import CA, CF, TF, alphas_nloop, beta


def cusp0() -> float:
    return 2.0 * CF


def cusp1(Nf: int = 3) -> float:
    return 2.0 * CF * ((67.0 / 9.0 - (np.pi**2) / 3.0) * CA - 20.0 / 9.0 * TF * Nf)


def cusp2(Nf: int = 3) -> float:
    return (
        2.0 
        * CF
        * (
            CA**2
            * (245.0 / 6.0 - 134.0 * np.pi**2 / 27.0 + 11.0 * np.pi**4 / 45.0 + 22.0 / 3.0 * zeta(3))
            + CA * TF * Nf * (-418.0 / 27.0 + 40.0 * np.pi**2 / 27.0 - 56.0 / 3.0 * zeta(3))
            + CF * TF * Nf * (-55.0 / 3.0 + 16.0 * zeta(3))
            - 16.0 / 27.0 * TF**2 * Nf**2
        )
    )


def CG_gamma_c() -> float:
    """Soft anomalous dimension piece (CG convention).

    Eq. (A4) in arXiv:2504.04625; CG analogue of ``gamma_mu`` in arXiv:2307.12359, Eq. (C9).
    """
    return -6.0 * CF


def sudakov_kernel(Q: float, mu: float = 2.0, epsrel: float = 1e-3) -> float:
    """Sudakov factor from one-loop anomalous dimension (exponential form).

    Eq. (20) and Eq. (37) of arXiv:hep-ph/0309176.
    """

    def gamma_1(u: float) -> float:
        return (
            -CF
            * alphas_nloop(u, order=2, Nf=3)
            / (4.0 * np.pi)
            * (4.0 * np.log(u**2 / Q**2) + 6.0)
        )

    def integrand(u: float) -> float:
        return gamma_1(u) / u

    sudakov_integral, _err = quad(integrand, Q, mu, epsrel=epsrel)

    # NLL: one-loop gamma_C (alpha_s * L) + two-loop cusp (alpha_s^2 * L^2) + tree hard (1).
    # Finite O(alpha_s) hard term is dropped at this order.
    return 1.0 * np.exp(sudakov_integral) #! Since the log term can be large, we should use exp(temp) instead of 1 + temp for resummed results.


def CG_tmd_kernel_fixed(x: float, pz_gev: float, mu: float = 2.0) -> float:
    """Fixed-order CG TMD kernel (NLO-style expansion in alpha_s).

    arXiv:2311.01391; ``pz_gev`` is longitudinal momentum in GeV.
    """
    zeta_scale = (2.0 * x * pz_gev) ** 2

    temp = (
        0.5 * (np.log((mu**2) / zeta_scale)) ** 2
        + 3.0 * np.log((mu**2) / zeta_scale)
        + 12.0
        - np.pi**2 * 7.0 / 12.0
    )

    h = -CF * alphas_nloop(mu, order=2, Nf=3) / (4.0 * np.pi) * temp

    return 1.0 + h


def CG_tmd_kernel_RGR(x: float, pz_gev: float, mu: float = 2.0, vary_eps: float = 1.0) -> float:
    """RG-resummed CG TMD kernel (NLL).

    App. D.2 of arXiv:1002.2213; ``pz_gev`` in GeV.
    """
    zeta_scale = (2.0 * x * pz_gev * vary_eps) ** 2
    b0 = beta(order=0, Nf=3)
    b1 = beta(order=1, Nf=3)

    # NLL uses two-loop cusp with two-loop alpha_s at the low scale.
    a0_order0 = alphas_nloop(np.sqrt(zeta_scale), order=0, Nf=3) # 1-loop
    a0_order1 = alphas_nloop(np.sqrt(zeta_scale), order=1, Nf=3) # 2-loop
    amu_order0 = alphas_nloop(mu, order=0, Nf=3)
    amu_order1 = alphas_nloop(mu, order=1, Nf=3)
    r_order0 = amu_order0 / a0_order0
    r_order1 = amu_order1 / a0_order1

    term1 = 4.0 * np.pi / a0_order1 * (1.0 - 1.0 / r_order1 - np.log(r_order1))
    term2 = (cusp1() / cusp0() - b1 / b0) * (1.0 - r_order1 + np.log(r_order1))
    term3 = b1 / (2.0 * b0) * (np.log(r_order1) ** 2)

    k_cusp = -cusp0() / (4.0 * (b0**2)) * (term1 + term2 + term3)

    k_gammac = -CG_gamma_c() / (2.0 * b0) * np.log(r_order0)

    integral = -2.0 * k_cusp + k_gammac

    # NLL: same counting as in :func:`sudakov_kernel`; use exp for large logs.
    return 1.0 * np.exp(integral) #! Since the log term can be large, we should use exp(temp) instead of 1 + temp for resummed results.


def CG_tmdwf_kernel_fixed(x: float, pz_gev: float, mu: float = 2.0) -> float:
    return CG_tmd_kernel_fixed(x, pz_gev, mu) * CG_tmd_kernel_fixed(1.0 - x, pz_gev, mu)


def CG_tmdwf_kernel_RGR(x: float, pz_gev: float, mu: float = 2.0, vary_eps: float = 1.0) -> float:
    return CG_tmd_kernel_RGR(x, pz_gev, mu, vary_eps) * CG_tmd_kernel_RGR(
        1.0 - x, pz_gev, mu, vary_eps
    )


def CG_ff_kernel_RGR(x1: float, x2: float, pz_gev: float, mu: float) -> float:
    """Fragmentation-region kernel HF / (C C C C).

    Eq. (30) in arXiv:2311.01391v2; ``pz_gev`` in GeV.
    """
    sud_q = np.sqrt(4.0 * x1 * x2 * pz_gev * pz_gev)
    sud_qbar = np.sqrt(4.0 * (1.0 - x1) * (1.0 - x2) * pz_gev * pz_gev)

    kernel = (
        sudakov_kernel(sud_q, mu)
        * sudakov_kernel(sud_qbar, mu)
        / CG_tmd_kernel_RGR(x1, pz_gev, mu)
        / CG_tmd_kernel_RGR(x2, pz_gev, mu)
        / CG_tmd_kernel_RGR(1.0 - x1, pz_gev, mu)
        / CG_tmd_kernel_RGR(1.0 - x2, pz_gev, mu)
    )

    return kernel


def CG_tmdpdf_kernel_fixed(x: float, pz_gev: float, mu: float = 2.0) -> float:
    """Fixed-order CG TMD PDF kernel (cf. :func:`CG_tmd_kernel_fixed` with PDF normalization).

    arXiv:2311.01391; ``pz_gev`` in GeV.
    """
    zeta_scale = (2.0 * x * pz_gev) ** 2

    temp = (
        0.5 * (np.log((mu**2) / zeta_scale)) ** 2
        + 3.0 * np.log((mu**2) / zeta_scale)
        + 12.0
        - np.pi**2 * 7.0 / 12.0
    )

    h = -CF * alphas_nloop(mu, order=2, Nf=3) / (2.0 * np.pi) * temp

    return 1.0 + h


def CG_tmdpdf_kernel_RGR(x: float, pz_gev: float, mu: float = 2.0, vary_eps: float = 1.0) -> float:
    """RG-resummed CG TMD PDF kernel (product form).

    App. D.2 of arXiv:1002.2213; ``pz_gev`` in GeV.
    """
    return CG_tmd_kernel_RGR(x, pz_gev, mu, vary_eps) * CG_tmd_kernel_RGR(
        x, pz_gev, mu, vary_eps
    )
