# %%
import numpy as np
from lametlat.utils.constants import *
from scipy.integrate import quad


def cusp0():
    return 2 * CF


def cusp1(Nf=3):
    return 2 * CF * ((67 / 9 - (np.pi**2) / 3) * CA - 20 / 9 * TF * Nf)


def cusp2(Nf=3):
    return (
        2
        * CF
        * (
            CA**2
            * (245 / 6 - 134 * np.pi**2 / 27 + 11 * np.pi**4 / 45 + 22 / 3 * sp.zeta(3))
            + CA * TF * Nf * (-418 / 27 + 40 * np.pi**2 / 27 - 56 / 3 * sp.zeta(3))
            + CF * TF * Nf * (-55 / 3 + 16 * sp.zeta(3))
            - 16 / 27 * TF**2 * Nf**2
        )
    )


def gamma_c():
    return -6 * CF


def sudakov_kernel(Q, mu=2, epsrel=1e-3):
    """
    From Eq.20 and Eq.37 of https://arxiv.org/pdf/hep-ph/0309176
    """

    def gamma_1(u):
        return (
            -CF
            * alphas_nloop(u, order=2, Nf=3)
            / (4 * np.pi)
            * (4 * np.log(u**2 / Q**2) + 6)
        )

    def integrand(u):
        return gamma_1(u) / u

    # do the integral on gamma_1 with specified precision
    int, error = quad(integrand, Q, mu, epsrel=epsrel)

    # * till NLL order, the finite term at alphas^1 of the hard kernel should be dropped.
    # * NLL means: 1-loop gammaC (alphas * L) + 2-loop cusp anomalous dimension (alphas^2 * L^2) + tree level hard kernel (1)
    return 1 * np.exp(int)


def CG_tmd_kernel_RGR(x, pz_gev, mu=2, vary_eps=1):
    """
    From App.D.2 of https://arxiv.org/pdf/1002.2213, pz in GeV
    """
    zeta = (2 * x * pz_gev * vary_eps) ** 2
    b0 = beta(order=0, Nf=3)
    b1 = beta(order=1, Nf=3)
    a0 = alphas_nloop(np.sqrt(zeta), order=2, Nf=3)
    amu = alphas_nloop(mu, order=2, Nf=3)

    r = amu / a0

    term1 = 4 * np.pi / a0 * (1 - 1 / r - np.log(r))
    term2 = (cusp1() / cusp0() - b1 / b0) * (1 - r + np.log(r))
    term3 = b1 / (2 * b0) * (np.log(r) ** 2)

    k_cusp = -cusp0() / (4 * (b0**2)) * (term1 + term2 + term3)

    k_gammac = -gamma_c() / (2 * b0) * np.log(r)

    integral = -2 * k_cusp + 1 * k_gammac  # * for DA, it is the Wilson coefficient

    # * till NLL order, the finite term at alphas^1 of the hard kernel should be dropped.
    # * NLL means: 1-loop gammaC (alphas * L) + 2-loop cusp anomalous dimension (alphas^2 * L^2) + tree level hard kernel (1)
    h = 1 * np.exp(integral)

    return h


def CG_tmdwf_kernel_RGR(x, pz_gev, mu=2, vary_eps=1):
    return CG_tmd_kernel_RGR(x, pz_gev, mu, vary_eps) * CG_tmd_kernel_RGR(
        1 - x, pz_gev, mu, vary_eps
    )


def CG_ff_kernel_RGR(x1, x2, pz_gev, mu):
    """
    Check Eq.30 in https://arxiv.org/pdf/2311.01391v2, this kernel is HF / (C C C C), pz in GeV
    """

    sud_q = np.sqrt(4 * x1 * x2 * pz_gev * pz_gev)
    sud_qbar = np.sqrt(4 * (1 - x1) * (1 - x2) * pz_gev * pz_gev)

    kernel = (
        sudakov_kernel(sud_q, mu)
        * sudakov_kernel(sud_qbar, mu)
        / CG_tmd_kernel_RGR(x1, pz_gev, mu)
        / CG_tmd_kernel_RGR(x2, pz_gev, mu)
        / CG_tmd_kernel_RGR(1 - x1, pz_gev, mu)
        / CG_tmd_kernel_RGR(1 - x2, pz_gev, mu)
    )

    return kernel


def CG_tmdpdf_kernel_fixed(x, pz_gev, mu=2):
    """
    From 2311.01391, pz in GeV
    """
    zeta = (2 * x * pz_gev) ** 2

    temp = (
        1 / 2 * (np.log((mu**2) / zeta)) ** 2
        + 3 * np.log((mu**2) / zeta)
        + 12
        - np.pi**2 * 7 / 12
    )

    h = -CF * alphas_nloop(mu, order=2, Nf=3) / (2 * np.pi) * temp

    return 1 + h


def CG_tmdpdf_kernel_RGR(x, pz_gev, mu=2, vary_eps=1):
    """
    From App.D.2 of https://arxiv.org/pdf/1002.2213, pz in GeV
    """
    return CG_tmd_kernel_RGR(x, pz_gev, mu, vary_eps) * CG_tmd_kernel_RGR(
        x, pz_gev, mu, vary_eps
    )


#!########################################################################################
#! Below are the GI matching kernels, from Xiang, check Appx. of https://arxiv.org/pdf/2307.12359
import scipy.special as sp


def GI_noncuspMu_func(order, Nf=3):
    if order == 0:
        return -2 * CF
    elif order == 1:
        return CF * (
            CF * (-4 + 14 / 3 * np.pi**2 - 24 * sp.zeta(3))
            + CA * (-554 / 27 - 11 * np.pi**2 / 6 + 22 * sp.zeta(3))
            + Nf * (80 / 27 + np.pi**2 / 3)
        )
    else:
        raise ValueError("GI_noncuspMu only implemented up to order 1")


def GI_noncuspC_func(order, Nf=3):
    if order == 0:
        return 2 * CF - complex(0, np.pi * cusp0())
    elif order == 1:
        return CF * (
            CF * (4 - 14 / 3 * np.pi**2 + 24 * sp.zeta(3))
            + CA * (950 / 27 + 11 * np.pi**2 / 9 - 22 * sp.zeta(3))
            + Nf * (-152 / 27 - 2 * np.pi**2 / 9)
        ) - complex(0, np.pi * (cusp1() + beta(0, Nf=3) * (2 * 2 * CF - cusp0())))
    else:
        raise ValueError("GI_noncuspC only implemented up to order 1")


def kernel_GI(mu0, mu, order, Nf=3):

    Lz = np.log(mu0**2 / mu**2) + complex(0, np.pi)
    aS = alphas_nloop(mu, order - 1) / (4 * np.pi)
    kernel = 1

    if order > 0:
        Lz2 = -0.5 * Lz**2
        Lz1 = Lz
        Lz0 = -2 - 5 * np.pi**2 / 12
        kernel += aS * CF * (Lz2 + Lz1 + Lz0)

    if order > 1:
        Lz4 = CF / 4 * Lz**4
        Lz3 = -(CF - 11 / 9 * CA + 2 / 9 * Nf) * Lz**3
        Lz2 = (
            (3 + 5 * np.pi**2 / 12) * CF + (np.pi**2 / 3 - 100 / 9) * CA + 16 / 9 * Nf
        ) * Lz**2
        Lz1 = (
            -(
                (11 * np.pi**2 / 2 - 24 * sp.zeta(3)) * CF
                + (22 * sp.zeta(3) - 44 * np.pi**2 / 9 - 950 / 27) * CA
                + (152 / 27 + 8 * np.pi**2 / 9) * Nf
            )
            * Lz
        )
        Lz0 = (
            (-30 * sp.zeta(3) + 65 * np.pi**2 / 3 - 167 * np.pi**4 / 144 - 16) * CF
            + (
                241 * sp.zeta(3) / 9
                + 53 * np.pi**4 / 60
                - 1759 * np.pi**2 / 108
                - 3884 / 81
            )
            * CA
            + (2 * sp.zeta(3) / 9 + 113 * np.pi**2 / 54 + 656 / 81) * Nf
        )
        kernel += aS**2 * CF / 2 * (Lz4 + Lz3 + Lz2 + Lz1 + Lz0)

    if order > 2:
        raise ValueError("NNNLO GI matching not avaible")

    return kernel


# RG cusp
def kernel_GI_K1(mu0, mu, order, Nf=3):

    r = alphas_nloop(mu, order) / alphas_nloop(mu0, order)
    aS = alphas_nloop(mu0, order) / (4 * np.pi)
    kernel = 1 / aS * (1 - 1 / r - np.log(r))

    if order > 0:
        kernel += (cusp1(Nf=Nf) / cusp0() - beta(1, Nf=Nf) / beta(0, Nf=Nf)) * (
            1 - r + np.log(r)
        ) + beta(1, Nf=Nf) / (2 * beta(0, Nf=Nf)) * np.log(r) ** 2
    if order > 1:
        kernel += aS * (
            ((beta(1, Nf=Nf) / beta(0, Nf=Nf)) ** 2 - beta(2, Nf=Nf) / beta(0, Nf=Nf))
            * ((1 - r**2) / 2 + np.log(r))
            + (
                beta(1, Nf=Nf) / beta(0, Nf=Nf) * cusp1(Nf=Nf) / cusp0()
                - (beta(1, Nf=Nf) / beta(0, Nf=Nf)) ** 2
            )
            * (1 - r + r * np.log(r))
            - (cusp2() / cusp0() - beta(1, Nf=Nf) / beta(0, Nf=Nf) * cusp1() / cusp0())
            * (1 - r) ** 2
            / 2
        )
    if order > 2:
        raise ValueError("NNNLL GI matching not avaible")

    return -cusp0() / (4 * beta(0, Nf=Nf) ** 2) * kernel


# RG GI_noncuspC
def kernel_GI_K2(mu0, mu, order):

    r = alphas_nloop(mu, order - 1) / alphas_nloop(mu0, order - 1)
    aS = alphas_nloop(mu0, order - 1) / (4 * np.pi)
    kernel = 0

    if order > 0:
        kernel += np.log(r)
    if order > 1:
        kernel += (
            aS
            * (
                GI_noncuspC_func(1, Nf=3) / GI_noncuspC_func(0, Nf=3)
                - beta(1, Nf=3) / beta(0, Nf=3)
            )
            * (r - 1)
        )
    if order > 2:
        raise ValueError("NNNLL GI matching not avaible")
    # print('>>',mu0, mu,r)

    return -GI_noncuspC_func(0, Nf=3) / (2 * beta(0, Nf=3)) * kernel


# RG noncusp_mu
def kernel_GI_K3(mu0, mu, order):

    r = alphas_nloop(mu, order - 1) / alphas_nloop(mu0, order - 1)
    aS = alphas_nloop(mu0, order - 1) / (4 * np.pi)
    kernel = 0

    if order > 0:
        kernel += np.log(r)
    if order > 1:
        kernel += (
            aS
            * (
                GI_noncuspMu_func(1, Nf=3) / GI_noncuspMu_func(0, Nf=3)
                - beta(1, Nf=3) / beta(0, Nf=3)
            )
            * (r - 1)
        )
    if order > 2:
        raise ValueError("NNNLL GI matching not avaible")

    return -GI_noncuspMu_func(0, Nf=3) / (2 * beta(0, Nf=3)) * kernel


# RG noncusp_eta
def kernel_GI_eta(mu0, mu, order):

    r = alphas_nloop(mu, order - 1) / alphas_nloop(mu0, order - 1)
    aS = alphas_nloop(mu0, order - 1) / (4 * np.pi)
    kernel = 0

    if order > 0:
        kernel += np.log(r)
    if order > 1:
        kernel += aS * (cusp1() / cusp0() - beta(1, Nf=3) / beta(0, Nf=3)) * (r - 1)
    if order > 2:
        raise ValueError("NNNLL GI matching not avaible")

    return -complex(0, np.pi) * cusp0() / (2 * beta(0, Nf=3)) * kernel


def GI_tmd_kernel_fixed(x, pz_gev, mu=2):
    """
    NNLO GI matching kernel, fixed order
    From Xiang, check Appx. of https://arxiv.org/pdf/2307.12359
    """
    temp1 = kernel_GI(2 * x * pz_gev, mu, 2, Nf=3)
    temp2 = -0.5 * (kernel_GI(2 * x * pz_gev, mu, 1) - 1) ** 2

    return temp1 + temp2


def GI_tmd_kernel_RGR(x, pz_gev, mu=2, vary_eps=1):
    """
    NNLL GI matching kernel, RGR
    From Xiang, check Appx. of https://arxiv.org/pdf/2307.12359
    """
    temp1 = -2 * kernel_GI_K1(2 * x * pz_gev * vary_eps, mu, 2)
    temp2 = kernel_GI_K3(2 * x * pz_gev * vary_eps, mu, 2)
    temp3 = kernel_GI_eta(2 * x * pz_gev * vary_eps, mu, 2)
    temp4 = kernel_GI(2 * x * pz_gev * vary_eps, 2 * x * pz_gev * vary_eps, 1)

    return temp1 + temp2 + temp3 + temp4

def GI_tmdwf_kernel_fixed(x, pz_gev, mu=2):
    return GI_tmd_kernel_fixed(x, pz_gev, mu) * GI_tmd_kernel_fixed(1-x, pz_gev, mu)

def GI_tmdwf_kernel_RGR(x, pz_gev, mu=2, vary_eps=1):
    return GI_tmd_kernel_RGR(x, pz_gev, mu, vary_eps) * GI_tmd_kernel_RGR(1-x, pz_gev, mu, vary_eps)

# %%

