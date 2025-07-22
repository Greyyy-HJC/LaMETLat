import numpy as np
from lametlat.utils.constants import *
from scipy.integrate import quad

def cusp1():
    return 2 * CF

def cusp2(Nf=3):
    return 2 * CF * ( (67/9 - (np.pi**2) / 3) * CA - 20/9 * TF * Nf )

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

def CG_tmd_kernel_RGR(x, pz, mu=2, vary_eps=1):
    '''
    From App.D.2 of https://arxiv.org/pdf/1002.2213, pz in GeV
    '''
    zeta = (2 * x * pz * vary_eps)**2
    b0 = beta(order=0, Nf=3)
    b1 = beta(order=1, Nf=3)
    a0 = alphas_nloop(np.sqrt(zeta), order=2, Nf=3)
    amu = alphas_nloop(mu, order=2, Nf=3)

    r = amu / a0

    term1 = 4 * np.pi / a0 * (1 - 1/r - np.log(r))
    term2 = ( cusp2()/cusp1() - b1/b0 ) * (1 - r + np.log(r))
    term3 = b1 / (2 * b0) * ( np.log(r)**2 )

    k_cusp = - cusp1() / ( 4 * (b0**2) ) * ( term1 + term2 + term3 )

    k_gammac = - gamma_c() / ( 2 * b0 ) * np.log(r)

    integral = - 2 * k_cusp + 1 * k_gammac  # * for DA, it is the Wilson coefficient

    #* till NLL order, the finite term at alphas^1 of the hard kernel should be dropped. 
    # * NLL means: 1-loop gammaC (alphas * L) + 2-loop cusp anomalous dimension (alphas^2 * L^2) + tree level hard kernel (1)
    h = 1 * np.exp( integral )

    return h

def CG_tmdwf_kernel_RGR(x, pz, mu=2, vary_eps=1):
    return CG_tmd_kernel_RGR(x, pz, mu, vary_eps) * CG_tmd_kernel_RGR(1-x, pz, mu, vary_eps)


def CG_ff_kernel_RGR(x1, x2, pz, mu):
    """
    Check Eq.30 in https://arxiv.org/pdf/2311.01391v2, this kernel is HF / (C C C C), pz in GeV
    """

    sud_q = np.sqrt(4 * x1 * x2 * pz * pz)
    sud_qbar = np.sqrt(4 * (1 - x1) * (1 - x2) * pz * pz)

    kernel = (
        sudakov_kernel(sud_q, mu)
        * sudakov_kernel(sud_qbar, mu)
        / CG_tmd_kernel_RGR(x1, pz, mu)
        / CG_tmd_kernel_RGR(x2, pz, mu)
        / CG_tmd_kernel_RGR(1 - x1, pz, mu)
        / CG_tmd_kernel_RGR(1 - x2, pz, mu)
    )

    return kernel

def CG_tmdpdf_kernel_fixed(x, pz_gev, mu=2):
    '''
    From 2311.01391, pz in GeV
    '''
    zeta = (2 * x * pz_gev)**2

    temp = 1/2 * (np.log((mu**2) / zeta))**2 + 3 * np.log((mu**2) / zeta) + 12 - np.pi**2 * 7 / 12
    
    h = - CF * alphas_nloop(mu, order=2, Nf=3) / (2*np.pi) * temp
    
    return 1 + h

def CG_tmdpdf_kernel_RGR(x, pz, mu=2, vary_eps=1):
    '''
    From App.D.2 of https://arxiv.org/pdf/1002.2213, pz in GeV
    '''
    return CG_tmd_kernel_RGR(x, pz, mu, vary_eps) * CG_tmd_kernel_RGR(x, pz, mu, vary_eps)