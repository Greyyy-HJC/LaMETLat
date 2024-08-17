"""
Hybrid renormalization means ratio for the short distance and MS-bar for the long distance.
"""

import numpy as np
import gvar as gv
import lsqfit as lsf
from lametlat.utils.constants import *

k = 3.320 # coefficient for PDF, from Yushan
lqcd = 0.1  # Lattice QCD in GeV
d_pdf = -0.08183 # coefficient for PDF, from Yushan


def pdf_ms_bar(z, mu):
    """
    1-loop PDF in the MS-bar scheme.

    Args:
        z (float): z in fm
        mu (float): mu in GeV

    Returns:
        float: 1-loop PDF in the MS-bar scheme
    """

    alphas = alphas_nloop(mu, order=2, Nf=3) # this is different from the alphas from Yushan

    val = 1 + alphas * CF / (2 * np.pi) * (
        3 / 2 * np.log(mu**2 * z**2 / GEV_FM**2 * np.exp(2 * np.euler_gamma) / 4)
        + 5 / 2
    )
    return val

def da_ms_bar(z, mu):
    """
    1-loop DA in the MS-bar scheme.

    Args:
        z (float): z in fm
        mu (float): mu in GeV

    Returns:
        float: 1-loop DA in the MS-bar scheme
    """

    alphas = alphas_nloop(mu, order=2, Nf=3)

    val = 1 + alphas * CF / (2 * np.pi) * (
        3 / 2 * np.log(mu**2 * z**2 / GEV_FM**2 * np.exp(2 * np.euler_gamma) / 4)
        + 7 / 2
    )
    return val

def bare_pdf_a_dep_fcn(z, mu):
    """
    Bare PDF fit function for a dependence.

    Args:
        z (float): z in fm
        a_ls (float): a_ls in fm
        mu (float): mu in GeV
        prior (dict): prior parameters

    Returns:
        float: Log of the bare PDF fit function
    """

    beta0 = beta(order=0, Nf=3)

    def fcn(a, p):

        val = (
        k * z / a * GEV_FM / np.log(a * lqcd / GEV_FM)
        + p["g"]
        + p["f"] * a / GEV_FM
        + 3 * CF / beta0 * np.log(np.log(GEV_FM / a / lqcd) / np.log(mu / lqcd))
            + np.log(1 + d_pdf / (np.log(a * lqcd / GEV_FM)))
        )

        return val

    return fcn

def renormalization_factor(z, a, f, m0, mu=2):
    """
    Renormalization factor for the self-renormalization.

    Args:
        z (float): z in fm
        a (float): a in fm
        mu (float): mu in GeV
        f (float): posterior f for linear divergence
        m0 (float): m0 in GeV

    Returns:
        float: Renormalization factor for the self-renormalization
    """

    beta0 = beta(order=0, Nf=3)

    val = (
        k * z / a * GEV_FM / np.log(a * lqcd / GEV_FM)
        + m0 * z / GEV_FM
        + f * a / GEV_FM
        + 3 * CF / beta0 * np.log(np.log(GEV_FM / a / lqcd) / np.log(mu / lqcd))
            + np.log(1 + d_pdf / (np.log(a * lqcd / GEV_FM)))
    )

    return np.exp(val)

def fit_m0(z_array, g_array, mu=2):
    """
    Fit m0 * z = g(z) - log(ms_bar)
    
    Args:
        z_array (np.array): z array in fm
        g_array (np.array): g array in GeV
        mu (float): mu in GeV

    Returns:
        fit_res (lsf.FitResult): fit result, in which m0 in GeV
    """

    priors = gv.BufferDict()
    priors["m0"] = gv.gvar(0, 1)
    priors["b"] = gv.gvar(0, 5)

    def fcn(z, p):
        return p["m0"] * z / GEV_FM + p["b"]

    m0z = g_array - np.log( pdf_ms_bar(z_array, mu) )

    fit_res = lsf.nonlinear_fit(data=(z_array, m0z), fcn=fcn, prior=priors, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')

    return fit_res

class SELF_RENORM:
    """
    Class for self-renormalization of quasi-PDFs.

    Attributes:
        bare_quasi (dict): Dictionary containing bare quasi-PDF data.
            Keys are lattice spacings (str), values are gvar arrays.
        z_interp (numpy.ndarray): Array of z values in fm.
        mu (float): Renormalization scale in GeV.
        log_data (dict): Dictionary to store logarithmic data.
        fit_results (list): List to store fit results.
        m0_fit_result (lsf.FitResult): Fit result for m0.
        renorm_quasi (list): List of renormalized quasi-PDFs.
        Z_R_dict (dict): Dictionary to store renormalization factors.

    Args:
        bare_quasi (dict): Dictionary containing bare quasi-PDF data.
            Keys are lattice spacings (str), values are gvar arrays.
        z_interp (array-like): Array of z values in fm.
        mu (float, optional): Renormalization scale in GeV. Defaults to 2.0.
    """
    def __init__(self, bare_quasi, z_interp, mu=2.0):
        self.bare_quasi = bare_quasi
        self.z_interp = np.array(z_interp)
        self.mu = mu
        self.log_data = {}
        self.fit_results = []
        self.m0_fit_result = None
        self.renorm_quasi = []
        self.Z_R_dict = {}

    def take_log(self):
        for a, data in self.bare_quasi.items():
            log_gvar_data = gv.log(data)
            self.log_data[a] = {
                'z': self.z_interp,
                '1/a': GEV_FM / float(a),
                'log_mean': gv.mean(log_gvar_data),
                'log_err': gv.sdev(log_gvar_data)
            }

    def fit_g_and_f(self):
        priors = gv.BufferDict({'g': gv.gvar(0, 20), 'f': gv.gvar(0, 5)})
        for z_idx, z in enumerate(self.z_interp):
            a_ls = []
            log_gvar_data = []
            for a, data in self.log_data.items():
                a_ls.append(float(a))
                log_gvar_data.append(gv.gvar(data['log_mean'][z_idx], data['log_err'][z_idx]))

            fcn = bare_pdf_a_dep_fcn(z, self.mu)
            fit_res = lsf.nonlinear_fit(data=(np.array(a_ls), log_gvar_data), fcn=fcn, prior=priors)
            self.fit_results.append(fit_res)

    def fit_m0(self):
        g_ls = [fit_res.p['g'] for fit_res in self.fit_results]
        ms_bar_ls = [pdf_ms_bar(z, self.mu) for z in self.z_interp]
        
        max_z_idx = 3
        z_array = self.z_interp[:max_z_idx]
        g_array = np.array(g_ls[:max_z_idx])
        self.m0_fit_result = fit_m0(z_array, g_array)

    def calculate_renormalization(self):
        g_ls = [fit_res.p['g'] for fit_res in self.fit_results]
        m0 = self.m0_fit_result.p['m0']

        self.renorm_quasi = [gv.exp(g - m0 * z / GEV_FM) for z, g in zip(self.z_interp, g_ls)]

        for a, data in self.bare_quasi.items():
            renorm_factors = []
            for z in self.z_interp:
                f = self.fit_results[np.where(self.z_interp == z)[0][0]].p['f']
                renorm_factor = renormalization_factor(z, float(a), f, m0, self.mu)
                renorm_factors.append(renorm_factor)
            self.Z_R_dict[a] = renorm_factors

    def main(self):
        self.take_log()
        self.fit_g_and_f()
        self.fit_m0()
        self.calculate_renormalization()
        return self.renorm_quasi, self.Z_R_dict