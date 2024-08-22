"""
All kinds of fit functions for the gs fit.
"""

import numpy as np

#! check to make sure be consistent with the paper
def pt2_re_fcn(pt2_t, p, Lt):
    """
    Calculate the real part of the two-point correlator function.

    Args:
        pt2_t (float): The time separation between the two points.
        p (dict): Priors.
        Lt (int): The temporal size of the lattice.

    Returns:
        float: The value of the real part of the two-point correlator function.
    """
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["re_z0"]
    z1 = p["re_z1"]

    val = z0 ** 2 * ( np.exp( -e0 * pt2_t ) + np.exp( -e0 * ( Lt - pt2_t ) ) ) + z1 ** 2 * ( np.exp( -e1 * pt2_t ) + np.exp( -e1 * ( Lt - pt2_t ) ) )

    return val


def ra_re_fcn(ra_t, ra_tau, p, Lt, nstate=2):
    """
    Calculate the value of the ra_re_fcn function.

    Parameters:
    - ra_t (float): The value of ra_t.
    - ra_tau (float): The value of ra_tau.
    - p (dict): Priors.
    - Lt (int): The temporal size of the lattice.
    - nstate (int, optional): The number of states. Default is 2. 1 means a constant fit.

    Returns:
    - val (float): The calculated value of the function.

    """
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["re_z0"]
    z1 = p["re_z1"]

    if nstate == 1:
        z1 = 0

    numerator = (
        p["pdf_re"] * z0 ** 2 * np.exp(-e0 * ra_t)
        + p["O01_re"] * z0 * z1 * np.exp(-e0 * (ra_t - ra_tau)) * np.exp(-e1 * ra_tau)
        + p["O01_re"] * z1 * z0 * np.exp(-e1 * (ra_t - ra_tau)) * np.exp(-e0 * ra_tau)
        + p["O11_re"] * z1 ** 2 * np.exp(-e1 * ra_t)
    )
    denominator = z0 ** 2 * ( np.exp( -e0 * ra_t ) + np.exp( -e0 * ( Lt - ra_t ) ) ) + z1 ** 2 * ( np.exp( -e1 * ra_t ) + np.exp( -e1 * ( Lt - ra_t ) ) )

    val = numerator / denominator

    return val

def ra_im_fcn(ra_t, ra_tau, p, Lt, nstate=2):
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["re_z0"]
    z1 = p["re_z1"]

    if nstate == 1:
        z1 = 0

    numerator = (
        p["pdf_im"] * z0 ** 2 * np.exp(-e0 * ra_t)
        + p["O01_im"] * z0 * z1 * np.exp(-e0 * (ra_t - ra_tau)) * np.exp(-e1 * ra_tau)
        + p["O01_im"] * z1 * z0 * np.exp(-e1 * (ra_t - ra_tau)) * np.exp(-e0 * ra_tau)
        + p["O11_im"] * z1 ** 2 * np.exp(-e1 * ra_t)
    )
    denominator = z0 ** 2 * ( np.exp( -e0 * ra_t ) + np.exp( -e0 * ( Lt - ra_t ) ) ) + z1 ** 2 * ( np.exp( -e1 * ra_t ) + np.exp( -e1 * ( Lt - ra_t ) ) )

    val = numerator / denominator

    return val



# * This is for the summation fit

def sum_re_fcn(t, tau_cut, p):
    val = p["pdf_re"] * (t - 2 * tau_cut + 1) + p["re_b1"]
    return val

def sum_im_fcn(t, tau_cut, p):
    val = p["pdf_im"] * (t - 2 * tau_cut + 1) + p["im_b1"]
    return val

def sum_re_2state_fcn(t, tau_cut, p):
    val = p["pdf_re"] * (t - 2 * tau_cut + 1) + p["re_b1"] + p["re_b2"] * np.exp( -p['dE1'] * t )
    return val

def sum_im_2state_fcn(t, tau_cut, p):
    val = p["pdf_im"] * (t - 2 * tau_cut + 1) + p["im_b1"] + p["im_b2"] * np.exp( -p['dE1'] * t )
    return val


# * This is for the FH fit, FH = sum(t + 1) - sum(t)

def fh_re_fcn(t, p):
    val = p["pdf_re"] + t * 0
    return val

def fh_im_fcn(t, p):
    val = p["pdf_im"] + t * 0
    return val