"""
All kinds of fit functions for the gs fit.
"""

#! check to make sure be consistent with the paper

import numpy as np


# * 2pt
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
    z0 = p["z0"]
    z1 = p["z1"]

    val = z0 ** 2 / (2 * e0) * ( np.exp( -e0 * pt2_t ) + np.exp( -e0 * ( Lt - pt2_t ) ) ) + z1 ** 2 / (2 * e1) * ( np.exp( -e1 * pt2_t ) + np.exp( -e1 * ( Lt - pt2_t ) ) )

    return val

# * ratio fit

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
    z0 = p["z0"]
    z1 = p["z1"]

    if nstate == 1:
        z1 = 0

    numerator = (
        p["O00_re"] * z0 ** 2 * np.exp(-e0 * ra_t) / (2 * e0) / (2 * e0)
        + p["O01_re"] * z0 * z1 * np.exp(-e0 * (ra_t - ra_tau)) * np.exp(-e1 * ra_tau) / (2 * e0) / (2 * e1)
        + p["O01_re"] * z1 * z0 * np.exp(-e1 * (ra_t - ra_tau)) * np.exp(-e0 * ra_tau) / (2 * e1) / (2 * e0)
        + p["O11_re"] * z1 ** 2 * np.exp(-e1 * ra_t) / (2 * e1) / (2 * e1)
    )
    denominator = z0 ** 2 / (2 * e0) * ( np.exp( -e0 * ra_t ) + np.exp( -e0 * ( Lt - ra_t ) ) ) + z1 ** 2 / (2 * e1) * ( np.exp( -e1 * ra_t ) + np.exp( -e1 * ( Lt - ra_t ) ) )

    val = numerator / denominator

    return val

def ra_im_fcn(ra_t, ra_tau, p, Lt, nstate=2):
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["z0"]
    z1 = p["z1"]

    if nstate == 1:
        z1 = 0

    numerator = (
        p["O00_im"] * z0 ** 2 * np.exp(-e0 * ra_t) / (2 * e0) / (2 * e0)
        + p["O01_im"] * z0 * z1 * np.exp(-e0 * (ra_t - ra_tau)) * np.exp(-e1 * ra_tau) / (2 * e0) / (2 * e1)
        + p["O01_im"] * z1 * z0 * np.exp(-e1 * (ra_t - ra_tau)) * np.exp(-e0 * ra_tau) / (2 * e1) / (2 * e0)
        + p["O11_im"] * z1 ** 2 * np.exp(-e1 * ra_t) / (2 * e1) / (2 * e1)
    )
    denominator = z0 ** 2 / (2 * e0) * ( np.exp( -e0 * ra_t ) + np.exp( -e0 * ( Lt - ra_t ) ) ) + z1 ** 2 / (2 * e1) * ( np.exp( -e1 * ra_t ) + np.exp( -e1 * ( Lt - ra_t ) ) )

    val = numerator / denominator

    return val

# * This is for the summation fit

def sum_re_1state_fcn(t, tau_cut, p):
    e0 = p["E0"]
    b1 = p["re_b1"]
    val = p["O00_re"] * (t - 2 * tau_cut + 1) / (2 * e0) + b1
    return val

def sum_im_1state_fcn(t, tau_cut, p):
    e0 = p["E0"]
    b1 = p["im_b1"]
    val = p["O00_im"] * (t - 2 * tau_cut + 1) / (2 * e0) + b1
    return val

def sum_re_2state_fcn(t, tau_cut, p, dE1=None):
    if dE1 is None:
        dE1 = p['dE1']
    e0 = p["E0"]
    b1 = p["re_b1"]
    b2 = p["re_b2"]
    b3 = p["re_b3"]
    c1 = p["re_c1"]

    numerator = p["O00_re"] * (t - 2 * tau_cut + 1) * ( 1 + b1 * np.exp(-dE1 * t) ) + b2 + b3 * np.exp( -dE1 * t )
    denominator = 2 * e0 * (1 + c1 * np.exp( -dE1 * t) )
    val = numerator / denominator
    return val

def sum_im_2state_fcn(t, tau_cut, p, dE1=None):
    if dE1 is None:
        dE1 = p['dE1']
    e0 = p["E0"]
    b1 = p["im_b1"]
    b2 = p["im_b2"]
    b3 = p["im_b3"]
    c1 = p["re_c1"]
    
    numerator = p["O00_im"] * (t - 2 * tau_cut + 1) * ( 1 + b1 * np.exp(-dE1 * t) ) + b2 + b3 * np.exp( -dE1 * t )
    denominator = 2 * e0 * (1 + c1 * np.exp( -dE1 * t) )
    val = numerator / denominator
    return val


# * This is for the FH fit, FH = sum(t + 1) - sum(t)

def fh_re_1state_fcn(t, p):
    e0 = p["E0"]
    val = p["O00_re"] / (2 * e0) + t * 0
    return val

def fh_im_1state_fcn(t, p):
    e0 = p["E0"]
    val = p["O00_im"] / (2 * e0) + t * 0
    return val

def fh_re_2state_fcn(t, tau_cut, p, dt=1):
    term1 = sum_re_2state_fcn(t + dt, tau_cut, p)
    term2 = sum_re_2state_fcn(t, tau_cut, p)
    val = (term1 - term2) / dt
    return val

def fh_im_2state_fcn(t, tau_cut, p, dt=1):
    term1 = sum_im_2state_fcn(t + dt, tau_cut, p)
    term2 = sum_im_2state_fcn(t, tau_cut, p)
    val = (term1 - term2) / dt
    return val

# * This is for DA, TMDWF fit

def da_re_fcn(da_t, p, Lt):
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["z0"]
    z1 = p["z1"]
    
    val = z0 / (2 * e0) * p["O00_re"] * ( np.exp( -e0 * da_t ) + np.exp( -e0 * ( Lt - da_t ) ) ) + z1 / (2 * e1) * p["O01_re"] * ( np.exp( -e1 * da_t ) + np.exp( -e1 * ( Lt - da_t ) ) )

    return val

def da_im_fcn(da_t, p, Lt):
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["z0"]
    z1 = p["z1"]

    val = z0 / (2 * e0) * p["O00_im"] * ( np.exp( -e0 * da_t ) + np.exp( -e0 * ( Lt - da_t ) ) ) + z1 / (2 * e1) * p["O01_im"] * ( np.exp( -e1 * da_t ) + np.exp( -e1 * ( Lt - da_t ) ) )

    return val

#* This is for the FF fit, check Eq. (D2) in https://arxiv.org/pdf/2504.04625

def ff_ratio_fcn(x, p):
    dE = p["dE1"]
    
    c1 = p["re_c1"]
    c2 = p["re_c2"]
    
    ra_t, ra_tau = x
    
    #* -1 for sign convention to keep ff positive
    val = -1 * p["ff"] * (1 + c1 * ( np.exp( - dE * ra_tau ) + np.exp( - dE * ( ra_t - ra_tau ) ) ) ) / ( 1 + c2 * np.exp( - dE * ra_t / 2 )  ) 
    return val
