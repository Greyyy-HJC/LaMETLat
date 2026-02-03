"""
All kinds of fit functions for the z-dependence extrapolation.
"""

import numpy as np


def exp_asym_re_fcn(m0=0):
    """
    Asymptotic form from https://arxiv.org/pdf/2601.12189, Eq.(2.4), the 1 / lambda^n term is for CG cases.

    Args:
        m0 (int, optional): minimum value of meff, usually set to 0.1 GeV / P_mom.
    """
    def fcn(lam_ls, p):
        return ( p["b"] * np.cos(p["c"]) + p["d"] * np.cos(p["e"]) / abs(lam_ls) ) * np.exp( -lam_ls * (p["m"] + m0) ) / ( lam_ls**p["n"] )
    
    return fcn


def exp_asym_im_fcn(m0=0):
    """
    Asymptotic form from https://arxiv.org/pdf/2601.12189, Eq.(2.4), the 1 / lambda^n term is for CG cases.

    Args:
        m0 (int, optional): minimum value of meff, usually set to 0.1 GeV / P_mom.
    """
    def fcn(lam_ls, p):
        return ( p["b"] * np.sin(p["c"]) + p["d"] * np.sin(p["e"]) / abs(lam_ls) ) * np.exp( -lam_ls * (p["m"] + m0) ) / ( lam_ls**p["n"] )
    
    return fcn

def exp_sin_fcn(m0=0):
    """
    No Regge behavior.

    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    def fcn(lam_ls, p):
        return ( p["b"] * np.sin(p["c"] * lam_ls + p["d"]) ) * np.exp( -lam_ls * (p["m"] + m0) ) / ( lam_ls**p["n"] )
    
    return fcn


def exp_poly_fcn(lam_ls, p):
    """
    No Regge behavior.

    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    return ( p["b"] + p["c"] * lam_ls + p["d"] * lam_ls**2 + p["e"] * lam_ls**3 ) * np.exp( -lam_ls * p["m"] ) / ( lam_ls**(3 + p["n"]) )


def exp_power_fcn(lam_ls, p):
    """
    No Regge behavior.

    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    return p["a"] * np.exp( -lam_ls * p["m"] ) / ( lam_ls**p["n"] )


def poly_fcn(lam_ls, p):
    """
    No Regge behavior.

    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    return ( p["b"] + p["c"] * lam_ls + p["d"] * lam_ls**2 + p["e"] * lam_ls**3 ) / ( lam_ls**(4 + p["n"]) )


def Regge_exp_re_fcn(lam_ls, p):
    """
    From Regge behavior in x-space.
    
    The extrapolation formular with exponential decay / finite correlation length, usually used for quasi distribution.
    Note here the input is lambda list, not z list.

    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    val = (
        p["c1"] / (lam_ls ** p["n1"]) * np.cos(np.pi / 2 * p["n1"])
        + p["c2"] / (lam_ls ** p["n2"]) * np.cos(lam_ls - np.pi / 2 * p["n2"])
    ) * np.exp(-lam_ls / p["lam0"])

    return val


def Regge_exp_im_fcn(lam_ls, p):
    """
    From Regge behavior in x-space.
    
    The extrapolation formular with exponential decay / finite correlation length, usually used for quasi distribution.
    Note here the input is lambda list, not z list.
    Note here imag part should go upward when z increase from z = 0.


    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    val = p["c1"] / (lam_ls ** p["n1"]) * np.sin(np.pi / 2 * p["n1"]) + p["c2"] / (
        lam_ls ** p["n2"]
    ) * np.sin(lam_ls - np.pi / 2 * p["n2"]) * np.exp(-lam_ls / p["lam0"])

    return val


def Regge_re_fcn(lam_ls, p):
    """
    From Regge behavior in x-space.
    
    The extrapolation formular without exponential decay, usually used for light-cone distribution.
    Note here the input is lambda list, not z list.

    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    val = p["c1"] / (lam_ls ** p["n1"]) * np.cos(np.pi / 2 * p["n1"]) + p["c2"] / (
        lam_ls ** p["n2"]
    ) * np.cos(lam_ls - np.pi / 2 * p["n2"])

    return val


def Regge_im_fcn(lam_ls, p):
    """
    From Regge behavior in x-space.
    
    The extrapolation formular without exponential decay, usually used for light-cone distribution.
    Note here the input is lambda list, not z list.
    Note here imag part should go upward when z increase from z = 0.


    Args:
        lam_ls (list): lambda = z * Pz
        p (prior): prior dict

    Returns:
        val: function value
    """
    val = p["c1"] / (lam_ls ** p["n1"]) * np.sin(np.pi / 2 * p["n1"]) + p["c2"] / (
        lam_ls ** p["n2"]
    ) * np.sin(lam_ls - np.pi / 2 * p["n2"])

    return val
