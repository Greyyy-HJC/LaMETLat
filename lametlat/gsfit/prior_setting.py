"""
All prior dicts for gs fit should be defined here.
"""
import gvar as gv

def two_state_fit():
    priors = gv.BufferDict()
    priors["E0"] = gv.gvar(1, 10)
    priors["log(dE1)"] = gv.gvar(0, 10)

    priors["pdf_re"] = gv.gvar(1, 10)
    priors["pdf_im"] = gv.gvar(1, 10)
    priors["O01_re"] = gv.gvar(1, 10)
    priors["O01_im"] = gv.gvar(1, 10)
    # priors["O10_re"] = gv.gvar(1, 10)
    # priors["O10_im"] = gv.gvar(1, 10)
    priors["O11_re"] = gv.gvar(1, 10)
    priors["O11_im"] = gv.gvar(1, 10)

    priors["re_z0"] = gv.gvar(1, 10)
    priors["re_z1"] = gv.gvar(1, 10)

    return priors

def summation_fit():
    priors = gv.BufferDict()
    priors["pdf_re"] = gv.gvar(0, 10)
    priors["pdf_im"] = gv.gvar(0, 10)
    priors['re_b1'] = gv.gvar(0, 10)
    priors['im_b1'] = gv.gvar(0, 10)
    priors['re_b2'] = gv.gvar(0, 10)
    priors['im_b2'] = gv.gvar(0, 10)
    priors["log(dE1)"] = gv.gvar(0, 10)

    return priors
