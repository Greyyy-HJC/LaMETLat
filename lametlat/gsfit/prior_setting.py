"""
All prior dicts for gs fit should be defined here.
"""
import gvar as gv

def two_state_fit():
    priors = gv.BufferDict()
    
    # * energy
    priors["E0"] = gv.gvar(1, 10)
    priors["log(dE1)"] = gv.gvar(0, 10)

    # * matrix elements
    priors["O00_re"] = gv.gvar(1, 10)
    priors["O00_im"] = gv.gvar(1, 10)
    priors["O01_re"] = gv.gvar(1, 10)
    priors["O01_im"] = gv.gvar(1, 10)
    # priors["O10_re"] = gv.gvar(1, 10)
    # priors["O10_im"] = gv.gvar(1, 10)
    priors["O11_re"] = gv.gvar(1, 10)
    priors["O11_im"] = gv.gvar(1, 10)

    # * overlap
    priors["z0"] = gv.gvar(1, 10)
    priors["z1"] = gv.gvar(1, 10)
    
    # * add-on coefficients
    priors["re_b1"] = gv.gvar(0, 10)
    priors["re_b2"] = gv.gvar(0, 10)
    priors["re_b3"] = gv.gvar(0, 10)
    priors["re_c1"] = gv.gvar(0, 10)
    priors["re_c2"] = gv.gvar(0, 10)
    
    priors["im_b1"] = gv.gvar(0, 10)
    priors["im_b2"] = gv.gvar(0, 10)
    priors["im_b3"] = gv.gvar(0, 10)
    priors["im_c1"] = gv.gvar(0, 10)
    priors["im_c2"] = gv.gvar(0, 10)
    
    # * FF fit
    priors["log(ff)"] = gv.gvar(-2, 5)
    
    return priors