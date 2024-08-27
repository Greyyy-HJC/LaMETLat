"""
All prior dicts for z-dependence extrapolation should be defined here.
"""
import gvar as gv

def Regge_prior(): # For Regge behavior
    priors = gv.BufferDict()
    priors['c1'] = gv.gvar(1, 10)
    priors['c2'] = gv.gvar(1, 10)
    priors['n1'] = gv.gvar(1, 10)
    priors['n2'] = gv.gvar(1, 10)
    priors['log(lam0)'] = gv.gvar(2.4, 10)

    return priors

def exp_decay_prior():
    priors = gv.BufferDict()
    priors["b"] = gv.gvar(1, 10)
    priors["c"] = gv.gvar(0, 10)
    priors["d"] = gv.gvar(0, 10)
    priors["e"] = gv.gvar(0, 10)
    priors["log(n)"] = gv.gvar(0, 10)
    priors["log(m)"] = gv.gvar(-2.4, 2.4) # gv.gvar(-2.4, 10)
    
    return priors



    