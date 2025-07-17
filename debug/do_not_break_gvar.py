# %%
import numpy as np
import gvar as gv
import lsqfit
import scipy.special as sp

def gamma_gvar_wrong(x):
    try:
        return sp.gamma(x)
    except TypeError:
        val = sp.gamma(gv.mean(x))
        dx = 1e-8
        deriv = (sp.gamma(gv.mean(x) + dx) - sp.gamma(gv.mean(x) - dx)) / (2*dx)
        return gv.gvar(val, abs(deriv * gv.sdev(x)))

def gamma_gvar_right(x):
    try:
        return sp.gamma(x)
    except TypeError:
        val = sp.gamma(gv.mean(x))
        dx = 1e-8
        deriv = (sp.gamma(gv.mean(x) + dx) - sp.gamma(gv.mean(x) - dx)) / (2*dx)
        return val + deriv * (x - gv.mean(x))

# 拟合函数
def fcn_wrong(x, p):
    return {'y': gamma_gvar_wrong(p['a']) * x['x']}

def fcn_right(x, p):
    return {'y': gamma_gvar_right(p['a']) * x['x']}

# 拟合数据
xdata = {'x': np.array([2.0, 3.0, 4.0])}
ydata = {'y': np.array([gv.gvar(2.0, 0.1), gv.gvar(3.0, 0.1), gv.gvar(4.0, 0.1)])}
prior = {'a': gv.gvar(2, 0.5)}

print("==== Wrong (manually break gvar) ====")
try:
    fit_wrong = lsqfit.nonlinear_fit(data=(xdata, ydata), prior=prior, fcn=fcn_wrong)
    print(fit_wrong)
except Exception as e:
    print("lsqfit error:", e)

print("\n==== Right (linear expression) ====")
try:
    fit_right = lsqfit.nonlinear_fit(data=(xdata, ydata), prior=prior, fcn=fcn_right)
    print(fit_right)
except Exception as e:
    print("lsqfit error:", e)

# %%
