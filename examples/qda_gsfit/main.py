# %%
import numpy as np
import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt

from utils import default_plot, COLOR_CYCLE, ERRORBAR_STYLE
from utils import jk_ls_avg

Lt = 64

pt2_re = np.loadtxt("pt2_re.txt")
qda_re = np.loadtxt("qda_re.txt")
qda_im = np.loadtxt("qda_im.txt")

pt2_avg = jk_ls_avg(pt2_re)
qda_re_avg = jk_ls_avg(qda_re)
qda_im_avg = jk_ls_avg(qda_im)

print(pt2_re.shape) # here 553 is number of jackknife samples, 64 is number of time slices
print(qda_re.shape)
print(qda_im.shape)

qda_ratio_re = qda_re / pt2_re
qda_ratio_im = qda_im / pt2_re

qda_ratio_re_avg = jk_ls_avg(qda_ratio_re)[:11] # just take the first 11 time slices
qda_ratio_im_avg = jk_ls_avg(qda_ratio_im)[:11]

# %%
#! first plot the data
fig, ax = default_plot()
ax.errorbar(np.arange(11), gv.mean(qda_ratio_re_avg), yerr=gv.sdev(qda_ratio_re_avg), label="QDA ratio real", **ERRORBAR_STYLE)
ax.errorbar(np.arange(11), gv.mean(qda_ratio_im_avg), yerr=gv.sdev(qda_ratio_im_avg), label="QDA ratio imag", **ERRORBAR_STYLE)
ax.legend()
ax.set_xlabel("Time slice")
ax.set_ylabel("QDA ratio")
ax.set_title("QDA ratio data")
ax.grid(True)
plt.show()

# %%
#! then do the fit and print the fit results

##! define priors
priors = gv.BufferDict()
priors["E0"] = gv.gvar(1, 0.5)
priors["log(dE1)"] = gv.gvar(-1, 0.3)
priors["da_re"] = gv.gvar(0, 0.5)
priors["da_im"] = gv.gvar(0, 0.5)
priors["O01_re"] = gv.gvar(0, 0.5)
priors["O01_im"] = gv.gvar(0, 0.5)

priors["re_z0"] = gv.gvar(0, 1)
priors["re_z1"] = gv.gvar(0, 1)
priors["re_z0_"] = gv.gvar(0, 1)
priors["re_z1_"] = gv.gvar(0, 1)

##! define fit functions
def pt2_ss_fcn(pt2_t, p, Lt):
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["re_z0"]
    z1 = p["re_z1"]

    val = z0 ** 2 / (2 * e0) * ( np.exp( -e0 * pt2_t ) + np.exp( -e0 * ( Lt - pt2_t ) ) ) + z1 ** 2 / (2 * e1) * ( np.exp( -e1 * pt2_t ) + np.exp( -e1 * ( Lt - pt2_t ) ) )

    return val

def da_re_fcn(da_t, p, Lt, nstate=2):
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["re_z0"]
    z1 = p["re_z1"]
    
    if nstate == 1:
        val = p["da_re"] * ( np.exp( -e0 * da_t ) + np.exp( -e0 * ( Lt - da_t ) ) )
        
    elif nstate == 2:
        val = z0 / (2 * e0) * p["da_re"] * e0 * ( np.exp( -e0 * da_t ) + np.exp( -e0 * ( Lt - da_t ) ) ) + z1 / (2 * e1) * p["O01_re"] * ( np.exp( -e1 * da_t ) + np.exp( -e1 * ( Lt - da_t ) ) )

    return val

def da_im_fcn(da_t, p, Lt, nstate=2):
    e0 = p["E0"]
    e1 = p["E0"] + p["dE1"]
    z0 = p["re_z0"]
    z1 = p["re_z1"]

    if nstate == 1:
        val = p["da_im"] * ( np.exp( -e0 * da_t ) + np.exp( -e0 * ( Lt - da_t ) ) )
        
    elif nstate == 2:
        val = z0 / (2 * e0) * p["da_im"] * e0 * ( np.exp( -e0 * da_t ) + np.exp( -e0 * ( Lt - da_t ) ) ) + z1 / (2 * e1) * p["O01_im"] * ( np.exp( -e1 * da_t ) + np.exp( -e1 * ( Lt - da_t ) ) )
        
    return val

def joint_fcn(x, p):
    pt2_t = x[0]
    da_t = x[1]
    return {
        "pt2": pt2_ss_fcn(pt2_t, p, Lt),
        "re": da_re_fcn(da_t, p, Lt, nstate=2),
        "im": da_im_fcn(da_t, p, Lt, nstate=2),
    }

##! prepare data for the fit
pt2_trange = np.arange(3, 11)
da_trange = np.arange(3, 11)
fit_pt2 = pt2_avg[pt2_trange]
fit_da_re = qda_re_avg[da_trange]
fit_da_im = qda_im_avg[da_trange]

data_x = [pt2_trange, da_trange]
fit_data = {"pt2": fit_pt2, "re": fit_da_re, "im": fit_da_im}

fit_res = lsf.nonlinear_fit(
    data=(data_x, fit_data), prior=priors, fcn=joint_fcn, maxit=10000,
    svdcut=1e-6,
)

if fit_res.Q < 0.05:
    print(f"\n>>> Bad DA joint fit with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")
else:
    print(f"\n>>> Good DA joint fit with Q = {fit_res.Q:.3f}, Chi2/dof = {fit_res.chi2/fit_res.dof:.3f}")

print(fit_res.format(100))


# %%
#! plot the fit results on data

fit_t = np.linspace(3, 10, 100)
fit_pt2 = pt2_ss_fcn(fit_t, fit_res.p, Lt)
fit_da_re = da_re_fcn(fit_t, fit_res.p, Lt, nstate=2)
fit_da_im = da_im_fcn(fit_t, fit_res.p, Lt, nstate=2)

fit_ratio_re = fit_da_re / fit_pt2
fit_ratio_im = fit_da_im / fit_pt2

fig, ax = default_plot()
ax.errorbar(np.arange(11), gv.mean(qda_ratio_re_avg), yerr=gv.sdev(qda_ratio_re_avg), color=COLOR_CYCLE[0], label="QDA ratio real", **ERRORBAR_STYLE)
ax.errorbar(np.arange(11), gv.mean(qda_ratio_im_avg), yerr=gv.sdev(qda_ratio_im_avg), color=COLOR_CYCLE[1], label="QDA ratio imag", **ERRORBAR_STYLE)

ax.fill_between(fit_t, gv.mean(fit_ratio_re) - gv.sdev(fit_ratio_re), gv.mean(fit_ratio_re) + gv.sdev(fit_ratio_re), color=COLOR_CYCLE[0], alpha=0.5, label="Fit ratio real")
ax.fill_between(fit_t, gv.mean(fit_ratio_im) - gv.sdev(fit_ratio_im), gv.mean(fit_ratio_im) + gv.sdev(fit_ratio_im), color=COLOR_CYCLE[1], alpha=0.5, label="Fit ratio imag")

ax.legend()
ax.set_xlabel("Time slice")
ax.set_ylabel("QDA ratio")
ax.set_title("QDA ratio data")
ax.grid(True)
plt.show()
# %%
