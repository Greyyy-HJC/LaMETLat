"""End-to-end ground-state fitting example using the fake HDF5 data."""
# %%
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np

from lametlat.correlators import (  # noqa: E402
    bs_dict_avg,
    bs_ls_avg,
    get_fh_data,
    get_pt3_ratio_data,
    get_qda_ratio_data,
    read_pt2_h5,
    read_pt3_h5,
    read_qda_h5,
)
from lametlat.ground_state import fh_fit, pt2_fit, pt3_ratio_fit, qda_fit  # noqa: E402
from lametlat.plotting import fh_plot, pt2_plot, pt3_ratio_plot, qda_ratio_plot  # noqa: E402
from lametlat.utils.logger import setup_logger  # noqa: E402


def priors() -> gv.BufferDict:
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(0.45, 0.12)
    prior["log(dE1)"] = gv.gvar(np.log(0.55), 0.35)
    prior["z0"] = gv.gvar(1.10, 0.30)
    prior["z1"] = gv.gvar(0.45, 0.25)
    return prior


def pt3_priors() -> gv.BufferDict:
    prior = priors()
    prior["O00_re"] = gv.gvar(0.25, 0.30)
    prior["O01_re"] = gv.gvar(0.0, 0.50)
    prior["O11_re"] = gv.gvar(0.0, 0.50)
    prior["O00_im"] = gv.gvar(0.02, 0.10)
    prior["O01_im"] = gv.gvar(0.0, 0.20)
    prior["O11_im"] = gv.gvar(0.0, 0.20)
    return prior


def qda_priors() -> gv.BufferDict:
    prior = priors()
    prior["O00_re"] = gv.gvar(3.0e9, 3.0e9)
    prior["O01_re"] = gv.gvar(-8.0e9, 8.0e9)
    prior["O00_im"] = gv.gvar(0.0, 1.0e8)
    prior["O01_im"] = gv.gvar(0.0, 1.0e8)
    return prior


def fh_priors() -> gv.BufferDict:
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(0.45, 0.12)
    prior["O00_re"] = gv.gvar(0.25, 0.30)
    prior["O00_im"] = gv.gvar(0.02, 0.10)
    return prior


DATA_DIR = Path(__file__).resolve().parent / "data"
PLOT_DIR = Path(__file__).resolve().parent / "plots" / "gsfit"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
setup_logger(PLOT_DIR / "gsfit.log")

LT = 32
N_BOOT = 96
BOOT_SEED = 1984
TSEP_FILES = {
    4: DATA_DIR / "fake_pt3_tsep4.h5",
    6: DATA_DIR / "fake_pt3_tsep6.h5",
    8: DATA_DIR / "fake_pt3_tsep8.h5",
    10: DATA_DIR / "fake_pt3_tsep10.h5",
}

pt2 = read_pt2_h5(
    DATA_DIR / "fake_pt2.h5",
    source_sink="SS",
    gamma="5",
    momentum="PX0PY0PZ0",
    resampling="bs",
    n_samples=N_BOOT,
    seed=BOOT_SEED,
)
qda = read_qda_h5(
    DATA_DIR / "fake_qda.h5",
    bT="bT0",
    bz="bz0",
    resampling="bs",
    n_samples=N_BOOT,
    seed=BOOT_SEED,
)
pt3 = {
    tsep: read_pt3_h5(
        path,
        bT="bT0",
        bz="bz0",
        resampling="bs",
        n_samples=N_BOOT,
        seed=BOOT_SEED,
    )
    for tsep, path in TSEP_FILES.items()
}

pt2_gv = bs_ls_avg(np.asarray(pt2.real, dtype=float), axis=0)
qda_real_gv = bs_ls_avg(np.asarray(qda.real, dtype=float), axis=0)
qda_imag_gv = bs_ls_avg(np.asarray(qda.imag, dtype=float), axis=0)

qda_ratio_real, qda_ratio_imag = get_qda_ratio_data(
    pt2.real,
    pt2.imag,
    qda.real,
    qda.imag,
)
qda_ratio_real_gv = bs_ls_avg(np.asarray(qda_ratio_real, dtype=float), axis=0)
qda_ratio_imag_gv = bs_ls_avg(np.asarray(qda_ratio_imag, dtype=float), axis=0)

pt3_ratio_real, pt3_ratio_imag = get_pt3_ratio_data(
    pt2.real,
    pt2.imag,
    {tsep: data.real for tsep, data in pt3.items()},
    {tsep: data.imag for tsep, data in pt3.items()},
)
pt3_ratio_real_gv = bs_dict_avg(pt3_ratio_real)
pt3_ratio_imag_gv = bs_dict_avg(pt3_ratio_imag)
tau_dict = {tsep: np.arange(tsep + 2) for tsep in TSEP_FILES}

fh_real, fh_imag = get_fh_data(
    pt2.real,
    pt2.imag,
    {tsep: data.real for tsep, data in pt3.items()},
    {tsep: data.imag for tsep, data in pt3.items()},
    tau_cut=1,
)
fh_real_gv = bs_ls_avg(np.asarray(fh_real, dtype=float), axis=0)
fh_imag_gv = bs_ls_avg(np.asarray(fh_imag, dtype=float), axis=0)
tsep_ls = sorted(TSEP_FILES)
fh_tsep_ls = tsep_ls[:-1]

pt2_fit_res = pt2_fit(
    pt2_gv,
    tmin=5,
    tmax=14,
    Lt=LT,
    prior=priors(),
    label="2-state fit",
)
pt3_fit_res = pt3_ratio_fit(
    tsep_ls=[6, 8, 10],
    tau_cut=3,
    ratio_real=pt3_ratio_real_gv,
    ratio_imag=pt3_ratio_imag_gv,
    Lt=LT,
    prior=pt3_priors(),
    pt2_fit_res=pt2_fit_res,
    label="3pt ratio fit",
)
qda_fit_res = qda_fit(
    qda_real_gv,
    qda_imag_gv,
    5,
    12,
    LT,
    prior=qda_priors(),
    label="qDA fit",
    part="re",
)
fh_fit_res = fh_fit(
    fh_real_gv,
    fh_imag_gv,
    tsep_ls,
    0,
    nstate=1,
    prior=fh_priors(),
    pt2_fit_res=pt2_fit_res,
    label="FH fit",
    dt=2,
)

(fig_pt2, _), (fig_meff, _) = pt2_plot(
    [pt2_gv],
    boundary="periodic",
    trange=(1, 18),
    fit_results=pt2_fit_res,
    fit_tmin=5,
    fit_tmax=14,
    Lt=LT,
    fit_label="2-state fit",
    save_path=PLOT_DIR / "pt2",
)
plt.close(fig_pt2)
plt.close(fig_meff)

(fig_qda_re, _), (fig_qda_im, _) = qda_ratio_plot(
    np.arange(13),
    qda_ratio_real_gv[:13],
    qda_ratio_imag_gv[:13],
    fit_result=qda_fit_res,
    pt2_fit_result=pt2_fit_res,
    fit_trange=np.arange(5, 12),
    fit_label="qDA fit",
    Lt=LT,
    id_label={"fake": "qDA", "bT": 0, "bz": 0},
    save_path=PLOT_DIR / "qda_ratio",
)
plt.close(fig_qda_re)
plt.close(fig_qda_im)

(fig_pt3_re, _), (fig_pt3_im, _) = pt3_ratio_plot(
    tau_dict,
    pt3_ratio_real_gv,
    pt3_ratio_imag_gv,
    fit_result=pt3_fit_res,
    fit_tsep_ls=tsep_ls,
    fit_tau_cut=1,
    fit_label="3pt ratio fit",
    Lt=LT,
    save_path=PLOT_DIR / "pt3_ratio",
)
plt.close(fig_pt3_re)
plt.close(fig_pt3_im)

(fig_fh_re, _), (fig_fh_im, _) = fh_plot(
    fh_tsep_ls,
    fh_real_gv,
    fh_imag_gv,
    fit_result=fh_fit_res,
    fit_tsep_ls=fh_tsep_ls,
    fit_tau_cut=1,
    fit_label="FH fit",
    dt=2,
    save_path=PLOT_DIR / "fh",
)
plt.close(fig_fh_re)
plt.close(fig_fh_im)

print("Read bootstrap shapes:")
print(f"  pt2: {pt2.shape}")
print(f"  qda: {qda.shape}")
print(f"  pt3: { {tsep: data.shape for tsep, data in pt3.items()} }")
print("2pt fit:")
print(f"  Q = {pt2_fit_res.Q:.3f}")
print(f"  chi2/dof = {pt2_fit_res.chi2 / pt2_fit_res.dof:.3f}")
print("3pt ratio fit:")
print(f"  Q = {pt3_fit_res.Q:.3f}")
print(f"  chi2/dof = {pt3_fit_res.chi2 / pt3_fit_res.dof:.3f}")
print("qDA fit:")
print(f"  Q = {qda_fit_res.Q:.3f}")
print(f"  chi2/dof = {qda_fit_res.chi2 / qda_fit_res.dof:.3f}")
print("FH fit:")
print(f"  Q = {fh_fit_res.Q:.3f}")
print(f"  chi2/dof = {fh_fit_res.chi2 / fh_fit_res.dof:.3f}")
print(f"Wrote plots to {PLOT_DIR}")

# %%
