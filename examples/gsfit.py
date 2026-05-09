# %%
from __future__ import annotations

import gvar as gv
import matplotlib.pyplot as plt
import numpy as np

from lametlat.correlators import (  # noqa: E402
    get_pt3_ratio_data,
    get_qda_ratio_data,
    read_pt2_h5,
    read_pt3_h5,
    read_qda_h5,
)
from lametlat.ground_state import pt2_fit  # noqa: E402
from lametlat.plotting import pt2_plot, pt3_ratio_plot, qda_ratio_plot  # noqa: E402
from lametlat.correlators import bs_dict_avg, bs_ls_avg  # noqa: E402


def priors() -> gv.BufferDict:
    prior = gv.BufferDict()
    prior["E0"] = gv.gvar(0.45, 0.12)
    prior["log(dE1)"] = gv.gvar(np.log(0.55), 0.35)
    prior["z0"] = gv.gvar(1.10, 0.30)
    prior["z1"] = gv.gvar(0.45, 0.25)
    return prior

LT = 32
N_BOOT = 96
BOOT_SEED = 1984
TSEP_FILES = {
    4: "data/fake_pt3_tsep4.h5",
    6: "data/fake_pt3_tsep6.h5",
    8: "data/fake_pt3_tsep8.h5",
    10: "data/fake_pt3_tsep10.h5",
}

pt2 = read_pt2_h5(
    "data/fake_pt2.h5",
    source_sink="SS",
    gamma="5",
    momentum="PX0PY0PZ0",
    resampling="bs",
    n_samples=N_BOOT,
    seed=BOOT_SEED,
)
qda = read_qda_h5(
    "data/fake_qda.h5",
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

pt2_fit_res = pt2_fit(
    pt2_gv,
    3,
    14,
    LT,
    prior=priors(),
    label="2-state fit",
)

pt2_plot(
    [pt2_gv],
    boundary="periodic",
    trange=(1, 18),
    fit_results=pt2_fit_res,
    fit_tmin=3,
    fit_tmax=14,
    Lt=LT,
    fit_label="2-state fit",
    show=True,
)

qda_ratio_plot(
    np.arange(13),
    qda_ratio_real_gv[:13],
    qda_ratio_imag_gv[:13],
    id_label={"fake": "qDA", "bT": 0, "bz": 0},
    show=True,
)

pt3_ratio_plot(
    tau_dict,
    pt3_ratio_real_gv,
    pt3_ratio_imag_gv,
    show=True,
)

print("Read bootstrap shapes:")
print(f"  pt2: {pt2.shape}")
print(f"  qda: {qda.shape}")
print(f"  pt3: { {tsep: data.shape for tsep, data in pt3.items()} }")
print("2pt fit:")
print(f"  Q = {pt2_fit_res.Q:.3f}")
print(f"  chi2/dof = {pt2_fit_res.chi2 / pt2_fit_res.dof:.3f}")
print(f"Wrote plots to plots/gsfit")

# %%
