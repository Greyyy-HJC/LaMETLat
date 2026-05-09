"""Convert gvar containers into correlated Gaussian samples."""

from __future__ import annotations

import gvar as gv
import numpy as np


def gvar_ls_to_samples_corr(gvar_ls: list[gv.GVar], n_samp: int) -> np.ndarray:
    """Draw correlated Gaussian samples from a list of gvar values.

    Parameters
    ----------
    gv_ls:
        Sequence of correlated gvar values.
    n_samp:
        Number of samples to generate.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_samp, len(gv_ls))``.
    """
    if n_samp < 1:
        raise ValueError("n_samp must be a positive integer")
    if not gvar_ls:
        raise ValueError("gv_ls must be non-empty")

    mean = np.asarray([item.mean for item in gvar_ls], dtype=float)
    cov = gv.evalcov(gvar_ls)
    rng = np.random.default_rng()
    return rng.multivariate_normal(mean, cov, size=n_samp)


def gvar_dic_to_samples_corr(gvar_dic: dict[str, list[gv.GVar]], n_samp: int) -> dict[str, np.ndarray]:
    """Draw correlated samples for each entry of a gvar dictionary.

    Parameters
    ----------
    gv_dic:
        Dictionary where each key maps to a list of gvar values.
    n_samp:
        Number of samples to generate.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary with the same keys as ``gv_dic``. Each value has shape
        ``(n_samp, len(gv_dic[key]))``.
    """
    if not gvar_dic:
        raise ValueError("gv_dic must be non-empty")

    key_order = list(gvar_dic.keys())
    lengths = {key: len(gvar_dic[key]) for key in key_order}
    flattened = [item for key in key_order for item in gvar_dic[key]]

    samples_flat = gvar_ls_to_samples_corr(flattened, n_samp)
    samples_by_element = list(np.swapaxes(samples_flat, 0, 1))

    out: dict[str, np.ndarray] = {}
    for key in key_order:
        key_samples = [samples_by_element.pop(0) for _ in range(lengths[key])]
        out[key] = np.swapaxes(np.asarray(key_samples), 0, 1)

    return out