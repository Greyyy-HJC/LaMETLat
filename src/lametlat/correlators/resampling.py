"""Bootstrap, jackknife, and reader-side resampling for correlator data."""

from __future__ import annotations

from typing import Literal

import gvar as gv
import numpy as np

ResamplingMode = Literal["none", "jk", "bs"]


def bad_point_filter(data: np.ndarray, threshold: float = 1) -> np.ndarray:
    """Replace entries with absolute value above threshold by random signs."""
    filtered = np.array(data, copy=True)
    mask = np.abs(filtered) > threshold
    bad_loc = np.argwhere(mask)

    for loc in bad_loc:
        filtered[tuple(loc)] = np.random.choice([-1, 1])

    return filtered


def bin_data(data: np.ndarray, bin_size: int, axis: int = 0) -> np.ndarray:
    """Average adjacent configurations into bins.

    Parameters
    ----------
    data:
        Input ensemble data.
    bin_size:
        Number of configurations per bin.
    axis:
        Configuration axis.

    Returns
    -------
    numpy.ndarray
        Binned data.
    """
    data = np.asarray(data)

    if bin_size < 1:
        raise ValueError("bin_size must be a positive integer")

    data = np.moveaxis(data, axis, 0)
    n_bins = data.shape[0] // bin_size
    data = data[: n_bins * bin_size]
    data = data.reshape(n_bins, bin_size, *data.shape[1:]).mean(axis=1)

    return np.moveaxis(data, 0, axis)


def bootstrap(
    data: np.ndarray,
    n_samples: int,
    sample_size: int | None = None,
    axis: int = 0,
    bin_size: int = 1,
    seed: int | None = 1984,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate bootstrap samples from ensemble data.

    Parameters
    ----------
    data:
        Input ensemble data.
    n_samples:
        Number of bootstrap samples to generate.
    sample_size:
        Number of configurations drawn per bootstrap sample. Defaults to the
        number of configurations.
    axis:
        Configuration axis.
    bin_size:
        Optional bin size applied before resampling.
    seed:
        Random seed for reproducible sampling.
    return_indices:
        Whether to return the sampled configuration indices with the samples.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, numpy.ndarray]
        Bootstrap sample averages, optionally with sampled indices.
    """
    data = np.asarray(data)

    if bin_size > 1:
        data = bin_data(data, bin_size, axis=axis)

    n_conf = data.shape[axis]

    if sample_size is None:
        sample_size = n_conf

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_conf, (n_samples, sample_size), replace=True)
    samples = np.take(data, indices, axis=axis).mean(axis=axis + 1)

    if return_indices:
        return samples, indices

    return samples


def jackknife(data: np.ndarray, axis: int = 0, bin_size: int = 1) -> np.ndarray:
    """Generate leave-one-bin-out jackknife samples.

    Parameters
    ----------
    data:
        Input ensemble data.
    axis:
        Configuration axis.
    bin_size:
        Optional bin size applied before jackknife resampling.

    Returns
    -------
    numpy.ndarray
        Jackknife sample averages.
    """
    data = np.asarray(data)

    if bin_size > 1:
        data = bin_data(data, bin_size, axis=axis)

    n_conf = data.shape[axis]

    if n_conf < 2:
        raise ValueError("jackknife needs at least two samples")

    total = data.sum(axis=axis, keepdims=True)

    return (total - data) / (n_conf - 1)


def apply_resampling(
    data: np.ndarray,
    mode: ResamplingMode = "none",
    *,
    sample_axis: int = 0,
    n_samples: int = 200,
    bin_size: int = 5,
    seed: int | None = 1984,
) -> np.ndarray:
    """Apply optional resampling on a correlator array."""
    data = np.asarray(data)
    if mode == "none":
        return data
    if mode == "jk":
        return jackknife(data, axis=sample_axis, bin_size=bin_size)
    if mode == "bs":
        return bootstrap(
            data,
            n_samples=n_samples,
            axis=sample_axis,
            bin_size=bin_size,
            seed=seed,
        )
    raise ValueError(f"unsupported resampling mode: {mode!r}")


def jk_ls_avg(jk_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average jackknife samples into gvar values."""
    jk_arr = np.asarray(jk_ls)
    assert np.isrealobj(jk_arr), "jk_ls must contain real-valued samples"
    if axis != 0:
        jk_arr = np.swapaxes(jk_arr, 0, axis)

    shape = jk_arr.shape
    jk_flat = jk_arr.reshape(shape[0], -1)
    n_sample = jk_flat.shape[0]
    mean = np.mean(jk_flat, axis=0)

    if jk_flat.shape[1] == 1:
        sdev = np.std(jk_flat, axis=0) * np.sqrt(n_sample - 1)
        return gv.gvar(mean, sdev)

    cov = np.cov(jk_flat, rowvar=False) * (n_sample - 1)
    out = gv.gvar(mean, cov)
    return out.reshape(shape[1:])


def bs_ls_avg(bs_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average bootstrap samples into gvar values."""
    bs_arr = np.asarray(bs_ls)
    assert np.isrealobj(bs_arr), "bs_ls must contain real-valued samples"
    if axis != 0:
        bs_arr = np.swapaxes(bs_arr, 0, axis)

    shape = bs_arr.shape
    bs_flat = bs_arr.reshape(shape[0], -1)
    mean = np.mean(bs_flat, axis=0)

    if bs_flat.shape[1] == 1:
        sdev = np.std(bs_flat, axis=0)
        return gv.gvar(mean, sdev)

    cov = np.cov(bs_flat, rowvar=False)
    out = gv.gvar(mean, cov)
    return out.reshape(shape[1:])


def jk_dict_avg(data: dict[str, np.ndarray]) -> dict[str, list[gv.GVar]]:
    """Average a dict of jackknife arrays into a dict of gvar lists."""
    key_order = list(data.keys())
    lengths = {key: len(data[key][0]) for key in key_order}
    n_sample = len(data[key_order[0]])

    merged: list[list[float]] = []
    for idx in range(n_sample):
        row: list[float] = []
        for key in key_order:
            row.extend(list(data[key][idx]))
        merged.append(row)

    gv_ls = list(jk_ls_avg(np.asarray(merged)))
    out: dict[str, list[gv.GVar]] = {}
    for key in key_order:
        out[key] = [gv_ls.pop(0) for _ in range(lengths[key])]
    return out


def bs_dict_avg(data: dict[str, np.ndarray]) -> dict[str, list[gv.GVar]]:
    """Average a dict of bootstrap arrays into a dict of gvar lists."""
    key_order = list(data.keys())
    lengths = {key: len(data[key][0]) for key in key_order}
    n_sample = len(data[key_order[0]])

    merged: list[list[float]] = []
    for idx in range(n_sample):
        row: list[float] = []
        for key in key_order:
            row.extend(list(data[key][idx]))
        merged.append(row)

    gv_ls = list(bs_ls_avg(np.asarray(merged)))
    out: dict[str, list[gv.GVar]] = {}
    for key in key_order:
        out[key] = [gv_ls.pop(0) for _ in range(lengths[key])]
    return out
