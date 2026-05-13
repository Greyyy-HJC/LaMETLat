"""Bootstrap, jackknife, and reader-side resampling for correlator data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import gvar as gv
import numpy as np
from scipy.interpolate import interp1d

ResamplingMode = Literal["none", "jk", "bs"]
SampleCovarianceMode = Literal["jk", "bs"]


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


def bs_ls_avg_percentile(bs_ls: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average bootstrap samples into gvar values without cross-correlations.

    The central value is the per-entry median across samples and the symmetric
    error is half the 16-84 percentile range, i.e. the 1-sigma width of a
    normal distribution. No covariance is propagated between entries.

    Parameters
    ----------
    bs_ls:
        Bootstrap samples.
    axis:
        Axis indexing independent bootstrap samples.

    Returns
    -------
    numpy.ndarray
        ``gvar`` array shaped like ``bs_ls`` with ``axis`` removed.
    """
    bs_arr = np.asarray(bs_ls)
    assert np.isrealobj(bs_arr), "bs_ls must contain real-valued samples"
    if axis != 0:
        bs_arr = np.swapaxes(bs_arr, 0, axis)

    shape = bs_arr.shape
    bs_flat = bs_arr.reshape(shape[0], -1)

    mid = np.median(bs_flat, axis=0)
    p16, p84 = np.percentile(bs_flat, [16, 84], axis=0)
    sdev = 0.5 * (p84 - p16)

    out = gv.gvar(mid, sdev)
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


def add_error_to_sample(
    samples: np.ndarray,
    mode: SampleCovarianceMode = "bs",
    *,
    axis: int = 0,
) -> np.ndarray:
    """Attach the covariance from jackknife or bootstrap averages to each resample.

    The covariance is taken from ``gv.evalcov`` applied to the ensemble average
    computed by ``jk_ls_avg`` or ``bs_ls_avg``; each independent resample slice
    is turned into a ``gvar`` array with that shared covariance structure.

    Parameters
    ----------
    samples:
        Jackknife or bootstrap samples (same layout as for ``jk_ls_avg`` /
        ``bs_ls_avg``).
    mode:
        Use jackknife (``"jk"``) or bootstrap (``"bs"``) covariance conventions.
    axis:
        Axis indexing independent resamples.

    Returns
    -------
    numpy.ndarray
        Same shape as ``samples``; each slice along ``axis`` is a ``gvar`` array
        built with the shared covariance from ``gv.evalcov(avg)``.
    """
    arr = np.asarray(samples)
    if axis != 0:
        arr = np.swapaxes(arr, 0, axis)

    if mode == "bs":
        avg = bs_ls_avg(arr, axis=0)
    elif mode == "jk":
        avg = jk_ls_avg(arr, axis=0)
    else:
        raise ValueError(f"unsupported sample covariance mode: {mode!r}")

    cov = gv.evalcov(avg)
    out = np.array([gv.gvar(row, cov) for row in arr])

    if axis != 0:
        out = np.swapaxes(out, 0, axis)
    return out


def gvar_ls_interpolate(
    x_ls: Sequence[float] | np.ndarray,
    gv_ls: list[gv.GVar],
    x_ls_new: Sequence[float] | np.ndarray,
    *,
    n_samp: int | None = None,
    kind: str = "cubic",
) -> np.ndarray:
    """Interpolate ``gvar`` means and standard deviations to new coordinates.

    This is an envelope interpolation: the central values and one-sigma widths
    are interpolated independently, and the result is rebuilt with
    :func:`gvar.gvar`. Cross-correlations from the input are not propagated.

    Parameters
    ----------
    x_ls:
        Original coordinates; must match ``gv_ls`` in length.
    gv_ls:
        Correlated ``gvar`` values at ``x_ls``.
    x_ls_new:
        Target coordinates (scalar or 1-d; shape of the interpolator output).
    n_samp:
        Deprecated compatibility argument. Envelope interpolation is
        deterministic and does not draw samples, so this value is ignored.
    kind:
        Spline order passed to :class:`scipy.interpolate.interp1d` (e.g.
        ``"linear"``, ``"cubic"``).

    Returns
    -------
    numpy.ndarray
        ``gvar`` array evaluated at ``x_ls_new``, shaped like ``numpy.asarray(x_ls_new)``.
    """
    x_array = np.asarray(x_ls, dtype=float)
    x_new = np.asarray(x_ls_new, dtype=float)
    y_mean = np.asarray(gv.mean(gv_ls), dtype=float)
    y_sdev = np.asarray(gv.sdev(gv_ls), dtype=float)

    mean_interp = interp1d(x_array, y_mean, kind=kind)(x_new)
    sdev_interp = interp1d(x_array, y_sdev, kind=kind)(x_new)
    return gv.gvar(mean_interp, sdev_interp)


def sample_ls_interpolate(
    x_ls: Sequence[float] | np.ndarray,
    sample_ls: np.ndarray,
    x_ls_new: Sequence[float] | np.ndarray,
    *,
    kind: str = "cubic",
) -> np.ndarray:
    """Interpolate each sample in a sample list to new coordinates.

    Parameters
    ----------
    x_ls:
        Original coordinates; must match the length of each sample.
    sample_ls:
        Sample list with shape ``(n_sample, len(x_ls))``.
    x_ls_new:
        Target coordinates (scalar or 1-d; shape of the interpolator output).
    kind:
        Spline order passed to :class:`scipy.interpolate.interp1d` (e.g.
        ``"linear"``, ``"cubic"``).

    Returns
    -------
    numpy.ndarray
        Interpolated sample list with shape
        ``(n_sample, *numpy.asarray(x_ls_new).shape)``.
    """
    x_array = np.asarray(x_ls, dtype=float)
    x_new = np.asarray(x_ls_new, dtype=float)
    samples = np.asarray(sample_ls, dtype=float)

    if samples.ndim != 2:
        raise ValueError("sample_ls must be a 2-d array shaped as (n_sample, n_x)")
    if samples.shape[1] != x_array.shape[0]:
        raise ValueError("the length of x_ls must match sample_ls.shape[1]")

    return interp1d(x_array, samples, kind=kind, axis=1)(x_new)
