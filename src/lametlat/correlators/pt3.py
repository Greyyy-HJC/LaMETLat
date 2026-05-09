"""HDF5 readers for 3-point qTMD correlator data."""

from __future__ import annotations

from os import PathLike

import h5py
import numpy as np

from .resampling import ResamplingMode, apply_resampling, bad_point_filter


def read_pt3_h5(
    path: str | PathLike[str],
    source_sink: str = "SS",
    gamma: str = "T",
    momentum: str = "PX0PY0PZ0",
    b_dir: str = "b_X",
    eta: str = "eta0",
    bT: str | None = None,
    bz: str | None = None,
    resampling: ResamplingMode = "none",
    n_samples: int = 200,
    bin_size: int = 5,
    seed: int | None = 1984,
    threshold: float | None = None,
) -> np.ndarray | dict[str, np.ndarray] | dict[str, dict[str, np.ndarray]]:
    """Read qTMD 3-point datasets from hierarchical HDF5 groups.

    Parameters
    ----------
    path:
        HDF5 file path.
    source_sink:
        Top-level source/sink group, default ``"SS"``.
    gamma:
        Gamma channel group, default ``"T"``.
    momentum:
        Momentum group, default ``"PX0PY0PZ0"``.
    b_dir:
        Wilson-line direction group, default ``"b_X"``.
    eta:
        Eta group, default ``"eta0"``.
    bT:
        Optional bT group selector (e.g. ``"bT0"``). If omitted, all ``bT*`` are returned.
    bz:
        Optional bz dataset selector (e.g. ``"bz0"``). Requires ``bT``.
    resampling:
        Resampling mode: ``"none"``, ``"jk"``, or ``"bs"``.
    n_samples:
        Number of bootstrap samples when ``resampling="bs"``.
    bin_size:
        Optional binning size before jackknife/bootstrap.
    seed:
        Random seed for bootstrap sampling.
    threshold:
        If provided, apply ``bad_point_filter`` before resampling.

    Returns
    -------
    numpy.ndarray or dict
        - If ``bT`` and ``bz`` are both provided: one dataset array.
        - If only ``bT`` is provided: ``dict[bz, array]``.
        - If neither is provided: ``dict[bT, dict[bz, array]]``.
    """
    if bz is not None and bT is None:
        raise ValueError("bT must be provided when bz is specified")

    with h5py.File(path, "r") as h5f:
        base = _require_group(h5f, source_sink, "source_sink")
        base = _require_group(base, gamma, "gamma")
        base = _require_group(base, momentum, "momentum")
        base = _require_group(base, b_dir, "b_dir")
        eta_group = _require_group(base, eta, "eta")

        if bT is None:
            return {
                bt_key: _read_dataset_group(
                    eta_group[bt_key],
                    resampling=resampling,
                    n_samples=n_samples,
                    bin_size=bin_size,
                    seed=seed,
                    threshold=threshold,
                )
                for bt_key in eta_group.keys()
            }

        bt_group = _require_group(eta_group, bT, "bT")
        if bz is None:
            return _read_dataset_group(
                bt_group,
                resampling=resampling,
                n_samples=n_samples,
                bin_size=bin_size,
                seed=seed,
                threshold=threshold,
            )

        if bz not in bt_group:
            raise KeyError(f"bz key not found under {bT!r}: {bz!r}")

        data = np.swapaxes(np.asarray(bt_group[bz]), 0, 1)
        if threshold is not None:
            data = bad_point_filter(data, threshold=threshold)
        return apply_resampling(
            data,
            resampling,
            sample_axis=0,
            n_samples=n_samples,
            bin_size=bin_size,
            seed=seed,
        )


def _read_dataset_group(
    group: h5py.Group,
    *,
    resampling: ResamplingMode,
    n_samples: int,
    bin_size: int,
    seed: int | None,
    threshold: float | None,
) -> dict[str, np.ndarray]:
    """Read all datasets in a group into NumPy arrays."""
    data_by_key: dict[str, np.ndarray] = {}
    for key in group.keys():
        data = np.swapaxes(np.asarray(group[key]), 0, 1)
        if threshold is not None:
            data = bad_point_filter(data, threshold=threshold)
        data_by_key[key] = apply_resampling(
            data,
            resampling,
            sample_axis=0,
            n_samples=n_samples,
            bin_size=bin_size,
            seed=seed,
        )
    return data_by_key


def _require_group(parent: h5py.File | h5py.Group, key: str, level: str) -> h5py.Group:
    """Return required child group or raise a descriptive error."""
    if key not in parent:
        raise KeyError(f"{level} key not found: {key!r}")

    child = parent[key]
    if not isinstance(child, h5py.Group):
        raise KeyError(f"{level} key is not a group: {key!r}")

    return child
