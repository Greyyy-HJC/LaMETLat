# lametlat — project specification

## What this is

**lametlat** is a Python package for **lattice QCD data analysis** in the **LaMET** (Large Momentum Effective Theory) setting: correlators from ensembles, fits, ratios, and downstream steps toward physical observables.

This repository is the library code; experiment-specific preprocessing pipelines live outside this package.

## Layout

- **`lametlat.correlators`**: reading correlators from HDF5 (and similar), building ratios and combined observables, and **resampling** (`bootstrap`, `jackknife`, binning, `gvar` aggregation) in `correlators/resampling.py`.
- **`lametlat.ground_state`**: ground-state extraction and fit helpers.
- **`lametlat.plotting`**: plotting helpers that stay separate from core numerics.
- **`lametlat.utils`**: small shared utilities (logging, converters, etc.).
- **`lametlat.fourier_transform`**: long-distance extrapolation and Fourier-transform steps (expand here as algorithms land).
- **`lametlat.perturbative_matching`**: perturbative matching utilities (expand here as needed).

## Code style

- Prefer **straight-line, readable** code over deep abstraction trees.
- **Avoid** thin helpers whose only job is to call another function one level down; **avoid** long chains of delegation.
- **Avoid** blanket `try`/`except` around normal control flow; use exceptions where failures are truly exceptional and unexpected.
- Keep **numerical** routines separate from plotting when practical.

See also the human-oriented overview in [README.md](README.md).
