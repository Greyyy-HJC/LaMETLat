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
- **`examples/`**: runnable notebooks-in-disguise scripts (e.g. `# %%` cells). Prefer **top-level** steps and small helpers (constants, `priors()`, plot calls) like [`examples/gsfit.py`](examples/gsfit.py); **avoid** wrapping the whole script in a `main()` unless there is a strong reason.

## Code style

- Prefer **straight-line, readable** code over deep abstraction trees.
- **Avoid** thin helpers whose only job is to call another function one level down; **avoid** long chains of delegation.
- **Avoid** blanket `try`/`except` around normal control flow; use exceptions where failures are truly exceptional and unexpected.
- For internal numerics, prefer letting Python/NumPy raise natural errors; if a check is optional, **do not add explicit exception guards** unless it protects a real user-facing failure mode.
- Default rule: **do not add `ValueError`/manual validation branches** for shape/type/range checks that NumPy/Python will already fail on.
- Keep **numerical** routines separate from plotting when practical.
- Before each commit, check whether public modules/functions changed and update [`docs/api_overview.html`](docs/api_overview.html) if the API overview would otherwise become stale.

See also the human-oriented overview in [README.md](README.md).
