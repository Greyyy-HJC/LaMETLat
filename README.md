# lametlat

**lametlat** is a Python package for analyzing lattice QCD data in the **LaMET** framework: correlators from HDF5 ensembles, bootstrap/jackknife handling, ratios, ground-state fits, and plotting. Fourier extrapolation/transform tools and perturbative matching live in dedicated subpackages and grow with the analysis workflow.

If you previously used the name **`latcorr`**, migrate imports to **`lametlat`** (the PyPI/install name matches the import name).

## Installation

With [uv](https://github.com/astral-sh/uv):

```bash
uv sync --dev
```

Use the project environment when running code:

```bash
uv run python
uv run pytest
```

Equivalently, activate `.venv` and call `python` / `pytest` from there.

With pip:

```bash
python -m pip install -e ".[dev]"
```

## Package layout

| Module | Role |
|--------|------|
| `lametlat.correlators` | HDF5 readers (`read_pt2_h5`, `read_pt3_h5`, `read_qda_h5`), ratio builders, and **resampling** (`bootstrap`, `jackknife`, binning, `gvar` averages) |
| `lametlat.ground_state` | Ground-state fits and related models |
| `lametlat.plotting` | Figures built on top of correlator and fit outputs |
| `lametlat.utils` | Small shared helpers |
| `lametlat.fourier_transform` | Long-distance extrapolation and Fourier transform steps (under active development) |
| `lametlat.perturbative_matching` | Perturbative matching (under active development) |

For a browsable module/function guide with short usage snippets, see the
[API overview](docs/api_overview.html).

Machine-oriented conventions and agent guidance are summarized in [SPEC.md](SPEC.md).

## Minimal usage

```python
import lametlat
from lametlat.correlators import bootstrap, jackknife, read_pt2_h5

print(lametlat.__version__)
```

Resampling helpers are imported from **`lametlat.correlators`** (same namespace as the HDF5 readers).

## Examples

Example data files are generated locally and are not tracked in git. To build
the fake qTMD dataset used by the examples, run:

```bash
python examples/generate_fake_data.py
```

This writes ignored files under `examples/data/`, including `fake_pt2.h5`,
`fake_qda.h5`, `fake_pt3_tsep{4,6,8,10}.h5`, and
`fake_soft_function.npy`. The generator also runs shape checks for the
fixed `z`, `b`, and `tsep` layouts.

## Development

```bash
python -m pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
