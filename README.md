# LaMETLat

Python package for lattice calculations in Large Momentum Effective Theory (LaMET). This package provides tools for lattice QCD data analysis, focusing on LaMET applications.

## Philosophy

- Separate evaluation and visualization:
  - Use `.py` files for evaluation
  - Use `.ipynb` files for visualization
- Decompose large functions into smaller, more manageable functions
- Save numerical cache for plotting and future evaluations
- Implement standard logging and cache output

## Features

- Preprocessing of lattice QCD data
- Fitting tools for Ground State Extraction (GSE)
- Extrapolation methods for lattice data
- Renormalization techniques for LaMET
- Utilities for data manipulation and analysis
- Bootstrap resampling for statistical error estimation

## Installation

```bash
# Clone the repository
git clone https://github.com/heji-the/LaMETLat.git
cd LaMETLat

# Install dependencies and the package
bash init.sh
```

## Usage

Check out examples in the `examples/` directory, which includes:
- `bootstrap.ipynb`: Bootstrap resampling methods
- `lanczos.ipynb`: Lanczos algorithm implementation
- `pdf_self_renorm.ipynb`: Self-renormalization techniques for PDFs

Basic usage:
```python
import lametlat as lml

# Use the preprocessing tools
# Example usage will depend on your specific needs
```

## Package Structure

- `lametlat/preprocess/`: Data preprocessing tools
- `lametlat/gsfit/`: Ground state fitting algorithms, including Lanczos
- `lametlat/extrapolation/`: Extrapolation methods
- `lametlat/renormalization/`: Renormalization techniques
- `lametlat/utils/`: Utility functions
- `examples/`: Jupyter notebooks examples of using the package
- `example_plots/`: Examples plots from the [pion TMD paper](https://arxiv.org/pdf/2504.04625)
- `debug/`: Debug tests

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

```
hejinchen17@gmail.com
```