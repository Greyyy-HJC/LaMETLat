"""Call and sanity-check ``lametlat.fourier_transform.core`` (Peskin-sum conventions).

Mirrors the structure of ``lametlat/utils/fourier_transform.py`` ``main``, without plotting.
"""
# %%
from __future__ import annotations

import time

import numpy as np
from lametlat.plotting.plot_settings import *
from lametlat.fourier_transform import core as ft_core
from lametlat.fourier_transform.core import sum_ft, sum_inv_ft


# Example function: Gaussian
def gaussian(x, mu=0, sigma=1):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

start_time = time.time()

# Generate coordinate space data
x = np.linspace(-10, 10, 1000)
fx = gaussian(x)

# Perform Fourier Transform
k = np.linspace(-2, 2, 1000)
fk = np.array([sum_ft(x, fx, ki) for ki in k])

# Perform Inverse Fourier Transform
fx_reconstructed = np.array([sum_inv_ft(k, fk, xi).real for xi in x])

# Plotting

# Coordinate Space
ax1 = default_plot()[1]
ax1.plot(x, fx, label="Original", color=COLOR_CYCLE[0])
ax1.plot(x, fx_reconstructed, "--", label="Reconstructed", color=COLOR_CYCLE[1])
ax1.set_xlabel("x", **FONT_SIZE)
ax1.set_ylabel("f(x)", **FONT_SIZE)
ax1.legend()
ax1.set_title("Coordinate Space", **FONT_SIZE)

# Momentum Space
ax2 = default_plot()[1]
ax2.plot(k, np.real(fk), label="Real", color=COLOR_CYCLE[2])
ax2.plot(k, np.imag(fk), label="Imaginary", color=COLOR_CYCLE[3])
ax2.set_xlabel("k", **FONT_SIZE)
ax2.set_ylabel("F(k)", **FONT_SIZE)
ax2.legend()
ax2.set_title("Momentum Space", **FONT_SIZE)

plt.tight_layout()
plt.show()

end_time = time.time()
total_time = end_time - start_time

print(
    "Maximum difference between original and reconstructed function:",
    np.max(np.abs(fx - fx_reconstructed)),
)
print(f"Total time used: {total_time:.4f} seconds")

# %%
