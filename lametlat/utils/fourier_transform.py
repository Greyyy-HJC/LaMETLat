"""
FT and inverse FT, using the convention in Peskin.
"""
# %%
import numpy as np

# * Without numba is even faster


def sum_ft(x_ls, fx_ls, delta_x, output_k):
    """FT: f(x) -> f(k), coordinate to momentum by discrete sum, produce complex numbers
    the f(x) cannot be gvar list, because of the complex calculation

    Args:
        x_ls (list): x list of f(x)
        fx_ls (list): y list of f(x)
        delta_x (float): the gap between two values in x_ls
        output_k (float): the k value to output f(k)

    Returns:
        float: f(k)
    """
    x_ls = np.array(x_ls)
    fx_ls = np.array(fx_ls)
    val = delta_x / (2 * np.pi) * np.sum(np.exp(1j * x_ls * output_k) * fx_ls)

    return val


def sum_ft_re_im(x_ls, fx_re_ls, fx_im_ls, delta_x, output_k):
    """FT: f(x) -> f(k), coordinate to momentum by discrete sum, produce real and imaginary part separately
    the f(x) can be gvar list

    Args:
        x_ls (list): x list of f(x)
        fx_re_ls (list): y list of the real part of f(x)
        fx_im_ls (list): y list of the imaginary part of f(x)
        delta_x (float): the gap between two values in x_ls
        output_k (float): the k value to output f(k)

    Returns:
        float: f(k) real and imaginary part separately
    """
    x_ls = np.array(x_ls)
    fx_re_ls = np.array(fx_re_ls)
    fx_im_ls = np.array(fx_im_ls)
    val_re = delta_x / (2 * np.pi) * np.sum(
        np.cos(x_ls * output_k) * fx_re_ls
    ) - delta_x / (2 * np.pi) * np.sum(np.sin(x_ls * output_k) * fx_im_ls)
    val_im = delta_x / (2 * np.pi) * np.sum(
        np.sin(x_ls * output_k) * fx_re_ls
    ) + delta_x / (2 * np.pi) * np.sum(np.cos(x_ls * output_k) * fx_im_ls)

    return val_re, val_im


def sum_inv_ft(k_ls, fk_ls, delta_k, output_x):
    """Inverse FT: f(k) -> f(x), momentum to coordinate by discrete sum, produce complex numbers

    Args:
        k_ls (list): k list of f(k)
        fk_ls (list): y list of f(k)
        delta_k (float): the gap between two values in k_ls
        output_x (float): the x value to output f(x)

    Returns:
        float: f(x)
    """
    k_ls = np.array(k_ls)
    fk_ls = np.array(fk_ls)
    val = delta_k * np.sum(np.exp(-1j * k_ls * output_x) * fk_ls)

    return val


def sum_inv_ft_re_im(k_ls, fk_re_ls, fk_im_ls, delta_k, output_x):
    """Inverse FT: f(k) -> f(x), momentum to coordinate by discrete sum, produce real and imaginary part separately

    Args:
        k_ls (list): k list of f(k)
        fk_re_ls (list): y list of the real part of f(k)
        fk_im_ls (list): y list of the imaginary part of f(k)
        delta_k (float): the gap between two values in k_ls
        output_x (float): the x value to output f(x)

    Returns:
        float: f(x) real and imaginary part separately
    """
    k_ls = np.array(k_ls)
    fk_re_ls = np.array(fk_re_ls)
    fk_im_ls = np.array(fk_im_ls)
    val_re = delta_k * np.sum(np.cos(k_ls * output_x) * fk_re_ls) + delta_k * np.sum(
        np.sin(k_ls * output_x) * fk_im_ls
    )
    val_im = delta_k * np.sum(np.cos(k_ls * output_x) * fk_im_ls) - delta_k * np.sum(
        np.sin(k_ls * output_x) * fk_re_ls
    )

    return val_re, val_im

# * From bT to kT space
def two_dim_ft(bT_gev, kT, f_bdep):
    """Two dimensional Fourier transform from bT to kT space.
    
    Args:
        bT_gev (float): bT value in GeV^-1
        kT (float): kT value in GeV
        f_bdep (function): Function that takes bT as input and returns f(bT)
        
    Returns:
        float: f(kT) value
    """
    from scipy.special import j0
    
    # Convert input lists to numpy arrays
    bT_gev = np.array(bT_gev)
    f_bdep = np.array(f_bdep)
    j0_kTbT = j0( kT * bT_gev )
    
    # Calculate the Fourier transform
    delta_bT = abs(bT_gev[1] - bT_gev[0])
    
    # Calculate the Fourier transform
    result = delta_bT / (2 * np.pi) * np.sum( bT_gev * f_bdep * j0_kTbT )
    
    return result

def two_dim_inv_ft(kT_gev, bT, f_kdep):
    """Two dimensional inverse Fourier transform from kT to bT space.
    
    Args:
        kT_gev (float): kT value in GeV
        bT (float): bT value in GeV^-1
        f_kdep (function): Function that takes kT as input and returns f(kT)
        
    Returns:
        float: f(kT) value
    """
    from scipy.special import j0
    
    # Convert input lists to numpy arrays
    kT_gev = np.array(kT_gev)
    f_kdep = np.array(f_kdep)
    j0_kTbT = j0( kT_gev * bT )
    
    # Calculate the Fourier transform
    delta_kT = abs(kT_gev[1] - kT_gev[0])
    
    # Calculate the Fourier transform
    result = delta_kT * (2 * np.pi) * np.sum( kT_gev * f_kdep * j0_kTbT )
    
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from lametlat.utils.plot_settings import *

    import time

    # Example function: Gaussian
    def gaussian(x, mu=0, sigma=1):
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    start_time = time.time()

    # Generate coordinate space data
    x = np.linspace(-10, 10, 1000)
    fx = gaussian(x)
    delta_x = x[1] - x[0]

    # Perform Fourier Transform
    k = np.linspace(-2, 2, 1000)
    fk = np.array([sum_ft(x, fx, delta_x, ki) for ki in k])

    # Perform Inverse Fourier Transform
    delta_k = k[1] - k[0]
    fx_reconstructed = np.array([sum_inv_ft(k, fk, delta_k, xi).real for xi in x])

    # Plotting

    # Coordinate Space
    ax1 = default_plot()[1]
    ax1.plot(x, fx, label="Original", color=color_ls[0])
    ax1.plot(x, fx_reconstructed, "--", label="Reconstructed", color=color_ls[1])
    ax1.set_xlabel("x", **fs_p)
    ax1.set_ylabel("f(x)", **fs_p)
    ax1.legend()
    ax1.set_title("Coordinate Space", **fs_p)

    # Momentum Space
    ax2 = default_plot()[1]
    ax2.plot(k, np.real(fk), label="Real", color=color_ls[2])
    ax2.plot(k, np.imag(fk), label="Imaginary", color=color_ls[3])
    ax2.set_xlabel("k", **fs_p)
    ax2.set_ylabel("F(k)", **fs_p)
    ax2.legend()
    ax2.set_title("Momentum Space", **fs_p)

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
