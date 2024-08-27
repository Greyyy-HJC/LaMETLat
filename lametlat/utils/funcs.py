import h5py as h5
import numpy as np
import gvar as gv
import lsqfit as lsf
from scipy import interpolate

from lametlat.utils.resampling import gv_ls_to_samples_corr, bs_ls_avg, jk_ls_avg

def gv_dic_save_to_h5(gv_dic, N_samp, file_path):
    """convert each key of a gvar dictionary to samples, then save the dict to a h5 file

    Args:
        gv_dic (dict): gvar dictionary
        N_samp (int): number of samples
        file_path (str): the path to save the h5 file
    """
    with h5.File(file_path, 'w') as f:
        for key, gv_ls in gv_dic.items():
            f.create_dataset(key, data=gv_ls_to_samples_corr(gv_ls, N_samp))

def constant_fit(data):
    """do a constant fit to the data

    Args:
        data (list): a list of data to do the constant fit

    Returns:
        gvar: the result of the constant fit
    """
    def fcn(x, p):
        return x * 0 + p['const']
    
    priors = gv.BufferDict({'const': gv.gvar(0, 100)})
    x = np.arange(len(data))
    
    fit_res = lsf.nonlinear_fit(data=(x, data), prior=priors, fcn=fcn, maxit=10000, svdcut=1e-100, fitter='scipy_least_squares')
    
    return fit_res.p['const']

def add_error_to_sample(sample_ls, jk_bs="bs"):
    """
    Add error to each sample in the sample list by combining the sample with the correlation matrix.

    Args:
        sample_ls (list): List of bootstrap samples, where each sample is a 1D array-like object.

    Returns:
        list: List of samples with errors, where each sample is a gvar object representing the sample with error.

    """
    if jk_bs == "bs":
        avg = bs_ls_avg(sample_ls)
    elif jk_bs == "jk":
        avg = jk_ls_avg(sample_ls)
    else:
        raise ValueError(f"Invalid jk_bs: {jk_bs}")
    
    cov = gv.evalcov(avg)
    return np.array([gv.gvar(sample, cov) for sample in sample_ls])

def gv_ls_interpolate(x_ls, gv_ls, x_new, N_samp=100, method="cubic"):
    """
    Interpolate a list of gvar objects to a new x list.

    Args:
        x_ls (list): List of x values.
        gv_ls (list): List of gvar objects.
        x_new (list): New x values to interpolate to.
        N_samp (int, optional): Number of samples. Defaults to 100.
        method (str, optional): Interpolation method. Defaults to "cubic".

    Returns:
        gvar: Interpolated gvar object.

    """
    x_array = np.array(x_ls)
    y_ls_samp = gv_ls_to_samples_corr(gv_ls, N_samp)
    
    interp_func = interpolate.interp1d(x_array, y_ls_samp, axis=1, kind=method)
    y_new_samp = interp_func(x_new)
    
    return bs_ls_avg(y_new_samp.T)
