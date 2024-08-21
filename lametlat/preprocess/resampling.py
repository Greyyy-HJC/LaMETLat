"""
Here are functions related to resampling, including bootstrap and jackknife.
You can find an example usage at the end of this file.
"""

import numpy as np
import gvar as gv


def bin_data(data, bin_size, axis=0):
    """Bin the data by averaging every bin_size samples along the specified axis.

    Args:
        data (np.ndarray): Data to be binned.
        bin_size (int): Number of samples per bin.
        axis (int, optional): Axis along which to bin the data. Defaults to 0.

    Returns:
        np.ndarray: Binned data.
    """
    shape = data.shape
    # Calculate the length of the axis after binning
    bin_length = shape[axis] // bin_size
    # Truncate the data to make its length a multiple of bin_size
    truncated_length = bin_length * bin_size
    truncated_data = np.take(data, range(truncated_length), axis=axis)
    new_shape = list(truncated_data.shape)
    new_shape[axis] = bin_length
    new_shape.insert(axis + 1, bin_size)
    binned_data = (
        truncated_data.swapaxes(0, axis).reshape(new_shape).mean(axis=axis + 1)
    )
    return binned_data.swapaxes(0, axis)


def bootstrap(data, samp_times, axis=0, bin=1):
    """Do bootstrap resampling on the data, take random samples from the data and average them.

    Args:
        data (list): Data to be resampled.
        samp_times (int): How many times to sample, i.e., how many bootstrap samples to generate.
        axis (int, optional): Which axis to resample on. Defaults to 0.
        bin (int, optional): Bin size to reduce autocorrelation. Defaults to 1.

    Returns:
        np.ndarray: Bootstrap samples.
    """
    data = np.array(data)

    #* Set the random seed for reproducibility
    np.random.seed(2020)

    # Bin the data to reduce autocorrelation
    if bin > 1:
        data = bin_data(data, bin, axis=axis)

    N_conf = data.shape[axis]
    conf_bs = np.random.choice(N_conf, (samp_times, N_conf), replace=True)
    bs_ls = np.take(data, conf_bs, axis=axis)
    bs_ls = np.mean(bs_ls, axis=axis + 1)

    return bs_ls


def bootstrap_with_seed(data, seed, axis=0):
    """Do bootstrap resampling on the data, take samples as the seed and average them.

    Args:
        data (list): data to be resampled
        seed (list): seed to generate bootstrap samples, shape should be (samp_times, N_conf)
        axis (int, optional): which axis to resample on. Defaults to 0.

    Returns:
        array: bootstrap samples
    """
    data = np.array(data)

    bs_ls = np.take(data, seed, axis=axis)
    bs_ls = np.mean(bs_ls, axis=axis + 1)

    return bs_ls


def jackknife(data, axis=0):
    """Do jackknife resampling on the data, drop one data each time and average the rest.

    Args:
        data (list): data to be resampled
        axis (int, optional): which axis to resample on. Defaults to 0.

    Returns:
        array: jackknife samples
    """
    data = np.array(data)

    N_conf = data.shape[axis]
    temp = np.swapaxes(data, 0, axis)
    conf_jk = [np.delete(temp, i, axis=0) for i in range(N_conf)]
    jk_ls = np.mean(conf_jk, axis=1)

    jk_ls = np.swapaxes(jk_ls, 0, axis)

    return jk_ls


def jk_ls_avg(jk_ls):
    """Average the jackknife list, the axis=0 is the jackknife samples.

    Args:
        jk_ls (list): jackknife samples, can be multi-dimensional
        axis (int, optional): which axis to average on. Defaults to 0.

    Returns:
        gvar list: gvar list after averaging
    """
    jk_ls = np.array(jk_ls)
    shape = np.shape(jk_ls)
    jk_ls = np.reshape(jk_ls, (shape[0], -1))

    N_sample = len(jk_ls)
    mean = np.mean(jk_ls, axis=0)
    cov = np.cov(jk_ls, rowvar=False) * (N_sample - 1)
    gv_ls = gv.gvar(mean, cov)

    return np.reshape(gv_ls, shape[1:])


def jk_dic_avg(dic):
    """Average the jackknife dictionary, the axis=0 of each key is the jackknife samples.

    Args:
        dic (dict): dict of jackknife lists

    Returns:
        dict: dict of gvar list after averaging
    """
    # * length of each key
    l_dic = {key: len(dic[key][0]) for key in dic}

    conf_ls = []
    for n_conf in range(len(dic[key])):
        temp = []
        for key in dic:
            temp.append(list(dic[key][n_conf]))

        conf_ls.append(sum(temp, []))  # * flatten the list

    gv_ls = list(jk_ls_avg(conf_ls))

    gv_dic = {}
    for key in l_dic:
        gv_dic[key] = []
        for i in range(l_dic[key]):
            temp = gv_ls.pop(0)
            gv_dic[key].append(temp)

    return gv_dic


def bs_ls_avg(bs_ls):
    """Average the bootstrap list, the axis=0 is the bootstrap samples.

    Args:
        bs_ls (list): bootstrap samples, can be multi-dimensional
        axis (int, optional): which axis to average on. Defaults to 0.

    Returns:
        gvar list: gvar list after averaging
    """
    bs_ls = np.array(bs_ls)
    shape = np.shape(bs_ls)
    bs_ls = np.reshape(bs_ls, (shape[0], -1))

    mean = np.mean(bs_ls, axis=0)

    # * if only one variable, use standard deviation
    if len(shape) == 1: 
        sdev = np.std(bs_ls, axis=0)
        gv_ls = gv.gvar(mean, sdev)
    else:
        cov = np.cov(bs_ls, rowvar=False)
        gv_ls = gv.gvar(mean, cov)

    return np.reshape(gv_ls, shape[1:])


def bs_dic_avg(dic):
    """Average the bootstrap dictionary, the axis=0 of each key is the bootstrap samples.

    Args:
        dic (dict): dict of bootstrap lists

    Returns:
        dict: dict of gvar list after averaging
    """
    # * length of each key
    key_ls = list(dic.keys())
    l_dic = {key: len(dic[key][0]) for key in key_ls}
    N_conf = len(dic[key_ls[0]])

    conf_ls = []
    for n in range(N_conf):
        temp = []
        for key in dic:
            temp.append(list(dic[key][n]))

        conf_ls.append(sum(temp, []))  # * flatten the list

    gv_ls = list(bs_ls_avg(conf_ls))

    gv_dic = {}
    for key in l_dic:
        gv_dic[key] = []
        for i in range(l_dic[key]):
            temp = gv_ls.pop(0)
            gv_dic[key].append(temp)

    return gv_dic


def gv_ls_to_samples_corr(gv_ls, N_samp):
    """Convert gvar list to gaussian distribution with correlation.

    Args:
        gv_ls (list): gvar list
        N_samp (int): how many samples to generate

    Returns:
        list: samp_ls with one more dimension than gv_ls
    """
    mean = np.array([gv.mean for gv in gv_ls])
    cov = gv.evalcov(gv_ls)
    rng = np.random.default_rng()

    samp_ls = rng.multivariate_normal(mean, cov, size=N_samp)

    return samp_ls


def gv_dic_to_samples_corr(gv_dic, N_samp):
    """Convert each key under the gvar dictionary to gaussian distribution with correlation.

    Args:
        gv_dic (dict): gvar dictionary
        N_samp (int): how many samples to generate

    Returns:
        dict: samp_dic with one more dimension than gv_dic
    """

    # * length of each key
    l_dic = {key: len(gv_dic[key]) for key in gv_dic}

    flatten_ls = []
    for key in gv_dic:
        flatten_ls.append(list(gv_dic[key]))

    flatten_ls = sum(flatten_ls, [])  ## flat

    samp_all = gv_ls_to_samples_corr(flatten_ls, N_samp)
    samp_all = list(np.swapaxes(samp_all, 0, 1))  # shape = len(all), N_samp

    samp_dic = {}
    for key in l_dic:
        samp_ls = []
        for i in range(l_dic[key]):
            temp = samp_all.pop(0)
            samp_ls.append(temp)

        samp_ls = np.swapaxes(np.array(samp_ls), 0, 1)  # shape = N_samp, len(key)
        samp_dic[key] = samp_ls

    return samp_dic
