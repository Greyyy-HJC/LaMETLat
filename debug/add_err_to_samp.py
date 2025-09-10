# %%
from lametlat.utils.resampling import jackknife, bootstrap, jk_ls_avg, bs_ls_avg
from lametlat.utils.funcs import add_error_to_sample
from lametlat.utils.plot_settings import *

import numpy as np
import gvar as gv

if __name__ == "__main__":
    # Generate random array with shape (100, 20)
    samples = np.random.randn(100, 20)
    
    samples_jk = jackknife(samples, axis=0)
    samples_jk_avg = jk_ls_avg(samples_jk)
    samples_add_err = add_error_to_sample(samples_jk, jk_bs="jk")
    
    
    fig, ax = default_plot()
    ax.errorbar(np.arange(20), gv.mean(samples_jk_avg), yerr=gv.sdev(samples_jk_avg), label='Jackknife', marker='o', **errorb)
    ax.errorbar(np.arange(20), gv.mean(samples_add_err[0]), yerr=gv.sdev(samples_add_err[0]), label='Jackknife with error', marker='x', **errorb)
    ax.legend()
    plt.show()
    
    samples_bs, conf_bs = bootstrap(samples, 50, axis=0)
    samples_bs_avg = bs_ls_avg(samples_bs)
    samples_add_err_bs = add_error_to_sample(samples_bs, jk_bs="bs")
    
    fig, ax = default_plot()
    ax.errorbar(np.arange(20), gv.mean(samples_bs_avg), yerr=gv.sdev(samples_bs_avg), label='Bootstrap', marker='s', **errorb)
    ax.errorbar(np.arange(20), gv.mean(samples_add_err_bs[0]), yerr=gv.sdev(samples_add_err_bs[0]), label='Bootstrap with error', marker='d', **errorb)
    ax.legend()
    plt.show()
# %%
