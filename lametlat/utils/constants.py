"""
Constants used in the LaMETLat analysis.
"""
# %%
import numpy as np

GEV_FM = 0.1973269631  # 1 = 0.197 GeV . fm
CF = 4 / 3  # color factor
NF = 3  # number of flavors
CA = 3
TF = 1 / 2

def lat_unit_convert(val, a, Ls, dimension):
    """Convert Lattice unit to GeV / fm.

    Args:
        val (float): The value to be converted.
        a (float): The lattice spacing in fm.
        Ls (int): The lattice size in the space directions.
        dimension (str): 'P'(like P=8), 'M'(like effective mass).
    """
    if dimension == "P":
        #! mom * (2pi * 0.197 / Ls / a)
        return val * 2 * np.pi * GEV_FM / Ls / a  # return in GeV

    elif dimension == "M":
        return val / a * GEV_FM  # return in GeV

    else:
        print("dimension not recognized")
        return None


def beta(order=0, Nf=3):
    if order == 0:
        return 11 / 3 * CA - 4 / 3 * TF * Nf
    elif order == 1:
        return 34 / 3 * CA**2 - (20 / 3 * CA + 4 * CF) * TF * Nf
    elif order == 2:
        return (
            2857 / 54 * CA**3
            + (2 * CF**2 - 205 / 9 * CF * CA - 1415 / 27 * CA**2) * TF * Nf
            + (44 / 9 * CF + 158 / 27 * CA) * TF**2 * Nf**2
        )
    else:
        print(">>> NNNLO beta not coded.")


# n-loop alphas; mu = [GeV]
def alphas_nloop(mu, order=0, Nf=3):
    aS = 0.293 / (4 * np.pi)
    temp = 1 + aS * beta(0, Nf) * np.log((mu / 2) ** 2)

    if order == 0:
        return aS * 4 * np.pi / temp
    elif order == 1:
        return aS * 4 * np.pi / (temp + aS * beta(1, Nf) / beta(0, Nf) * np.log(temp))
    elif order == 2:
        return (
            aS
            * 4
            * np.pi
            / (
                temp
                + aS * beta(1, Nf) / beta(0, Nf) * np.log(temp)
                + aS**2
                * (
                    beta(2, Nf) / beta(0, Nf) * (1 - 1 / temp)
                    + beta(1, Nf) ** 2 / beta(0, Nf) ** 2 * (np.log(temp) / temp + 1 / temp - 1)
                )
            )
        )
    else:
        print("NNNLO running coupling not coded.")

def Lz_func(z_fm, mu=2):
    '''
    This is the log term that commonly appears in the perturbation theory.
    z_fm: z-coordinate in fm
    mu: renormalization scale, GeV
    '''
    z2mu2 = z_fm ** 2 * mu ** 2 / ( GEV_FM ** 2 )
    val = z2mu2 * np.exp(2 * np.euler_gamma) / 4
    
    return np.log(val)

def CG_c0_func_nlo(z_fm, mu=2, pol="unpolarized", asorder=0):
    '''
    This is the c0 function that appears in the SDF.
    asorder: order of alpha_s, 0 for LO, 1 for NLO, 2 for NNLO
    z_fm: z-coordinate in fm
    mu: renormalization scale, GeV
    '''
    alphas = alphas_nloop(mu=mu, order=asorder, Nf=3)
    
    if pol == "unpolarized":
        # gamma t / gamma t gamma 5
        const_term = 1
    elif pol == "helicity":
        # gamma z / gamma z gamma 5
        const_term = 3
    elif pol == "transversity":
        # gamma z gamma y
        const_term = 0
        
    return 1 + alphas * CF / (4 * np.pi) * (const_term - Lz_func(z_fm, mu))

def CG_c0_func_ll(z_fm, mu=2, coeff=1, Nf=3, asorder=0):
    '''
    This is the c0 function that appears in the SDF, after RGR up to LL, see Eq. (B6) in 2504.04625
    z_fm: z-coordinate in fm
    mu: renormalization scale, GeV
    coeff: coefficient of the initial scale mu0, can vary from 0.8 to 1.2
    Nf: number of flavors
    '''
    mu0 = 2 * coeff * np.exp(- np.euler_gamma) / z_fm * GEV_FM # GeV
    a0 = alphas_nloop(mu=mu0, order=asorder, Nf=Nf)
    a1 = alphas_nloop(mu=mu, order=asorder, Nf=Nf)
    
    temp = CF / beta(0, Nf) * np.log( a1 / a0 )
    
    return np.exp(temp) #! Since the log term can be large, we should use exp(temp) instead of 1 + temp for resummed results.

# %%
if __name__ == "__main__":
    from lametlat.utils.plot_settings import *
    
    fig, ax = default_plot()
    
    mu_range = np.linspace(0.5, 4, 100)
    
    for order in [0,1,2]:
        alphas = [alphas_nloop(mu=mu, order=order, Nf=3) for mu in mu_range]
        ax.plot(mu_range, alphas, label=f'Order {order}')
    
    ax.set_xlabel(r'$\mu$ [GeV]', **fs_p)
    ax.set_ylabel(r'$\alpha_s(\mu)$', **fs_p)
    ax.legend(loc="upper right", ncol=1, **fs_small_p)
    ax.grid(True)
    plt.show()
# %%
