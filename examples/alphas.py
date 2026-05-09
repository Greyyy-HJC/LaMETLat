# %%
from __future__ import annotations

import numpy as np
from lametlat.plotting.plot_settings import FONT_SIZE, default_plot, plt
from lametlat.utils.constants import alphas_nloop

_, ax = default_plot()
mu_range = np.linspace(0.5, 4.0, 100)

for order in (0, 1, 2):
    alpha_s_vals = [alphas_nloop(mu=mu, order=order, Nf=3) for mu in mu_range]
    ax.plot(mu_range, alpha_s_vals, label=f"Order {order}")

ax.set_xlabel(r"$\mu$ [GeV]", **FONT_SIZE)
ax.set_ylabel(r"$\alpha_s(\mu)$", **FONT_SIZE)
ax.legend(loc="upper right", ncol=1)
ax.grid(True)
plt.show()
# %%
