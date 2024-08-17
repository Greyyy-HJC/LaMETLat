"""
Settings for visualization.
"""

import matplotlib.pyplot as plt

# Color settings
grey = "#808080"
red = "#FF6F6F"
peach = "#FF9E6F"
orange = "#FFBC6F"
sunkist = "#FFDF6F"
yellow = "#FFEE6F"
lime = "#CBF169"
green = "#5CD25C"
turquoise = "#4AAB89"
blue = "#508EAD"
grape = "#635BB1"
violet = "#7C5AB8"
fuschia = "#C3559F"

color_ls = [
    blue, orange, green, red, violet, fuschia,
    turquoise, grape, lime, peach, sunkist, yellow,
]

# Marker settings
marker_ls = [
    ".", "o", "s", "P", "X", "*", "p", "D",
    "<", ">", "^", "v", "1", "2", "3", "4", "+", "x",
]

# Font settings
font_config = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["Times New Roman"],
}

from matplotlib import rcParams
rcParams.update(font_config)

# Figure size settings
fig_width = 6.75  # in inches, 2x as wide as APS column
gr = 1.618034333  # golden ratio
fig_size = (fig_width, fig_width / gr)

# Default plot axes for general plots
plt_axes = [0.15, 0.15, 0.8, 0.8]  # left, bottom, width, height
fs_p = {"fontsize": 16}  # font size of text, label, ticks
ls_p = {"labelsize": 16}

# Errorbar plot settings
errorb = {
    "markersize": 5,
    "mfc": "none",
    "linestyle": "none",
    "capsize": 3,
    "elinewidth": 1,
}  # circle

# Common used labels
tmin_label = r"$t_{\mathrm{min}}~/~a$"
tmax_label = r"$t_{\mathrm{max}}~/~a$"
tau_center_label = r"$(\tau - t_{\rm{sep}}/2)~/~a$"
tsep_label = r'${t_{\mathrm{sep}}~/~a}$'
z_label = r'${z~/~a}$'
lambda_label = r"$\lambda = z P^z$"
meff_label = r'${m}_{\mathrm{eff}}~/~\mathrm{GeV}$'

def default_plot():
    """
    Create a default plot.

    Returns:
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
    """
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes()
    ax.tick_params(direction="in", top="on", right="on", **ls_p)
    ax.grid(linestyle=":")
    return fig, ax

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = default_plot()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel(tau_center_label, **fs_p)
    ax.set_ylabel(meff_label, **fs_p)
    plt.tight_layout()
    plt.show()