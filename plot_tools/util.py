import numpy as _np
import matplotlib.pyplot as _plt

_plt.style.use('seaborn-white')
_plt.rc('font', size=35)
_plt.rc('axes', labelsize=30)

def gen_axes(extent, ax=None, fmt='{:0.2f}', n_ticks=7, fontsize=25, xlim=None, ylim=None, xlabel=None, ylabel=None, xticklabels=True, yticklabels=True):
    # Build axis object
    if ax is None:
        fig = _plt.figure()
        ax = fig.addsubplot(111)

    # Set scale limits
    if not xlim is None:
        ax.set_xlim(*xlim)

    if not ylim is None:
        ax.set_xlim(*ylim)

    # Set labels

    if not xlabel is None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if not n_ticks is None:
        xticks = _np.linspace(extent[0], extent[1], n_ticks)
        ax.set_xticks(xticks)
        if xticklabels:
            xtickslabels = [fmt.format(xtick) for xtick in xticks]
            ax.set_xticklabels(xtickslabels, fontsize=0.75*fontsize)
        else:
            xtickslabels = ['' for xtick in xticks]
            ax.set_xticklabels(xtickslabels, fontsize=0.75*fontsize)

    if not ylabel is None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if not n_ticks is None:
        yticks = _np.linspace(extent[2], extent[3], n_ticks)
        ax.set_yticks(yticks)
        if yticklabels:
            ytickslabels = [fmt.format(ytick) for ytick in yticks]
            ax.set_yticklabels(ytickslabels, fontsize=0.75*fontsize)
        else:
            ytickslabels = ['' for ytick in yticks]
            ax.set_yticklabels(ytickslabels, fontsize=0.75*fontsize)
