import numpy as _np
from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler

import matplotlib.pyplot as _plt
from matplotlib import patches as _patches
_plt.style.use('seaborn-white')
_plt.rc('font', size=35)
_plt.rc('axes', labelsize=30)


def flux(model, flux, fraction=1.0, pos=None, mu=None, threshold=1e-4, figsize=(15,15), n_pathways=None, ax=None, legend=False, fancy_alpha=False):
    '''Assumes flux is calcualted from PyEMMA'''
    paths, capacities = flux.pathways(fraction)
    A = paths[0][0]
    B = paths[0][-1]

    if pos is None:
        pos = _np.zeros((model.n_macrostates, model.n_features))
        x = 0
        for i in range(model.n_macrostates):
            if flux.committor[i] == 0 or flux.committor[i] == 1:
                pos[i,:] = [flux.committor[i], 0.5]
            else:
                pos[i,:] = [flux.committor[i], x]
                x += 1/(model.n_macrostates-2)

    # Scale stationary distribution to reduce impact from states significantly more populated than others
    if mu is None:
        scale = -1.25*7.5e1/_np.log(model.macrostates.stationary_distribution)[:]
    else:
        scale = -1.25*7.5e1/_np.log(mu)[:]

    # Define opacity scale for pathways
    if fancy_alpha == True:
        F_min = _np.ma.array(flux.net_flux, mask=flux.net_flux==0).min()
        F_max = _np.ma.array(flux.net_flux, mask=flux.net_flux==0).max()
        alpha = _MinMaxScaler(feature_range=(0.5,0.95)).fit(_np.linspace(F_min, F_max, 100)[:,None])
    else:
        alpha = 0.75

    color = _plt.cm.rainbow(_np.linspace(0,1, model.n_macrostates))

    if ax is None:
        fig = _plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    for i in range(model.n_macrostates):
        ax.plot(pos[i,0], pos[i,1], 'o',
                color=color[i], markeredgecolor='k',
                markersize=scale[i], markeredgewidth=1)
        if i == A:
            ax.text(pos[i,0], pos[i,1], 'A',
               horizontalalignment = 'center',
               verticalalignment = 'center',
                fontweight='bold',
               color='w', fontsize=scale[i]-0.1*scale[i])
        elif i == B:
            ax.text(pos[i,0], pos[i,1], 'B',
                   horizontalalignment = 'center',
                   verticalalignment = 'center',
                    fontweight='bold',
                   color='w', fontsize=scale[i]-0.1*scale[i])
        else:
            ax.text(pos[i,0], pos[i,1], str(i),
                   horizontalalignment = 'center',
                   verticalalignment = 'center',
                    fontweight='bold',
                   color='w', fontsize=scale[i]-0.1*scale[i])
        for j in range(model.n_macrostates):
            if not i==j and flux.net_flux[i,j] > threshold:
                #ax.text(_np.mean([pos[i,0], pos[j,0]]), _np.mean([pos[i,1], pos[j,1]]), '{:0.6e}'.format(F.net_flux[i,j]))
                p = _patches.FancyArrowPatch((pos[j,0], pos[j,1]), (pos[i,0], pos[i,1]),
                                            connectionstyle='arc3, rad=0.45',
                                            arrowstyle='wedge',
                                            mutation_scale=5e4*flux.net_flux[i,j],
                                            color='k',
                                            alpha=alpha) #alpha.transform(flux.net_flux[i,j].reshape(-1,1)))
                ax.add_patch(p)

    if n_pathways is None:
        n_pathways = len(paths)
    elif not n_pathways is None and n_pathways > len(paths):
        n_pathways = len(paths)

    path_color = _plt.cm.coolwarm_r(_np.linspace(0, 1, n_pathways))
    for i,path in enumerate(paths[:n_pathways]):
        for j,k in zip(path[:-1], path[1:]):
            if j == path[0]:
                p = _patches.FancyArrowPatch((pos[j,0], pos[j,1]), (pos[k,0], pos[k,1]),
                                            connectionstyle='arc3, rad=-0.45',
                                            mutation_scale=5,
                                            color=path_color[i],
                                            alpha=0.5,
                                            label=(capacities[i]/capacities[0], path))
            else:
                p = _patches.FancyArrowPatch((pos[j,0], pos[j,1]), (pos[k,0], pos[k,1]),
                                            connectionstyle='arc3, rad=-0.45',
                                            mutation_scale=5,
                                            color=path_color[i],
                                            alpha=0.5)
            ax.add_patch(p)
    if legend == True:
        _plt.legend()

def timescale_separation(its, xlim=None):
    if its[0] == _np.inf:
        its = its[1:]
    sep = its[:-1]/its[1:]
    _plt.plot(_np.arange(1,len(sep)+1), sep, '.')
    if not xlim is None:
        _plt.xlim(xlim)

def timescales(x, y=None, yerr=None, tau=None, scale=None):
    if y is None:
        lags = _np.arange(len(x))
        its = x
    else:
        lags = x
        its = y
    if not tau is None:
        _plt.vlines(tau, 0, its.max()+its.max()*0.1)

    _plt.plot(lags, its, color='k')
    _plt.fill_between(lags, _np.zeros(len(lags)), lags)
    if not yerr is None:
        _plt.fill_between(lags, its-yerr, its+yerr, color='k', alpha=0.25)
    if not scale is None:
        _plt.yscale(scale)

def conditional_transition_map(A, B, data, labels, lag=1, bins=100, figsize=(10,5), text_labels=None):
    origin = list(A)
    target = list(B)
    n_sets = len(data)
    if not len(data) == len(labels):
        raise ValueError('Size of data and labels do not match.')

    for i in origin:
        _plt.figure(i+1, figsize=figsize)
        for count,j in enumerate(target):
            data_cat = []
            for n in range(n_sets):
                for t in range(len(data[n])-lag):
                    if labels[n][t] == i and labels[n][t+lag] == j:
                        data_cat.append([data[n][t], data[n][t+lag]])
            data_cat = _np.array(data_cat)
            if not len(data_cat.shape) == 2:
                print('Transition '+str(i)+' to '+str(j)+' not sampled.')
                continue

            his, extent = _np.histogramdd(data_cat, bins=bins)
            amin, amax = min([min(extent[0]), min(extent[1])]), max([max(extent[0]), max(extent[1])])
            extent = [amin, amax, amin, amax]
            P = his.T/his.sum()

            _plt.subplot(1,len(target),count+1)
            _plt.title(r'$P(\chi_i(t+\tau)=$'+str(j)+'$|\chi_i(t)=$'+str(i)+'$)$', fontsize=50*(1/len(target)))
            _plt.contourf(P, 30, cmap=_plt.cm.gray_r, alpha=0.75, extent=extent)
            _plt.contour (P, 10, colors='k', extent=extent)
            _plt.plot([amin,amax], [amin,amax], '--', color='k', linewidth=2, alpha=0.75)

            if not text_labels is None:
                n_labels = len(text_labels.items())
                label_colors = _plt.cm.rainbow(_np.linspace(0,1,n_labels))
                n_label = 0
                for key,val in text_labels.items():
                    _plt.hlines(val[1], amin, amax, linestyles='-', color=label_colors[n_label])
                    _plt.vlines(val[0], amin, amax, linestyles='-', color=label_colors[n_label])
                    if i == origin[-1] and j == target[-1]:
                        _plt.text(0.9*(amax-amin), (n_label+1)*0.05*(amax-amin), key, color=label_colors[n_label], ha='center')
                    n_label += 1

            _plt.xlabel(r'$\chi(t)$')
            _plt.xlim(amin, amax)
            if count+1 == 1:
                _plt.ylabel(r'$\chi(t+\tau)$')
            _plt.ylim(amin, amax)
