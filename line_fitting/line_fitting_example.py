import getdist.plots
import numpy
import line_fitting.plot_settings
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.backends.backend_pdf import PdfPages
from line_fitting import data
from line_fitting import polychord
from line_fitting import fgivenx
from mpi4py import MPI

#%matplotlib
#%load_ext autoreload
#%autoreload 2

w = 4
h = w/4*3
figsize = (h,h)


fig, ax = plt.subplots(figsize=figsize)
data.plot_points(ax,errors=False)
data.label_axes(ax)
data.plot_function(ax,lambda x: data.f(x,[0.7,1]))
data.plot_function(ax,lambda x: data.f(x,[0.8,0,1]))
fig.tight_layout()
fig.savefig('../figures/data_points.pdf')

fig, ax = plt.subplots(figsize=figsize)
data.plot_diff(ax,lambda x: data.f(x,[0.7,1]))
data.label_axes(ax)
fig.tight_layout()
fig.savefig('../figures/data_diff.pdf')

fig, ax = plt.subplots(figsize=figsize)
data.plot_diff(ax,lambda x: data.f(x,[0.9,0,1]),1)
data.label_axes(ax)
fig.tight_layout()
fig.savefig('../figures/data_diff_1.pdf')

fig, ax = plt.subplots(figsize=figsize)
data.plot_diff(ax,lambda x: data.f(x,[0.9,0,1]))
data.label_axes(ax)
fig.tight_layout()
fig.savefig('../figures/data_diff_2.pdf')

fig, ax = plt.subplots(figsize=figsize)
data.plot_points(ax)
data.label_axes(ax)
fig.tight_layout()
fig.savefig('../figures/data.pdf')


if os.path.exists('data.pkl'):
    outputs = pickle.load(open('data.pkl','rb'))
else:
    outputs = {}
    nDims = 5
    for i in range(1,2**nDims):
        root = format(i, '#0%ib' %(nDims+2))[2:]
        outputs[root] = polychord.run(root)
        pickle.dump(outputs,open('data.pkl','wb'))

roots = [root for _, root in sorted([(output.logZ,root) for root, output in outputs.items()],reverse=True)]


# Evidence plots
ind = [i for i, _ in enumerate(roots)]
evs = numpy.array([outputs[root].logZ for root in roots])
evs -= evs.max()
errs = [outputs[root].logZerr for root in roots]
labels = [data.poly(root) for root in roots]


fig, ax = plt.subplots(figsize=(figsize[0]*2,figsize[1]*2))
ax.set_xticks(ind)
ax.set_xticklabels(labels)

ax.grid(color='b', linestyle=':', linewidth=0.1)
ax.xaxis.set_tick_params(rotation=90)
ax.plot(ind,evs,'k.-')

ax.set_ylabel(r'$\log Z$')
fig.tight_layout()
fig.savefig('../figures/evidences_log.pdf')


fig, ax = plt.subplots(figsize=(figsize[0]*2,figsize[1]*2))
ax.set_xticks(ind)
ax.set_xticklabels(labels)

ax.grid(color='b', linestyle=':', linewidth=0.1)
ax.xaxis.set_tick_params(rotation=90)
ax.plot(ind,numpy.exp(evs),'k.-')

ax.set_ylabel(r'Betting ratio')
fig.tight_layout()
fig.savefig('../figures/evidences_lin.pdf')



# parameter plots
g = getdist.plots.getSubplotPlotter()

fig, axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=figsize)
for i, (root, ax) in enumerate(zip(['11000','10100','10010','10001'],axes.ravel())):
    g.plot_2d(outputs[root].root + '_equal_weights','p0','p%i'% (i+1),filled=True,contour_args={'ax':ax})
    ax.set_title(data.poly(root))

axes[0,0].set_xlim(0.6,1.15)
axes[0,0].set_ylim(0.7,1.3)
axes[0,0].set_ylabel(r'$b$')
axes[1,0].set_ylabel(r'$b$')
axes[1,0].set_xlabel(r'$a$')
axes[1,1].set_xlabel(r'$a$')

fig.tight_layout()
fig.savefig('../figures/parameters.pdf')


# fgivenx plots
fig, axes = plt.subplots(2,2,sharex=True,sharey=True,figsize=figsize)
axes[0,0].set_ylabel(r'$P(y|x)$')
axes[1,0].set_ylabel(r'$P(y|x)$')
axes[1,0].set_xlabel(r'$x$')
axes[1,1].set_xlabel(r'$x$')
for root, ax in zip(['11000','10100','10010','10001'],axes.ravel()):
    data.plot_points(ax)
    cbar = fgivenx.plot(outputs[root], ax)
    ax.set_yticks([])
    ax.set_xticks([])
fig.tight_layout()


fig.savefig('../figures/fgivenx.pdf')
