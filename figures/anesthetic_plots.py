from anesthetic import read_chains, make_2d_axes

samples = read_chains("tests/example_data/pc_250")
prior = samples.prior()
params = ['x0', 'x1', 'x2', 'x3', 'x4']
fig, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w')
prior.plot_2d(axes, label="prior")
samples.plot_2d(axes, label="posterior")
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncols=2)
fig.tight_layout()
fig.savefig("anesthetic_kde.pdf")

kind={'lower':'hist_2d', 'diagonal':'hist_1d', 'upper':'scatter_2d'}

fig, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w')
prior.plot_2d(axes, label="prior", kind=kind)
samples.plot_2d(axes, label="posterior", kind=kind)
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncols=2)
fig.tight_layout()
fig.savefig("anesthetic_hist.pdf")

lower_kwargs={'levels': [0.95, 0.68], 'bins':20}
fig, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w')
prior.plot_2d(axes, label="prior", kind=kind)
samples.plot_2d(axes, label="posterior", kind=kind, lower_kwargs=lower_kwargs)
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncols=2)
fig.tight_layout()
fig.savefig("anesthetic_hist_levels.pdf")


from anesthetic.convert import to_getdist
from getdist import plots
import matplotlib.pyplot as plt

gd_samples = to_getdist(samples)
gd_samples.label = 'posterior'
gd_prior = to_getdist(prior)
gd_prior.label = 'prior'
g = plots.get_subplot_plotter()
g.triangle_plot([gd_prior, gd_samples], params, filled=True, colors=['C0', 'C1'])
plt.gcf().set_size_inches(6,6)
plt.tight_layout()
g.export("getdist.pdf")
plt.close()

import corner
#corner.corner(samples[params].compress('equal').to_numpy(), color='C1', label='posterior', labels=samples.get_labels()[:5])
corner.corner(samples[params].to_numpy(), weights=samples.get_weights(), color='C1', label='posterior', labels=samples.get_labels()[:5])
plt.gcf().set_size_inches(6,6)
plt.savefig("corner.pdf")
