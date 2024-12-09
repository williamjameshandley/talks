import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal
from matplotlib.backends.backend_pgf import PdfPages
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True 
mpl.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'cm'

pdf = PdfPages('anatomy.pdf')

np.random.seed(0)
fig, axes = plt.subplots(1, 2, sharey=True)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].set_xticks([])



dist = norm()
x = dist.rvs(size=10000)
x = np.sort(x)
axes[0].plot(x, dist.pdf(x), linewidth=2)

axes[0].set_xlabel(r'$\theta$')
axes[0].set_ylabel(r'$\mathcal{P}(\theta)$')
fig.set_size_inches(5,5)
fig.tight_layout()
pdf.savefig(fig)

axes[0].hist(x, bins=100, density=True, alpha=0.8, color='C0')

pdf.savefig(fig)

axes[1].set_xlabel(r'$\mathcal{P}(\mathcal{P})$')
axes[1].hist(dist.pdf(x), bins=100, alpha=0.8, orientation='horizontal', label='$d=1$')
pdf.savefig(fig)
loc = 'upper right'
legend = axes[1].legend(loc=loc)

for d in range(2, 7):
    dist_d = multivariate_normal(mean=np.zeros(d))
    x = dist_d.rvs(size=10000)
    axes[1].hist(dist_d.pdf(x)/dist_d.pdf(0)*dist.pdf(0), bins=100, alpha=0.8, orientation='horizontal', label=f'$d={d}$')
    legend = axes[1].legend(loc=loc)
    pdf.savefig(fig)



fig, ax = plt.subplots()

logPmax = 0
d = 30
n = 1000
avlogP = logPmax-d/2

logP = np.linspace(logPmax-d/2-np.sqrt(d/2)*3.1,0,1000)
logpdf = logP - logPmax + (d/2-1)*np.log(logPmax-logP)
logpdf -= np.max(logpdf)
ax.plot(logP, np.exp(logpdf), label='posterior: $\mathcal{P}(\log\mathcal{P})$')


logPmaxlive = logPmax - d/2/np.e + np.log(n)/np.e

l1 = 1.3
ax.axvline(avlogP, color='k', linestyle=':')
ax.axvline(logPmax, color='k', linestyle=':')
ax.plot([avlogP, logPmax], [l1,l1], 'k-')
ax.annotate(r'$\frac{d}{2}$', ((avlogP+logPmax)/2,l1), ha='center', va='bottom')

l2 = 0.7
ax.plot([avlogP-np.sqrt(d/2), avlogP+np.sqrt(d/2)], [l2, l2], 'k-')
ax.annotate(r'$\pm\sqrt{\frac{d}{2}}$', (avlogP,l2), ha='left', va='bottom')

l3 = 1.05
ax.plot([avlogP, avlogP+1], [l3, l3], 'k-')
ax.annotate(r'$+1$', (avlogP+1,l3), ha='right', va='bottom')

ax.set_xticks([avlogP, 
               logPmax])
ax.set_xticklabels([r'$\langle\log\mathcal{P}\rangle_\mathcal{P}$',
                    r'$\log\mathcal{P}_\mathrm{max}$'])

ax.set_yticks([])
#ax.legend(loc='upper left')
ax.set_ylim([0,1.4])

fig.set_size_inches(5,5)
fig.tight_layout()


fig, ax = plt.subplots()
ax.set_yticks([])
ax.set_xlabel(r'$\log\mathcal{P}/\mathcal{P}_\mathrm{max}$')
fig.set_size_inches(5,5)
fig.tight_layout()

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))



for d in [1,2, 6, 19, 50, 500]:
    logP = np.linspace(logPmax-d/2-np.sqrt(d/2)*3.1,0,1000)
    logpdf = logP - logPmax + (d/2-1)*np.log(logPmax-logP)
    logpdf[np.isinf(logpdf)] = np.nan
    logpdf -= np.nanmax(logpdf)
    ax.plot(logP, np.exp(logpdf), label=f'$d={d}$')
    ax.legend(loc='upper left')
    pdf.savefig(fig)


pdf.close()
