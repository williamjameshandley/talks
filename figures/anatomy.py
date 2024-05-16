import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'cm'


fig, ax = plt.subplots()
DKL = 0
n = 10
logr = np.linspace(DKL-np.sqrt(n/2)*3,DKL+n/2-1e-5,1000)
logpdf = logr - DKL - n/2 + (n/2-1)*np.log(n/2+DKL-logr)
logpdf -= np.max(logpdf)
ax.plot(logr, np.exp(logpdf), label='$\mathcal{P}(\log r)$')

l1 = 1.35
ax.axvline(DKL, color='k', linestyle=':')
ax.axvline(DKL+n/2, color='k', linestyle=':')
ax.plot([DKL, DKL+n/2], [l1,l1], 'k-')
ax.annotate(r'$\frac{n}{2}$', (DKL+n/4,l1+0.01), ha='center', va='bottom')

l3 = 1.2
ax.plot([DKL-np.sqrt(n/2), DKL+np.sqrt(n/2)], [l3, l3], 'k-')
ax.annotate(r'$\sigma = \sqrt{\frac{n}{2}}$', (DKL-0.2,l3+0.01), ha='right', va='bottom')

l3 = 1.05
ax.plot([DKL, DKL+1], [l3, l3], 'k-')
ax.annotate(r'$+1$', (DKL+1,l3), ha='right', va='bottom')

ax.set_xticks([DKL, 
               DKL+n/2])
ax.set_xticklabels([r'$\langle\log r\rangle_\mathcal{P} = \mathcal{D}_\mathrm{KL}$',
                    r'$\log r_\mathrm{max}$'])

ax.set_yticks([])
ax.set_ylim([0,1.5])

fig.set_size_inches(3,3)
ax.legend(loc='center left')
fig.tight_layout()
fig.savefig('anatomy.pdf')
