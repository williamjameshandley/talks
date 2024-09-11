#| Define LSBI inference function

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import invwishart, matrix_normal, multivariate_normal
from lsbi.model import LinearModel

def LSBI(θ, D, *args, **kwargs):
    shape = kwargs.pop('shape', ())
    if isinstance(shape, int):
        shape = (shape,)
    k, n = θ.shape
    d = D.shape[1]
    θD = np.concatenate([θ, D], axis=1)
    mean = θD.mean(axis=0)
    θbar = mean[:n]
    Dbar = mean[n:]

    cov = np.cov(θD.T)
    Θ = cov[:n, :n]
    Δ = cov[n:, n:]
    Ψ = cov[n:, :n]
    ν = k - d - n - 2
    invΘ = np.linalg.inv(Θ)

    C_ = invwishart(df=ν, scale=k*(Δ-Ψ @ invΘ @ Ψ.T)).rvs(shape)
    L1 = np.linalg.cholesky(C_/k)
    L2 = np.linalg.cholesky(invΘ)
    M_ = Ψ @ invΘ + np.einsum('...jk,...kl,ml->...jm', L1, np.random.randn(*shape, d, n), L2)
    m_ = Dbar - M_ @ θbar + np.einsum('...jk,...k->...j', L1, np.random.randn(*shape, d))
    return LinearModel(m=m_, M=M_, C=C_, *args, **kwargs)

#| Define CMB sampling class

from scipy.stats import chi2

class CMB(object):
    def __init__(self, Cl):
        self.Cl = Cl

    def rvs(self, shape=()):
        shape = tuple(np.atleast_1d(shape))
        return chi2(2*l+1).rvs(shape + self.Cl.shape)*self.Cl/(2*l+1)

    def logpdf(self, x):
        return (chi2(2*l+1).logpdf((2*l+1)*x/self.Cl)  + np.log(2*l+1)-np.log(self.Cl)).sum(axis=-1) 

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX 
emulator = CosmoPowerJAX(probe='cmb_tt')

#| Generate some simulations

np.random.seed(0)
paramnames = [('Ωbh2', r'\Omega_b h^2'), ('Ωch2', r'\Omega_c h^2'), ('h', 'h'), ('τ', r'\tau'), ('ns', r'n_s'), ('lnA', r'\ln(10^{10}A_s)')]
params = ['Ωbh2', 'Ωch2', 'h', 'τ', 'ns', 'lnA']
θmin, θmax = np.array([[0.01865, 0.02625], [0.05, 0.255], [0.64, 0.82], [0.04, 0.12], [0.84, 1.1], [1.61, 3.91]]).T
Nsim = 10000
θ = np.random.uniform(θmin, θmax, size=(Nsim, 6))
l = np.arange(2, 2509)
Cl = emulator.predict(θ)
D = CMB(Cl).rvs()

#| Define the observed variables
θobs = θ[0]
Dobs = D[0]

#| If you want to reproduce the ground-truth yourself, uncomment and run the below (takes about an hour on four cores)

#from pypolychord import run
#samples = run(lambda θ: CMB(emulator.predict(θ)).logpdf(Dobs), len(θmin), prior=lambda x: θmin + (θmax-θmin)*x, paramnames=paramnames)
#samples.to_csv('lcdm.csv')

#| Otherwise just load these chains

from anesthetic import read_chains
samples = read_chains('lcdm.csv')

#| Run sequential LSBI

import tqdm
for i in tqdm.trange(4):
    if i == 0:
        models = [LSBI(θ, D, μ= (θmin + θmax)/2, Σ= (θmax - θmin)**2)]
    else:
        θ_ = models[-1].posterior(Dobs).rvs(Nsim)
        D_ = CMB(emulator.predict(θ_)).rvs() 
        models.append(LSBI(θ_, D_, μ=models[-1].μ, Σ=models[-1].Σ))


#| Plot the results
from anesthetic.plot import make_2d_axes

pdf = PdfPages('cosmo_update.pdf')

with PdfPages('cosmo_update.pdf') as pdf:
    fig, axes = make_2d_axes(params, labels=samples.get_labels_map())
    fig.set_size_inches(7, 7)
    fig.tight_layout()
    axes = models[0].prior().plot_2d(axes, label='prior')
    samples.plot_2d(axes, color='k', alpha=0.5, label='ground truth', zorder=1000)
    axes.axlines(dict(zip(params, θobs)), color='k', ls='--')
    pdf.savefig(fig)
    for i, model in enumerate(models):
        posterior = model.posterior(Dobs)
        axes = posterior.plot_2d(axes, label=f'round {i}')
        samps = posterior.rvs(1000)
        pdf.savefig(fig)
        for p, x in zip(params, samps.T):
            axes.loc[p, p].set_xlim(x.mean() - 5* x.std(), x.mean() + 5* x.std())
        pdf.savefig(fig)

