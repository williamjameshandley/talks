import numpy as np
import copy
import matplotlib.pyplot as plt
#from sbi import LinearModel, plot_2d, plot_1d, flip
from matplotlib.backends.backend_pdf import PdfPages
from labellines import labelLine, labelLines

from lsbi.model import LinearModel, MixtureModel
from lsbi.plot import pdf_plot_2d, pdf_plot_1d
from anesthetic.plot import AxesSeries, AxesDataFrame

from scipy.stats import invwishart, matrix_normal, multivariate_normal
from numpy.linalg import cholesky

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
    C_ = np.atleast_2d(C_)
    L1 = np.linalg.cholesky(C_/k)
    L2 = np.linalg.cholesky(invΘ)
    M_ = Ψ @ invΘ + np.einsum('...jk,...kl,ml->...jm', L1, np.random.randn(*shape, d, n), L2)
    m_ = Dbar - M_ @ θbar + np.einsum('...jk,...k->...j', L1, np.random.randn(*shape, d))
    return LinearModel(m=m_, M=M_, C=C_, *args, **kwargs)


pdf = PdfPages('lsbi_plot.pdf')

with PdfPages('lsbi_plot.pdf') as pdf:
    mu = np.array([np.linspace(-1, 3, 100)]).T
    m = mu**2
    model = MixtureModel(M=0, m=m, C=0.01, mu=mu, Sigma=0.01)
    fig = plt.figure(figsize=(6,6))
    gs = fig.add_gridspec(2,2, width_ratios=(1,2), height_ratios=(1,2))
    ax_D = fig.add_subplot(gs[1,0])
    ax_joint = fig.add_subplot(gs[1,1], sharey=ax_D)
    ax_theta = fig.add_subplot(gs[0,1], sharex=ax_joint)
    
    ax_theta.set_xlim(np.min(model.prior().mean - model.prior().cov**0.5*3),
                      np.max(model.prior().mean + model.prior().cov**0.5*3)) 
    ax_D.set_ylim(np.min(model.evidence().mean - model.evidence().cov**0.5*3),
                  np.max(model.evidence().mean + model.evidence().cov**0.5*3))


    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    ax_D.set_ylabel(r'$D$')
    ax_joint.set_xlabel(r'$\theta$')
    fig.tight_layout()

    np.random.seed(1)
    pdf_plot_1d(ax_theta, model.prior(), color='C1',nplot_1d=1000)
    pdf_plot_1d(ax_D, model.evidence(), orientation='vertical', color='C3')
    k = 300
    data = model.joint().rvs(k)
    ax_joint.plot(*data.T, 'C4o')

    pdf.savefig(fig)

    Do=[5]
    ax_D.axhline(Do, color='k', ls='--')
    ax_joint.axhline(Do, color='k', ls='--')
    pdf_plot_1d(ax_theta, model.posterior(Do), color='C0')[-1]

    pdf.savefig(fig)

    θ, D = np.split(data, 2, axis=1)
    lsbi_model = LSBI(θ, D, μ=θ.mean(), Σ=np.cov(θ.T))

    to_remove = []
    to_remove += list(pdf_plot_2d(ax_joint, lsbi_model.joint(), color='C4', nplot_2d=100000, label=r'$\mathcal{J}$'))

    pdf.savefig(fig)
    to_remove.append(pdf_plot_1d(ax_theta, lsbi_model.posterior(Do), ls='--', color='C0',nplot_1d=1000)[-1])
    pdf.savefig(fig)

    for _ in range(3):
        for x in to_remove:
            x.remove()

        to_remove = []
        to_remove.append(pdf_plot_1d(ax_theta, lsbi_model.posterior(Do), ls='--', color='C5',nplot_1d=1000)[-1])

        pdf.savefig(fig)

        θ = lsbi_model.posterior(Do).rvs(300)
        D = model.likelihood(θ).rvs()

        to_remove += ax_joint.plot(θ, D, 'C5o')

        pdf.savefig(fig)

        lsbi_model =  LSBI(θ, D, μ=lsbi_model.μ, Σ=lsbi_model.Σ)
        to_remove.pop().remove()
        to_remove += list(pdf_plot_2d(ax_joint, lsbi_model.joint(), color='C4', nplot_2d=100000, label=r'$\mathcal{J}$'))
        pdf.savefig(fig)
        to_remove.append(pdf_plot_1d(ax_theta, lsbi_model.posterior(Do), ls='--', color='C0',nplot_1d=1000)[-1])
        pdf.savefig(fig)
