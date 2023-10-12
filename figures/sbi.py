import numpy as np
from functools import cached_property
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages


class LinearModel(object):
    """
    A linear model:

    - Parameters theta (n dims)
        - Prior mean mu
        - Prior covariance Sigma
    - Data D (d dims)
        - D = m + M theta +/- sqrt(C)
    Model M
    Data offset m
    Covariance C

    """
    def __init__(self, M, m=None, C=None, mu=None, Sigma=None):

        self.M = np.atleast_2d(M)

        if m is None:
            m = np.zeros(self.M.shape[0])
        if C is None:
            C = np.eye(self.M.shape[0])
        if mu is None:
            mu = np.zeros(self.M.shape[1])
        if Sigma is None:
            Sigma = np.eye(self.M.shape[1])

        self.m = np.atleast_1d(m)
        self.C = np.atleast_2d(C)
        self.mu = np.atleast_1d(mu)
        self.Sigma = np.atleast_2d(Sigma)

    @cached_property
    def invSigma(self):
        return np.linalg.inv(self.Sigma)

    @cached_property
    def invC(self):
        return np.linalg.inv(self.C)

    def likelihood(self, theta):
        return multivariate_normal(self.D(theta), self.C)

    def prior(self):
        return multivariate_normal(self.mu, self.Sigma)

    def posterior(self, D):
        Sigma = np.linalg.inv(self.invSigma + self.M.T @ self.invC @ self.M)
        mu = Sigma @ (self.invSigma @ self.mu
                      + self.M.T @ self.invC @ (D-self.m))
        return multivariate_normal(mu, Sigma)

    def evidence(self):
        return multivariate_normal(self.D(self.mu),
                                   self.C + self.M @ self.Sigma @ self.M.T)

    def joint(self):
        mu = np.concatenate([self.m+self.M @ self.mu, self.mu])
        Sigma = np.block([[self.C+self.M @ self.Sigma @ self.M.T,
                           self.M @ self.Sigma],
                          [self.Sigma @ self.M.T, self.Sigma]])
        return multivariate_normal(mu, Sigma)

    def D(self, theta):
        return self.m + self.M @ theta

    def DKL(self, D):
        cov_p = self.posterior(D).cov
        cov_q = self.prior().cov
        mu_p = self.posterior(D).mean
        mu_q = self.prior().mean
        return 0.5 * (np.linalg.slogdet(cov_p)[1] - np.linalg.slogdet(cov_q)[1]
                      + np.trace(np.linalg.inv(cov_q) @ cov_p)
                      + (mu_q - mu_p) @ np.linalg.inv(cov_q) @ (mu_q - mu_p)
                      - len(mu_p))


model = LinearModel(M=1, m=0, C=1, mu=0, Sigma=1).joint()

def plot_2d(model, ax, n=100, color='C0'):
    cov = model.cov
    mean = model.mean
    xmin, xmax = mean[0]-3*cov[0,0]**0.5, mean[0]+3*cov[0,0]**0.5
    ymin, ymax = mean[1]-3*cov[1,1]**0.5, mean[1]+3*cov[1,1]**0.5
    x = np.linspace(xmin, xmax, n+1)
    y = np.linspace(ymin, ymax, n+1)
    x, y = np.meshgrid(x, y)
    levels = [0.95, 0.67, 0.0]
    logpdf= model.logpdf(np.array([x, y]).T)
    m = np.exp(-(-2*logpdf - np.linalg.slogdet(2*np.pi*cov)[1])/2)
    cmap = LinearSegmentedColormap.from_list(str(color), ['#ffffff', color]) 
    return ax.contourf(x, y, m, levels=1-np.array(levels), cmap=cmap)

def plot_1d(model, ax, n=1000, orientation='horizontal', normalise=False, *args, **kwargs):
    cov = model.cov
    mean = model.mean
    xmin, xmax = mean[0]-3*cov[0,0]**0.5, mean[0]+3*cov[0,0]**0.5
    x = np.linspace(xmin, xmax, n+1)
    y = np.exp(model.logpdf(x))
    if normalise is True:
        y /= y.max()
    if orientation == 'vertical':
        x, y = y, x
    return ax.plot(x, y, *args, **kwargs)


#A = MCMCSamples(LinearModel(M=1, m=0, C=1, mu=0, Sigma=1).joint().rvs(10000), columns=['D', 'theta'], labels=['$D$', r'$\theta_A$'])
#B = MCMCSamples(LinearModel(M=5, m=5, C=1, mu=1, Sigma=1).joint().rvs(10000), columns=['D', 'theta'], labels=['$D$', r'$\theta_B$'])
#B_ = pd.DataFrame(LinearModel(M=5, m=5, C=1, mu=1, Sigma=1).joint().rvs(10000), columns=['D', 'theta'])

def flip(model):
    return multivariate_normal(model.mean[::-1], model.cov[::-1, ::-1])


pdf = PdfPages('sbi.pdf')
with PdfPages('sbi.pdf') as pdf:
    A = LinearModel(M=1, m=0, C=1, mu=0, Sigma=1)
    B = LinearModel(M=5, m=5, C=1, mu=1, Sigma=1)
    fig = plt.figure(figsize=(6,3))
    gs = fig.add_gridspec(2,3, width_ratios=(1,2,2), height_ratios=(1,2))
    ax_D = fig.add_subplot(gs[1,0])
    ax_joint_A = fig.add_subplot(gs[1,1], sharey=ax_D)
    ax_joint_B = fig.add_subplot(gs[1,2], sharey=ax_D)
    ax_theta_A = fig.add_subplot(gs[0,1], sharex=ax_joint_A)
    ax_theta_B = fig.add_subplot(gs[0,2], sharex=ax_joint_B)

    plot_2d(flip(A.joint()), ax_joint_A, color='C0')
    plot_2d(flip(B.joint()), ax_joint_B, color='C1')
    plot_1d(A.evidence(), ax_D, orientation='vertical', color='C0')
    plot_1d(B.evidence(), ax_D, orientation='vertical', color='C1')
    plot_1d(A.prior(), ax_theta_A, color='C0', normalise=True, label=r'$P(\theta_A)$')
    plot_1d(B.prior(), ax_theta_B, color='C1', normalise=True, label=r'$P(\theta_B)$')


    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    ax_D.set_xlabel(r'$P(D)$')
    ax_D.set_ylabel(r'$D$')
    ax_theta_A.set_ylabel(r'$P^*$')
    ax_joint_A.set_xlabel(r'$\theta_A$')
    ax_theta_A.set_title('Model A')
    ax_theta_B.set_title('Model B')
    #ax_theta_A.legend()
    #ax_theta_B.legend()
    ax_joint_B.set_xlabel(r'$\theta_B$')
    fig.tight_layout()
    pdf.savefig(fig)

    def plot_data(D):
        to_remove = []
        to_remove.append(ax_D.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint_A.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint_B.axhline(D, color='k', ls='--'))

        to_remove.append(plot_1d(A.posterior(D), ax_theta_A, color='C0', normalise=True, lw=4, label=r'$P(\theta_A|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(ax_joint_A.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        to_remove.append(plot_1d(B.posterior(D), ax_theta_B, color='C1', normalise=True, lw=4, label=r'$P(\theta_B|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(ax_joint_B.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        return to_remove

    to_remove = plot_data(1)
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()

    to_remove = plot_data(12)
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()

    #--------------------------------------------------------------------------
    A = LinearModel(M=1, m=0, C=1, mu=0, Sigma=1)
    B = LinearModel(M=5, m=20, C=1, mu=1, Sigma=1)
    fig = plt.figure(figsize=(6,3))
    gs = fig.add_gridspec(2,3, width_ratios=(1,2,2), height_ratios=(1,2))
    ax_D = fig.add_subplot(gs[1,0])
    ax_joint_A = fig.add_subplot(gs[1,1], sharey=ax_D)
    ax_joint_B = fig.add_subplot(gs[1,2], sharey=ax_D)
    ax_theta_A = fig.add_subplot(gs[0,1], sharex=ax_joint_A)
    ax_theta_B = fig.add_subplot(gs[0,2], sharex=ax_joint_B)

    plot_2d(flip(A.joint()), ax_joint_A, color='C0')
    plot_2d(flip(B.joint()), ax_joint_B, color='C1')
    plot_1d(A.evidence(), ax_D, orientation='vertical', color='C0')
    plot_1d(B.evidence(), ax_D, orientation='vertical', color='C1')
    plot_1d(A.prior(), ax_theta_A, color='C0', normalise=True, label=r'$P(\theta_A)$')
    plot_1d(B.prior(), ax_theta_B, color='C1', normalise=True, label=r'$P(\theta_B)$')


    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    ax_D.set_xlabel(r'$P(D)$')
    ax_D.set_ylabel(r'$D$')
    ax_theta_A.set_ylabel(r'$P^*$')
    ax_joint_A.set_xlabel(r'$\theta_A$')
    ax_theta_A.set_title('Model A')
    ax_theta_B.set_title('Model B')
    #ax_theta_A.legend()
    #ax_theta_B.legend()
    ax_joint_B.set_xlabel(r'$\theta_B$')
    fig.tight_layout()
    pdf.savefig(fig)

    def plot_data(D):
        to_remove = []
        to_remove.append(ax_D.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint_A.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint_B.axhline(D, color='k', ls='--'))

        to_remove.append(plot_1d(A.posterior(D), ax_theta_A, color='C0', normalise=True, lw=4, label=r'$P(\theta_A|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(ax_joint_A.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        to_remove.append(plot_1d(B.posterior(D), ax_theta_B, color='C1', normalise=True, lw=4, label=r'$P(\theta_B|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(ax_joint_B.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        return to_remove

    to_remove = plot_data(7)
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()
