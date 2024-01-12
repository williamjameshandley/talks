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
    if orientation == 'horizontal':
        i = 0
    elif orientation == 'vertical':
        i = -1
    xmin, xmax = mean[i]-3*cov[i,i]**0.5, mean[i]+3*cov[i,i]**0.5
    x = np.linspace(xmin, xmax, n+1)
    y = np.exp(model.logpdf(x))
    if normalise is True:
        y /= y.max()
    if orientation == 'vertical':
        x, y = y[::-1], x[::-1]
    return ax.plot(x, y, *args, **kwargs)


def flip(model):
    return multivariate_normal(model.mean[::-1], model.cov[::-1, ::-1])


