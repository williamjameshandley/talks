from anesthetic import MCMCSamples
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property
from scipy.stats import multivariate_normal


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
    def __init__(self, M, m, C, mu, Sigma):
        self.M = np.atleast_2d(M)
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
        return multivariate_normal(self.m + self.M@theta, self.C)

    def prior(self):
        return multivariate_normal(self.mu, self.Sigma)

    def posterior(self, D):
        Sigma = np.linalg.inv(self.invSigma + self.M.T @ self.invC @ self.M)
        mu = Sigma @ (self.invSigma @ self.mu + self.M.T @ self.invC @ (D-self.m))
        return multivariate_normal(mu, Sigma)

    def evidence(self, D):
        return multivariate_normal(self.m + self.M @ self.mu,
                                   self.C + self.M @ self.Sigma @ self.M.T)

    def joint(self):
        mu = np.concatenate([self.m+self.M @ self.mu, self.mu])
        Sigma = np.block([[self.C+self.M @ self.Sigma @ self.M.T,
                           self.M @ self.Sigma],
                          [self.Sigma @ self.M.T, self.Sigma]])
        return multivariate_normal(mu, Sigma)



#M = np.array([[1]])
#m = np.array([0])
#C = np.array([[1]])
#mu = np.array([0])
#Sigma = np.array([[1]])


def joint(M, m, C, mu, Sigma, dist='joint'):
    model = LinearModel(M, m, C, mu, Sigma)
    columns = ['D', 'theta']
    labels = ['$D$', r'$\theta$']
    return MCMCSamples(model.joint().rvs(10000), columns=columns, labels=labels)


def posterior(M, m, C, mu, Sigma, D, dist='joint'):
    model = LinearModel(M, m, C, mu, Sigma)
    columns = ['theta']
    labels = [r'$\theta$']
    return MCMCSamples(model.posterior(D).rvs(10000), columns=columns, labels=labels)


from anesthetic.plot import kde_plot_1d

kwargs = {'diagonal_kwargs':dict(density=True)}
axes = joint(1,0,1,0,1).plot_2d(['theta','D'],**kwargs)
axes = joint(5,5,1,0,1).plot_2d(axes, alpha=0.9,**kwargs) 

axes.axlines({'D':1},color='k',ls='--')
axes = posterior(1,0,1,0,1,1).plot_2d(axes,color='C0',ls='--',**kwargs)
axes = posterior(5,5,1,0,1,1).plot_2d(axes,color='C1',ls='--',**kwargs)

axes.axlines({'D':7},color='k',ls=':')
axes = posterior(1,0,1,0,1,7).plot_2d(axes,color='C0',ls=':',**kwargs)
axes = posterior(5,5,1,0,1,7).plot_2d(axes,color='C1',ls=':',**kwargs)

axes.loc['theta','theta'].twin.autoscale(True)
axes.loc['theta','theta'].twin.set_ylim(0,None)
axes.loc['theta','theta'].set_xlim(None,None)

for ax in axes.iloc[:,0]:
    ax.set_yticks([])
for ax in axes.iloc[-1,:]:
    ax.set_xticks([])

fig = plt.gcf()
fig.set_size_inches(3,3)
fig.tight_layout()
fig.savefig('joint.pdf')

kwargs = {'diagonal_kwargs':dict(density=True)}
axes = joint(1,0,1,0,1).plot_2d(['theta','D'],**kwargs)
axes = joint(5,50,1,0,1).plot_2d(axes, alpha=0.9, **kwargs) 

axes.axlines({'D':25},color='k',ls='-.')
axes = posterior(1,0,1,0,1,25).plot_2d(axes,color='C0',ls='-.',**kwargs)
axes = posterior(5,50,1,0,1,25).plot_2d(axes,color='C1',ls='-.',**kwargs)

axes.loc['theta','theta'].twin.autoscale(True)
axes.loc['theta','theta'].twin.set_ylim(0,None)
axes.loc['theta','theta'].set_xlim(None,None)

for ax in axes.iloc[:,0]:
    ax.set_yticks([])
for ax in axes.iloc[-1,:]:
    ax.set_xticks([])

fig = plt.gcf()
fig.set_size_inches(3,3)
fig.tight_layout()
fig.savefig('joint_highd.pdf')
