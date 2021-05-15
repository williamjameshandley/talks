import matplotlib.pyplot as plt
import getdist.plots
from numpy import pi,log,sqrt,exp
import numpy
import sys
import PyPolyChord as PolyChord
from PyPolyChord.settings import PolyChordSettings
from PyPolyChord.priors import UniformPrior

import PyPolyChord.priors as priors

import ligo_likelihood

nDims = 9
nDerived = 0

def likelihood(physical):

    derived = [0.0] * nDerived

    #m1 = ligo_likelihood.m1
    #m2 = ligo_likelihood.m2
    #r = ligo_likelihood.r
    #theta = ligo_likelihood.theta
    #phi = ligo_likelihood.phi
    #p = ligo_likelihood.p
    #i = ligo_likelihood.i
    #t_c = ligo_likelihood.t_c
    #Phi_c = ligo_likelihood.Phi_c

    m1 = physical[1]
    m2 = physical[0]

    r, theta, phi, p, i, t_c, Phi_c = physical[2:]
    #theta, phi = physical

    return ligo_likelihood.ligo_loglikelihood(m1, m2, r, theta, phi, p, i, t_c, Phi_c), derived

class CosinePrior:
    def __call__(self, x):
        return numpy.arccos(1-2*x)

m_prior = priors.SortedUniformPrior(15*ligo_likelihood.m0,45*ligo_likelihood.m0)
r_prior = priors.LogUniformPrior(ligo_likelihood.r/2, ligo_likelihood.r*2)
theta_prior = CosinePrior()
phi_prior = priors.UniformPrior(0,2*pi)
p_prior = priors.UniformPrior(0,2*pi)
i_prior = CosinePrior()
t_c_prior = priors.UniformPrior(ligo_likelihood.tmin, ligo_likelihood.tmax)
Phi_c_prior = priors.UniformPrior(0,2*pi)

def prior(x):
    physical = [0.]*nDims
    #physical[0] = theta_prior(x[0])
    #physical[1] = phi_prior(x[1])
    physical[0:2] = m_prior(x[0:2])
    physical[2] = r_prior(x[2])
    physical[3] = theta_prior(x[3])
    physical[4] = phi_prior(x[4])
    physical[5] = p_prior(x[5])
    physical[6] = i_prior(x[6])
    physical[7] = t_c_prior(x[7])
    physical[8] = Phi_c_prior(x[8])
    return physical


#root = 'pi_2_pi_reduced'
#root = '0_0_reduced'
#root = '0_0_reduced'

settings = PolyChordSettings(nDims, nDerived)
#settings.file_root = 'ligo_%s' % root
settings.file_root = 'ligo'
#settings.nlive = 1000
#settings.update_files = 50
settings.do_clustering = True

output = PolyChord.run_polychord(likelihood, nDims, nDerived, settings, prior)

paramnames = [
        ('m2',r'm_2'),
        ('m1',r'm_1'),
        ('r',r'r'),
        ('theta',r'\theta'),
        ('phi',r'\phi'),
        ('p',r'p'),
        ('i',r'i'),
        ('tc',r't_c'),
        ('Phic',r'\Phi_c')
        ]

#paramnames = [
#        ('theta',r'\theta'),
#        ('phi',r'\phi')
#        ]

output.make_paramnames_files(paramnames)
posterior = output.posterior
#posterior1 = output.cluster_posterior(1)
#posterior2 = output.cluster_posterior(2)
#
g = getdist.plots.getSubplotPlotter()
g.triangle_plot(posterior,filled=True)
g.export('plot_ligo.pdf')
