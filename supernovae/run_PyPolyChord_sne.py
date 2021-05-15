import matplotlib.pyplot as plt
import getdist.plots

from PyPolyChord import run_polychord
from PyPolyChord.priors import UniformPrior
from PyPolyChord.settings import PolyChordSettings

from SNE.supernova_data import loglikelihood_sys, loglikelihood_nosys
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

nDims = 2
nDerived = 1


# Define the likelihoods
# ----------------------

def likelihood_sys(theta):
    """ Likelihood with systematic errors.

    This is the likelihood for a model of a flat LCDM universe with only
    baryonic matter and dark energy components. It treats the dark energy as a
    derived parameter.

    """

    H0, O_m = theta
    O_r, O_k, O_L = 0., 0., 1. - O_m
    derived_params = [O_L]

    return loglikelihood_sys(H0, O_r, O_m, O_L, O_k), derived_params


def likelihood_nosys(theta):
    """ Likelihood with no systematic errors. """

    H0, O_m = theta
    O_r, O_k, O_L = 0., 0., 1. - O_m
    derived_params = [O_L]

    return loglikelihood_nosys(H0, O_r, O_m, O_L, O_k), derived_params


# Define the priors
# -----------------

H_prior = UniformPrior(50, 100)
O_m_prior = UniformPrior(0, 1)

def prior(x):
    physical = [0.]*nDims
    physical[0] = H_prior(x[0])
    physical[1] = O_m_prior(x[1])
    return physical

        
# Set up PolyChord
# -----------------

settings = PolyChordSettings(nDims, nDerived)
settings.feedback = 1
settings.num_repeats = 2*nDims
settings.do_clustering = False


# Run PolyChord for the systematic likelihood
# -------------------------------------------

settings.file_root = 'sys'
output_sys = run_polychord(
        likelihood_sys, nDims, nDerived, settings, prior
        )


# Run PolyChord for the non-systematic likelihood
# -----------------------------------------------

settings.file_root = 'nosys'
output_nosys = run_polychord(
        likelihood_nosys, nDims, nDerived, settings, prior
        )


# Produce paramnames files for getdist
# ------------------------------------

paramnames = [('H0', r'H_0'), ('O_m', r'\Omega_m'), ('O_L*', r'\Omega_\Lambda')]
if rank == 0:
    output_sys.make_paramnames_files(paramnames)
    output_nosys.make_paramnames_files(paramnames)


# Plot chains using getdist on the same plot
# ------------------------------------------

if rank == 0:
    posterior_sys = output_sys.posterior
    posterior_nosys = output_nosys.posterior
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot([posterior_sys, posterior_nosys], filled=True)
    plt.show()
