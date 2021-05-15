""" Compute contours.

    Things to specify:

    xmin: float
        minimum of x range
    xmax: float
        maximum of x range
    chains_file: str
        Name of the file where the posterior samples are stored. These should
        be a text data file with columns:
            weight  log-likelihood  <parameters>
        Typically this file is produced by getdist.
    paramnames_file: str
        Where the names of the the parameters are stored. This should be a text
        file with one parameter name per line (no spaces), in the order they
        appear in chains_file.
        Typically this file is produced by getdist.

    f: function
        The function to be inferred f(x;theta)
        Parameters
        ----------
        x: float
            independent variable
        theta: List[float]
            parameters of function

    chosen_parameters: List[str]
        The parameter names that the function depends on, in the order f as
        determined by f.
"""
from fgivenx.contours import Contours
from fgivenx.data_storage import FunctionalPosterior

from SNE.distance import distance_modulus, D_L

# setup
# -----
xmin = 0.01
xmax = 1.4
chains_file = 'chains/mat_DE.txt'              # posterior files
paramnames_file = 'chains/mat_DE.paramnames'   # paramnames file

def f(x, theta):
    O_m = theta[0]
    return D_L(x, O_m=O_m, O_L=1.-O_m)

chosen_parameters = ['mat']


# Computing contours
# ------------------
# load the posteriors from file, set the function, and trim them
nsamp = 500
posterior = FunctionalPosterior(chains_file, paramnames_file).set_function(f, chosen_parameters)
trimmed_posterior = posterior.trim_samples(nsamp)
# Create the contours and save
contourfile = 'posterior_mat_DE.pkl'
contours = Contours(trimmed_posterior, (xmin, xmax), ny=1000).save(contourfile)
