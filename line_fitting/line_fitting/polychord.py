import PyPolyChord
from PyPolyChord.settings import PolyChordSettings
import numpy
from line_fitting import data

# gaussian likelihood
def gaussian(x, mu, sig):
    """ Normalised gaussian log pdf """
    return -numpy.log(2*numpy.pi*sig**2)/2 - (x-mu)**2/sig**2/2

def run(root):
        nDerived = 0
        ranges = [(-3*int(c),+3*int(c)) for c in root]
        nDims = len(ranges)

        settings = PolyChordSettings(nDims, nDerived)
        settings.do_clustering = True
        settings.read_resume = False
        settings.feedback = 0
        settings.file_root = root

        def prior(cube):
            """ Logarithmic priors in mass """
            theta = [0] * nDims
            for i, (amin, amax)  in enumerate(ranges):
                theta[i] = amin + (amax-amin)*cube[i]
            return theta

        def likelihood(theta):
            """ Likelihood for the normal hierarchy """
            phi = [0.] * nDerived
            logl = sum(gaussian(data.y, data.f(data.x, theta), data.sigma))
            return logl, phi

        output = PyPolyChord.run_polychord(likelihood, nDims, nDerived, settings, prior)

        params = ['_']*nDims
        letters = 'abcde'
        for i, c in enumerate(root):
            if c=='1':
                params[i] = letters[0]
                letters=letters[1:]
        paramnames = [('p%i' % i, param) for i, param in enumerate(params)]
        output.make_paramnames_file(paramnames,output.root + '_equal_weights.paramnames')

        return output
