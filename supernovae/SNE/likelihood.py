from SNE.supernova_data import loglikelihood_sys, loglikelihood_nosys
from PyPolyChord.priors import UniformPrior


def create_likelihood(choice):
    """ Create a likelihood with specific choice suitable for PolyChord.

    Parameters
    ----------
    choice: list
        List with strings 'rad', 'mat', 'DE', 'k' to indicate components 
        of the universe to be consider. At least two components must be provided, 
        and the last component in the list will be set so that the top value is fixed

    Returns:
    likelihood, prior, paramnames, nDims, nDerived
    """
    nDims = len(choice) # N.B. Hubble parameter is an extra parameter
    nDerived = 1

    def likelihood(theta):

        O_m, O_L, O_r, O_k = 0., 0., 0., 0.
        for i, omega in enumerate(choice[:-1]):
            if omega is 'rad': O_r = theta[i]
            elif omega is 'mat': O_m = theta[i]
            elif omega is 'DE': O_L = theta[i]
            elif omega is 'k': O_k = theta[i]
            else: raise ValueError("%s is an unknown component." % omega)

        H0 = theta[-1]
        O = 1 - sum(theta[:-1])
        omega = choice[-1]
        if omega is 'rad': O_r = O
        elif omega is 'mat': O_m = O
        elif omega is 'DE': O_L = O
        elif omega is 'k': O_k = O
        else: raise ValueError("%s is an unknown component." % omega)

        return loglikelihood_sys(H0, O_r, O_m, O_L, O_k), [O]


    prior_map = {'H': UniformPrior(50, 100),
                 'rad': UniformPrior(0,1),
                 'mat': UniformPrior(0,1),
                 'DE': UniformPrior(0,2),
                 'k': UniformPrior(-1,1)}

    def prior(x):
        physical = [prior_map[omega](xi) for xi, omega in zip(x, choice[:-1])]
        physical += [prior_map['H'](x[-1])]
        return physical


    latex = {'H': 'H_0',
             'rad': r'\Omega_r',
             'mat': r'\Omega_m',
             'DE': r'\Omega_\Lambda',
             'k': r'\Omega_k'}
    paramnames =[(c, latex[c]) for c in choice[:-1]]
    paramnames += [('H', 'H_0')]
    paramnames += [(choice[-1] + '*', latex[choice[-1]])]

    return likelihood, prior, paramnames, nDims, nDerived
