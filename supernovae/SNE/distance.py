""" Distance calculations."""
import numpy
import scipy.integrate


def check_O(O_r, O_m, O_L, O_k):
    """ Check that relative densities sum to 1. """

    O = O_r + O_m + O_L + O_k
    if not numpy.isclose(O,1):
        print("Warning: total relative energy today should be close to 1.\n\
                O_r + O_m + O_L + O_k = %f + %f + %f + %f = %f " %
                (O_r, O_m, O_L, O_k, O))
        O_r = O_r / O
        O_m = O_m / O
        O_L = O_L / O
        O_k = O_k / O

    return O_r, O_m, O_L, O_k


def D_L(z, H0=66, O_m=0.25, O_r=0.0, O_L=0.75, O_k=0.0):
    """ Luminosity distance as a function of redshift z.

    For more detail, see:
    https://en.wikipedia.org/wiki/Distance_measures_(cosmology)

    Parameters
    ----------
    z : float
        redshift of object

    Optional Parameters
    -------------------
    H0 : float
        Local Hubble constant, measured in km/s / Mpc
    O_m:
        proportion of density today accounted for by baryonic matter.
    O_r:
        proportion of density today accounted for by radiation.
    O_L:
        proportion of density today accounted for by dark energy.

    """
    O_r, O_m, O_L, O_k = check_O(O_r, O_m, O_L, O_k)
    d_H = D_H(H0)

    def E(zz):
        a = 1./(1.+zz)
        return numpy.sqrt(O_r * a**-4. + O_m * a**-3. + O_L)

    d_C = scipy.integrate.quad(lambda zz: d_H/E(zz), 0, z)[0]
    d_M = D_M(d_C, O_k, d_H)

    return d_M * (1.+z)


def distance_modulus(d_L):
    """ Distance modulus, defined as m-M.

    For more detail see:
    https://en.wikipedia.org/wiki/Distance_modulus

    Parameters
    ----------
    d_L : float
        Luminosity distance, measured in MegaParsecs.
    """

    return 5*(numpy.log10(d_L*1e+6)-1)

def luminosity_distance(mod):
    """ luminosity distance from distance modulus."""
    return 10.**(mod/5. + 1.)

def D_M(d_C, O_k, d_H):
    """Transverse comoving distance from cosmology distance d_C."""
    if O_k > 0:
        return d_H / numpy.sqrt(O_k) * numpy.sinh(numpy.sqrt(O_k) * d_C/d_H)
    elif O_k < 0:
        return d_H / numpy.sqrt(-O_k) * numpy.sin(numpy.sqrt(-O_k) * d_C/d_H)
    else:
        return d_C

def D_C(d_M, O_k, d_H):
    """cosmology distance from Transverse comoving distance d_M."""
    if O_k > 0:
        return  d_H/numpy.sqrt(O_k) * numpy.arcsinh(d_M/d_H * numpy.sqrt(O_k))
    elif O_k < 0:
        return  d_H/numpy.sqrt(-O_k) * numpy.arcsin(d_M/d_H * numpy.sqrt(-O_k))
    else:
        return d_M


def D_H(H0):
    """ Hubble distance in Mpc. 

    Parameters
    ----------
    H0 : float
        Hubble constant measured in km/s/Mpc.
    """
    c = 299792.458  # Speed of light in km/s
    return c/H0

def D_Ls(zs, H0=66, O_m=0.25, O_r=0.0, O_L=0.75, O_k=0.0):
    """ Compute Luminosity distance for several redshifts zs.

    For more detail, see D_L:

    """
    O_r, O_m, O_L, O_k = check_O(O_r, O_m, O_L, O_k)
    d_H = D_H(H0)

    def deriv_D_C(D_C, z):
        a = 1./(1.+z)
        return d_H/numpy.sqrt(O_r * a**-4. + O_m * a**-3. + O_L)

    Zs = numpy.insert(zs, 0, 0.)
    d_Cs = scipy.integrate.odeint(deriv_D_C, 0., Zs)[1:, 0]

    d_Ms = D_M(d_Cs, O_k, d_H)

    return d_Ms * (1.+zs)
