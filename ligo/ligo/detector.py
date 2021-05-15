from numpy import pi
from ligo.utils import VectorOnSphere

class Detector(VectorOnSphere):
    def __init__(self, theta, phi, x_theta):
        """ A detector situated at Earth coordinates theta, phi.

        Parameters
        ----------
        theta: float
            Earth latitude of detector in radians.
        phi: float
            Earth longitude of detector in radians.
        x_theta: float
            Angle of x arm from north in radians.
        """

        # Earth radius
        R0 = 6.371e6

        # Initialise base class
        VectorOnSphere.__init__(self, R0, theta, phi, x_theta)


def radians(d,m=0,s=0):
    """ Convert degrees, minutes and seconds to radians. """
    return (d + m/60. + s/3600.)/360 *2*pi

def latitude_radians(d,m,s,D):
    """ Convert latitude coordinate to radians """
    if D == 'N':
        return pi-radians(d,m,s)
    elif D == 'S':
        return pi+radians(d,m,s)
    else:
        raise ValueError("Unknown latitude coordinate %s. Options are 'N' or 'S'" % D)

def longitude_radians(d,m,s,D):
    """ Convert longitude coordinate to radians """
    if D == 'E':
        return radians(d,m,s)
    elif D == 'W':
        return -radians(d,m,s)
    else:
        raise ValueError("Unknown longitude coordinate %s. Options are 'E' or 'W'" % D)


# http://www.ligo.org/scientists/GW100916/GW100916-geometry.html
LHO = Detector(latitude_radians(46,27,19,'N'),longitude_radians(119,24,28,'W'),radians(36))
LLO = Detector(latitude_radians(30,33,46,'N'),longitude_radians(90,46,27,'W'),radians(90+18))
Virgo = Detector(latitude_radians(43,37,53,'N'),longitude_radians(10,30,16,'E'),radians(-19))

