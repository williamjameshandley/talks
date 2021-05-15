import numpy as np
from numpy import sin, cos

class VectorOnSphere(object):
    def __init__(self, r, theta, phi, v_theta):
        """ Vector on a sphere. 
        Parameters
        ----------
        r: float
            Radius of sphere
        theta: float
            Latitude of detector in radians.
        phi: float
            Longitude of detector in radians.
        v_theta:
            Rotation of vector on sphere as measured from North.
        """
        self.r = r
        self.theta = theta
        self.phi = phi
        self.v_theta = v_theta

    @property 
    def n(self):
        """ Unit vector pointing to theta, phi. """
        theta = self.theta
        phi = self.phi
        return np.array([
            sin(theta)*cos(phi),
            sin(theta)*sin(phi),
            cos(theta)
            ])
    @property
    def x(self):
        """ Position vector of object. """
        return self.r * self.n

    @property
    def e(self):
        """ Frame of vectors at this location.

        Returns
        -------
        3 x 3 numpy.array 
            set of basis vectors:
                x: defined by self.v_theta
                z: defined by self.n

        """
        theta = self.theta
        phi = self.phi
        t = self.v_theta
        p = np.array([
            cos(t)*cos(theta)*cos(phi)-sin(t)*sin(phi),
            cos(t)*cos(theta)*sin(phi)+sin(t)*cos(phi),
            -cos(t)*sin(theta)
            ])
        n = self.n
        q = np.cross(n,p)
        return np.stack((p,q,n)).T
