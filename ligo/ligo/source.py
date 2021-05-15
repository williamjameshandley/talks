import numpy as np

from scipy.constants import G, c 
from numpy import sin, cos, log, pi
from ligo.utils import VectorOnSphere

class BinaryMass(object): 
    def __init__(self, m1, m2):
        """ Define an Binary Mass system.

            Parameters
            ----------
            m1 : float
                mass of greater mass object.
            m2 : float
                mass of lesser mass object.
        """
        if m1 < m2:
            raise ValueError("Invalid mass combination: m1 < m2 (%f < %f)" % (m1, m2))
        self.m1 = m1
        self.m2 = m2

    @property
    def m(self):
        """ The combined mass of the binary. """
        return self.m1 + self.m2

    @m.setter
    def m(self,m):
        """ Set the combined mass, keeping nu constant. """
        nu = self.nu
        self.m1 = m/2. * (1 + np.sqrt(1-4*nu))
        self.m2 = m/2. * (1 - np.sqrt(1-4*nu))

    @property
    def nu(self):
        """ Conjugate to combined mass. """
        return self.m1*self.m2 / self.m**2

    @nu.setter
    def nu(self,nu):
        """ Set nu, keeping the combined mass m constant. """
        if nu > 0.25 or nu <= 0:
            raise ValueError("Invalid nu: %f. nu must be 0 < nu <= 0.25" % nu)
        m = self.m
        self.m1 = m/2. * (1 + np.sqrt(1-4*nu))
        self.m2 = m/2. * (1 - np.sqrt(1-4*nu))

    @property
    def dm(self):
        """ The difference in mass. """
        return self.m1 - self.m2

    @dm.setter
    def dm(self, dm):
        """ Set the difference in mass, keeping the combined mass constant. """
        if dm/2 > self.m:
            raise ValueError("Invalid dm: %f. This dm would result in negative m2 (m=%f)" % (dm, self.m))
        m = self.m
        self.m1 = m + abs(dm)/2.
        self.m2 = m - abs(dm)/2.

    @property
    def Delta(self):
        """ Dimensionless difference in mass. """
        return self.dm/self.m

    @Delta.setter
    def Delta(self,Delta):
        """ Set the value of the Dimensionless mass difference, holding the combined mass constant. """
        self.dm = Delta * self.m



class BinaryCircle(VectorOnSphere):
    def __init__(self, r, theta, phi, p, i):
        """ Define a binary circular orbiting system. 
        """
        VectorOnSphere.__init__(self, r, theta, phi, p)
        self.i = i

class BinaryMerger(BinaryMass, BinaryCircle):
    def __init__(self, m1, m2, r, theta, phi, p, i, t_c, Phi_c, order=1):
        """ Define a binary merger source. """
        BinaryCircle.__init__(self, r, theta, phi, p, i)
        BinaryMass.__init__(self, m1, m2)

        self.Phi_c = Phi_c
        self.t_c = t_c

        self.w0 = 1
        self.order = order

    def x(self, t):
        return (G*self.m*self.w(t)/c**3)**(2./3)


    def tau(self, t):
        """ Dimensionless time parameter """
        return c**3 * self.nu / (5*G*self.m) * (self.t_c - t)

    def Phi(self, t):
        """ Orbital phase """
        tau = self.tau(t)
        return self.Phi_c - 1./self.nu * (
                tau**(5./8)
                + (3715./8064 + 55./96*self.nu)*tau**(3./8)
                - 3*pi/4 * tau**(1./4)
                + (9275495./14450688 + 284875./258048*self.nu + 1855./2048*self.nu**2)*tau**(1./8)
                )

    def w(self, t):
        """ Orbital frequency """
        tau = self.tau(t)
        nu = self.nu
        return c**3/(8*G*self.m) * (
                tau**(-3./8)
                + (743./2688 + 11./32*nu)*tau**(-5./8)
                - 3*pi/10*tau**(-3./4)
                + (1855099./14450688 + 56975./258048*nu + 371./2048*nu**2)*tau**(-7./8)
                )

    def psi(self, t):
        """ Tail-less phase. """
        w = self.w(t)
        return self.Phi(t) - 2*G*self.m*w/c**3*log(w/self.w0)

    def ppn(self, t, pol, o):
        psi = self.psi(t)
        nu = self.nu
        Delta = self.Delta
        ci = cos(self.i)
        si = sin(self.i)
        if pol == '+':
            if o == 0:
                return -(1+ci**2) * cos(2*psi) #- 1./96 * si**2 * (17 + ci**2)
            elif o == 0.5:
                return -si * Delta * (
                        cos(psi) * (5./8 + 1./8*ci**2) 
                        - cos(3*psi) *(9./8 + 9./8*ci**2)
                        )
            elif o == 1:
                return (
                        cos(2*psi) * (
                            19./6 + 3./2 * ci**2 - 1./3 * ci**4 
                            + nu * (-19./6 + 11./6*ci**2 + ci**4)
                            ) 
                        - cos(4*psi) * (4./3 * si**2 *(1+ci**2)*(1-3*nu))
                        )
            elif o == 1.5:
                return (
                        si*Delta * cos(psi) * (
                            19./64 + 5./16*ci**2 - 1./192*ci**4 
                            + nu*(-49./96 + 1./8 * ci**2 + 1./96*ci**4)
                            )
                        + cos(2*psi) * (-2*pi*(1+ci**2))
                        + si * Delta * cos(3*psi)*(
                            -657./128 - 45./16*ci**2 + 81./128*ci**4 
                            + nu*(225./64 - 9./8*ci**2 - 81./64*ci**4)
                            )
                        + si * Delta * cos(5*psi)*(
                            625./384 * si**2*(1+ci**2)*(1-2*nu)
                            )
                        )
            elif o == 2:
                return (
                        pi*si*Delta*cos(psi)*(-5./8-1./8*ci**2)
                        + cos(2*psi)*(
                            11./60 + 33./10*ci**2 + 29./24*ci**4-1./24*ci**6
                            + nu*(353./36 - 3*ci**2 - 251./72*ci**4 + 5/24*ci**6)
                            + nu**2*(-49./12 + 9./2*ci**2 -7./24*ci**4 - 5./24*ci**6)
                            )
                        + pi*si*Delta*cos(3*psi)*(27./8*(1+ci**2))
                        +2./15*si**2*cos(4*psi)*(
                            59 + 35*ci**2 - 8*ci**4
                            -5./3*nu*(131+59*ci**2-24*ci**4)
                            +5*nu**2*(21-3*ci**2-8*ci**4)
                            )
                        +cos(6*psi)*(-81./40*si**4*(1+ci**2)*(1-5*nu+5*nu**2))
                        +si*Delta*sin(psi)*(11./40 + 5*log(2)/4 + ci**2*(7./40+log(2)/4))
                        +si*Delta*sin(3*psi)*(-189./40+27./4*log(3./2))*(1+ci**2)
                        )

            else:
                raise NotImplementedError("order %f ppn has not been implemented" % o)

        elif pol == 'x':
            if o == 0:
                return -2*ci*sin(2*psi)
            elif o == 0.5:
                return si*ci*Delta*( -3./4*sin(psi)+9./4*sin(3*psi))
            elif o == 1:
                return (
                        ci*sin(2*psi)*(
                            17./3 - 4./3*ci**2 
                            + nu*(-13./3 + 4*ci**2)) 
                        + ci*si**2 * sin(4*psi) * (-8./3*(1-3*nu))
                        )
            elif o == 1.5:
                return (
                        si*ci*Delta * sin(psi) * (
                            21./32 - 5./96*ci**2 
                            + nu*(-23./48 + 5./48*ci**2)
                            )
                        -4*pi*ci*sin(2*psi)
                        +si*ci*Delta*sin(3*psi)*(
                            -603./64 + 135./64*ci**2 + nu*(171./32 - 135./32*ci**2)
                            )
                        +si*ci*Delta*sin(5*psi)*(625./192*(1-2*nu)*si**2)
                        )
            elif o == 2:
                return (
                        si*ci*Delta*cos(psi)*(-9./20 - 3./2*log(2))
                        +si*ci*Delta*cos(3*psi)*(189./20 - 27./2*log(3./2))
                        -si*ci*Delta*3*pi/4*sin(psi)
                        +ci*sin(2*psi)*(
                            17./15 + 113./30*ci**2 - 1./4*ci**4
                            + nu*(143./9 - 245./18*ci**2 + 5./4*ci**4)
                            + nu**2*(-14./3 + 35./6*ci**2 - 5./4*ci**4)
                            )
                        +si*ci*sin(3*psi)*27*pi/4
                        +4./15*ci*si**2*sin(4*psi)*(
                            55-12*ci**2
                            -5./3*nu*(119-36*ci**2)
                            +5*nu**2*(17-12*ci**2)
                            )
                        +ci*sin(6*psi)*(-81./20*si**4*(1-5*nu+5*nu**2))
                        )
            else:
                raise NotImplementedError("order %f ppn has not been implemented" % o)
        else:
            raise ValueError("Unrecognised polarisation: %si. Must be either '+' or 'x'" % pol)

    def h(self, t, pol):
        o = 0.
        ans = 0.
        x = self.x(t)
        while o <= self.order:
            ans += x**o * self.ppn(t, pol, o)
            o += 0.5
        return 2 * G * self.m * self.nu/c**2 / self.r * x * ans
