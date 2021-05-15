import numpy as np
from scipy.constants import c
from numpy import pi
from ligo.source import BinaryMerger

class Likelihood:
    def __init__(self, source, detectors, noise, timedata):
        """ Initialise a likelihood.

        Parameters
        ----------
        source: BinaryMerger
            The details of the binary merger source.

        detectors: List[Detector]
            The detectors that we are observing with.

        noise: Float
            The Gaussian white noise level (i.e. standard deviation).

        timedata: numpy.array
            The times that the detectors observe at.
        """

        # Load in the details
        self.detectors = detectors
        self.noise = noise
        self.ts = timedata

        # Compute the underlying signal from the true source.
        self.signal0 = self.signal(source)

        # Add the noise
        np.random.seed(0)
        self.signal0 += np.random.normal(0,self.noise,(len(self.detectors),len(self.ts)))


    def signal(self, source, t=None):
        """ Compute the signal as seen by the detectors. 

        Parameters
        ----------
        source: Binary Merger
            The details of the trial source.

        Returns
        -------
        N x M array where:
            N: number of detectors.
            M: number of time points.
        """

        ans = []
        x0 = self.detectors[0].x

        for detector in self.detectors:
            # plus and cross GW polarisations in GW frame
            h_plus = np.array([[1,0,0],[0,-1,0],[0,0,0]])
            h_cross = np.array([[0,1,0],[1,0,0],[0,0,0]])

            # Sensitivity of detector is just h_plus in its own frame
            A = h_plus

            # source and detector frames in earth coordinates
            ep = source.e 
            e = detector.e # detector frame in earth coordinates

            # Rotate plus and cross polarisations into the detector frame
            F_plus =  0.5*np.trace(A.dot(e.T.dot(ep.dot(h_plus.dot(ep.T.dot(e))))))
            F_cross = 0.5*np.trace(A.dot(e.T.dot(ep.dot(h_cross.dot(ep.T.dot(e))))))

            # Compute time from detector to baseline
            dt = source.n.dot(detector.x-x0)/c

            # Compute plus and cross polarisations at time points
            if t is None:
                h_plus = source.h(self.ts-dt,'+')
                h_cross = source.h(self.ts-dt,'x')
            else:
                h_plus = source.h(t-dt,'+')
                h_cross = source.h(t-dt,'x')

            # Compute strain as seen by detector
            h = F_plus * h_plus + F_cross * h_cross

            ans.append(h)

        # Return an array of strains as seen by each detector, zeroing out points beyond t_c
        return np.nan_to_num(np.array(ans))
        

    def __call__(self, m1, m2, r, theta, phi, p, i, t_c, Phi_c):
        """ The log-likelihood of a trial source."""

        # Construct Binary merger for these parameters
        source = BinaryMerger(m1, m2, r, theta, phi, p, i, t_c, Phi_c)

        # Compute signal for these parameters
        signal1 = self.signal(source)

        # Compute difference between observed signal and trial
        ds = signal1 - self.signal0
        
        # Return log likelihood
        return -len(self.detectors)*len(self.ts)/2. * np.log(2*pi*self.noise**2) - np.trace(ds.dot(ds.T))/self.noise**2/2
