from ligo.source import BinaryMerger
from ligo.detector import LHO,LLO,Virgo
from ligo.likelihood import Likelihood
from numpy import pi
from scipy.constants import c, G
import sys
import numpy as np

# Source
# ------
m0 = 2e30               # Solar mass
Mpc = 3e22              # MegaParsecs in meters

m1 = 35 * m0            # 35 solar mass black hole
m2 = 25 * m0            # 25 solar mass black hole
phi, theta = pi, pi/2  # angular sky location of merger
phi, theta = 0, 0
r = 390 * Mpc           # Luminosity distance of source

i = pi/4                # inclination
p = pi/4                # angle of major axis (measured in radians from north)
t_c = 0                 # Merger time
Phi_c = 0               # Merger phase

# Construct source
order = 1
source = BinaryMerger(m1, m2, r, theta, phi, p, i, t_c, Phi_c, order)


# Likelihood
# ----------
T = G * source.m / c**3 # Define relevant timescale of problem

noise = 1e-21           # Noise level
t_c = source.t_c        # Get actual merger time

detectors = [LLO,Virgo] # Choice of detectors
tmin = t_c-1000*T       # Start time
tmax = t_c+1000*T       # End time
ndata = 1000
timedata = np.linspace(tmin, tmax, ndata)

# Construct likelihood
ligo_loglikelihood = Likelihood(source, detectors, noise, timedata)

l0 = ligo_loglikelihood(m1, m2, r, theta, phi, p, i, t_c, Phi_c)
