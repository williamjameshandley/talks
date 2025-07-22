#!/usr/bin/env python3
"""
Analyzing exoplanet transit data - demonstrating autocomplete in action
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits

# Load transit photometry data
data = np.loadtxt('transit_lightcurve.dat')
time = data[:, 0]  # Time in days
flux = data[:, 1]  # Normalized flux
flux_error = data[:, 2]  # Flux uncertainties

# Define transit parameters (autocomplete helps with parameter names)
planet_radius = 1.2 * u.R_earth    # Planet radius
stellar_radius = 0.8 * u.R_sun      # Stellar radius
orbital_period = 3.52 * u.day       # Orbital period
transit_midtime = Time('2024-01-15T12:34:56')

# Calculate transit depth (autocomplete suggests methods and attributes)
transit_depth = (planet_radius / stellar_radius)**2
print(f"Expected transit depth: {transit_depth:.6f}")

# Fit transit model (autocomplete helps with function parameters)
def transit_model(t, depth, duration, t0):
    # Simple box-shaped transit model
    model = np.ones_like(t)
    in_transit = np.abs(t - t0) < duration/2
    model[in_transit] = 1 - depth
    return model

# Phase fold the data (autocomplete suggests numpy functions)
phase = np.mod(time - transit_midtime.jd, orbital_period.to(u.day).value)
phase[phase > orbital_period.to(u.day).value/2] -= orbital_period.to(u.day).value

# Create plots (autocomplete helps with matplotlib methods)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Raw lightcurve
ax1.scatter(time, flux, s=1, alpha=0.6, c='steelblue')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Normalized Flux')
ax1.set_title('Transit Lightcurve')

# Phase-folded data
ax2.scatter(phase, flux, s=1, alpha=0.6, c='darkred')
ax2.set_xlabel('Orbital Phase (days)')
ax2.set_ylabel('Normalized Flux')
ax2.set_title('Phase-folded Transit')

# TODO: Fit the transit model to data
# TODO: Calculate planet properties from best-fit parameters
# TODO: Estimate uncertainties using MCMC

plt.tight_layout()
plt.show()