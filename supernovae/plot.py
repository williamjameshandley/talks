""" Plot the contours.

    Note that you need to generate the contours first
    (see compute_contours.py for an example).
"""
import matplotlib.pyplot
import numpy
from fgivenx.contours import Contours
from SNE.supernova_data import zs, mods, mod_errs

# Set up the grid of axes
fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(1,1,1)

# plot the contours
contourfile = 'posterior_mat_DE.pkl'
contours = Contours.load(contourfile)
colours = contours.plot(ax)

# x & y labels
ax.set_xlabel('$z$')
ax.set_ylabel('luminosity distance')

# Add a colorbar (essential to do this after tight_layout)
cbar = fig.colorbar(colours, ax=ax, ticks=[1, 2, 3], pad=0.01)
cbar.ax.set_yticklabels(['$1\\sigma$', '$2\\sigma$', '$3\\sigma$'])

lums = 10.**(mods/5.+1.) * 1e-6
lum_errs = lums * 0.2*numpy.log(10) * mod_errs 
ax.errorbar(zs, lums, yerr=lum_errs, fmt='.')

# Plot to file
matplotlib.pyplot.show()
