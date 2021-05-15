import matplotlib.pyplot as plt
import numpy

from SNE.distance import D_Ls, distance_modulus
from SNE.supernova_data import zs, mods, mod_errs


# Plot Supernovae distance modulus data against redshift
fig, ax = plt.subplots()
ax.errorbar(zs, mods, yerr=mod_errs, fmt='.')
ax.set_xlabel(r'redshift ($z$)')
ax.set_ylabel(r'distance modulus ($\mu=m-M$)')

# Plot theoretical curve
d_Ls_theory = D_Ls(zs, O_L=0.7, O_m=0.3)
mods_theory = distance_modulus(d_Ls_theory)

ax.plot(zs, mods_theory)
plt.show()


# Plot Supernovae luminosity distance against redshift
lums = 10.**(mods/5.+1.)* 1e-9
lum_errs = lums * 0.2*numpy.log(10) * mod_errs

fig, ax = plt.subplots()
ax.errorbar(zs, lums, yerr=lum_errs, fmt='.')
ax.set_xlabel(r'redshift ($z$)')
ax.set_ylabel('luminosity distance ($D_L$ / Gpc)')

# Plot theoretical curve
ax.plot(zs, d_Ls_theory * 1e-3)
plt.show()
