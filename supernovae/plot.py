from SNE.distance import D_Ls, distance_modulus
from SNE.supernova_data import zs, mods, mod_errs

# Plot Supernovae distance modulus data against redshift
fig, ax = plt.subplots(figsize=figsize)
ax.errorbar(zs, mods, yerr=mod_errs,capthick=0.1,markersize=0,linewidth=0.1)
ax.set_xlabel(r'redshift ($z$)')
ax.set_ylabel(r'distance modulus ($\mu=m-M$)')

# Plot theoretical curve
d_Ls_theory = D_Ls(zs, O_L=0.7, O_m=0.3)
mods_theory = distance_modulus(d_Ls_theory)

ax.plot(zs, mods_theory)
fig.tight_layout()
fig.savefig('../figures/supernovae.pdf')
