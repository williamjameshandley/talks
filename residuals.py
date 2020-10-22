import numpy
import matplotlib.pyplot as plt
import pandas

fit = numpy.loadtxt('./COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')
fit = pandas.DataFrame(fit, columns=['l','TT','TE','EE','BB','PP'])
fit.l = fit.l.astype(int)
fit.set_index('l',inplace=True)


data_lowl = numpy.loadtxt('./planck_spectrum_TT_lowl.txt')
data_highl = numpy.loadtxt('./planck_spectrum_TT_highl.txt')
data = pandas.DataFrame(numpy.concatenate((data_lowl,data_highl)), columns=['l','Dl','sigm','sigp'])
data.l = data.l.astype(int)
data.set_index('l',inplace=True)

shift = ((data.Dl-fit.loc[data.index].TT)/numpy.where(fit.loc[data.index].TT-data.Dl>0,data.sigp,data.sigm))

fig, axes = plt.subplots(1,2,sharey=True, gridspec_kw={'wspace':0, 'width_ratios':[1,2]})


axes[0].errorbar(shift.index[:28], shift[:28], yerr=1, fmt='.', color='r', ecolor='b',marker='o',linestyle='None',capsize=0, markersize=3,zorder=2000,elinewidth=1, markeredgecolor='k', markeredgewidth=0.5)
axes[0].set_xlim(1.5,30)
axes[0].axhline(0)
axes[0].set_xscale('log')
axes[0].set_xticks([2,10,30])
axes[0].set_xticklabels([2,10,30])
axes[0].set_yticks([-3,-2,-1,1,2,3])

#for ax in axes:
#    for tick in ax.get_yticks():
#        ax.axhline(tick, color='k', linestyle='--', linewidth=0.5)


axes[1].axhline(0)
axes[1].errorbar(shift.index[28:], shift[28:], yerr=1, fmt='.', color='r', ecolor='b',marker='o',linestyle='None',capsize=0, markersize=3,zorder=2000,elinewidth=1, markeredgecolor='k', markeredgewidth=0.5)
axes[1].set_xlim(30,2600)

axes[1].set_xlabel('$\ell$')
axes[0].set_ylabel('Normalised residual $(\Delta\mathcal{D}_\ell^{TT}/\sigma_\ell^{TT})$')
fig.set_size_inches(7,3)
fig.tight_layout()
fig.savefig('residuals.pdf')
