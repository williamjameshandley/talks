from anesthetic import MCMCSamples, NestedSamples, make_2d_axes
import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = "cm"

#rootdir = '/data/will/data/pablo/runs_default/chains/'
#
#DES = NestedSamples(root=rootdir + 'DES')
#planck = NestedSamples(root=rootdir + 'planck')
#SH0ES = NestedSamples(root=rootdir + 'SH0ES')
#SH0ES_planck = NestedSamples(root=rootdir + 'SH0ES_planck')
#BAO = NestedSamples(root=rootdir + 'BAO')
#
#fig, axes = planck.plot_1d('H0', label=r'Planck')
#SH0ES.plot_1d(axes, label='S$H_0$ES')
#axes['H0'].set_xticks([67.4,74])
#axes['H0'].set_xticklabels(['$67.4\pm0.5$','$74\pm1.4$'])
#legend = fig.legend()
#fig.set_size_inches(2.3,2.3)
#fig.tight_layout()
#fig.savefig('H0.pdf', bbox_inches='tight')
#
#SH0ES_planck.plot_1d(axes, label='both')
#legend.remove()
#fig.legend()
#fig.savefig('H0_combined.pdf', bbox_inches='tight')
#
#fig, axes = planck.plot_2d([['omegam'],['sigma8']], label=r'Planck',zorder=1000)
#BAO.plot_2d(axes,label='BAO')
#DES.plot_2d(axes,label='DES')
#fig.legend()
#fig.set_size_inches(2.3,2.3)
#fig.tight_layout()
#fig.savefig('DES_planck.pdf', bbox_inches='tight')
#ymin, ymax = fig.axes[0].get_ylim()
#xmin, xmax = fig.axes[0].get_xlim()
#
#fig, axes = planck.plot_2d([['omegam'],['sigma8']], label=r'Planck',zorder=1000)
#fig.axes[0].set_ylim(ymin, ymax)
#fig.axes[0].set_xlim(xmin, xmax)
#BAO.plot_2d(axes,label='BAO')
#fig.legend()
#fig.set_size_inches(2.3,2.3)
#fig.tight_layout()
#fig.savefig('BAO_planck.pdf', bbox_inches='tight')
#
#planck_mcmc = MCMCSamples(root='/data/will/PLA_3/COM_CosmoParams_fullGrid_R3.00/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE')
#planck_lensing_mcmc = MCMCSamples(root='/data/will/PLA_3/COM_CosmoParams_fullGrid_R3.00/base_omegak/plikHM_TTTEEE_lowl_lowE_lensing/base_omegak_plikHM_TTTEEE_lowl_lowE_lensing')
#
#fig, axes = planck_mcmc.plot_1d('omegak', label='Planck')
#fig.axes[0].plot([],[])
#fig.axes[0].set_xlim(-0.1,0.1)
#planck_lensing_mcmc.plot_1d(axes, label='Planck+lensing')
#fig.legend()
#fig.set_size_inches(2.3,2.3)
#fig.tight_layout()
#fig.savefig('curvature.pdf', bbox_inches='tight')
#
#DES.plot_2d([['omegamh2'],['S8']], label=r'DES')
#
#
#rootdir = '/data/will/data/pablo/runs_default/chains/'
#planck = NestedSamples(root=rootdir + 'planck')
#fig, axes = planck.set_beta(0).plot_2d([['omegam'],['sigma8']])
#rootdir = '/data/will/data/pablo/runs_medium/chains/'
#planck = NestedSamples(root=rootdir + 'planck')
#planck.set_beta(0).plot_2d(axes)
#rootdir = '/data/will/data/pablo/runs_narrow/chains/'
#planck = NestedSamples(root=rootdir + 'planck')
#planck.set_beta(0).plot_2d(axes)
#
#
#
#
#
import tqdm



data = {}
for typ in tqdm.tqdm(['default','medium','narrow']):
    rootdir = '/data/will/data/pablo/runs_%s/chains/' % typ
    data[typ] = {}
    for root in tqdm.tqdm(['DES','planck','DES_planck']):
        data[typ][root] = NestedSamples(root=rootdir + root) 



fig, axes = plt.subplots(1,3)
ax = axes[0]
typ = 'default'
for ax, typ in zip(axes,data):

    planck = data[typ]['planck']
    DES = data[typ]['DES']
    DES_planck = data[typ]['DES_planck']
    planck.plot(ax, 'omegam','sigma8',zorder=1000) 
    DES.plot(ax, 'omegam','sigma8') 
    ax.set_ylim(planck.set_beta(0)['sigma8'].quantile([0.01,0.99])) 
    ax.set_xlim(planck.set_beta(0)['omegam'].quantile([1e-4,0.99]))
    ax.set_xticks([])
    ax.set_yticks([])

    des = DES.ns_output()
    plk = planck.ns_output() 
    des_plk = DES_planck.ns_output() 

    logR = des_plk.logZ - (des.logZ + plk.logZ)
    ax.set_title('$\log R=%.2f \pm %.2f$' % (logR.mean(),logR.std()))

axes[0].set_ylabel(planck.tex['sigma8']) 
axes[1].set_xlabel(planck.tex['omegam']) 
fig.set_size_inches(5.4,1.8)
fig.tight_layout()
fig.savefig('prior_dependency.pdf', bbox_inches='tight') 

logR = des_plk.logZ - (des.logZ + plk.logZ)
logI = (des.D + plk.D) - des_plk.D 
logS = logR - logI
d = (des.d + plk.d) - des_plk.d
(d - 2*logS)
import scipy.stats
(1-scipy.stats.chi2.cdf(d[d>0]-2*logS[d>0],d[d>0]) ).mean()


logR = des_plk.logZ - (des.logZ + plk.logZ)
logI = (des.D + plk.D) - des_plk.D 
logS = logR - logI
d = (des.d + plk.d) - des_plk.d
((d - 2*logS)/d).mean()

