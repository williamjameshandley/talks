"""
You will need to download and unpack these files from zenodo:
    https://zenodo.org/record/4116393/files/data.1902.04029.tar.gz
    https://zenodo.org/record/3371152/files/data.1908.09139.tar.gz
    https://zenodo.org/record/4554118/files/data.2007.08496.tar.gz

For the tehran talk plots anesthetic 2.0.0-beta.5 was used:
    https://github.com/williamjameshandley/anesthetic/releases/tag/2.0.0-beta.5
"""

from anesthetic import NestedSamples, MCMCSamples
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup ----------------------------------------
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams["figure.figsize"] = (2.87343,2.87343)
plt.rcParams["savefig.bbox"] = "tight"


np.random.seed(0)
if not os.path.exists('figures'):
    os.makedirs('figures')


# DES vs Planck figure -------------------------
DES = NestedSamples(root='data.1902.04029/runs_default/chains/DES',label='DES')
planck = NestedSamples(root='data.1902.04029/runs_default/chains/planck', label=r'\textit{Planck}')
DES_planck = NestedSamples(root='data.1902.04029/runs_default/chains/DES_planck',label=r'DES+\textit{Planck}')

fig, axes = DES.plot_2d([['omegam'], ['sigma8']])
planck.plot_2d(axes,alpha=0.9)
ax = axes.iloc[-1,0]
def reset():
    ax.set_xlim(0.17,0.35)
    ax.set_ylim(0.7,1.08)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    fig.tight_layout()

reset()
fig.savefig('figures/DES_planck_1.pdf')

DES_planck.plot_2d(axes,alpha=0.8)
ax.get_legend().remove()
reset()
fig.savefig('figures/DES_planck_2.pdf')



# Curvature figure -----------------------------
planck = NestedSamples(root='data.1908.09139/klcdm/chains/planck',label=r'\textit{Planck}')
planck_lensing = NestedSamples(root='data.1908.09139/klcdm/chains/planck_lensing',label=r'+CMB lensing')
planck_lensing_BAO = NestedSamples(root='data.1908.09139/klcdm/chains/planck_lensing_BAO',label=r'+BAO')
BAO = NestedSamples(root='data.1908.09139/klcdm/chains/BAO', label=r'BAO')

fig, axes = BAO.plot_2d([['omegak'], ['H0']])
planck.plot_2d(axes)
ax = axes.iloc[-1,0]
def reset():
    ax.set_xlim(-0.085,0.035)
    ax.set_ylim(46,73)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left')
    fig.tight_layout()

reset()
fig.savefig('figures/curvature_1.pdf')

planck_lensing.plot_2d(axes,alpha=0.7)
ax.get_legend().remove()
reset()
fig.savefig('figures/curvature_2.pdf')

planck_lensing_BAO.plot_2d(axes)
ax.axvline(0,linestyle=':',color='k')
ax.get_legend().remove()
reset()
fig.savefig('figures/curvature_3.pdf')


# Planck vs ACT figure -------------------------

act = MCMCSamples(root='data.2007.08496/chains/ACTPol_lcdm', label='ACT')
planck = MCMCSamples(root='data.2007.08496/chains/base_plikHM_TTTEEE_lowl_lowE_lensing', label=r'\textit{Planck}') 

params = planck.columns[:6]
fig, axes = act.plot_2d(params,types={'lower':'kde','diagonal':'kde'})
planck.plot_2d(axes)
ax = axes.iloc[-1,0]
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.tight_layout()
size = fig.get_size_inches()
fig.set_size_inches(2*size[0],2*size[1])
fig.savefig('figures/act_planck.pdf')

Sig_planck = planck[params].cov()
mu_planck = planck[params].mean()

Sig_act = act[params].cov() 
mu_act = act[params].mean() 

def n(muA, SigA, muB, SigB):
    return np.linalg.solve(SigA + SigB,muA-muB)

nhat = n(mu_act, Sig_act, mu_planck, Sig_planck)
nhat /= abs(nhat).max()

planck['t'] = nhat @ planck[params].T 
planck.tex['t'] = '$t$'
act['t'] = nhat @ act[params].T 
act.tex['t'] = '$t$'
params=['ns', 't']
fig, axes = act.plot_2d(params,types={'lower':'kde','diagonal':'kde'})
planck.plot_2d(axes)
ax = axes.iloc[-1,0]
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.tight_layout()
fig.savefig('figures/act_planck_t.pdf')
