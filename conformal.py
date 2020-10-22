import numpy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Omegabh2 = 0.022383
Omegach2 = 0.12011
H0 = 67.32

h = H0/100
Omegar = 4.18343e-5/h**2
Omegam = (Omegach2 + Omegabh2)/h**2
OmegaL = 1-Omegar-Omegam

def f_a_t(t, y):
    a, = y
    return H0*numpy.sqrt(Omegar*a**-2 + Omegam*a**-1 + OmegaL*a**2)

def today(t, y):
    a, = y
    return a-1

t = numpy.logspace(-8,0,1000)
t0 = t[0]
t1 = t[-1]
a0 = numpy.sqrt(2*H0*numpy.sqrt(Omegar)*t0)
sol = solve_ivp(f_a_t, [t0,t1], [a0], events=[today], t_eval=t)

Mpc = 3.086e19

sol.t_events[0][0] * Mpc / (60*60*24*365) /1e9
t0 = sol.t_events[0][0]

fig, axes = plt.subplots(2,2,sharey='row', sharex='col', gridspec_kw={'hspace':0,'wspace':0})

axes[0,0].plot(sol.t,sol.y[0])
axes[1,0].plot(sol.t,1/sol.y[0])

#ax.set_yscale('log')
#ax.set_ylim(1e-3,1e3)
#ax.set_yticklabels([1])
#ax.set_yticks([1])
#ax.set_xlim(-t0*0.1, t0*3)
#ax.set_xticks([0,t0,2*t0])
#ax.set_xticklabels([0,'$t_0$',r'$\to\infty$'])



def f_s_eta(t, y):
    s, = y
    return -H0*numpy.sqrt(Omegar*s**4 + Omegam*numpy.abs(s)**3 + OmegaL)

def fcb(t, y):
    s, = y
    return s

fcb.terminal = True


#eta = numpy.logspace(-8,0,1000)
eta = numpy.linspace(1e-8,1,10000)
eta0 = eta[0]
eta1 = eta[-1]
s0 = 1/H0/numpy.sqrt(Omegar)/eta0
sol = solve_ivp(f_s_eta, [eta0,eta1], [s0], events=[fcb,today], t_eval=eta)
eta_fcb = sol.t_events[0][0]
eta_0 = sol.t_events[1][0]

#fig, ax = plt.subplots()
axes[0,1].plot(sol.t,1/sol.y[0])
axes[0,1].plot(2*eta_fcb - sol.t,1/sol.y[0])
#ax.set_yscale('log')
#ax.set_xticks([0,eta_0,eta_fcb])
#ax.set_xticklabels([0,'$\eta_0$','FCB'])

axes[1,1].plot(sol.t,sol.y[0])
axes[1,1].plot(2*eta_fcb-sol.t,-sol.y[0])
#ax.set_ylim(-10,10)

axes[1,0].set_xlabel('$t$')
axes[1,0].set_xlim(-t0*0.1, t0*2.2)
axes[1,0].set_xticks([0,t0,2*t0])
axes[1,0].set_xticklabels([0,'$t_0$',r'$\to\infty$'])


axes[1,1].set_xlabel(r'$\eta$')
axes[1,1].set_xticks([0,eta_0,eta_fcb,2*eta_fcb])
axes[1,1].set_xticklabels([0,'$\eta_0$','FCB', r'$\eta_\mathrm{end}$'])

axes[0,0].set_ylabel('$|a|$')
axes[0,0].set_yscale('log')
axes[0,0].set_ylim(1e-3,1e3)
axes[0,0].set_yticks([1e-2,1,1e2])


axes[1,0].set_ylabel('$s=1/a$')
axes[1,0].set_ylim(-10,10)
axes[1,0].set_yticks([-5,0,5])

for ax in axes[:,1]:
    ax.axvline(eta_fcb, color='k', linestyle='--', linewidth=0.5)

fig.set_size_inches(5,3.5)
fig.tight_layout()
fig.savefig('evolution.pdf')
