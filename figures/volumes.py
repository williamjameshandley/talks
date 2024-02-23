import lecture_style
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()


def pi(x,y):
    return -(x**2+y-11)**2 - (x+y**2-7)**2

def P(x,y):
    x0 = 1
    y0 = 1
    return -(x-x0)**2-(x-x0)*(y-y0)-(y-y0)**2

x = np.linspace(-6,6,100)
y = x.copy()
x, y = np.meshgrid(x, y)
ax.contourf(x, y, pi(x,y), levels=[-200,1e3], colors='C0')

x = np.linspace(-6,6,100)
y = x.copy()
x, y = np.meshgrid(x, y)
ax.contourf(x, y, P(x,y), levels=[-2,1e3], colors='C1')

ax.text(-3,3, '$\mathcal{R}_\pi$')
ax.text(0,1, '$\mathcal{R}_\mathcal{P}$')
ax.text(5,5, r'$\theta$')

ax.annotate('', (-5.5,-5.3), (5.5,-5.3), arrowprops=dict(arrowstyle='<->'))
ax.text(0,-5.5,'$\ell_\pi$')

ax.annotate('', (-1,-1), (3,-1), arrowprops=dict(arrowstyle='<->'))
ax.text(0.5,-1.5,'$\ell_\mathcal{P}$')

ax.set_xticks([])
ax.set_yticks([])
fig.set_size_inches(lecture_style.width*0.4, lecture_style.width*0.4)
fig.tight_layout()
fig.savefig('volumes.pdf')
