import lecture_style
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects, rcParams

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
ax.contourf(x, y, pi(x,y), levels=[-200,1e3], colors='C1')
ax.contourf(x, y, P(x,y), levels=[-5,1e3], colors='C0')

ax.text(-4,-3, r'$\pi = \frac{1}{V_\pi}$')
ax.text(0,1, r'$\mathcal{P} = \frac{1}{V_\mathcal{P}}$')
ax.text(5,5, r'$\theta$')

ax.set_xticks([])
ax.set_yticks([])
fig.set_size_inches(lecture_style.width*0.5, lecture_style.width*0.5)
fig.tight_layout()
fig.savefig('volumes.pdf')

np.random.seed(1)
fig, ax = plt.subplots()
ax.contourf(x, y, pi(x,y), levels=[-200,1e3], colors='C1')
x, y = np.random.uniform(-6,6, (2,400))
i = P(x,y) > -5
ax.plot(x[i], y[i], '.', color='C0')
ax.text(5,5, r'$\theta$')

ax.set_xticks([])
ax.set_yticks([])
fig.set_size_inches(lecture_style.width*0.5, lecture_style.width*0.5)
fig.tight_layout()
fig.savefig('samples.pdf')


rcParams['path.effects'] = [patheffects.withStroke(linewidth=1, foreground='white')]
np.random.seed(1)
fig, ax = plt.subplots()
x = np.linspace(-6,6,100)
y = x.copy()
x, y = np.meshgrid(x, y)
ax.contourf(x, y, pi(x,y), levels=[-1e5,-200], colors='C0')
ax.contourf(x, y, pi(x,y), levels=[-200,1e3], colors='C1')
x, y = np.random.uniform(-6,6, (2,400))
ax.set_xticks([])
ax.set_yticks([])
fig.set_size_inches(lecture_style.width*0.5, lecture_style.width*0.5)
fig.tight_layout()
fig.savefig('compression_1.pdf')

np.random.seed(1)
fig, ax = plt.subplots()
i = pi(x,y) > -200
ax.plot(x[i], y[i], '.', color='C1')
ax.plot(x[~i], y[~i], '.', color='C0')
ax.set_xticks([])
ax.set_yticks([])
fig.set_size_inches(lecture_style.width*0.5, lecture_style.width*0.5)
fig.tight_layout()
fig.savefig('compression_2.pdf')

