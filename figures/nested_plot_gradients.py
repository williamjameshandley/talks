import lecture_style
from anesthetic import read_chains
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib import patheffects, rcParams
rcParams['path.effects'] = [patheffects.withStroke(linewidth=1, foreground='white')]

def f(x,y):
    return - ( (x**2 + y - 11)**2 + (x+y**2-7)**2 )

def gradf(x,y):
    return - np.array([ 4*x*(x**2 + y - 11) + 2*(x+y**2-7), 2*(x**2 + y - 11) + 4*y*(x+y**2-7) ])

data = read_chains(root='./chains/himmelblau')

x = np.linspace(-6,6,100)
y = np.linspace(-6,6,100)
x, y = np.meshgrid(x, y)
z = f(x,y)


pdf = PdfPages('himmelblau_gradient.pdf')
with PdfPages('himmelblau_gradient.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    live_points = data.live_points(0)
    live_grads = gradf(live_points.loc[:,'theta0'],live_points.loc[:,'theta1'])
    #live_grads/= np.linalg.norm(live_grads, axis=0)
    grads = ax.quiver(*live_points.loc[:,:'theta1'].values.T, *live_grads)
    lines, = ax.plot(*live_points.loc[:,:'theta1'].values.T, 'C0.')

    pdf.savefig()
    lines.remove()
    grads.remove()

    for i in [50] + np.arange(125,len(data),125).tolist():
        live_points = data.live_points(i)
        live_grads = gradf(live_points.loc[:,'theta0'],live_points.loc[:,'theta1'])
        #live_grads/= np.linalg.norm(live_grads, axis=0)
        logL = live_points.logL.min()
        dead_points = data[data.logL < logL]

        grads = ax.quiver(*live_points.loc[:,:'theta1'].values.T, *live_grads)
        ax.contour(x, y, z, levels=[logL], colors='k', linestyles='solid', linewidths=0.3)
        live_lines, = ax.plot(*live_points.loc[:,:'theta1'].values.T, 'C0.')
        pdf.savefig()
        live_lines.remove()
        grads.remove()
