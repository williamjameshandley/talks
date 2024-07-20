import lecture_style
from anesthetic import read_chains
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib import patheffects, rcParams
rcParams['path.effects'] = [patheffects.withStroke(linewidth=1, foreground='white')]

def f(x,y):
    return - ( (x**2 + y - 11)**2 + (x+y**2-7)**2 )

data = read_chains('./chains/himmelblau')

x = np.linspace(-6,6,100)
y = np.linspace(-6,6,100)
x, y = np.meshgrid(x, y)
z = f(x,y)


pdf = PdfPages('himmelblau.pdf')
with PdfPages('himmelblau.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    live_points = data.live_points(0)
    lines, = ax.plot(*live_points.loc[:,:'theta1'].values.T, 'C0.')
    pdf.savefig()
    lines.remove()

    for i in [50] + np.arange(125,len(data),125).tolist():
        live_points = data.live_points(i)
        logL = live_points.logL.min()
        dead_points = data[data.logL < logL]

        live_lines, = ax.plot(*live_points.loc[:,:'theta1'].values.T, 'C0.')
        ax.contour(x, y, z, levels=[logL], colors='k', linestyles='solid')
        pdf.savefig()
        live_lines.remove()

    dead_lines, = ax.plot(*dead_points.loc[:,:'theta1'].values.T, 'k.')
    pdf.savefig()

    posterior_points = data.posterior_points()
    posterior_lines, = ax.plot(*posterior_points.loc[:,:'theta1'].values.T,'C3.')
    pdf.savefig()

    dead_lines.remove()
    posterior_lines.remove()


    sig = 2
    x, y, w = [0.], [0.], [1]
    logP = f(x[-1], y[-1])/sig**2
    np.random.seed(5)
    for i in [0,5,10,20,23,30]:
        while len(w) < i:
            x_, y_ = [x[-1], y[-1]] + sig*np.random.randn(2)
            logP_ = f(x_,y_)/sig**2
            if logP_ - logP > np.log(np.random.rand()):
                x.append(x_)
                y.append(y_)
                w.append(1)
                logP = logP_
            else:
                w[-1] += 1

        lines, = ax.plot(x, y, 'C3o-')
        pdf.savefig()
        lines.remove()


