import lecture_style
from anesthetic import NestedSamples
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def f(x,y):
    return - ( (x**2 + y - 11)**2 + (x+y**2-7)**2 )

data = NestedSamples(root='./chains/himmelblau')

x = np.linspace(-6,6,100)
y = np.linspace(-6,6,100)
x, y = np.meshgrid(x, y)
z = f(x,y)


with PdfPages('himmelblau.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    live_points = data.live_points(logL=-np.inf)
    lines, = ax.plot(*live_points.loc[:,:'theta1'].values.T, 'C0.')
    pdf.savefig()
    lines.remove()

    for i in [50] + np.arange(125,len(data),125).tolist():
        live_points = data.live_points(i)
        logL = live_points.logL.min()
        dead_points = data[data.logL < logL]

        live_lines, = ax.plot(*live_points.loc[:,:'theta1'].values.T, 'C0.')
        ax.contour(x, y, z, levels=[logL], colors='k', linestyles='solid', linewidths=0.3)
        pdf.savefig()
        #dead_lines.remove()
        live_lines.remove()

    dead_lines, = ax.plot(*dead_points.loc[:,:'theta1'].values.T, 'k.', ms=2)
    pdf.savefig()

    posterior_points = data.posterior_points()
    posterior_lines, = ax.plot(*posterior_points.loc[:,:'theta1'].values.T,'C3.', linewidth=0.3)
    pdf.savefig()

    dead_lines.remove()
    posterior_lines.remove()
    data.plot(ax,'theta0','theta1', color='C0', ncompress=10000)
    pdf.savefig()
