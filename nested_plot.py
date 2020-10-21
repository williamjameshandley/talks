from anesthetic import NestedSamples
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def f(x,y):
    return - ( (x**2 + y - 11)**2 + (x+y**2-7)**2 )

data = NestedSamples(root='/home/will/gtd/code/anesthetic/tests/example_data/himmelblau')

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

data.plot_2d(['theta0','theta1'])

data_up = NestedSamples(root='./figure_11_chains/log_spiral_6hr_G_up/12_regions/test', 
                        columns=['beta%i' % i for i in range(12)] + ['nu0', 'w', 'A', 'sigma'])

data_down = NestedSamples(root='./figure_11_chains/log_spiral_6hr_G_down/9_regions/test', 
                          columns=['beta%i' % i for i in range(9)] + ['nu0', 'w', 'A', 'sigma'])

for data in [data_down, data_up]:
    data.tex['nu0'] = '$\\nu_0$ [MHz]'
    data.tex['A'] = '$A$ [K]'
    data.tex['w'] = '$w$ [MHz]'
    for i in range(len(data.columns)-7):
        data.tex['beta%i' % i] = '$\\beta_{%i}$' % i

fig, axes = data_down.plot_2d(['nu0', 'w', 'A'])
#data_down.plot_2d(axes)
fig.set_size_inches(4,4)
fig.tight_layout()
fig.savefig('params.pdf')
data.gui(params=['nu0','A','w'])


fig, axes = data_down.plot_2d(data_down.columns[:-3])
#data_down.plot_2d(axes)
fig.set_size_inches(8,8)
fig.tight_layout()
fig.savefig('params_full.pdf')
data.gui(params=['nu0','A','w'])
