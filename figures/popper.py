import lecture_style
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects, rcParams
rcParams['path.effects'] = [patheffects.withStroke(linewidth=1, foreground='white')]
from labellines import labelLine, labelLines

# Plot a curved top-hat gaussian in python

# Define the function
def rounded_top_hat(x, x0, sigma, alpha):
    return np.tanh((x-(x0-sigma/2))/alpha)/2/sigma + np.tanh(((x0+sigma/2)-x)/alpha)/2/sigma

x = np.linspace(-3, 3, 1000)

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('popper.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('$D$')
    ax.plot(x, rounded_top_hat(x, 0, 5, 0.1), label=r'$P(D|M_1)$')
    labelLine(ax.get_lines()[-1], 1.5, label=r'$P(D|M_1)$')
    ax.plot(x, rounded_top_hat(x, -1, 1, 0.1), label=r'$P(D|M_2)$')
    labelLine(ax.get_lines()[-1], -0.5, label=r'$P(D|M_2)$')
    pdf.savefig(fig, bbox_inches='tight')

    to_delete = []
    to_delete.append(ax.axvline(1, color='black', ls=':'))
    ax.set_xticks([1])
    ax.set_xlabel(None)
    ax.set_xticklabels([r'$D_\mathrm{obs}$'])
    # Put axis text in top right of figure
    to_delete.append(ax.text(0.95, 0.95, r'Prefer Model $M_1$', transform=ax.transAxes, ha='right', va='top'))
    pdf.savefig(fig, bbox_inches='tight')
    for line in to_delete:
        line.remove()

    to_delete = []
    to_delete.append(ax.axvline(-1, color='black', ls=':'))
    ax.set_xticks([-1])
    ax.set_xticklabels([r'$D_\mathrm{obs}$'])
    to_delete.append(ax.text(0.95, 0.95, r'Prefer Model $M_2$', transform=ax.transAxes, ha='right', va='top'))
    pdf.savefig(fig, bbox_inches='tight')
    for line in to_delete:
        line.remove()
