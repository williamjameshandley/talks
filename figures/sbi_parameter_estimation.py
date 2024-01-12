import numpy as np
import matplotlib.pyplot as plt
from sbi import LinearModel, plot_2d, plot_1d, flip
from matplotlib.backends.backend_pdf import PdfPages
from labellines import labelLine, labelLines



with PdfPages('sbi_parameter_estimation.pdf') as pdf:
    model = LinearModel(M=5, m=5, C=1, mu=1, Sigma=1)
    fig = plt.figure(figsize=(6,6))
    gs = fig.add_gridspec(2,2, width_ratios=(1,2), height_ratios=(1,2))
    ax_D = fig.add_subplot(gs[1,0])
    ax_joint = fig.add_subplot(gs[1,1], sharey=ax_D)
    ax_theta = fig.add_subplot(gs[0,1], sharex=ax_joint)

    plot_2d(flip(model.joint()), ax_joint, color='C0')
    plot_1d(model.evidence(), ax_D, orientation='vertical', normalise=True, color='C0')
    plot_1d(model.prior(), ax_theta, color='C0', normalise=True)


    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    ax_D.set_ylabel(r'$D$')
    labelLine(ax_D.get_lines()[-1], 0.5, label=r'$\mathcal{Z}(D)$')
    ax_joint.set_xlabel(r'$\theta$')
    labelLine(ax_theta.get_lines()[-1], 2.0, label=r'$\pi(\theta)$')
    fig.tight_layout()
    pdf.savefig(fig)

    def plot_data(D):
        to_remove = []
        to_remove.append(ax_D.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint.axhline(D, color='k', ls='--'))

        to_remove.append(plot_1d(model.posterior(D), ax_theta, color='C0', normalise=True, lw=4, label=r'$\mathcal{P}(\theta|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(ax_joint.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        to_remove.append(ax_joint.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        to_remove.append(labelLine(ax_theta.get_lines()[-1], xdata[0]*0.3+xdata[-1]*0.7, label=r'$\mathcal{P}(D|\theta)$'))
        return to_remove

    to_remove = plot_data(16)
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()

    to_remove = plot_data(2)
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()

    to_remove = plot_data(6)
    pdf.savefig(fig)

    theta = [-0.0]
    to_remove.append(ax_theta.axvline(theta, color='k', ls='--'))
    to_remove.append(ax_joint.axvline(theta, color='k', ls='--'))

    to_remove.append(plot_1d(model.likelihood(theta), ax_D, color='C0', normalise=True, orientation='vertical', lw=4, label=r'$\mathcal{P}(\theta|D)$')[0])
    ydata = to_remove[-1].get_ydata()
    to_remove.append(ax_joint.plot(theta* np.ones_like(ydata), ydata, color='k', ls='-', lw=3, alpha=0.5)[0])
    to_remove.append(ax_joint.plot(theta* np.ones_like(ydata), ydata, color='k', ls='-', lw=3, alpha=0.5)[0])

    to_remove.append(labelLine(ax_D.get_lines()[-1], 0.5, label=r'$\mathcal{L}(D|\theta)$'))
    pdf.savefig(fig)

    for x in to_remove:
        x.remove()

    to_remove = plot_data(30)
    pdf.savefig(fig)
