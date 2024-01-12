import numpy as np
import matplotlib.pyplot as plt
from sbi import LinearModel, plot_2d, plot_1d, flip
from matplotlib.backends.backend_pdf import PdfPages
from labellines import labelLine


with PdfPages('sbi_model_comparison.pdf') as pdf:
    A = LinearModel(M=1, m=0, C=1, mu=0, Sigma=1)
    B = LinearModel(M=5, m=5, C=1, mu=1, Sigma=1)
    fig = plt.figure(figsize=(6*0.7/0.5,6))
    gs = fig.add_gridspec(2,3, width_ratios=(1,2,2), height_ratios=(1,2))
    ax_D = fig.add_subplot(gs[1,0])
    ax_joint_A = fig.add_subplot(gs[1,1], sharey=ax_D)
    ax_joint_B = fig.add_subplot(gs[1,2], sharey=ax_D)
    ax_theta_A = fig.add_subplot(gs[0,1], sharex=ax_joint_A)
    ax_theta_B = fig.add_subplot(gs[0,2], sharex=ax_joint_B)

    plot_2d(flip(A.joint()), ax_joint_A, color='C0')
    plot_2d(flip(B.joint()), ax_joint_B, color='C1')
    plot_1d(A.evidence(), ax_D, orientation='vertical', color='C0')
    plot_1d(B.evidence(), ax_D, orientation='vertical', color='C1')
    plot_1d(A.prior(), ax_theta_A, color='C0', normalise=True, label=r'$\pi(\theta_A)$')
    plot_1d(B.prior(), ax_theta_B, color='C1', normalise=True, label=r'$\pi(\theta_B)$')


    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    ax_D.set_ylabel(r'$D$')
    labelLine(ax_D.get_lines()[-1], 0.05, label=r'$\mathcal{Z}(D|B)$')
    labelLine(ax_D.get_lines()[-2], 0.15, label=r'$\mathcal{Z}(D|A)$')
    labelLine(ax_theta_A.get_lines()[-1], -2.0, label=r'$\pi(\theta|A)$')
    labelLine(ax_theta_B.get_lines()[-1], 2.0, label=r'$\pi(\theta|B)$')
    ax_joint_A.set_xlabel(r'$\theta_A$')
    ax_theta_A.set_title('Model A')
    ax_theta_B.set_title('Model B')
    ax_joint_B.set_xlabel(r'$\theta_B$')
    fig.tight_layout()
    pdf.savefig(fig)

    def plot_data(D):
        to_remove = []
        to_remove.append(ax_D.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint_A.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint_B.axhline(D, color='k', ls='--'))

        to_remove.append(plot_1d(A.posterior(D), ax_theta_A, color='C0', normalise=True, lw=4, label=r'$\mathcal{P}(\theta_A|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(labelLine(to_remove[-1], xdata[0]*0.3+xdata[-1]*0.7, label=r'$\mathcal{P}(D|\theta|B)$'))
        to_remove.append(ax_joint_A.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        to_remove.append(plot_1d(B.posterior(D), ax_theta_B, color='C1', normalise=True, lw=4, label=r'$\mathcal{P}(\theta_B|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(labelLine(to_remove[-1], xdata[0]*0.3+xdata[-1]*0.7, label=r'$\mathcal{P}(D|\theta|A)$'))
        to_remove.append(ax_joint_B.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        return to_remove

    to_remove = plot_data(1)
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()

    to_remove = plot_data(12)
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()

    #--------------------------------------------------------------------------
    A = LinearModel(M=1, m=0, C=1, mu=0, Sigma=1)
    B = LinearModel(M=5, m=20, C=1, mu=1, Sigma=1)
    fig = plt.figure(figsize=(6*0.7/0.5,6))
    gs = fig.add_gridspec(2,3, width_ratios=(1,2,2), height_ratios=(1,2))
    ax_D = fig.add_subplot(gs[1,0])
    ax_joint_A = fig.add_subplot(gs[1,1], sharey=ax_D)
    ax_joint_B = fig.add_subplot(gs[1,2], sharey=ax_D)
    ax_theta_A = fig.add_subplot(gs[0,1], sharex=ax_joint_A)
    ax_theta_B = fig.add_subplot(gs[0,2], sharex=ax_joint_B)

    plot_2d(flip(A.joint()), ax_joint_A, color='C0')
    plot_2d(flip(B.joint()), ax_joint_B, color='C1')
    plot_1d(A.evidence(), ax_D, orientation='vertical', color='C0')
    plot_1d(B.evidence(), ax_D, orientation='vertical', color='C1')
    plot_1d(A.prior(), ax_theta_A, color='C0', normalise=True, label=r'$\pi(\theta_A)$')
    plot_1d(B.prior(), ax_theta_B, color='C1', normalise=True, label=r'$\pi(\theta_B)$')


    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    ax_D.set_ylabel(r'$D$')
    labelLine(ax_D.get_lines()[-1], 0.05, label=r'$\mathcal{Z}(D|B)$')
    labelLine(ax_D.get_lines()[-2], 0.15, label=r'$\mathcal{Z}(D|A)$')
    labelLine(ax_theta_A.get_lines()[-1], -2.0, label=r'$\pi(\theta|A)$')
    labelLine(ax_theta_B.get_lines()[-1], 2.0, label=r'$\pi(\theta|B)$')
    ax_joint_A.set_xlabel(r'$\theta_A$')
    ax_theta_A.set_title('Model A')
    ax_theta_B.set_title('Model B')
    ax_joint_B.set_xlabel(r'$\theta_B$')
    fig.tight_layout()
    pdf.savefig(fig)

    def plot_data(D):
        to_remove = []
        to_remove.append(ax_D.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint_A.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint_B.axhline(D, color='k', ls='--'))

        to_remove.append(plot_1d(A.posterior(D), ax_theta_A, color='C0', normalise=True, lw=4, label=r'$\mathcal{P}(\theta_A|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(labelLine(to_remove[-1], xdata[0]*0.3+xdata[-1]*0.7, label=r'$\mathcal{P}(D|\theta|B)$'))
        to_remove.append(ax_joint_A.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        to_remove.append(plot_1d(B.posterior(D), ax_theta_B, color='C1', normalise=True, lw=4, label=r'$\mathcal{P}(\theta_B|D)$')[0])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(labelLine(to_remove[-1], xdata[0]*0.3+xdata[-1]*0.7, label=r'$\mathcal{P}(D|\theta|A)$'))
        to_remove.append(ax_joint_B.plot(xdata, D* np.ones_like(xdata), color='k', ls='-', lw=3, alpha=0.5)[0])
        return to_remove


    to_remove = plot_data(7)
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()

