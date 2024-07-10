import numpy as np
import matplotlib.pyplot as plt
#from sbi import LinearModel, plot_2d, plot_1d, flip
from matplotlib.backends.backend_pdf import PdfPages
from labellines import labelLine, labelLines

from lsbi.model import LinearModel
from lsbi.plot import pdf_plot_2d
from anesthetic.plot import AxesSeries, AxesDataFrame


pdf = PdfPages('sbi_parameter_estimation_.pdf')

with PdfPages('sbi_parameter_estimation.pdf') as pdf:
    model = LinearModel(M=5, m=5, C=1, mu=1, Sigma=1)
    fig = plt.figure(figsize=(6,6))
    gs = fig.add_gridspec(2,2, width_ratios=(1,2), height_ratios=(1,2))
    ax_D = fig.add_subplot(gs[1,0])
    ax_joint = fig.add_subplot(gs[1,1], sharey=ax_D)
    ax_theta = fig.add_subplot(gs[0,1], sharex=ax_joint)
    ax_theta.set_xlim(model.prior().mean - model.prior().cov**0.5*3,
                      model.prior().mean + model.prior().cov**0.5*3) 
    ax_D.set_ylim(model.evidence().mean - model.evidence().cov**0.5*3,
                  model.evidence().mean + model.evidence().cov**0.5*3) 


    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    ax_D.set_ylabel(r'$D$')
    ax_joint.set_xlabel(r'$\theta$')
    fig.tight_layout()
    pdf.savefig(fig)

    def plot_data(D):
        to_remove = []
        to_remove.append(ax_D.axhline(D, color='k', ls='--'))
        to_remove.append(ax_joint.axhline(D, color='k', ls='--'))
        to_remove.append(model.posterior(D).plot_1d(ax_theta, color='C0')[0].lines[-1])
        xdata = to_remove[-1].get_xdata()
        to_remove.append(ax_joint.plot(xdata, D* np.ones_like(xdata), color='C0', ls='-', lw=3)[0])
        to_remove.append(labelLine(ax_theta.get_lines()[-1], xdata[0]*0.3+xdata[-1]*0.7, label=r'$\mathcal{P}(D|\theta)$'))
        return to_remove

    def plot_theta(theta):
        to_remove = []
        to_remove.append(ax_theta.axvline(theta, color='k', ls='--'))
        to_remove.append(ax_joint.axvline(theta, color='k', ls='--'))

        to_remove.append(model.likelihood(theta).plot_1d(ax_D, color='C2', orientation='vertical')[0].lines[-1])
        ydata = to_remove[-1].get_ydata() 
        to_remove.append(ax_joint.plot(theta* np.ones_like(ydata), ydata, color='C2', ls='-', lw=3)[0])

        to_remove.append(labelLine(ax_D.get_lines()[-1], 0.5, label=r'$\mathcal{L}(D|\theta)$'))
        return to_remove

    np.random.seed(1)
    points = []

    theta = model.prior().rvs()
    to_remove = plot_theta(theta)
    pdf.savefig(fig)
    data = model.likelihood(theta).rvs()
    points += ax_joint.plot(theta, data, 'C4o')
    pdf.savefig(fig)
    for x in to_remove:
        x.remove()

    points += [model.prior().plot_1d(ax_theta, color='C1')[0].lines[-1]]
    labelLine(ax_theta.get_lines()[-1], 2.0, label=r'$\pi(\theta)$')
    pdf.savefig(fig)

    for _ in range(5):
        theta = model.prior().rvs()
        to_remove = plot_theta(theta)
        pdf.savefig(fig)
        data = model.likelihood(theta).rvs()
        points += ax_joint.plot(theta, data, 'C4o')
        pdf.savefig(fig)
        for x in to_remove:
            x.remove()

    points += ax_joint.plot(*model.joint().rvs(300).T, 'C4o')
    pdf.savefig(fig)

    for x in points:
        x.remove()


    pdf_plot_2d(ax_joint, model.joint(), color='C4', nplot_2d=100000)
    model.prior().plot_1d(ax_theta, color='C1')
    labelLine(ax_theta.get_lines()[-1], 2.0, label=r'$\pi(\theta)$')
    pdf.savefig(fig)

    model.evidence().plot_1d(ax_D, orientation='vertical', color='C3')
    labelLine(ax_D.get_lines()[-1], 0.5, label=r'$\mathcal{Z}(D)$')
    pdf.savefig(fig)

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
    to_remove += plot_theta(theta)
    pdf.savefig(fig)

    for x in to_remove:
        x.remove()

    pdf.savefig(fig)
