import numpy
from SNE.distance import D_Ls, distance_modulus

sne = numpy.genfromtxt('./SNE/data/SCPUnion2.1_mu_vs_z.txt')

order = sne[:, 1].argsort()

zs = sne[:, 1][order]
mods = sne[:, 2][order]

Sigma_nosys = numpy.genfromtxt(
        './SNE/data/SCPUnion2.1_covmat_nosys.txt'
        )[order, :][:, order]
Sigma_sys = numpy.genfromtxt(
        './SNE/data/SCPUnion2.1_covmat_sys.txt'
        )[order, :][:, order]

mod_errs = [numpy.sqrt(Sigma_sys[i, i]) for i, _ in enumerate(Sigma_sys)]

mu = mods
invSigma_sys = numpy.linalg.inv(Sigma_sys)
invSigma_nosys = numpy.linalg.inv(Sigma_nosys)
_, LogDetTwoPiSigma_sys = numpy.linalg.slogdet(2*numpy.pi*Sigma_sys)
_, LogDetTwoPiSigma_nosys = numpy.linalg.slogdet(2*numpy.pi*Sigma_nosys)


def loglikelihood_sys(H0, O_r, O_m, O_L, O_k):
    dLs = D_Ls(zs, H0=H0, O_L=O_L, O_m=O_m, O_r=O_r, O_k=O_k)
    mods = distance_modulus(dLs)
    Delta = mu - mods
    return -LogDetTwoPiSigma_sys/2. - Delta.dot(invSigma_sys.dot(Delta))/2.


def loglikelihood_nosys(H0, O_r, O_m, O_L, O_k):
    dLs = D_Ls(zs, H0=H0, O_L=O_L, O_m=O_m, O_r=O_r, O_k=O_k)
    mods = distance_modulus(dLs)
    Delta = mu - mods
    return -LogDetTwoPiSigma_nosys/2. - Delta.dot(invSigma_nosys.dot(Delta))/2.
