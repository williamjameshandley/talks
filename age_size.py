import os
import matplotlib.cm
import numpy
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.stats
import kde

root = '/home/will/Data/base_omegak/plikHM_TTTEEE_lowl_lowE_lensing/base_omegak_plikHM_TTTEEE_lowl_lowE_lensing'
samples = numpy.concatenate([numpy.loadtxt(root + '_%i.txt' % i) for i in range(1,5)])
paramnames = [line.split('\t')[0] for line in open(root + '.paramnames', 'r')]
w = samples[:,0]

H0 = samples[:,2+paramnames.index('H0*')]
omegam = samples[:,2+paramnames.index('omegam*')]
omegal = samples[:,2+paramnames.index('omegal*')]
omegak = samples[:,2+paramnames.index('omegak')]
omegar = numpy.ones_like(omegam) * 4.18343e-5 / (H0/100)**2
zstar = samples[:,2+paramnames.index('zstar*')]

H0 *= 1000 / 3.086e+22

D = 3e8/H0 / 9.461e+15

integral = numpy.array([scipy.integrate.quad(lambda x: (r*(1+x)**4 + m*(1+x)**3 + k*(1+x)**2 + l)**-0.5, 0,z)[0] for m, l, r, k, z in zip(omegam, omegal, omegar, omegak, zstar)])
D*=integral /1e9

A = samples[:,2+paramnames.index('age*')]

curved = numpy.array([w, D, A])

root = '/home/will/Data/COM_CosmoParams_base-plikHM_R3.00/base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing'
samples = numpy.concatenate([numpy.loadtxt(root + '_%i.txt' % i) for i in range(1,5)])
paramnames = [line.split('\t')[0] for line in open(root + '.paramnames', 'r')]
w = samples[:,0]

H0 = samples[:,2+paramnames.index('H0*')]
omegam = samples[:,2+paramnames.index('omegam*')]
omegal = samples[:,2+paramnames.index('omegal*')]
omegar = numpy.ones_like(omegam) * 4.18343e-5 / (H0/100)**2
zstar = samples[:,2+paramnames.index('zstar*')]

H0 *= 1000 / 3.086e+22

D = 3e8/H0 / 9.461e+15

integral = numpy.array([scipy.integrate.quad(lambda x: (r*(1+x)**4 + m*(1+x)**3  + l)**-0.5, 0,z)[0] for m, l, r, z in zip(omegam, omegal, omegar, zstar)])
D*=integral /1e9

A = samples[:,2+paramnames.index('age*')]
flat = numpy.array([w,D,A])

def minmax(arr):
    mn = arr.min()
    mx = arr.max()
    return mn - (mx-mn)*0.1, mx + (mx-mn)*0.1 

def contour_plot(ax, samples, col='Blues'):
    n = 50
    KDE = kde.gaussian_kde(samples[1:],weights=samples[0])

    lpdf = KDE.logpdf(samples[1:])
    i = numpy.argsort(lpdf)
    samples = samples[:,i]
    lpdf = lpdf[i]
    cdf = numpy.cumsum(samples[0]/samples[0].sum())
    contours = [lpdf[numpy.argmin(cdf<c)] for c in [0.05, 0.33, ]]

    d = numpy.linspace(*minmax(samples[1,:]),n)
    a = numpy.linspace(*minmax(samples[2,:]),n) 
    ad = numpy.array([[di, ai] for di in d for ai in a])
    lpdf_grid = KDE.logpdf(ad.transpose()).reshape(n,n).transpose()

    ax.contourf(d, a, lpdf_grid, levels=contours, cmap=col, extend='max')

fig, ax = plt.subplots(figsize=(4,4))
contour_plot(ax, flat, 'Blues')
ax.set_xlabel('size / Glyr')
ax.set_ylabel('age / Gyr')
fig.tight_layout()
fig.savefig('./figures/age_size.pdf')
     
fig, ax = plt.subplots(figsize=(4,4))
contour_plot(ax, curved, 'Greens')
contour_plot(ax, flat, 'Blues')
ax.set_xlabel('size / Glyr')
ax.set_ylabel('age / Gyr')
fig.tight_layout()
fig.savefig('./figures/age_size_curved.pdf')


w,D,A =  flat
i = numpy.random.rand(len(w)) < (w / w.max()/5)
D_flat = D[i]
A_flat = A[i]

w,D,A =  curved
i = numpy.random.rand(len(w)) < (w / w.max()/5)
D_curved = D[i]
A_curved = A[i]

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(D_curved, A_curved, 'g+')
ax.plot(D_flat, A_flat, 'b+')
ax.set_xlim(*minmax(curved[1,:]))
ax.set_ylim(*minmax(curved[2,:]))
ax.set_xlabel('size / Glyr')
ax.set_ylabel('age / Gyr')
fig.tight_layout()
fig.savefig('./figures/age_size_curved_samples.pdf')


