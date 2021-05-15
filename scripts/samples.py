import matplotlib.pyplot as plt
import numpy
import os

def log_gaussian(x,Sig):
    invSig = numpy.linalg.inv(Sig)
    logdet, _ = numpy.linalg.slogdet(2*numpy.pi*Sig)
    return -logdet/2 - x.transpose().dot(invSig).dot(x)/2

Sig = numpy.array([[1,-0.5],[-0.5,1]])
def two_dim(x,y):
    return log_gaussian(numpy.array([x,y]),Sig)

def x_dim(x):
    return log_gaussian(numpy.array([x]),Sig[0:1,0:1])

def y_dim(y):
    return log_gaussian(numpy.array([y]),Sig[1:2,1:2])


fig = plt.figure(figsize=(5,5))
margu = fig.add_subplot(221)
margl = fig.add_subplot(224)
mid = fig.add_subplot(223)

point_color = 'k'
line_color = 'b'

mid.set_xlim(-3,3)
mid.set_ylim(-3,3)
margu.set_xlim(*mid.get_xlim())
margl.set_ylim(*mid.get_ylim())
mid.tick_params(bottom='off',labelbottom='off',left='off',labelleft='off')
margu.tick_params(left='off',labelleft='off',length=20,direction='in',color=point_color)
margu.set_xticks([])
margu.set_xticklabels([])
margl.tick_params(bottom='off',labelbottom='off',length=20,direction='in',color=point_color)
margl.set_yticks([])
margl.set_yticklabels([])

x = numpy.linspace(*mid.get_xlim(),100)
y = numpy.linspace(*mid.get_ylim(),100)

i=0
z = numpy.array([[two_dim(xi,yi) for xi in x] for yi in y])
mid.contour(x,y,numpy.exp(z),colors=line_color,linewidths=0.5)
fig.savefig('plot_%i.pdf' % i)
i+=1

z = numpy.array([x_dim(xi) for xi in x])
margu.plot(x,numpy.exp(z),line_color,linewidth=0.5)

z = numpy.array([y_dim(yi) for yi in y])
margl.plot(numpy.exp(z),y,line_color,linewidth=0.5)
fig.savefig('plot_%i.pdf' % i)
i+=1

nsamples = 50
samples = numpy.random.multivariate_normal(numpy.zeros(2),Sig,nsamples)

mid.plot(samples.T[0],samples.T[1],'+'+point_color)
fig.savefig('plot_%i.pdf' % i)
i+=1

margl.set_yticklabels([])
margl.set_yticks(samples.T[1])

margu.set_xticklabels([])
margu.set_xticks(samples.T[0])
fig.savefig('plot_%i.pdf' % i)
i+=1

#os.execv('/usr/bin/pdfjoin ' + ' '.join(['plot_%i.pdf' % j for j in range(i)]) + ' --outfile ../figures/samples.pdf',[])
