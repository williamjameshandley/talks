from anesthetic import MCMCSamples
import numpy
import matplotlib.pyplot as plt

Sigma = numpy.array([[1,0.9],[0.9,1]]) 
data = numpy.random.multivariate_normal([0,0],Sigma,10000)
samples = MCMCSamples(data=data, columns=['x0','x1'], tex=['x_0','x_1']) 
fig, ax = plt.subplots()
samples.plot(ax,'x0','x1')
vals, vecs = numpy.linalg.eig(Sigma)
l1, l2 = vals
v1, v2 = vecs.T
ax.arrow(0,0, *v1*numpy.sqrt(l1),width=0.05)
ax.arrow(0,0, *v2*numpy.sqrt(l2),width=0.05)

fig.set_size_inches(2.5,2.5)
fig.tight_layout()
fig.savefig('pca.pdf')
