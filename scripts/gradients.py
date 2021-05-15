import matplotlib.pyplot as plt
import numpy

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


def logL(x,Sigma):
    return -0.05*numpy.linalg.inv(Sigma).dot(x)

Sigma1 = numpy.array([[1,0],[0,1]])
Sigma2 = numpy.array([[1,0.9],[0.9,1]])


for Sigma,ax in zip([Sigma1,Sigma2],[ax1,ax2]):
    circle_points_dense = numpy.array([[numpy.sin(theta),numpy.cos(theta)] for theta in numpy.linspace(0,numpy.pi*2,10001)])
    circle_points = numpy.array([[numpy.sin(theta),numpy.cos(theta)] for theta in numpy.linspace(0,numpy.pi*2,21)])

    chol = numpy.linalg.cholesky(Sigma)
    circle_points_dense = chol.dot(circle_points_dense.T).T
    circle_points = chol.dot(circle_points.T).T

    ax.plot(circle_points_dense.T[0], circle_points_dense.T[1])
    ax.plot(circle_points.T[0], circle_points.T[1],'o')
    for pt in circle_points:
        ax.arrow(*pt,*logL(pt,Sigma),head_width=0.02)
    ax.set_xticks([])
    ax.set_yticks([])



fig.savefig('gradients.pdf')
