import getdist.plots as gplot
import os
import matplotlib.pyplot as plt

g=gplot.getSubplotPlotter(chain_dir=r'/home/will/Data/planck_2017/historical/chains')
roots = ['planck_2015', 'planck_2016']
g.rectangle_plot(['logA','ns','r'],['tau'],roots=roots, filled=True, shaded=False)
g.export('./figures/tau.pdf')

g.triangle_plot(params=['logA','ns','r','tau'],roots=['planck_2015', 'planck_2016'], filled=True)
g.export('./figures/tau_triangle_both.pdf')

g.triangle_plot(params=['logA','ns','r','tau'],roots=['planck_2015'], filled=True)
g.export('./figures/tau_triangle.pdf')
