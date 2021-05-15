import numpy
from fgivenx import compute_samples, compute_pmf
import fgivenx.plot
import matplotlib.pyplot as plt
from line_fitting import data

def plot(output, ax, prior=False, lines=False, col=plt.cm.Reds_r):
    sample_file = output.root
    cache = output.root
    if prior:
        sample_file += '_prior.txt'
        cache += '_prior'
    else:
        sample_file +=  '_equal_weights.txt'
    samples = numpy.loadtxt(sample_file)[:,2:]
    x = numpy.linspace(0,1,100)
    fsamps = compute_samples(data.f, x, samples, cache=cache)
    y, pmf = compute_pmf(data.f, x, samples,cache=cache)
    if lines:
        fgivenx.plot.plot_lines(x, fsamps, ax, color=col)
    else:
        cbar = fgivenx.plot.plot(x, y, pmf, ax, colors=col)
        return cbar

