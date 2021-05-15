import matplotlib.pyplot as plt
import getdist.plots

from PyPolyChord import run_polychord
from PyPolyChord.settings import PolyChordSettings

from SNE.likelihood import create_likelihood
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

components = [
        ['mat','DE'],
        ['mat','DE','k'],
        ['mat','k']
        ]

outputs = {}
for choice in components:
    likelihood, prior, paramnames, nDims, nDerived = create_likelihood(choice)

    settings = PolyChordSettings(nDims, nDerived)
    settings.feedback = -1
    settings.do_clustering = False

    settings.file_root = '_'.join(choice)
    if rank is 0: print("Running %s" % settings.file_root)
    output = run_polychord(
            likelihood, nDims, nDerived, settings, prior
            )
    if rank is 0: output.make_paramnames_files(paramnames)

    outputs[settings.file_root] = output

if rank == 0:
    items = [(name, output.logZ, output.logZerr)
             for name, output in outputs.items()]
    items = sorted(items)
    for name, logZ, logZerr in items:
        print(name, logZ, '+/-', logZerr)

if rank == 0:
    g = getdist.plots.getSubplotPlotter()

    posteriors = [outputs[name].posterior for name in ['mat_DE_k', 'mat_DE']]

    g.triangle_plot(posteriors, filled=True)
    plt.show()
