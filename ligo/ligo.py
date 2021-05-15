import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from ligo_likelihood import source, timedata as ts, ligo_loglikelihood as l, m0

import ligo_likelihood


# Plot the 'data'
plt.plot(ts,l.signal0[0],'.')

# Plot the underlying signal
dense_ts = np.linspace(ts[0],ts[-1],1000)
hs = l.signal(source,dense_ts)
plt.plot(dense_ts,hs[0])

plt.show()
