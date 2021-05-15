import matplotlib.pyplot as plt
import numpy as np
from SNE.supernova_data import loglikelihood_sys, loglikelihood_nosys

def loglikelihood(H0, O_m):
    O_L = 1. - O_m
    O_k = 0.
    O_r = 0.
    return loglikelihood_sys(H0, O_r, O_m, O_L, O_k)

step_size = 0.05
def proposal_step():
    return step_size * np.random.randn(2)


N = 10000
O_m, H0 = 0.01, 10
l = loglikelihood(H0, O_m)
count = 1
O_m_array, H0_array, count_array = [], [], []

for _ in range(N):

    # Propose a new point
    O_m_new, H0_new = np.array([O_m, H0]) + proposal_step()
    l_new = loglikelihood(H0_new, O_m_new)
    
    # If the point is acceptable...
    if np.random.rand() < np.exp(l_new-l):
        # ... save the values of the parameters, and the count ...
        O_m_array.append(O_m)
        H0_array.append(H0)
        count_array.append(count)
        # ... and set the values of the parameters to the new values
        O_m, H0, count, l = O_m_new, H0_new, 1, l_new
    else:
        # Otherwise simply increment the count and repeat
        count += 1

plt.plot(O_m_array, H0_array)
plt.show()
