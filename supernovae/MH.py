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
    # INSERT METROPOLIS HASTINGS CODE HERE

plt.plot(O_m_array, H0_array)
plt.show()
