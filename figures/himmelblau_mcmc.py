#!/usr/bin/env python3
"""Generate Himmelblau function MCMC figures"""
# pip install git+https://github.com/handley-lab/blackjax@nested_sampling matplotlib

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import blackjax

jax.config.update("jax_enable_x64", True)

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def logprob(position):
    x, y = position
    return -himmelblau(x, y) / 10.0  # Scale for better acceptance

key = jax.random.PRNGKey(42)

x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = himmelblau(X, Y)

# MCMC parameters using BlackJAX
initial_position = jnp.array([0.0, 0.0])  # Start from middle of space
step_size = 1.0

# Create MCMC sampler
rw_sampler = blackjax.random_walk.normal_random_walk(logprob, step_size)

# Initialize
key, init_key = jax.random.split(key)
initial_state = rw_sampler.init(initial_position)

# Compile step function
step_fn = jax.jit(rw_sampler.step)

# Run MCMC
num_steps = 1500
mcmc_path = [initial_position]
state = initial_state

for i in range(num_steps):
    key, subkey = jax.random.split(key)
    state, info = step_fn(subkey, state)
    mcmc_path.append(state.position)

with PdfPages('himmelblau_mcmc.pdf') as pdf:
    # Page 1: Just contours
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.contour(X, Y, Z, levels=np.logspace(0, 3, 8), colors='black', linewidths=1.0)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # Progressive MCMC path visualization
    path_steps = [10, 30, 100, 300, 500, 800, 1200, 1500]
    
    for steps in path_steps:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.contour(X, Y, Z, levels=np.logspace(0, 3, 8), colors='black', linewidths=1.0)
        
        # Draw MCMC path up to this step
        path_array = jnp.array(mcmc_path[:steps+1])
        ax.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=1.5, alpha=0.5)
        ax.scatter(path_array[:, 0], path_array[:, 1], c='red', s=3, alpha=0.3)
        
        # Highlight current position
        ax.scatter(path_array[-1, 0], path_array[-1, 1], c='red', s=30, zorder=10)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
