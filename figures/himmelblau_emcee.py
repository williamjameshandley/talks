#!/usr/bin/env python3
"""Generate Himmelblau function emcee (ensemble) figures"""
# pip install git+https://github.com/handley-lab/blackjax@ensemble matplotlib

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
    beta = 100
    return -himmelblau(x, y) / (beta * 0.1)  # Scale for better acceptance

key = jax.random.PRNGKey(42)

x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = himmelblau(X, Y)

# Ensemble parameters using BlackJAX stretch sampler
num_walkers = 50
initial_positions = jax.random.normal(key, (num_walkers, 2)) * 0.5  # Start centered near origin

# Create ensemble sampler
stretch_sampler = blackjax.stretch(logprob, a=2.0)

# Initialize
key, init_key = jax.random.split(key)
initial_state = stretch_sampler.init(initial_positions)

# Compile step function
step_fn = jax.jit(stretch_sampler.step)

# Run ensemble sampling
num_steps = 200
walker_paths = [initial_positions]
state = initial_state

for i in range(num_steps):
    key, subkey = jax.random.split(key)
    state, info = step_fn(subkey, state)
    walker_paths.append(state.coords)

walker_paths = jnp.array(walker_paths)

with PdfPages('himmelblau_emcee.pdf') as pdf:
    # Progressive ensemble path visualization - detailed first steps then wider spacing
    path_steps = [0, 1, 2, 3, 4, 20, 50, 100, 200]
    
    for steps in path_steps:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.contour(X, Y, Z, levels=np.logspace(0, 3, 8), colors='black', linewidths=1.0)
        
        # No lines - just show current positions
            
        # Show current positions of all walkers
        current_positions = walker_paths[steps, :, :]
        ax.scatter(current_positions[:, 0], current_positions[:, 1], 
                  c='C2', s=15, alpha=0.7, zorder=10)
        
        # Add iteration count in corner
        ax.text(0.98, 0.98, f'Iteration: {steps}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
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