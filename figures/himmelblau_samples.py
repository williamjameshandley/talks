#!/usr/bin/env python3
"""Generate Himmelblau function true samples visualization"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

jax.config.update("jax_enable_x64", True)

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def logprob(position):
    x, y = position
    beta = 0.1
    return -himmelblau(x, y) * beta  # Scale for better sampling

key = jax.random.PRNGKey(42)

x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = himmelblau(X, Y)

# Generate true samples using simple rejection sampling
num_samples = 50
samples = []
attempts = 0
max_attempts = 100000

# Find approximate bounds for sampling
x_bounds = [-5, 5]
y_bounds = [-5, 5]

# Estimate maximum log probability for rejection sampling
max_logprob = 0  # Start with 0 since we're looking for maxima

while len(samples) < num_samples and attempts < max_attempts:
    # Sample uniformly from bounds
    key, subkey = jax.random.split(key)
    x_candidate = jax.random.uniform(subkey, minval=x_bounds[0], maxval=x_bounds[1])
    
    key, subkey = jax.random.split(key)
    y_candidate = jax.random.uniform(subkey, minval=y_bounds[0], maxval=y_bounds[1])
    
    candidate = jnp.array([x_candidate, y_candidate])
    candidate_logprob = logprob(candidate)
    
    # Update max if we found a better one
    if candidate_logprob > max_logprob:
        max_logprob = candidate_logprob
    
    # Rejection sampling: accept with probability proportional to exp(logprob)
    key, subkey = jax.random.split(key)
    accept_prob = jnp.exp(candidate_logprob - max_logprob + 5)  # Add offset to make reasonable acceptance
    
    if jax.random.uniform(subkey) < accept_prob:
        samples.append(candidate)
    
    attempts += 1

# Convert to array
samples = jnp.array(samples) if samples else jnp.array([]).reshape(0, 2)

print(f"Generated {len(samples)} samples in {attempts} attempts")

with PdfPages('himmelblau_samples.pdf') as pdf:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Draw contours
    ax.contour(X, Y, Z, levels=np.logspace(0, 3, 8), colors='black', linewidths=1.0)
    
    # Draw samples as grey + symbols
    if len(samples) > 0:
        ax.scatter(samples[:, 0], samples[:, 1], c='grey', marker='+', s=30, alpha=0.8, zorder=10)
    
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