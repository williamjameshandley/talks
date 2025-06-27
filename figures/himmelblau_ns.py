#!/usr/bin/env python3
"""Generate Himmelblau function figures"""
# pip install git+https://github.com/handley-lab/blackjax@nested_sampling matplotlib anesthetic

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import blackjax

jax.config.update("jax_enable_x64", True)

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def loglikelihood(params):
    x, y = params["x"], params["y"]
    return -himmelblau(x, y)

key = jax.random.PRNGKey(42)
num_live = 200
num_delete = 100

prior_bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}

x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = himmelblau(X, Y)

key, prior_key = jax.random.split(key)
particles, logprior_fn = blackjax.ns.utils.uniform_prior(prior_key, num_live, prior_bounds)

nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood,
    num_delete=num_delete,
    num_inner_steps=10,
)

init_fn = jax.jit(nested_sampler.init)
step_fn = jax.jit(nested_sampler.step)

live = init_fn(particles)
dead_points = []

with PdfPages('himmelblau_ns.pdf') as pdf:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    # No dead points yet, just live points
    ax.scatter(live.particles['x'], live.particles['y'], s=5, c='C0', zorder=2)
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
    
    for i in range(7):
        key, subkey = jax.random.split(key)
        live, dead_point = step_fn(subkey, live)
        dead_points.append(dead_point)
        
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        
        # Add contours for dead point likelihood thresholds
        if dead_points:
            # Get all dead point thresholds
            all_dead_logL = [float(dp.loglikelihood.max()) for dp in dead_points]
            all_threshold_vals = [-logL for logL in all_dead_logL]
            
            # Current threshold is the most recent (highest likelihood killed)
            current_threshold = all_threshold_vals[-1]
            previous_thresholds = all_threshold_vals[:-1]
            
            # Draw previous contours fainter
            if previous_thresholds:
                valid_prev = sorted([t for t in previous_thresholds if t > 0])
                if valid_prev:
                    ax.contour(X, Y, Z, levels=valid_prev, colors='black', linewidths=0.5, alpha=0.3)
            
            # Draw current threshold prominently
            if current_threshold > 0:
                ax.contour(X, Y, Z, levels=[current_threshold], colors='black', linewidths=1.0)
        
        # Plot dead points in faint grey
        if dead_points:
            all_dead_x = []
            all_dead_y = []
            for dp in dead_points:
                all_dead_x.extend(dp.particles['x'])
                all_dead_y.extend(dp.particles['y'])
            ax.scatter(all_dead_x, all_dead_y, s=3, c='lightgrey', alpha=0.4, zorder=1)
        
        # Plot live points in blue
        ax.scatter(live.particles['x'], live.particles['y'], s=5, c='C0', zorder=2)
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
