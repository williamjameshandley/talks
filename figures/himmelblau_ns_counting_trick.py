#!/usr/bin/env python3
"""Generate Himmelblau function counting trick figure for nested sampling"""
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
    beta = 100
    return -himmelblau(x, y) / (beta * 0.1)

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

with PdfPages('himmelblau_ns_counting_trick.pdf') as pdf:
    
    # Frame 1: Initial setup with 200 blue live points
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
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
    
    # Cache the current live points before stepping
    cached_live_x = np.array(live.particles['x'])
    cached_live_y = np.array(live.particles['y'])
    
    # Calculate likelihoods for all current live points
    cached_logL = np.array([loglikelihood({"x": x, "y": y}) for x, y in zip(cached_live_x, cached_live_y)])
    
    # Take one step to get the threshold
    key, subkey = jax.random.split(key)
    live_after_step, dead_point = step_fn(subkey, live)
    dead_points.append(dead_point)
    
    # The threshold is the worst likelihood that was just killed
    threshold_logL = float(dead_point.loglikelihood.max())
    threshold_val = -threshold_logL
    
    # Frame 2: Show contour and classify points for deletion
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Draw likelihood contour (dashed red line)
    if threshold_val > 0:
        ax.contour(X, Y, Z, levels=[threshold_val], colors='red', 
                  linewidths=1.5, linestyles='--', zorder=3)
    
    # Classify cached points based on threshold
    inside_mask = cached_logL >= threshold_logL
    outside_mask = ~inside_mask
    
    # Plot points: blue (inside/surviving), orange (outside/marked for deletion)
    if np.any(inside_mask):
        ax.scatter(cached_live_x[inside_mask], cached_live_y[inside_mask], 
                  s=5, c='C0', zorder=2, label='Inside (survive)')
    if np.any(outside_mask):
        ax.scatter(cached_live_x[outside_mask], cached_live_y[outside_mask], 
                  s=5, c='C1', zorder=2, label='Outside (marked for deletion)')
    
    n_inside = int(np.sum(inside_mask))
    n_outside = int(np.sum(outside_mask))
    
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
    
    # Frame 3: Show deleted points as grey, surviving points as blue
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Draw likelihood contour (dashed red line)
    if threshold_val > 0:
        ax.contour(X, Y, Z, levels=[threshold_val], colors='red', 
                  linewidths=1.5, linestyles='--', zorder=3)
    
    # Plot all accumulated dead points as grey circles
    if dead_points:
        all_dead_x = []
        all_dead_y = []
        for dp in dead_points:
            all_dead_x.extend(dp.particles['x'])
            all_dead_y.extend(dp.particles['y'])
        ax.scatter(all_dead_x, all_dead_y, s=3, c='grey', alpha=0.4, zorder=1)
    
    # Plot deleted points from this round as grey circles
    if np.any(outside_mask):
        ax.scatter(cached_live_x[outside_mask], cached_live_y[outside_mask], 
                  s=5, c='grey', alpha=0.6, zorder=1, label='Deleted')
    
    # Plot surviving points as blue
    if np.any(inside_mask):
        ax.scatter(cached_live_x[inside_mask], cached_live_y[inside_mask], 
                  s=5, c='C0', zorder=2, label='Surviving')
    
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
    
    # Frame 4: Show repopulated points (200 total again)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Draw likelihood contour (dashed red line)
    if threshold_val > 0:
        ax.contour(X, Y, Z, levels=[threshold_val], colors='red', 
                  linewidths=1.5, linestyles='--', zorder=3)
    
    # Plot all accumulated dead points as grey circles
    if dead_points:
        all_dead_x = []
        all_dead_y = []
        for dp in dead_points:
            all_dead_x.extend(dp.particles['x'])
            all_dead_y.extend(dp.particles['y'])
        ax.scatter(all_dead_x, all_dead_y, s=3, c='grey', alpha=0.4, zorder=1)
    
    # Plot all new live points (after repopulation)
    live = live_after_step  # Use the state after the step
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
    
    # Show the next iteration to demonstrate the pattern repeats
    # Cache the current live points for the next round
    cached_live_x = np.array(live.particles['x'])
    cached_live_y = np.array(live.particles['y'])
    cached_logL = np.array([loglikelihood({"x": x, "y": y}) for x, y in zip(cached_live_x, cached_live_y)])
    
    # Take another step
    key, subkey = jax.random.split(key)
    live_after_step, dead_point = step_fn(subkey, live)
    dead_points.append(dead_point)
    
    # The new threshold
    threshold_logL = float(dead_point.loglikelihood.max())
    threshold_val = -threshold_logL
    
    # Frame 5: Next iteration - show new contour and classification
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Draw previous contour faintly
    prev_threshold = -float(dead_points[0].loglikelihood.max())
    if prev_threshold > 0:
        ax.contour(X, Y, Z, levels=[prev_threshold], colors='black', 
                  linewidths=0.5, alpha=0.3, zorder=1)
    
    # Draw current contour
    if threshold_val > 0:
        ax.contour(X, Y, Z, levels=[threshold_val], colors='red', 
                  linewidths=1.5, linestyles='--', zorder=3)
    
    # Plot all accumulated dead points as grey circles
    if dead_points:
        all_dead_x = []
        all_dead_y = []
        for dp in dead_points:
            all_dead_x.extend(dp.particles['x'])
            all_dead_y.extend(dp.particles['y'])
        ax.scatter(all_dead_x, all_dead_y, s=3, c='grey', alpha=0.4, zorder=1)
    
    # Classify points for this round
    inside_mask = cached_logL >= threshold_logL
    outside_mask = ~inside_mask
    
    # Plot points: blue (inside), orange (marked for deletion)
    if np.any(inside_mask):
        ax.scatter(cached_live_x[inside_mask], cached_live_y[inside_mask], 
                  s=5, c='C0', zorder=2)
    if np.any(outside_mask):
        ax.scatter(cached_live_x[outside_mask], cached_live_y[outside_mask], 
                  s=5, c='C1', zorder=2)
    
    n_inside = int(np.sum(inside_mask))
    n_outside = int(np.sum(outside_mask))
    
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
    
    # Frame 6: Second step - delete points
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Draw previous contour faintly
    prev_threshold = -float(dead_points[0].loglikelihood.max())
    if prev_threshold > 0:
        ax.contour(X, Y, Z, levels=[prev_threshold], colors='black', 
                  linewidths=0.5, alpha=0.3, zorder=1)
    
    # Draw current contour
    if threshold_val > 0:
        ax.contour(X, Y, Z, levels=[threshold_val], colors='red', 
                  linewidths=1.5, linestyles='--', zorder=3)
    
    # Plot all accumulated dead points as grey circles
    if dead_points:
        all_dead_x = []
        all_dead_y = []
        for dp in dead_points:
            all_dead_x.extend(dp.particles['x'])
            all_dead_y.extend(dp.particles['y'])
        ax.scatter(all_dead_x, all_dead_y, s=3, c='grey', alpha=0.4, zorder=1)
    
    # Plot deleted points from this round as grey circles
    if np.any(outside_mask):
        ax.scatter(cached_live_x[outside_mask], cached_live_y[outside_mask], 
                  s=5, c='grey', alpha=0.6, zorder=1)
    
    # Plot surviving points as blue
    if np.any(inside_mask):
        ax.scatter(cached_live_x[inside_mask], cached_live_y[inside_mask], 
                  s=5, c='C0', zorder=2)
    
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
    
    # Frame 7: Second step - repopulate
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Draw previous contour faintly
    if prev_threshold > 0:
        ax.contour(X, Y, Z, levels=[prev_threshold], colors='black', 
                  linewidths=0.5, alpha=0.3, zorder=1)
    
    # Draw current contour
    if threshold_val > 0:
        ax.contour(X, Y, Z, levels=[threshold_val], colors='red', 
                  linewidths=1.5, linestyles='--', zorder=3)
    
    # Plot all accumulated dead points as grey circles
    if dead_points:
        all_dead_x = []
        all_dead_y = []
        for dp in dead_points:
            all_dead_x.extend(dp.particles['x'])
            all_dead_y.extend(dp.particles['y'])
        ax.scatter(all_dead_x, all_dead_y, s=3, c='grey', alpha=0.4, zorder=1)
    
    # Plot all new live points (after repopulation)
    live = live_after_step
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
    
    # Run nested sampling to completion to collect all dead points
    for i in range(10):  # Run more iterations to get more dead points
        key, subkey = jax.random.split(key)
        live, dead_point = step_fn(subkey, live)
        dead_points.append(dead_point)
    
    # Frame 8: All dead points after complete nested sampling
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Plot all accumulated dead points as grey circles
    if dead_points:
        all_dead_x = []
        all_dead_y = []
        for dp in dead_points:
            all_dead_x.extend(dp.particles['x'])
            all_dead_y.extend(dp.particles['y'])
        ax.scatter(all_dead_x, all_dead_y, s=3, c='grey', alpha=0.4, zorder=1)
    
    # Plot remaining live points
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
    
    # Frame 9: Posterior reweighting using proper BlackJAX utilities
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Combine dead points with final live points using BlackJAX finalise function
    nested_samples = blackjax.ns.utils.finalise(live, dead_points)
    
    # Calculate proper importance weights using BlackJAX log_weights function
    key, subkey = jax.random.split(key)
    logw = blackjax.ns.utils.log_weights(subkey, nested_samples)
    # Use mean across multiple volume realizations
    weights = jnp.exp(logw.mean(axis=-1) - jnp.max(logw.mean(axis=-1)))
    
    # Generate equally weighted posterior samples using BlackJAX sample function
    key, subkey = jax.random.split(key)
    n_posterior = 1000
    posterior_samples = blackjax.ns.utils.sample(subkey, nested_samples, shape=n_posterior)
    
    posterior_x = np.array(posterior_samples['x'])
    posterior_y = np.array(posterior_samples['y'])
    
    # Plot all dead points in light grey
    if dead_points:
        all_dead_x = []
        all_dead_y = []
        for dp in dead_points:
            all_dead_x.extend(dp.particles['x'])
            all_dead_y.extend(dp.particles['y'])
        ax.scatter(all_dead_x, all_dead_y, s=3, c='grey', alpha=0.2, zorder=1)
    
    # Plot remaining live points in light blue
    ax.scatter(live.particles['x'], live.particles['y'], s=5, c='C0', alpha=0.2, zorder=1)
    
    # Overlay posterior-weighted samples in darker blue
    ax.scatter(posterior_x, posterior_y, s=2, c='C0', alpha=0.8, zorder=2)
    
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