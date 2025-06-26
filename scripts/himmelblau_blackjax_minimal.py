#!/usr/bin/env python3
"""Ultra-minimal BlackJAX nested sampling of Himmelblau function"""

import jax
import jax.numpy as jnp
import blackjax

# Configure JAX
jax.config.update("jax_enable_x64", True)

# Himmelblau function as likelihood
def loglikelihood(params):
    """Himmelblau function: f(x,y) = -((x^2 + y - 11)^2 + (x + y^2 - 7)^2)"""
    x, y = params["x"], params["y"]
    return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

# Setup
key = jax.random.PRNGKey(42)
num_live = 100
num_delete = 50

# Prior bounds
prior_bounds = {"x": (-6.0, 6.0), "y": (-6.0, 6.0)}

# Initialize particles and prior
key, prior_key = jax.random.split(key)
particles, logprior_fn = blackjax.ns.utils.uniform_prior(prior_key, num_live, prior_bounds)

# Create nested sampler
nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood,
    num_delete=num_delete,
    num_inner_steps=10,
)

# JIT compile
init_fn = jax.jit(nested_sampler.init)
step_fn = jax.jit(nested_sampler.step)

# Run sampling
print("Running nested sampling on Himmelblau function...")
live = init_fn(particles)
dead = []

# Evidence threshold - stop when remaining evidence is small
evidence_threshold = 1e-3
max_steps = 500

for i in range(max_steps):
    key, subkey = jax.random.split(key)
    live, dead_point = step_fn(subkey, live)
    dead.append(dead_point)
    
    # Check convergence criterion from workshop
    if live.logZ_live - live.logZ < -3:
        print(f"Converged at step {i}: logZ_live - logZ = {live.logZ_live - live.logZ:.2f}")
        break
    
    if i % 50 == 0:
        print(f"Step {i}, logZ_live - logZ = {live.logZ_live - live.logZ:.2f}")

print(f"Completed {len(dead)} steps")
print(f"Final logL range: {live.loglikelihood.min():.2f} to {live.loglikelihood.max():.2f}")

# Finalize dead points (from workshop)
dead = blackjax.ns.utils.finalise(live, dead)

# Compute evidence using BlackJAX utilities
key, evidence_key = jax.random.split(key)
log_weights_samples = blackjax.ns.utils.log_weights(evidence_key, dead, shape=100)
# Evidence is log(sum(weights))
logZ_samples = jax.scipy.special.logsumexp(log_weights_samples, axis=0)
logZ_mean = jnp.mean(logZ_samples)
logZ_std = jnp.std(logZ_samples)

print(f"\nEvidence (BlackJAX): {logZ_mean:.2f} ± {logZ_std:.2f}")

# Convert to anesthetic for analysis
from anesthetic import NestedSamples

# Extract samples and weights
columns = ["x", "y"]
data = jnp.vstack([dead.particles[key] for key in columns]).T

# Create NestedSamples object
ns = NestedSamples(
    data,
    logL=dead.loglikelihood,
    logL_birth=dead.loglikelihood_birth,
    columns=columns,
    logzero=jnp.nan,
)

print(f"Evidence (anesthetic): {ns.logZ():.2f} ± {ns.logZ(100).std():.2f}")
print(f"Found {len(ns)} samples")
print(f"Parameter means: x={ns['x'].mean():.2f}, y={ns['y'].mean():.2f}")

print("Sampling complete!")
