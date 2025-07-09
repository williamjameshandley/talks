#!/usr/bin/env python3
"""Generate Himmelblau function HMC figures"""
# pip install git+https://github.com/handley-lab/blackjax@nested_sampling matplotlib

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import blackjax
import blackjax.mcmc.trajectory as trajectory
import blackjax.mcmc.integrators as integrators

jax.config.update("jax_enable_x64", True)

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def logprob(position):
    x, y = position
    beta = 0.1
    return -himmelblau(x, y) * beta  # Scale for better acceptance

key = jax.random.PRNGKey(42)

x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = himmelblau(X, Y)

# HMC parameters using BlackJAX
initial_position = jnp.array([0.0, 0.0])  # Start from middle of space
step_size = 0.1
num_integration_steps = 50

# Create HMC sampler
inverse_mass_matrix = jnp.eye(2)  # Identity matrix for 2D problem
hmc_sampler = blackjax.hmc(logprob, step_size, inverse_mass_matrix, num_integration_steps)

# Initialize
key, init_key = jax.random.split(key)
initial_state = hmc_sampler.init(initial_position)

# Compile step function
step_fn = jax.jit(hmc_sampler.step)

# Custom trajectory integration that captures intermediate states
def trajectory_integration_with_capture(integrator, initial_state, step_size, num_steps):
    """Integration that captures all intermediate states using BlackJAX integrator"""
    
    trajectory_states = [initial_state]
    current_state = initial_state
    
    for i in range(num_steps):
        current_state = integrator(current_state, step_size)
        trajectory_states.append(current_state)
    
    return trajectory_states

# Run HMC and capture trajectories
num_steps = 30  
hmc_path = [initial_position]
hmc_trajectories = []  
acceptance_info = []  # Store acceptance information
state = initial_state

for i in range(num_steps):
    key, subkey = jax.random.split(key)
    
    # Generate trajectory using BlackJAX integrator with capture
    key, momentum_key = jax.random.split(key)
    
    # Create integrator from BlackJAX
    from blackjax.mcmc import integrators, metrics
    metric = metrics.default_metric(inverse_mass_matrix)
    symplectic_integrator = integrators.velocity_verlet(logprob, metric.kinetic_energy)
    
    # Sample momentum and create initial integrator state
    momentum = metric.sample_momentum(momentum_key, state.position)
    integrator_state = integrators.IntegratorState(
        state.position, momentum, state.logdensity, state.logdensity_grad
    )
    
    # Capture trajectory states
    trajectory_states = trajectory_integration_with_capture(
        symplectic_integrator, integrator_state, step_size, num_integration_steps
    )
    
    # Extract positions for visualization
    traj_positions = jnp.array([s.position for s in trajectory_states])
    hmc_trajectories.append(traj_positions)
    
    # Use BlackJAX for accept/reject
    new_state, info = step_fn(subkey, state)
    acceptance_info.append(info.is_accepted)
    hmc_path.append(new_state.position)
    state = new_state

with PdfPages('himmelblau_hmc.pdf') as pdf:
    # Progressive HMC path visualization - detailed first steps then wider spacing
    path_steps = [0, 1, 2, 3, 4, 10, 15, 20, 30]
    
    for step_idx, steps in enumerate(path_steps):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.contour(X, Y, Z, levels=np.logspace(0, 3, 8), colors='black', linewidths=1.0)
        
        # Draw only the last HMC trajectory if we have any
        if steps > 0 and len(hmc_trajectories) > 0:
            last_idx = min(steps - 1, len(hmc_trajectories) - 1)
            if last_idx >= 0:
                traj = hmc_trajectories[last_idx]
                if traj is not None and len(traj) > 1:
                    # Draw the curved trajectory for the last HMC step
                    ax.plot(traj[:, 0], traj[:, 1], 'C4', linewidth=2.0, alpha=0.8)
        
        # Draw accepted HMC points (larger)
        path_array = jnp.array(hmc_path[:steps+1])
        ax.scatter(path_array[:-1, 0], path_array[:-1, 1], c='C4', s=25, alpha=0.8, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Highlight current position only if the last step was accepted
        if len(path_array) > 0:
            # Show large circle only if this step was accepted
            if steps > 0 and steps <= len(acceptance_info):
                last_step_idx = steps - 1
                if last_step_idx < len(acceptance_info) and acceptance_info[last_step_idx]:
                    # Last step was accepted, show large circle
                    ax.scatter(path_array[-1, 0], path_array[-1, 1], c='C4', s=50, zorder=10, edgecolors='black', linewidth=1)
                # If rejected, don't show the large circle
            else:
                # For step 0, always show the initial position
                ax.scatter(path_array[-1, 0], path_array[-1, 1], c='C4', s=50, zorder=10, edgecolors='black', linewidth=1)
        
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
