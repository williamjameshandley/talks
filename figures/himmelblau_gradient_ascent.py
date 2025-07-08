#!/usr/bin/env python3
"""Generate Himmelblau function gradient ascent figures"""

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
    return -himmelblau(x, y) / 10.0  # Scale for better visualization

# Gradient function
grad_logprob = jax.grad(logprob)

key = jax.random.PRNGKey(42)

x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = himmelblau(X, Y)

# Gradient ascent parameters
initial_position_1 = jnp.array([0.0, 0.0])  # Start from middle of space
initial_position_2 = jnp.array([-1.0, 0.0])  # Second starting point to the left
learning_rate = 0.15  # Faster learning rate for more visible steps
num_steps_1 = 10  # First particle runs for 10 steps
num_steps_2 = 20  # Second particle runs for 20 steps

# Run gradient ascent for first particle
gradient_path_1 = [initial_position_1]
current_position = initial_position_1

for i in range(num_steps_1):
    # Compute gradient
    gradient = grad_logprob(current_position)
    
    # Take gradient ascent step
    current_position = current_position + learning_rate * gradient
    gradient_path_1.append(current_position)

# Run gradient ascent for second particle
gradient_path_2 = [initial_position_2]
current_position = initial_position_2

for i in range(num_steps_2):
    # Compute gradient
    gradient = grad_logprob(current_position)
    
    # Take gradient ascent step
    current_position = current_position + learning_rate * gradient
    gradient_path_2.append(current_position)

with PdfPages('himmelblau_gradient_ascent.pdf') as pdf:
    # Progressive gradient ascent path visualization - detailed first steps then wider spacing
    path_steps = [0, 1, 2, 3, 4, 10, 11, 12, 13, 20]
    
    for frame_idx, steps in enumerate(path_steps):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.contour(X, Y, Z, levels=np.logspace(0, 3, 8), colors='black', linewidths=1.0)
        
        # Handle first particle (frames 0-5)
        if steps <= 10:
            # Draw first particle path
            path_array_1 = jnp.array(gradient_path_1[:steps+1])
            if len(path_array_1) > 1:
                # Draw arrows for each step
                for i in range(len(path_array_1) - 1):
                    start = path_array_1[i]
                    end = path_array_1[i + 1]
                    ax.annotate('', xy=end, xytext=start,
                               arrowprops=dict(arrowstyle='->', color='C1', lw=2.0, alpha=0.8))
            
            # Highlight current position
            if len(path_array_1) > 0:
                ax.scatter(path_array_1[-1, 0], path_array_1[-1, 1], c='C1', s=50, zorder=10, edgecolors='black', linewidth=1)
            
            # Add colored iteration count
            ax.text(0.98, 0.98, f'Iteration: {steps}', transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(boxstyle='round', facecolor='C1', alpha=0.3), color='C1')
        
        # Handle second particle (frames 6-9)
        else:
            # Show complete first particle path
            path_array_1 = jnp.array(gradient_path_1)
            if len(path_array_1) > 1:
                for i in range(len(path_array_1) - 1):
                    start = path_array_1[i]
                    end = path_array_1[i + 1]
                    ax.annotate('', xy=end, xytext=start,
                               arrowprops=dict(arrowstyle='->', color='C1', lw=2.0, alpha=0.5))
            ax.scatter(path_array_1[-1, 0], path_array_1[-1, 1], c='C1', s=50, zorder=10, edgecolors='black', linewidth=1, alpha=0.5)
            
            # Draw second particle path
            steps_2 = steps - 10  # Reset counter for second particle
            max_steps_2 = min(steps_2, 10)  # Cap at 10 iterations for second particle
            path_array_2 = jnp.array(gradient_path_2[:max_steps_2+1])
            if len(path_array_2) > 1:
                for i in range(len(path_array_2) - 1):
                    start = path_array_2[i]
                    end = path_array_2[i + 1]
                    ax.annotate('', xy=end, xytext=start,
                               arrowprops=dict(arrowstyle='->', color='C3', lw=2.0, alpha=0.8))
            
            if len(path_array_2) > 0:
                ax.scatter(path_array_2[-1, 0], path_array_2[-1, 1], c='C3', s=50, zorder=10, edgecolors='black', linewidth=1)
            
            # Add colored iteration count for second particle
            ax.text(0.98, 0.98, f'Iteration: {max_steps_2}', transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(boxstyle='round', facecolor='C3', alpha=0.3), color='C3')
        
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