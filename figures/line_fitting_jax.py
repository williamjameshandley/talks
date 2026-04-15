"""Generate line-fitting figures using JAX + BlackJAX NSS + anesthetic.

Port of the original numpy/PolyChord line_fitting.py to modern JAX toolchain.
Produces: data_points.pdf, data_diff.pdf, data_diff_1.pdf, data.pdf,
          parameters.pdf, fgivenx.pdf, evidences_lin.pdf

Uses a pickle cache so that JAX/BlackJAX are only imported when the cache
is missing. Subsequent runs only need numpy/matplotlib/anesthetic for plotting.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cache_file = 'line_fitting_cache.pkl'
use_cache = os.path.exists(cache_file)

# ============================================================================
# Data generation (matching original)
# ============================================================================
np.random.seed(0)
x_min, x_max = 0.0, 1.0
n = 100
sigma_min, sigma_max = 0.1, 0.5
sigma = np.random.uniform(low=sigma_min, high=sigma_max, size=n)
x = np.random.uniform(low=x_min, high=x_max, size=n)

def f_true(x):
    return 1 + x**3

y = np.random.normal(loc=f_true(x), scale=sigma)

# Beamer 16:9 half-column dimensions
# textwidth = 6.00in, textheight = 3.15in, minus ~0.6in for frametitle
figsize = (3.0, 2.55)

# ============================================================================
# Data plots
# ============================================================================

def plot_points(ax, errors=True, n_pts=None):
    x_, y_, s_ = x[:n_pts], y[:n_pts], sigma[:n_pts]
    if errors:
        ax.errorbar(x_, y_, yerr=s_, fmt='.', color='k', capthick=0.1, markersize=0, linewidth=0.1)
    else:
        ax.plot(x_, y_, 'k.')

def plot_function(ax, f_func, **kwargs):
    x_ = np.linspace(x_min, x_max, 100)
    ax.plot(x_, f_func(x_), **kwargs)

def label_axes(ax):
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 2.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 1, 2])

def plot_diff(ax, f_func, n_pts=None):
    for i, (xi, yi) in enumerate(zip(x, y)):
        if n_pts is not None and i >= n_pts:
            break
        ax.plot([xi, xi], [yi, f_func(xi)], 'k-')
    plot_points(ax, errors=False, n_pts=n_pts)
    plot_function(ax, f_func)

# data_points.pdf
fig, ax = plt.subplots(figsize=figsize)
plot_points(ax, errors=False)
label_axes(ax)
plot_function(ax, lambda x: 0.7 + x)
plot_function(ax, lambda x: 0.8 + x**2)
fig.tight_layout()
fig.savefig('data_points.pdf')
plt.close()

# data_diff.pdf
fig, ax = plt.subplots(figsize=figsize)
plot_diff(ax, lambda x: 0.7 + x)
label_axes(ax)
fig.tight_layout()
fig.savefig('data_diff.pdf')
plt.close()

# data_diff_1.pdf
fig, ax = plt.subplots(figsize=figsize)
plot_diff(ax, lambda x: 0.9 + x**2, n_pts=1)
label_axes(ax)
fig.tight_layout()
fig.savefig('data_diff_1.pdf')
plt.close()

# data_diff_2.pdf
fig, ax = plt.subplots(figsize=figsize)
plot_diff(ax, lambda x: 0.9 + x**2)
label_axes(ax)
fig.tight_layout()
fig.savefig('data_diff_2.pdf')
plt.close()

# data.pdf
fig, ax = plt.subplots(figsize=figsize)
plot_points(ax)
label_axes(ax)
fig.tight_layout()
fig.savefig('data.pdf')
plt.close()

print("Data plots done.")

# ============================================================================
# BlackJAX NSS inference (only imported when cache is missing)
# ============================================================================

def poly_label(root):
    letters = 'abcde'
    terms = []
    li = 0
    for i, c in enumerate(root):
        if c == '1':
            if i == 0:
                terms.append(f'{letters[li]}')
            elif i == 1:
                terms.append(f'{letters[li]}x')
            else:
                terms.append(f'{letters[li]}x^{i}')
            li += 1
    return '$' + '+'.join(terms) + '$'


def poly_model(x, coeffs):
    """Evaluate polynomial: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ..."""
    return sum(c * x**i for i, c in enumerate(coeffs))


if not use_cache:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    import blackjax
    import tqdm
    from anesthetic import NestedSamples

    jax.config.update("jax_enable_x64", True)

    x_jax = jnp.array(x)
    y_jax = jnp.array(y)
    sigma_jax = jnp.array(sigma)


def run_nss(active_terms, num_live=500, rng_seed=42):
    """Run nested sampling for a polynomial model.

    active_terms: string like '11000' indicating which polynomial terms are active.
    Returns: NestedSamples object, logZ.
    """
    param_names = []
    param_indices = []
    letters = 'abcde'
    letter_idx = 0
    for i, c in enumerate(active_terms):
        if c == '1':
            param_names.append(letters[letter_idx])
            param_indices.append(i)
            letter_idx += 1

    num_dims = len(param_names)
    num_delete = max(num_live // 10, 1)
    num_inner_steps = num_dims * 5

    # Prior: uniform [-3, 3] for each active coefficient
    prior_min = -3.0
    prior_width = 6.0

    def logprior_fn(params):
        lp = 0.0
        for name in param_names:
            lp = lp + jax.scipy.stats.uniform.logpdf(params[name], prior_min, prior_width)
        return lp

    def loglikelihood_fn(params):
        coeffs = [0.0] * 5
        for name, idx in zip(param_names, param_indices):
            coeffs[idx] = params[name]
        y_model = poly_model(x_jax, coeffs)
        return jnp.sum(jax.scipy.stats.norm.logpdf(y_jax, y_model, sigma_jax))

    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )

    rng_key = jax.random.PRNGKey(rng_seed)
    rng_key, *prior_keys = jax.random.split(rng_key, num_dims + 1)

    particles = {}
    for i, name in enumerate(param_names):
        particles[name] = jax.random.uniform(
            prior_keys[i], (num_live,), minval=prior_min, maxval=prior_min + prior_width
        )

    live = jax.jit(algo.init)(particles)
    step_fn = jax.jit(algo.step)

    dead = []
    with tqdm.tqdm(desc=f"NS [{active_terms}]", unit=" dead") as pbar:
        while not live.integrator.logZ_live - live.integrator.logZ < -3:
            rng_key, subkey = jax.random.split(rng_key)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(num_delete)

    dead = blackjax.ns.utils.finalise(live, dead)

    labels = {name: f"${name}$" for name in param_names}
    samples = NestedSamples(
        dead.particles.position,
        logL=dead.particles.loglikelihood,
        logL_birth=dead.particles.loglikelihood_birth,
        labels=labels,
    )

    logZ = float(samples.logZ())

    return samples, logZ, param_names, param_indices


# Run all 2^5 - 1 = 31 models (at least the key ones)
# For speed, run the 4 two-parameter models shown in the original + a few others
key_models = ['11000', '10100', '10010', '10001']
all_models = []

# Generate all non-zero binary strings of length 5
for i in range(1, 2**5):
    root = format(i, '05b')
    all_models.append(root)

if use_cache:
    print(f"Loading cached results from {cache_file}")
    results = pickle.load(open(cache_file, 'rb'))
else:
    results = {}
    for root in all_models:
        n_active = root.count('1')
        nlive = 500 if n_active <= 3 else 300
        print(f"\nRunning model {root} ({n_active}D)...")
        samples, logZ, pnames, pidx = run_nss(root, num_live=nlive)
        results[root] = {'samples': samples, 'logZ': logZ, 'param_names': pnames, 'param_indices': pidx}
    print("\nAll models done.")
    pickle.dump(results, open(cache_file, 'wb'))
    print(f"Cached results to {cache_file}")

# ============================================================================
# Parameter plots (2x2 grid for key two-parameter models)
# ============================================================================

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
for root, ax in zip(key_models, axes.ravel()):
    r = results[root]
    samples = r['samples']
    pnames = r['param_names']

    # Draw weighted posterior samples and scatter
    posterior = samples.sample(1000)
    ax.scatter(posterior[pnames[0]], posterior[pnames[1]], s=0.5, alpha=0.3)
    ax.set_title(poly_label(root))

axes[0, 0].set_xlim(0.6, 1.15)
axes[0, 0].set_ylim(0.7, 1.3)
axes[0, 0].set_ylabel(r'$b$')
axes[1, 0].set_ylabel(r'$b$')
axes[1, 0].set_xlabel(r'$a$')
axes[1, 1].set_xlabel(r'$a$')

fig.tight_layout()
fig.savefig('parameters.pdf')
plt.close()
print("Parameters plot done.")

# ============================================================================
# Evidence plot
# ============================================================================

sorted_roots = sorted(all_models, key=lambda r: results[r]['logZ'], reverse=True)
evs = np.array([results[root]['logZ'] for root in sorted_roots])
evs_norm = evs - evs.max()

# Log evidence plot
fig, ax = plt.subplots(figsize=figsize)
ind = range(len(sorted_roots))
ax.set_xticks(list(ind))
ax.set_xticklabels([poly_label(r) for r in sorted_roots])
ax.grid(color='b', linestyle=':', linewidth=0.1)
ax.xaxis.set_tick_params(rotation=90)
ax.plot(list(ind), evs_norm, 'k.-')
ax.set_ylabel(r'$\log \mathcal{Z}$')
fig.tight_layout()
fig.savefig('evidences_log.pdf')
plt.close()

# Linear evidence (betting odds) plot
from scipy.special import logsumexp
evs_odds = evs_norm - logsumexp(evs_norm)
fig, ax = plt.subplots(figsize=figsize)
ax.set_xticks(list(ind))
ax.set_xticklabels([poly_label(r) for r in sorted_roots])
ax.grid(color='b', linestyle=':', linewidth=0.1)
ax.xaxis.set_tick_params(rotation=90)
ax.plot(list(ind), np.exp(evs_odds), 'k.-')
ax.set_ylabel(r'Betting odds')
fig.tight_layout()
fig.savefig('evidences_lin.pdf')
plt.close()
print("Evidence plots done.")

# ============================================================================
# Predictive posterior (fgivenx-style) for the best model
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=figsize)
x_plot = np.linspace(x_min, x_max, 200)

for root, ax in zip(key_models, axes.ravel()):
    r = results[root]
    samples = r['samples']
    pnames = r['param_names']
    pidx = r['param_indices']

    # Draw posterior samples
    posterior_samples = samples.sample(500)

    # Plot data
    plot_points(ax)

    # Vectorised predictive posterior
    # Build (n_samples, n_x) prediction matrix
    coeffs_array = np.zeros((len(posterior_samples), 5))
    for name, idx in zip(pnames, pidx):
        coeffs_array[:, idx] = posterior_samples[name].values
    x_powers = np.array([x_plot**i for i in range(5)])  # (5, n_x)
    y_pred = coeffs_array @ x_powers  # (n_samples, n_x)
    ax.plot(x_plot, y_pred.T, 'C0-', alpha=0.02, linewidth=0.5)

    ax.set_yticks([])
    ax.set_xticks([])

fig.tight_layout()
fig.savefig('fgivenx.pdf')
plt.close()
print("Predictive posterior plot done.")

print("\nAll figures generated successfully.")
