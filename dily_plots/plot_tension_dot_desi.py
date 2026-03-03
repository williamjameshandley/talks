#%%
# --- 1. IMPORTS ---
import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from anesthetic import read_chains
from collections import OrderedDict

#%%
# --- 2. SETUP: LABELS, MODELS, AND PLOT CONFIGURATION ---

# Model labels
labels_model = {
    'lcdm': r'$\Lambda$CDM',
    'klcdm': r'$\Omega_k\Lambda$CDM',
    'nrunlcdm': r'$n_{\mathrm{run}}\Lambda$CDM',
    'rlcdm': r'$r\Lambda$CDM',
    'wlcdm': r'$w$CDM',
    'mlcdm': r'$m_\nu\Lambda$CDM',
    'walcdm': r'$w_0w_a$CDM',
    'Alcdm': r'$A_L\Lambda$CDM',
}

# Short dataset labels for y-axis
labels_dataset_short = {
    'bao.desi_2024_bao_all': 'DESI DR1',
    'bao.desi_dr2': 'DESI DR2',
    'sn.desy5': 'DES Y5',
    'sn.pantheonplus': r'Pantheon$^+$',
    'sn.union3': 'Union3',
    'des_y1.joint': 'DES Y1',
    'planck_2018_CamSpec': 'CamSpec+lens',
    'planck_2018_CamSpec_nolens': 'CamSpec',
    'planck_2018_lensing': 'CMB lensing',
    'planck_2018_plik': 'Plik+lens',
    'planck_2018_plik_nolens': 'Plik',
}


def get_dataset_label(dataset_name):
    """Generate a short y-axis label from a dataset combination key."""
    parts = dataset_name.split('+')
    return ' vs '.join(labels_dataset_short.get(p, p) for p in parts)


# Define datasets (same as plot_tension_stats_desi.py)
model_list = list(labels_model.keys())
combo_datasets = [
    "bao.desi_2024_bao_all+des_y1.joint", "bao.desi_2024_bao_all+planck_2018_CamSpec",
    "bao.desi_2024_bao_all+planck_2018_CamSpec_nolens", "bao.desi_2024_bao_all+planck_2018_lensing",
    "bao.desi_2024_bao_all+planck_2018_plik", "bao.desi_2024_bao_all+planck_2018_plik_nolens",
    "bao.desi_2024_bao_all+sn.desy5", "bao.desi_2024_bao_all+sn.pantheonplus",
    "bao.desi_2024_bao_all+sn.union3", "bao.desi_dr2+des_y1.joint",
    "bao.desi_dr2+planck_2018_CamSpec", "bao.desi_dr2+planck_2018_CamSpec_nolens",
    "bao.desi_dr2+planck_2018_lensing", "bao.desi_dr2+planck_2018_plik",
    "bao.desi_dr2+planck_2018_plik_nolens", "bao.desi_dr2+sn.desy5",
    "bao.desi_dr2+sn.pantheonplus", "bao.desi_dr2+sn.union3",
    "planck_2018_plik+sn.pantheonplus"
]
triplet_datasets = [
    "bao.desi_dr2+planck_2018_CamSpec+sn.desy5", "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
    "bao.desi_dr2+planck_2018_CamSpec+sn.union3", "bao.desi_dr2+planck_2018_plik+sn.desy5",
    "bao.desi_dr2+planck_2018_plik+sn.pantheonplus", "bao.desi_dr2+planck_2018_plik+sn.union3"
]
all_combinations = sorted(combo_datasets + triplet_datasets)

# Fixed model order based on D_KL from planck_2018_plik
FIXED_MODEL_ORDER = [
    'lcdm', 'wlcdm', 'walcdm', 'mlcdm',
    'klcdm', 'rlcdm', 'Alcdm', 'nrunlcdm'
]

# Marker styles for each model: marker shape, color, filled/open
model_styles = OrderedDict([
    ('lcdm',     {'marker': 'o', 'color': 'black',   'filled': True}),
    ('wlcdm',    {'marker': 'D', 'color': '#E69F00', 'filled': True}),
    ('walcdm',   {'marker': 'X', 'color': '#D55E00', 'filled': True}),
    ('mlcdm',    {'marker': 's', 'color': '#0072B2', 'filled': True}),
    ('klcdm',    {'marker': '^', 'color': '#009E73', 'filled': True}),
    ('rlcdm',    {'marker': 'v', 'color': '#CC79A7', 'filled': True}),
    ('Alcdm',    {'marker': '*', 'color': '#8B4513', 'filled': True}),
    ('nrunlcdm', {'marker': 'P', 'color': '#56B4E9', 'filled': True}),
])

# Dataset groups for the dot plot (ordered top-to-bottom)
dataset_groups = OrderedDict([
    (r'\textbf{BAO  vs  CMB lensing}', [
        "bao.desi_2024_bao_all+planck_2018_lensing",
        "bao.desi_dr2+planck_2018_lensing",
    ]),
    (r'\textbf{BAO  vs  CMB}', [
        "bao.desi_2024_bao_all+planck_2018_plik_nolens",
        "bao.desi_2024_bao_all+planck_2018_CamSpec_nolens",
        "bao.desi_dr2+planck_2018_plik_nolens",
        "bao.desi_dr2+planck_2018_CamSpec_nolens",
    ]),
    (r'\textbf{BAO  vs  CMB + lensing}', [
        "bao.desi_2024_bao_all+planck_2018_plik",
        "bao.desi_2024_bao_all+planck_2018_CamSpec",
        "bao.desi_dr2+planck_2018_plik",
        "bao.desi_dr2+planck_2018_CamSpec",
    ]),
    (r'\textbf{BAO  vs  SN}', [
        "bao.desi_2024_bao_all+sn.desy5",
        "bao.desi_2024_bao_all+sn.pantheonplus",
        "bao.desi_2024_bao_all+sn.union3",
        "bao.desi_dr2+sn.desy5",
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_dr2+sn.union3",
    ]),
    (r'\textbf{BAO  vs  galaxy survey}', [
        "bao.desi_2024_bao_all+des_y1.joint",
        "bao.desi_dr2+des_y1.joint",
    ]),
    (r'\textbf{CMB + lensing  vs  SN}', [
        "planck_2018_plik+sn.pantheonplus",
    ]),
    (r'\textbf{BAO  vs  CMB + lensing  vs  SN}', [
        "bao.desi_dr2+planck_2018_plik+sn.desy5",
        "bao.desi_dr2+planck_2018_plik+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_plik+sn.union3",
        "bao.desi_dr2+planck_2018_CamSpec+sn.desy5",
        "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
    ]),
])

# Tension parameters
parameters = ['logR', 'logI', 'logS', 'd_G', 'p', 'sigma']
statistics = ['mean', 'median', 'std']

# LaTeX rendering
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif=['Computer Modern Roman'])
    print("LaTeX font rendering is enabled.")
except RuntimeError:
    print("LaTeX font rendering failed.")

#%%
# --- 3. DATA LOADING AND PRE-PROCESSING ---
row_index = pd.MultiIndex.from_product([all_combinations, model_list], names=['dataset', 'model'])
dfs = {
    param: pd.DataFrame(
        index=row_index,
        columns=pd.MultiIndex.from_product(
            [[param], statistics],
            names=['parameter', 'statistic']
        )
    )
    for param in parameters
}

print("Loading raw tension data from 'tension_stats_desi/'...")
for model in model_list:
    for dataset in all_combinations:
        filepath = f'tension_stats_desi/{model}/tension_stats_{dataset}.csv'
        try:
            samples = read_chains(filepath)
            filtered_samples = samples[samples.d_G > 0].copy() if 'd_G' in samples else samples.copy()
        except FileNotFoundError:
            filtered_samples = pd.DataFrame()

        for param in parameters:
            if not filtered_samples.empty and param in filtered_samples:
                dfs[param].loc[(dataset, model), (param, 'mean')] = filtered_samples[param].mean()
                dfs[param].loc[(dataset, model), (param, 'median')] = filtered_samples[param].median()
                dfs[param].loc[(dataset, model), (param, 'std')] = filtered_samples[param].std()
            else:
                dfs[param].loc[(dataset, model), :] = np.nan
print("Processing complete.")

#%%
# --- 4. CREATE DOT PLOT ---

# Extract sigma mean and std as unstacked DataFrames (datasets x models)
sigma_mean = dfs['sigma'][('sigma', 'mean')].unstack(level='model').apply(pd.to_numeric, errors='coerce')
sigma_std = dfs['sigma'][('sigma', 'std')].unstack(level='model').apply(pd.to_numeric, errors='coerce')

# Build layout: assign y-positions for data rows and header bands
y_data = []      # list of (y_position, dataset_key)
y_bands = []     # list of (y_top, y_bottom, group_label)
current_y = 0.0
HEADER_HEIGHT = 0.7
GAP_AFTER_HEADER = 0.3
ROW_SPACING = 1.0
GAP_BETWEEN_GROUPS = 0.5

for group_name, datasets in dataset_groups.items():
    # Group header band
    y_bands.append((current_y, current_y + HEADER_HEIGHT, group_name))
    current_y += HEADER_HEIGHT + GAP_AFTER_HEADER

    # Data rows: only add ROW_SPACING between rows, not after the last one
    for i, ds in enumerate(datasets):
        y_data.append((current_y, ds))
        if i < len(datasets) - 1:
            current_y += ROW_SPACING

    current_y += GAP_BETWEEN_GROUPS

total_y = current_y

# Create figure
fig_height = max(total_y * 0.35, 8)
fig, ax = plt.subplots(figsize=(8, fig_height))

# x-axis range
x_min, x_max = -0.3, 4.0

# Draw group header bands
for y_top, y_bot, name in y_bands:
    ax.axhspan(y_top, y_bot, facecolor='#DCDCDC', edgecolor='#A0A0A0',
               linewidth=0.5, zorder=0)
    ax.text((x_min + x_max) / 2, (y_top + y_bot) / 2, name,
            ha='center', va='center', fontsize=10, zorder=3)

# Draw vertical grid lines at each sigma
for x_val in [0, 1, 2, 3, 4]:
    ax.axvline(x_val, color='#A0A0A0', linewidth=0.5, linestyle='-', zorder=0)

# Plot data points with error bars for each model
for model in FIXED_MODEL_ORDER:
    style = model_styles[model]
    xs, xerrs, ys = [], [], []

    for y_pos, ds in y_data:
        if ds in sigma_mean.index and model in sigma_mean.columns:
            mean_val = sigma_mean.loc[ds, model]
            std_val = sigma_std.loc[ds, model]
            if pd.notna(mean_val):
                xs.append(float(mean_val))
                xerrs.append(float(std_val) if pd.notna(std_val) else 0.0)
                ys.append(y_pos)

    if not xs:
        continue

    fc = style['color'] if style['filled'] else 'none'

    # Error bars (horizontal)
    ax.errorbar(xs, ys, xerr=xerrs, fmt='none',
                ecolor=style['color'], elinewidth=0.8, capsize=0, zorder=1)
    # Markers
    ax.scatter(xs, ys, marker=style['marker'],
               c=fc, edgecolors=style['color'],
               s=55, linewidths=1.0, label=labels_model[model], zorder=2)

# Configure y-axis
ax.set_yticks([y for y, _ in y_data])
ax.set_yticklabels([get_dataset_label(ds) for _, ds in y_data], fontsize=9)
ax.tick_params(axis='y', length=0)

# Configure x-axis
ax.set_xlim(x_min, x_max)
ax.set_ylim(total_y, 0)  # inverted so first group is at top
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels([r'$0\sigma$', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$'],
                    fontsize=11)
ax.set_xlabel(r'Tension ($p$-value $\sigma$)', fontsize=12)
ax.tick_params(axis='x', labelsize=11)

# Legend
ax.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=1,
          title=r'\textbf{Model}', title_fontsize=9,
          edgecolor='black', fancybox=False)

fig.tight_layout()

# Save
os.makedirs('paper_fig', exist_ok=True)
output_path = os.path.join('paper_fig', 'tension_dot_sigma_desi.pdf')
fig.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Figure saved to '{output_path}'")
plt.show()
plt.close(fig)
# %%
