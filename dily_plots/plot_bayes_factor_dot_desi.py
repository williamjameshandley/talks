#%%
# --- 1. IMPORTS ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

#%%
# --- 2. SETUP: LABELS, MODELS, AND PLOT CONFIGURATION ---

data_source_folder = "desi_results"

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
    return ' + '.join(labels_dataset_short.get(p, p) for p in parts)


model_list = list(labels_model.keys())

# Fixed model order based on D_KL
FIXED_MODEL_ORDER = [
    'lcdm', 'wlcdm', 'walcdm', 'mlcdm',
    'klcdm', 'rlcdm', 'Alcdm', 'nrunlcdm'
]

# Marker styles (same as plot_tension_dot_desi.py)
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

# Dataset groups by probe type
dataset_groups = OrderedDict([
    (r'\textbf{BAO only}', [
        "bao.desi_dr2",
        "bao.desi_2024_bao_all",
    ]),
    (r'\textbf{CMB only}', [
        "planck_2018_plik",
        "planck_2018_plik_nolens",
        "planck_2018_CamSpec",
        "planck_2018_CamSpec_nolens",
    ]),
    (r'\textbf{CMB lensing only}', [
        "planck_2018_lensing",
    ]),
    (r'\textbf{SN only}', [
        "sn.desy5",
        "sn.pantheonplus",
        "sn.union3",
    ]),
    (r'\textbf{Galaxy survey only}', [
        "des_y1.joint",
    ]),
    (r'\textbf{BAO + CMB}', [
        "bao.desi_dr2+planck_2018_plik",
        "bao.desi_2024_bao_all+planck_2018_plik",
        "bao.desi_dr2+planck_2018_plik_nolens",
        "bao.desi_2024_bao_all+planck_2018_plik_nolens",
        "bao.desi_dr2+planck_2018_CamSpec",
        "bao.desi_2024_bao_all+planck_2018_CamSpec",
        "bao.desi_dr2+planck_2018_CamSpec_nolens",
        "bao.desi_2024_bao_all+planck_2018_CamSpec_nolens",
    ]),
    (r'\textbf{BAO + CMB lensing}', [
        "bao.desi_dr2+planck_2018_lensing",
        "bao.desi_2024_bao_all+planck_2018_lensing",
    ]),
    (r'\textbf{BAO + SN}', [
        "bao.desi_dr2+sn.desy5",
        "bao.desi_2024_bao_all+sn.desy5",
        "bao.desi_dr2+sn.union3",
        "bao.desi_2024_bao_all+sn.union3",
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_2024_bao_all+sn.pantheonplus",
    ]),
    (r'\textbf{BAO + galaxy survey}', [
        "bao.desi_dr2+des_y1.joint",
        "bao.desi_2024_bao_all+des_y1.joint",
    ]),
    (r'\textbf{CMB + SN}', [
        "planck_2018_plik+sn.pantheonplus",
    ]),
    (r'\textbf{BAO + CMB + SN}', [
        "bao.desi_dr2+planck_2018_plik+sn.desy5",
        "bao.desi_dr2+planck_2018_plik+sn.union3",
        "bao.desi_dr2+planck_2018_plik+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_CamSpec+sn.desy5",
        "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
        "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
    ]),
])

# LaTeX rendering
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif=['Computer Modern Roman'])
    print("LaTeX font rendering is enabled.")
except RuntimeError:
    print("LaTeX font rendering failed.")

#%%
# --- 3. DATA LOADING ---

print("Loading logP data...")


def load_logP_multiindex(filepath):
    """Load a multi-index logP CSV and return a dict of {(dataset, model): (mean, std)}."""
    df = pd.read_csv(filepath, header=[0, 1], index_col=[0, 1])
    data = {}
    for (dataset, model), row in df.iterrows():
        mean_val = row[('logP', 'mean')]
        std_val = row[('logP', 'std')]
        data[(dataset, model)] = (float(mean_val), float(std_val))
    return data


# Load all logP data with mean and std
logP_data = {}
for fname in ['logP_single.csv', 'logP_combo.csv', 'logP_triplet.csv']:
    fpath = os.path.join(data_source_folder, fname)
    if os.path.exists(fpath):
        logP_data.update(load_logP_multiindex(fpath))
        print(f"  Loaded {fpath}")

print(f"Total (dataset, model) entries: {len(logP_data)}")

#%%
# --- 4. COMPUTE ln B = logP_model - logP_lcdm ---

# For each dataset: compute ln B and error for all models
# ln B = logP_model_mean - logP_lcdm_mean
# sigma(ln B) = sqrt(logP_model_std^2 + logP_lcdm_std^2)

# Collect all datasets from groups
all_datasets = []
for datasets in dataset_groups.values():
    all_datasets.extend(datasets)

ln_B_mean = {}   # (dataset, model) -> float
ln_B_err = {}    # (dataset, model) -> float

for ds in all_datasets:
    lcdm_key = (ds, 'lcdm')
    if lcdm_key not in logP_data:
        print(f"  Warning: no lcdm data for {ds}")
        continue

    lcdm_mean, lcdm_std = logP_data[lcdm_key]

    for model in FIXED_MODEL_ORDER:
        model_key = (ds, model)
        if model_key not in logP_data:
            continue

        model_mean, model_std = logP_data[model_key]
        ln_B_mean[(ds, model)] = model_mean - lcdm_mean
        ln_B_err[(ds, model)] = np.sqrt(model_std**2 + lcdm_std**2)

print(f"Computed ln B for {len(ln_B_mean)} (dataset, model) pairs.")

#%%
# --- 5. CREATE DOT PLOT ---

# Build layout
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

    # Data rows
    for i, ds in enumerate(datasets):
        y_data.append((current_y, ds))
        if i < len(datasets) - 1:
            current_y += ROW_SPACING

    current_y += GAP_BETWEEN_GROUPS

total_y = current_y

# Create figure
fig_height = max(total_y * 0.35, 8)
fig, ax = plt.subplots(figsize=(9, fig_height))

# x-axis range
x_min, x_max = -6, 4

# --- Colored background bands (Jeffreys scale) ---
# Green: positive ln B (favors alternative model over LCDM)
# Orange: negative ln B (favors LCDM)
green_alphas = [0.06, 0.12, 0.20, 0.30]
orange_alphas = [0.06, 0.12, 0.20, 0.30]
boundaries = [0, 1, 2.5, 5, max(abs(x_min), abs(x_max)) + 1]

for i in range(len(boundaries) - 1):
    # Green bands (positive ln B)
    ax.axvspan(boundaries[i], boundaries[i + 1],
               facecolor='green', alpha=green_alphas[i], edgecolor='none', zorder=0)
    # Orange bands (negative ln B)
    ax.axvspan(-boundaries[i + 1], -boundaries[i],
               facecolor='orange', alpha=orange_alphas[i], edgecolor='none', zorder=0)

# Zero line
ax.axvline(0, color='grey', linewidth=0.8, linestyle='-', zorder=0)

# Draw group header bands
for y_top, y_bot, name in y_bands:
    ax.axhspan(y_top, y_bot, facecolor='#DCDCDC', edgecolor='#A0A0A0',
               linewidth=0.5, zorder=1)
    ax.text((x_min + x_max) / 2, (y_top + y_bot) / 2, name,
            ha='center', va='center', fontsize=10, zorder=3)

# Draw vertical grid lines
for x_val in range(int(x_min), int(x_max) + 1):
    if x_val != 0:
        ax.axvline(x_val, color='#C0C0C0', linewidth=0.3, linestyle='-', zorder=0)

# Plot data points with error bars for each model
for model in FIXED_MODEL_ORDER:
    style = model_styles[model]
    xs, xerrs, ys = [], [], []

    for y_pos, ds in y_data:
        key = (ds, model)
        if key in ln_B_mean:
            mean_val = ln_B_mean[key]
            err_val = ln_B_err.get(key, 0.0)
            if not np.isnan(mean_val):
                xs.append(mean_val)
                xerrs.append(err_val)
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
ax.set_ylim(total_y, 0)
ax.set_xlabel(r'$\ln B = \ln \mathcal{Z}_i - \ln \mathcal{Z}_{\Lambda\mathrm{CDM}}$',
              fontsize=12)
ax.tick_params(axis='x', labelsize=11)

# Legend
ax.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=1,
          title=r'\textbf{Model}', title_fontsize=9,
          edgecolor='black', fancybox=False)

fig.tight_layout()

# Save
os.makedirs('paper_fig', exist_ok=True)
output_path = os.path.join('paper_fig', 'bayes_factor_dot_desi.pdf')
fig.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Figure saved to '{output_path}'")
plt.show()
plt.close(fig)
# %%
