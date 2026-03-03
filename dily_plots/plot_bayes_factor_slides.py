#%%
# Slide-friendly landscape Bayes factor dot plots
# Split into: singles (individual datasets) and pairs (two-dataset combos)
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

#%%
# --- SETUP ---

data_source_folder = "desi_results"

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
    parts = dataset_name.split('+')
    return ' + '.join(labels_dataset_short.get(p, p) for p in parts)


FIXED_MODEL_ORDER = [
    'lcdm', 'wlcdm', 'walcdm', 'mlcdm',
    'klcdm', 'rlcdm', 'Alcdm', 'nrunlcdm'
]

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

# Split dataset groups into singles and pairs
singles_groups = OrderedDict([
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
])

pairs_groups = OrderedDict([
    (r'\textbf{BAO + CMB}', [
        "bao.desi_dr2+planck_2018_plik",
        "bao.desi_2024_bao_all+planck_2018_plik",
        "bao.desi_dr2+planck_2018_CamSpec",
        "bao.desi_2024_bao_all+planck_2018_CamSpec",
    ]),
    (r'\textbf{BAO + SN}', [
        "bao.desi_dr2+sn.desy5",
        "bao.desi_2024_bao_all+sn.desy5",
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_2024_bao_all+sn.pantheonplus",
        "bao.desi_dr2+sn.union3",
        "bao.desi_2024_bao_all+sn.union3",
    ]),
    (r'\textbf{BAO + CMB lensing}', [
        "bao.desi_dr2+planck_2018_lensing",
        "bao.desi_2024_bao_all+planck_2018_lensing",
    ]),
    (r'\textbf{BAO + galaxy survey}', [
        "bao.desi_dr2+des_y1.joint",
        "bao.desi_2024_bao_all+des_y1.joint",
    ]),
    (r'\textbf{CMB + SN}', [
        "planck_2018_plik+sn.pantheonplus",
    ]),
])

# Match beamer default: Computer Modern Sans Serif
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\renewcommand{\familydefault}{\sfdefault}')
plt.rc('font', family='sans-serif', size=8)

#%%
# --- DATA LOADING ---

def load_logP_multiindex(filepath):
    df = pd.read_csv(filepath, header=[0, 1], index_col=[0, 1])
    data = {}
    for (dataset, model), row in df.iterrows():
        mean_val = row[('logP', 'mean')]
        std_val = row[('logP', 'std')]
        data[(dataset, model)] = (float(mean_val), float(std_val))
    return data


logP_data = {}
for fname in ['logP_single.csv', 'logP_combo.csv', 'logP_triplet.csv']:
    fpath = os.path.join(data_source_folder, fname)
    if os.path.exists(fpath):
        logP_data.update(load_logP_multiindex(fpath))

#%%
# --- COMPUTE ln B ---

def compute_ln_B(groups):
    all_datasets = []
    for datasets in groups.values():
        all_datasets.extend(datasets)

    ln_B_mean = {}
    ln_B_err = {}
    for ds in all_datasets:
        lcdm_key = (ds, 'lcdm')
        if lcdm_key not in logP_data:
            continue
        lcdm_mean, lcdm_std = logP_data[lcdm_key]
        for model in FIXED_MODEL_ORDER:
            model_key = (ds, model)
            if model_key not in logP_data:
                continue
            model_mean, model_std = logP_data[model_key]
            ln_B_mean[(ds, model)] = model_mean - lcdm_mean
            ln_B_err[(ds, model)] = np.sqrt(model_std**2 + lcdm_std**2)
    return ln_B_mean, ln_B_err


#%%
# --- PLOTTING ---

def plot_landscape(groups, ln_B_mean, ln_B_err, output_path,
                   fig_width=3.9):
    """fig_width in inches: 3.9 for 0.63 beamer column, 6.0 for full-width."""
    # Build layout
    y_data = []
    y_bands = []
    current_y = 0.0
    HEADER_HEIGHT = 0.7
    GAP_AFTER_HEADER = 0.3
    ROW_SPACING = 1.0
    GAP_BETWEEN_GROUPS = 0.5

    for group_name, datasets in groups.items():
        y_bands.append((current_y, current_y + HEADER_HEIGHT, group_name))
        current_y += HEADER_HEIGHT + GAP_AFTER_HEADER
        for i, ds in enumerate(datasets):
            y_data.append((current_y, ds))
            if i < len(datasets) - 1:
                current_y += ROW_SPACING
        current_y += GAP_BETWEEN_GROUPS

    total_y = current_y
    n_rows = len(y_data)

    # Physical size: width from argument, height fills column (~3.2" usable)
    fig_height = max(n_rows * 0.25 + len(groups) * 0.2, 1.5)
    fig_height = min(fig_height, 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_min, x_max = -6, 4

    # Jeffreys scale background
    green_alphas = [0.06, 0.12, 0.20, 0.30]
    orange_alphas = [0.06, 0.12, 0.20, 0.30]
    boundaries = [0, 1, 2.5, 5, max(abs(x_min), abs(x_max)) + 1]

    for i in range(len(boundaries) - 1):
        ax.axvspan(boundaries[i], boundaries[i + 1],
                   facecolor='green', alpha=green_alphas[i], edgecolor='none', zorder=0)
        ax.axvspan(-boundaries[i + 1], -boundaries[i],
                   facecolor='orange', alpha=orange_alphas[i], edgecolor='none', zorder=0)

    ax.axvline(0, color='grey', linewidth=0.8, linestyle='-', zorder=0)

    for y_top, y_bot, name in y_bands:
        ax.axhspan(y_top, y_bot, facecolor='#DCDCDC', edgecolor='#A0A0A0',
                   linewidth=0.5, zorder=1)
        ax.text((x_min + x_max) / 2, (y_top + y_bot) / 2, name,
                ha='center', va='center', fontsize=8, zorder=3)

    for x_val in range(int(x_min), int(x_max) + 1):
        if x_val != 0:
            ax.axvline(x_val, color='#C0C0C0', linewidth=0.3, linestyle='-', zorder=0)

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
        ax.errorbar(xs, ys, xerr=xerrs, fmt='none',
                    ecolor=style['color'], elinewidth=0.8, capsize=0, zorder=1)
        ax.scatter(xs, ys, marker=style['marker'],
                   c=fc, edgecolors=style['color'],
                   s=30, linewidths=0.8, label=labels_model[model], zorder=2)

    ax.set_yticks([y for y, _ in y_data])
    ax.set_yticklabels([get_dataset_label(ds) for _, ds in y_data], fontsize=7)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(total_y, 0)
    ax.set_xlabel(r'$\ln B = \ln \mathcal{Z}_i - \ln \mathcal{Z}_{\Lambda\mathrm{CDM}}$',
                  fontsize=8)
    ax.tick_params(axis='x', labelsize=7)

    ax.legend(loc='upper right', fontsize=6, framealpha=0.95, ncol=1,
              title=r'\textbf{Model}', title_fontsize=7,
              edgecolor='black', fancybox=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


#%%
# --- GENERATE ---

ln_B_mean_s, ln_B_err_s = compute_ln_B(singles_groups)
ln_B_mean_p, ln_B_err_p = compute_ln_B(pairs_groups)

plot_landscape(singles_groups, ln_B_mean_s, ln_B_err_s,
               '../figures/bayes_factor_singles.pdf')
plot_landscape(pairs_groups, ln_B_mean_p, ln_B_err_p,
               '../figures/bayes_factor_pairs.pdf')
