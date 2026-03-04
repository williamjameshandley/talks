#%%
# Slide-friendly landscape tension sigma dot plots
# Split into: pairs (two-dataset combos) and triplets (three-dataset combos)
import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anesthetic import read_chains
from collections import OrderedDict

#%%
# --- SETUP ---

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
    'sn.desdovekie': 'DES-SN5YR',
    'sn.pantheonplus': r'Pantheon$^+$',
    'sn.union3': 'Union3',
    'des_y1.joint': 'DES Y1',
    'planck_2018_CamSpec': 'CamSpec',
    'planck_2018_CamSpec_nolens': 'CamSpec (no lens)',
    'planck_2018_lensing': 'CMB lensing',
    'planck_2018_plik': 'Plik',
    'planck_2018_plik_nolens': 'Plik (no lens)',
}

# Common label for dovekie overlays (matched y-axis width)
labels_dovekie = dict(labels_dataset_short)
labels_dovekie['sn.desy5'] = 'DES SN'
labels_dovekie['sn.desdovekie'] = 'DES SN'


def get_dovekie_label(dataset_name):
    parts = dataset_name.split('+')
    return ' vs '.join(labels_dovekie.get(p, p) for p in parts)


def get_dataset_label(dataset_name):
    parts = dataset_name.split('+')
    return ' vs '.join(labels_dataset_short.get(p, p) for p in parts)


model_list = list(labels_model.keys())

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

pairs_groups = OrderedDict([
    (r'\textbf{BAO vs CMB}', [
        "bao.desi_2024_bao_all+planck_2018_plik",
        "bao.desi_2024_bao_all+planck_2018_CamSpec",
        "bao.desi_dr2+planck_2018_plik",
        "bao.desi_dr2+planck_2018_CamSpec",
    ]),
    (r'\textbf{BAO vs SN}', [
        "bao.desi_2024_bao_all+sn.desy5",
        "bao.desi_2024_bao_all+sn.pantheonplus",
        "bao.desi_2024_bao_all+sn.union3",
        "bao.desi_dr2+sn.desy5",
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_dr2+sn.union3",
    ]),
    (r'\textbf{BAO vs galaxy survey}', [
        "bao.desi_2024_bao_all+des_y1.joint",
        "bao.desi_dr2+des_y1.joint",
    ]),
])

triplets_groups = OrderedDict([
    (r'\textbf{BAO vs CMB vs SN}', [
        "bao.desi_dr2+planck_2018_plik+sn.desy5",
        "bao.desi_dr2+planck_2018_plik+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_plik+sn.union3",
        "bao.desi_dr2+planck_2018_CamSpec+sn.desy5",
        "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
        "bao.desi_2024_bao_all+planck_2018_plik+sn.desdovekie",
        "bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie",
    ]),
])

# Match beamer default: Computer Modern Sans Serif
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\renewcommand{\familydefault}{\sfdefault}')
plt.rc('font', family='sans-serif', size=8)

#%%
# --- DATA LOADING ---

combo_datasets = [
    "bao.desi_2024_bao_all+des_y1.joint", "bao.desi_2024_bao_all+planck_2018_CamSpec",
    "bao.desi_2024_bao_all+planck_2018_CamSpec_nolens", "bao.desi_2024_bao_all+planck_2018_lensing",
    "bao.desi_2024_bao_all+planck_2018_plik", "bao.desi_2024_bao_all+planck_2018_plik_nolens",
    "bao.desi_2024_bao_all+sn.desy5", "bao.desi_2024_bao_all+sn.desdovekie",
    "bao.desi_2024_bao_all+sn.pantheonplus",
    "bao.desi_2024_bao_all+sn.union3", "bao.desi_dr2+des_y1.joint",
    "bao.desi_dr2+planck_2018_CamSpec", "bao.desi_dr2+planck_2018_CamSpec_nolens",
    "bao.desi_dr2+planck_2018_lensing", "bao.desi_dr2+planck_2018_plik",
    "bao.desi_dr2+planck_2018_plik_nolens", "bao.desi_dr2+sn.desy5",
    "bao.desi_dr2+sn.desdovekie",
    "bao.desi_dr2+sn.pantheonplus", "bao.desi_dr2+sn.union3",
    "planck_2018_plik+sn.pantheonplus"
]
triplet_datasets = [
    "bao.desi_dr2+planck_2018_CamSpec+sn.desy5", "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
    "bao.desi_dr2+planck_2018_CamSpec+sn.union3", "bao.desi_dr2+planck_2018_plik+sn.desy5",
    "bao.desi_dr2+planck_2018_plik+sn.pantheonplus", "bao.desi_dr2+planck_2018_plik+sn.union3",
    "bao.desi_2024_bao_all+planck_2018_plik+sn.desdovekie",
    "bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie",
]
all_combinations = sorted(combo_datasets + triplet_datasets)

parameters = ['logR', 'logI', 'logS', 'd_G', 'p', 'sigma']
statistics = ['mean', 'median', 'std']

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

sigma_mean = dfs['sigma'][('sigma', 'mean')].unstack(level='model').apply(pd.to_numeric, errors='coerce')
sigma_std = dfs['sigma'][('sigma', 'std')].unstack(level='model').apply(pd.to_numeric, errors='coerce')

#%%
# --- PLOTTING ---

def plot_column(groups, sigma_mean, sigma_std, output_path,
                axes_width=2.3, left_margin=1.05, legend_loc='above',
                title=None, label_func=None):
    """axes_width: plot area width in inches. left_margin: space for y labels."""
    if label_func is None:
        label_func = get_dataset_label
    y_data = []
    y_bands = []
    current_y = 0.0
    HEADER_HEIGHT = 0.9
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

    right_margin = 0.15
    fig_width = left_margin + axes_width + right_margin
    fig_height = max(n_rows * 0.25 + len(groups) * 0.2, 1.5)
    fig_height = min(fig_height, 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=left_margin/fig_width,
                        right=1 - right_margin/fig_width)

    x_min, x_max = -0.3, 4.0

    for y_top, y_bot, name in y_bands:
        ax.axhspan(y_top, y_bot, facecolor='#DCDCDC', edgecolor='#A0A0A0',
                   linewidth=0.5, zorder=0)
        ax.text((x_min + x_max) / 2, (y_top + y_bot) / 2, name,
                ha='center', va='center', fontsize=8, zorder=3)

    for x_val in [0, 1, 2, 3, 4]:
        ax.axvline(x_val, color='#A0A0A0', linewidth=0.5, linestyle='-', zorder=0)

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
        ax.errorbar(xs, ys, xerr=xerrs, fmt='none',
                    ecolor=style['color'], elinewidth=0.8, capsize=0, zorder=1)
        ax.scatter(xs, ys, marker=style['marker'],
                   c=fc, edgecolors=style['color'],
                   s=30, linewidths=0.8, label=labels_model[model], zorder=2)

    ax.set_yticks([y for y, _ in y_data])
    ax.set_yticklabels([label_func(ds) for _, ds in y_data], fontsize=7)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(total_y, 0)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels([r'$0\sigma$', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$', r'$4\sigma$'],
                        fontsize=7)
    ax.set_xlabel(r'Tension ($p$-value $\sigma$)', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)

    if legend_loc == 'above':
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1.02),
                  fontsize=5, framealpha=0.95, ncol=4,
                  edgecolor='black', fancybox=False)
    elif legend_loc is not None:
        ax.legend(loc=legend_loc, fontsize=6, framealpha=0.95, ncol=1,
                  edgecolor='black', fancybox=False)

    if title is not None:
        sn_ys = [y for y, ds in y_data if 'desy5' in ds or 'desdovekie' in ds]
        if len(sn_ys) >= 2:
            title_y = np.mean(sn_ys)
        elif sn_ys:
            sn_y = sn_ys[0]
            next_ys = [y for y, ds in y_data if y > sn_y]
            title_y = (sn_y + min(next_ys)) / 2 if next_ys else sn_y
        else:
            title_y = y_data[0][0] if y_data else 0
        ax.text(x_max - 0.1, title_y, title,
                ha='right', va='center', fontsize=8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


#%%
# --- GENERATE ---

plot_column(pairs_groups, sigma_mean, sigma_std,
           '../figures/tension_sigma_pairs.pdf',
           axes_width=2.3, left_margin=1.05)
plot_column(triplets_groups, sigma_mean, sigma_std,
           '../figures/tension_sigma_triplets.pdf',
           axes_width=2.1, left_margin=1.25)

#%%
# --- DOVEKIE OVERLAY ---

dovekie_pre_groups = OrderedDict([
    (r'\textbf{BAO vs DES SN}', [
        "bao.desi_dr2+sn.desy5",
        "bao.desi_2024_bao_all+sn.desy5",
    ]),
    (r'\textbf{BAO vs other SN}', [
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_dr2+sn.union3",
    ]),
])

dovekie_post_groups = OrderedDict([
    (r'\textbf{BAO vs DES SN}', [
        "bao.desi_dr2+sn.desdovekie",
        "bao.desi_2024_bao_all+sn.desdovekie",
    ]),
    (r'\textbf{BAO vs other SN}', [
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_dr2+sn.union3",
    ]),
])

plot_column(dovekie_pre_groups, sigma_mean, sigma_std,
           '../figures/tension_sigma_predovekie.pdf',
           axes_width=2.3, left_margin=1.05,
           title=r'\textbf{Before Dovekie}', label_func=get_dovekie_label)
plot_column(dovekie_post_groups, sigma_mean, sigma_std,
           '../figures/tension_sigma_postdovekie.pdf',
           axes_width=2.3, left_margin=1.05,
           title=r'\textbf{After Dovekie}', label_func=get_dovekie_label)

#%%
# --- DOVEKIE TRIPLET OVERLAY ---

dovekie_pre_triplets_groups = OrderedDict([
    (r'\textbf{BAO vs CMB vs SN}', [
        "bao.desi_dr2+planck_2018_CamSpec+sn.desy5",
        "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
    ]),
])

dovekie_post_triplets_groups = OrderedDict([
    (r'\textbf{BAO vs CMB vs SN}', [
        "bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie",
        "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
    ]),
])

plot_column(dovekie_pre_triplets_groups, sigma_mean, sigma_std,
           '../figures/tension_sigma_triplets_predovekie.pdf',
           axes_width=2.1, left_margin=1.25,
           title=r'\textbf{Before Dovekie}', label_func=get_dovekie_label)
plot_column(dovekie_post_triplets_groups, sigma_mean, sigma_std,
           '../figures/tension_sigma_triplets_postdovekie.pdf',
           axes_width=2.1, left_margin=1.25,
           title=r'\textbf{After Dovekie}', label_func=get_dovekie_label)
