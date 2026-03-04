#%%
# Slide-friendly landscape Bayes factor dot plots
# Split into: singles, pairs, and triplets
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
    (r'\textbf{BAO + SN}', [
        "bao.desi_dr2+sn.desy5",
        "bao.desi_2024_bao_all+sn.desy5",
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_2024_bao_all+sn.pantheonplus",
        "bao.desi_dr2+sn.union3",
        "bao.desi_2024_bao_all+sn.union3",
    ]),
    (r'\textbf{BAO + galaxy survey}', [
        "bao.desi_dr2+des_y1.joint",
        "bao.desi_2024_bao_all+des_y1.joint",
    ]),
    (r'\textbf{BAO + CMB lensing}', [
        "bao.desi_dr2+planck_2018_lensing",
        "bao.desi_2024_bao_all+planck_2018_lensing",
    ]),
    (r'\textbf{BAO + CMB}', [
        "bao.desi_dr2+planck_2018_plik",
        "bao.desi_2024_bao_all+planck_2018_plik",
        "bao.desi_dr2+planck_2018_CamSpec",
        "bao.desi_2024_bao_all+planck_2018_CamSpec",
    ]),
    (r'\textbf{CMB + SN}', [
        "planck_2018_plik+sn.pantheonplus",
    ]),
])

triplets_groups = OrderedDict([
    (r'\textbf{BAO + CMB + SN}', [
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
                   axes_width=3.0, left_margin=1.0, legend_loc='lower left',
                   title=None, label_func=None):
    """axes_width: plot area width in inches. left_margin: space for y labels."""
    if label_func is None:
        label_func = get_dataset_label
    # Build layout
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

    # Physical size: fixed axes width, figure width adjusts for labels
    right_margin = 0.15
    fig_width = left_margin + axes_width + right_margin
    fig_height = max(n_rows * 0.25 + len(groups) * 0.2, 1.5)
    fig_height = min(fig_height, 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=left_margin/fig_width,
                        right=1 - right_margin/fig_width)

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
    ax.set_yticklabels([label_func(ds) for _, ds in y_data], fontsize=7)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(total_y, 0)
    ax.set_xlabel(r'$\ln B = \ln \mathcal{Z}_i - \ln \mathcal{Z}_{\Lambda\mathrm{CDM}}$',
                  fontsize=8)
    ax.tick_params(axis='x', labelsize=7)

    if legend_loc == 'below':
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

ln_B_mean_s, ln_B_err_s = compute_ln_B(singles_groups)
ln_B_mean_p, ln_B_err_p = compute_ln_B(pairs_groups)
ln_B_mean_t, ln_B_err_t = compute_ln_B(triplets_groups)

plot_landscape(singles_groups, ln_B_mean_s, ln_B_err_s,
               '../figures/bayes_factor_singles.pdf',
               axes_width=2.5, left_margin=0.85, legend_loc='below')
plot_landscape(pairs_groups, ln_B_mean_p, ln_B_err_p,
               '../figures/bayes_factor_pairs.pdf',
               axes_width=2.3, left_margin=1.05, legend_loc='below')
plot_landscape(triplets_groups, ln_B_mean_t, ln_B_err_t,
               '../figures/bayes_factor_triplets.pdf',
               axes_width=2.1, left_margin=1.25, legend_loc='below')

#%%
# --- DOVEKIE OVERLAY ---
# Matched pre/post figures for beamer \only<1>/\only<2>

# Use a common label mapping so both overlays have identical y-axis width
labels_dovekie = dict(labels_dataset_short)
labels_dovekie['sn.desy5'] = 'DES SN'
labels_dovekie['sn.desdovekie'] = 'DES SN'


def get_dovekie_label(dataset_name):
    parts = dataset_name.split('+')
    return ' + '.join(labels_dovekie.get(p, p) for p in parts)


dovekie_pre_singles_groups = OrderedDict([
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

dovekie_post_singles_groups = OrderedDict([
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
        "sn.desdovekie",
        "sn.pantheonplus",
        "sn.union3",
    ]),
    (r'\textbf{Galaxy survey only}', [
        "des_y1.joint",
    ]),
])

dovekie_predovekie_groups = OrderedDict([
    (r'\textbf{BAO + DES SN}', [
        "bao.desi_dr2+sn.desy5",
        "bao.desi_2024_bao_all+sn.desy5",
    ]),
    (r'\textbf{BAO + other SN}', [
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_dr2+sn.union3",
    ]),
])

dovekie_postdovekie_groups = OrderedDict([
    (r'\textbf{BAO + DES SN}', [
        "bao.desi_dr2+sn.desdovekie",
        "bao.desi_2024_bao_all+sn.desdovekie",
    ]),
    (r'\textbf{BAO + other SN}', [
        "bao.desi_dr2+sn.pantheonplus",
        "bao.desi_dr2+sn.union3",
    ]),
])

ln_B_mean_pre_s, ln_B_err_pre_s = compute_ln_B(dovekie_pre_singles_groups)
ln_B_mean_post_s, ln_B_err_post_s = compute_ln_B(dovekie_post_singles_groups)

plot_landscape(dovekie_pre_singles_groups, ln_B_mean_pre_s, ln_B_err_pre_s,
               '../figures/bayes_factor_singles_predovekie.pdf',
               axes_width=2.5, left_margin=0.85, legend_loc='below',
               title=r'\textbf{Before Dovekie}', label_func=get_dovekie_label)
plot_landscape(dovekie_post_singles_groups, ln_B_mean_post_s, ln_B_err_post_s,
               '../figures/bayes_factor_singles_postdovekie.pdf',
               axes_width=2.5, left_margin=0.85, legend_loc='below',
               title=r'\textbf{After Dovekie}', label_func=get_dovekie_label)

ln_B_mean_pre, ln_B_err_pre = compute_ln_B(dovekie_predovekie_groups)
ln_B_mean_post, ln_B_err_post = compute_ln_B(dovekie_postdovekie_groups)

plot_landscape(dovekie_predovekie_groups, ln_B_mean_pre, ln_B_err_pre,
               '../figures/bayes_factor_predovekie.pdf',
               axes_width=2.3, left_margin=1.05, legend_loc='below',
               title=r'\textbf{Before Dovekie}', label_func=get_dovekie_label)
plot_landscape(dovekie_postdovekie_groups, ln_B_mean_post, ln_B_err_post,
               '../figures/bayes_factor_postdovekie.pdf',
               axes_width=2.3, left_margin=1.05, legend_loc='below',
               title=r'\textbf{After Dovekie}', label_func=get_dovekie_label)

#%%
# --- DOVEKIE TRIPLET OVERLAY ---

dovekie_pre_triplets_groups = OrderedDict([
    (r'\textbf{BAO + CMB + SN}', [
        "bao.desi_dr2+planck_2018_CamSpec+sn.desy5",
        "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
    ]),
])

dovekie_post_triplets_groups = OrderedDict([
    (r'\textbf{BAO + CMB + SN}', [
        "bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie",
        "bao.desi_dr2+planck_2018_CamSpec+sn.pantheonplus",
        "bao.desi_dr2+planck_2018_CamSpec+sn.union3",
    ]),
])

ln_B_mean_pre_t, ln_B_err_pre_t = compute_ln_B(dovekie_pre_triplets_groups)
ln_B_mean_post_t, ln_B_err_post_t = compute_ln_B(dovekie_post_triplets_groups)

plot_landscape(dovekie_pre_triplets_groups, ln_B_mean_pre_t, ln_B_err_pre_t,
               '../figures/bayes_factor_triplets_predovekie.pdf',
               axes_width=2.1, left_margin=1.25, legend_loc='below',
               title=r'\textbf{Before Dovekie}', label_func=get_dovekie_label)
plot_landscape(dovekie_post_triplets_groups, ln_B_mean_post_t, ln_B_err_post_t,
               '../figures/bayes_factor_triplets_postdovekie.pdf',
               axes_width=2.1, left_margin=1.25, legend_loc='below',
               title=r'\textbf{After Dovekie}', label_func=get_dovekie_label)
