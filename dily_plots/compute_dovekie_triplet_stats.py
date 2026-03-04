#!/usr/bin/env python3
"""
Compute logP (evidence) and tension statistics for dovekie triplet combinations.

Prerequisites:
  ./rsync_dovekie_triplets.sh   (downloads chains to /data/will/new_grid/ns/)

Outputs:
  desi_results/logP_triplet.csv          (merged, original backed up to .bak)
  tension_stats_desi/{model}/tension_stats_{dataset}.csv   (new files)

Run from: cosmoverse_2026/dily_plots/
"""

import os
import sys

import numpy as np
import pandas as pd
from anesthetic import read_chains
from anesthetic.samples import Samples
from scipy.special import erfcinv
from scipy.stats import chi2

# ─── Configuration ───────────────────────────────────────────────────────────

CHAIN_BASE = '/data/will/new_grid/ns'
NSAMPLES = 1000  # match existing tension CSVs (1000 rows)

MODELS = [
    'lcdm', 'wlcdm', 'walcdm', 'mlcdm',
    'klcdm', 'rlcdm', 'Alcdm', 'nrunlcdm',
]

DOVEKIE_TRIPLETS = [
    'bao.desi_2024_bao_all+planck_2018_plik+sn.desdovekie',
    'bao.desi_dr2+planck_2018_CamSpec+sn.desdovekie',
]

LOGP_CSV = 'desi_results/logP_triplet.csv'
TENSION_DIR = 'tension_stats_desi'

# ─── Helpers ─────────────────────────────────────────────────────────────────


def polychord_root(base, model, dataset):
    """Return the anesthetic root path for a polychord run."""
    return os.path.join(
        base, model, dataset,
        f'{dataset}_polychord_raw', dataset,
    )


def load_chains(model, dataset, base=CHAIN_BASE):
    """Load NestedSamples from local polychord_raw directory."""
    root = polychord_root(base, model, dataset)
    return read_chains(root)


def load_prior_info(model, dataset, base=CHAIN_BASE):
    """Parse .prior_info → dict with nprior, ndiscarded."""
    path = polychord_root(base, model, dataset) + '.prior_info'
    info = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=')
                info[k.strip()] = int(v.strip())
    return info


def chains_exist(model, dataset, base=CHAIN_BASE):
    """Check whether chain files are present."""
    root = polychord_root(base, model, dataset)
    return os.path.exists(root + '_dead-birth.txt')



def compute_tension_from_stats(joint_stats, separate_stats_list,
                               joint_f=1.0, separate_fs=None):
    """
    Compute tension statistics from stats DataFrames.

    Each stats DataFrame has columns: logZ, D_KL, logL_P, d_G
    with NSAMPLES rows (Monte Carlo realisations from nested sampling).

    Returns a DataFrame with columns: logR, logI, logS, d_G, p, sigma
    """
    n = len(separate_stats_list)
    if separate_fs is None:
        separate_fs = [1.0] * n

    log_F = np.log(joint_f) - sum(np.log(f) for f in separate_fs)

    # logR = logZ_joint - sum(logZ_sep)
    logR = joint_stats['logZ'].values.copy()
    for s in separate_stats_list:
        logR = logR - s['logZ'].values

    # logS = logL_P_joint - sum(logL_P_sep)
    logS = joint_stats['logL_P'].values.copy()
    for s in separate_stats_list:
        logS = logS - s['logL_P'].values

    # d_G = sum(d_G_sep) - d_G_joint
    d_G = -joint_stats['d_G'].values.copy()
    for s in separate_stats_list:
        d_G = d_G + s['d_G'].values

    # F correction on logR (logS unchanged)
    logR = logR + log_F

    # I = logR - logS (inherits F correction)
    logI = logR - logS

    # p-value and sigma
    p = chi2.sf(d_G - 2 * logS, df=d_G)
    sigma = erfcinv(p) * np.sqrt(2)

    result = Samples(data={
        'logR': logR,
        'logI': logI,
        'logS': logS,
        'd_G': d_G,
        'p': p,
        'sigma': sigma,
    })
    result.set_label('logR', r'$\ln\mathcal{R}$')
    result.set_label('logI', r'$\log\mathcal{I}$')
    result.set_label('logS', r'$\ln\mathcal{S}$')
    result.set_label('d_G', r'$d_\mathrm{G}$')
    result.set_label('p', r'$p$')
    result.set_label('sigma', r'$\sigma$')
    return result


# ─── Prerequisite check ─────────────────────────────────────────────────────

def check_prerequisites():
    """Verify all required chains are downloaded."""
    missing = []
    for model in MODELS:
        for dataset in DOVEKIE_TRIPLETS:
            if not chains_exist(model, dataset):
                missing.append(f'{model}/{dataset}')
        if not chains_exist(model, 'sn.desdovekie'):
            missing.append(f'{model}/sn.desdovekie')
    if missing:
        print("Missing chains (run rsync_dovekie_triplets.sh first):")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)
    # Non-dovekie singles will be downloaded from Zenodo
    print("All local chains present.")


# ─── Part 1: logP (evidence) ────────────────────────────────────────────────

def compute_logP():
    """Compute log-evidence for each (model, triplet) and merge into CSV."""
    existing = pd.read_csv(LOGP_CSV, header=[0, 1], index_col=[0, 1])
    print(f"Existing logP_triplet.csv: {len(existing)} rows")

    new_rows = []
    for model in MODELS:
        for dataset in DOVEKIE_TRIPLETS:
            idx = (dataset, model)
            if idx in existing.index:
                print(f"  SKIP (exists): {model}/{dataset}")
                continue

            print(f"  Computing: {model}/{dataset} ...", end=' ', flush=True)
            samples = load_chains(model, dataset)
            stats = samples.stats(nsamples=NSAMPLES)
            logZ = stats['logZ']
            new_rows.append((dataset, model,
                             float(logZ.mean()),
                             float(logZ.median()),
                             float(logZ.std())))
            print(f"logZ = {logZ.mean():.3f} ± {logZ.std():.3f}")

    if not new_rows:
        print("No new logP rows to add.")
        return

    # Build DataFrame matching existing structure
    new_idx = pd.MultiIndex.from_tuples(
        [(d, m) for d, m, _, _, _ in new_rows],
        names=existing.index.names,
    )
    new_df = pd.DataFrame(
        [(mean, med, std) for _, _, mean, med, std in new_rows],
        index=new_idx,
        columns=existing.columns,
    )

    merged = pd.concat([existing, new_df])
    merged.sort_index(inplace=True)

    # Back up original, then save
    backup = LOGP_CSV + '.bak'
    if os.path.exists(backup):
        os.remove(backup)
    os.rename(LOGP_CSV, backup)
    merged.to_csv(LOGP_CSV)
    print(f"Saved {LOGP_CSV} ({len(merged)} rows), backup at {backup}")


# ─── Part 2: Tension statistics ──────────────────────────────────────────────

def compute_tension():
    """Compute tension stats for each (model, triplet) → CSV files."""
    singles_cache = {}

    def get_single_stats(model, dataset):
        key = (model, dataset)
        if key in singles_cache:
            return singles_cache[key]

        if not chains_exist(model, dataset):
            raise FileNotFoundError(
                f"Missing local chains: {model}/{dataset}\n"
                f"Run rsync_dovekie_triplets.sh to download."
            )
        print(f"    Loading local: {model}/{dataset}")
        samples = load_chains(model, dataset)

        stats = samples.stats(nsamples=NSAMPLES)

        try:
            pi = load_prior_info(model, dataset)
            f = pi['nprior'] / pi['ndiscarded']
        except Exception:
            f = 1.0

        singles_cache[key] = (stats, f)
        return stats, f

    for model in MODELS:
        for dataset in DOVEKIE_TRIPLETS:
            outpath = os.path.join(TENSION_DIR, model,
                                   f'tension_stats_{dataset}.csv')
            if os.path.exists(outpath):
                print(f"  SKIP (exists): {outpath}")
                continue

            print(f"  Computing tension: {model}/{dataset}")

            # Joint (triplet) chains
            joint = load_chains(model, dataset)
            joint_stats = joint.stats(nsamples=NSAMPLES)
            try:
                joint_pi = load_prior_info(model, dataset)
                joint_f = joint_pi['nprior'] / joint_pi['ndiscarded']
            except Exception:
                joint_f = 1.0

            # Constituent singles
            constituents = dataset.split('+')
            sep_stats = []
            sep_fs = []
            for ds in constituents:
                stats, f = get_single_stats(model, ds)
                sep_stats.append(stats)
                sep_fs.append(f)

            # Compute
            tension = compute_tension_from_stats(
                joint_stats, sep_stats,
                joint_f=joint_f, separate_fs=sep_fs,
            )

            # Save in anesthetic CSV format
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            tension.to_csv(outpath)
            print(f"    Saved: {outpath}")
            print(f"    sigma = {tension['sigma'].mean():.2f}"
                  f" ± {tension['sigma'].std():.2f}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    check_prerequisites()

    print()
    print("=" * 60)
    print("Part 1: Computing logP (evidence)")
    print("=" * 60)
    compute_logP()

    print()
    print("=" * 60)
    print("Part 2: Computing tension statistics")
    print("=" * 60)
    compute_tension()
