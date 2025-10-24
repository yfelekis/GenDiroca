#!/usr/bin/env python3
"""
Battery-specific Empirical Evaluation (for 90/10 per-seed splits)

Usage examples:
  python evaluate_battery_empirical.py --experiment battery
  python evaluate_battery_empirical.py --experiment battery --results-dir data/battery/results_empirical_9010 \
      --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 --noise_min 0.0 --noise_max 5.0 --noise_steps 20 --trials 20 \
      --shift_type additive --distribution gaussian
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import warnings
import scipy.stats as stats

warnings.filterwarnings('ignore')

# local imports
import utilities as ut
import evaluation_utils as evut

# -----------------------------
# Noise/shift helpers (reusing your logic)
# -----------------------------
def _apply_shift(clean_data, shift_config, all_var_names, model_level, seed=None):
    rng = np.random.default_rng(seed)
    shift_type = shift_config.get('type')
    dist_type  = shift_config.get('distribution', 'gaussian')
    n_samples, n_dims = clean_data.shape

    level_key = 'll_params' if model_level == 'L' else 'hl_params'
    params = shift_config.get(level_key, {})

    noise_matrix = np.zeros_like(clean_data)

    if dist_type == 'gaussian':
        mu = np.array(params.get('mu', np.zeros(n_dims)))
        sigma_def = params.get('sigma', np.eye(n_dims))
        sigma = np.diag(np.array(sigma_def)) if np.array(sigma_def).ndim == 1 else np.array(sigma_def)
        noise_matrix = rng.multivariate_normal(mean=mu, cov=sigma, size=n_samples)

    elif dist_type == 'student-t':
        df = params.get('df', 3)
        loc = np.array(params.get('loc', np.zeros(n_dims)))
        shape_def = params.get('shape', np.eye(n_dims))
        shape = np.diag(np.array(shape_def)) if np.array(shape_def).ndim == 1 else np.array(shape_def)
        noise_matrix = stats.multivariate_t.rvs(loc=loc, shape=shape, df=df, size=n_samples, random_state=rng)

    elif dist_type == 'exponential':
        scale = params.get('scale', 1.0)
        noise_matrix = rng.exponential(scale=scale, size=(n_samples, n_dims))

    elif dist_type == 'translation':
        c = params.get('c', 0.5)
        noise_matrix = np.ones((n_samples, n_dims)) * c

    elif dist_type == 'scaling':
        c = params.get('c', 1.5)
        noise_matrix = np.ones((n_samples, n_dims)) * c

    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

    final_noise = np.zeros_like(clean_data)
    vars_to_affect = params.get('apply_to_vars')
    if vars_to_affect is None:
        final_noise = noise_matrix
    else:
        indices_to_affect = [all_var_names.index(var) for var in vars_to_affect if var in all_var_names]
        final_noise[:, indices_to_affect] = noise_matrix[:, indices_to_affect]

    if shift_type == 'additive':
        return clean_data + final_noise
    elif shift_type == 'multiplicative':
        return clean_data * final_noise
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

def _apply_huber(clean_data, alpha, shift_config, all_var_names, model_level, seed=None):
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if alpha == 0:
        return clean_data

    noisy_data = _apply_shift(clean_data, shift_config, all_var_names, model_level, seed=seed)
    if alpha == 1:
        return noisy_data

    n = clean_data.shape[0]
    n_to_contaminate = int(alpha * n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, n_to_contaminate, replace=False)

    out = clean_data.copy()
    out[idx] = noisy_data[idx]
    return out

def _empirical_error(T, Dll, Dhl, metric='fro'):
    if Dll.shape[0] == 0 or Dhl.shape[0] == 0:
        return np.nan
    try:
        Dhl_pred = Dll @ T.T
        return evut.compute_empirical_distance(Dhl_pred.T, Dhl.T, metric)
    except Exception as e:
        print(f"  - Warning: distance failed ({e}); returning NaN.")
        return np.nan

# -----------------------------
# Load per-seed results (NOT k-fold)
# -----------------------------
def _gather_results(results_dir):
    """
    Looks for the *_empirical_splits.pkl files saved by battery_optimization.py and
    normalizes them into a dict:
      method_name -> { seed_key -> {'T_matrix': ..., 'test_indices': ... } }
    """
    candidates = {
        'DIROCA': os.path.join(results_dir, 'diroca_empirical_splits.pkl'),
        'GradCA': os.path.join(results_dir, 'gradca_empirical_splits.pkl'),
        'BARYCA': os.path.join(results_dir, 'baryca_empirical_splits.pkl'),
        'Abs-LiNGAM (Perfect)': os.path.join(results_dir, 'abslingam_empirical_splits.pkl'),
        'Abs-LiNGAM (Noisy)':   os.path.join(results_dir, 'abslingam_empirical_splits.pkl'),
    }
    out = {}

    for method, path in candidates.items():
        if not os.path.exists(path):
            continue
        blob = joblib.load(path)

        # We support two common shapes:
        #  A) {'seed_42': {'T_matrix':..., 'test_indices':...}, ...}
        #  B) {'seeds': [...], 'results': {42: {'T_matrix':..., 'test_indices':...}, ...}}
        per_seed = {}

        if isinstance(blob, dict) and 'results' in blob and 'seeds' in blob:
            for seed in blob['seeds']:
                rec = blob['results'].get(seed, None)
                if rec is None:
                    continue
                per_seed[f'seed_{seed}'] = {
                    'T_matrix': rec['T_matrix'],
                    'test_indices': rec['test_indices'],
                }
        elif isinstance(blob, dict):
            # try flatten dict of dicts
            for k, v in blob.items():
                if isinstance(v, dict) and 'T_matrix' in v and 'test_indices' in v:
                    per_seed[k] = {'T_matrix': v['T_matrix'], 'test_indices': v['test_indices']}
                elif isinstance(v, dict):
                    # maybe one more level
                    for kk, vv in v.items():
                        if isinstance(vv, dict) and 'T_matrix' in vv and 'test_indices' in vv:
                            per_seed[f'{k}_{kk}'] = {'T_matrix': vv['T_matrix'], 'test_indices': vv['test_indices']}

        # If Abs-LiNGAM file stores both styles, split them
        if method.startswith('Abs-LiNGAM') and per_seed:
            style = 'Perfect' if 'Perfect' in method else 'Noisy'
            # If the file separated styles under keys
            # we already tried flatten above; if nothing specific, reuse per_seed as-is.
            out[method] = per_seed
        elif per_seed:
            out[method] = per_seed

    if not out:
        raise FileNotFoundError(f"No per-seed results found in {results_dir}")
    return out

# -----------------------------
# Main evaluation
# -----------------------------
def run_evaluation(
    experiment='battery',
    results_dir=None,
    alpha_values=None,
    noise_levels=None,
    num_trials=2,
    zero_mean=True,
    shift_type='additive',
    distribution='gaussian',
    output_file=None,
):
    if results_dir is None:
        results_dir = f"data/{experiment}/results_empirical_9010"

    # Load model data (endogenous samples + mapping)
    all_data = ut.load_all_data(experiment)
    Dll_samples = all_data['LLmodel']['data']
    Dhl_samples = all_data['HLmodel']['data']
    I_ll_relevant = all_data['LLmodel']['intervention_set']
    omega = all_data['abstraction_data']['omega']
    ll_var_names = list(all_data['LLmodel']['graph'].nodes())
    hl_var_names = list(all_data['HLmodel']['graph'].nodes())

    # Base covariances for shift scaling
    base_sigma_L = np.eye(len(ll_var_names))
    base_sigma_H = np.eye(len(hl_var_names))

    # Gather learned results per seed
    results_to_evaluate = _gather_results(results_dir)

    # Defaults
    if alpha_values is None:
        alpha_values = np.linspace(0, 1.0, 10)
    if noise_levels is None:
        noise_levels = np.linspace(0, 5.0, 20)

    print("Starting battery empirical evaluation (per-seed 90/10 splits):")
    print(f"  - Experiment: {experiment}")
    print(f"  - Results dir: {results_dir}")
    print(f"  - Methods found: {list(results_to_evaluate.keys())}")
    print(f"  - Alpha grid: {len(alpha_values)}   [{alpha_values[0]:.2f} .. {alpha_values[-1]:.2f}]")
    print(f"  - Noise grid: {len(noise_levels)}   [{noise_levels[0]:.2f} .. {noise_levels[-1]:.2f}]")
    print(f"  - Trials per config: {num_trials}")
    print(f"  - Shift: {shift_type} / {distribution}")
    print()

    records = []

    # Iterate configs
    for alpha in tqdm(alpha_values, desc="Alpha"):
        for scale in noise_levels:
            for trial in range(num_trials):
                # Loop methods
                for method_name, per_seed in results_to_evaluate.items():
                    # Loop seeds (no folds now)
                    for seed_key, run_data in per_seed.items():
                        T_learned = run_data['T_matrix']
                        test_idx  = run_data['test_indices']

                        # Build shift config
                        if zero_mean:
                            mu_scale_L = np.zeros(base_sigma_L.shape[0])
                            mu_scale_H = np.zeros(base_sigma_H.shape[0])
                        else:
                            mu_scale_L = np.ones(base_sigma_L.shape[0]) * scale
                            mu_scale_H = np.ones(base_sigma_H.shape[0]) * scale

                        sigma_scale_L = base_sigma_L * (scale**2)
                        sigma_scale_H = base_sigma_H * (scale**2)

                        if distribution == 'gaussian':
                            shift_cfg = {
                                'type': shift_type, 'distribution': distribution,
                                'll_params': {'mu': mu_scale_L, 'sigma': sigma_scale_L},
                                'hl_params': {'mu': mu_scale_H, 'sigma': sigma_scale_H},
                            }
                        elif distribution == 'exponential':
                            shift_cfg = {
                                'type': shift_type, 'distribution': distribution,
                                'll_params': {'scale': scale}, 'hl_params': {'scale': scale},
                            }
                        elif distribution == 'student-t':
                            shift_cfg = {
                                'type': shift_type, 'distribution': distribution,
                                'll_params': {'df': 3, 'loc': np.zeros(base_sigma_L.shape[0]), 'shape': sigma_scale_L},
                                'hl_params': {'df': 3, 'loc': np.zeros(base_sigma_H.shape[0]), 'shape': sigma_scale_H},
                            }
                        elif distribution == 'translation':
                            shift_cfg = {
                                'type': 'additive', 'distribution': distribution,
                                'll_params': {'c': scale}, 'hl_params': {'c': scale},
                            }
                        elif distribution == 'scaling':
                            shift_cfg = {
                                'type': 'multiplicative', 'distribution': distribution,
                                'll_params': {'c': scale}, 'hl_params': {'c': scale},
                            }
                        else:
                            raise ValueError(f"Unknown distribution: {distribution}")

                        # Aggregate error across interventions
                        errs = []
                        for iota in I_ll_relevant:
                            Dll_clean = Dll_samples[iota][test_idx]
                            Dhl_clean = Dhl_samples[omega[iota]][test_idx]

                            Dll_cont = _apply_huber(Dll_clean, alpha, shift_cfg, ll_var_names, 'L', seed=trial)
                            Dhl_cont = _apply_huber(Dhl_clean, alpha, shift_cfg, hl_var_names, 'H', seed=trial)

                            e = _empirical_error(T_learned, Dll_cont, Dhl_cont)
                            if not np.isnan(e):
                                errs.append(e)

                        avg_error = np.mean(errs) if errs else np.nan
                        records.append({
                            'method': method_name,
                            'alpha': alpha,
                            'noise_scale': scale,
                            'trial': trial,
                            'seed': seed_key,
                            'error': avg_error,
                        })

    df = pd.DataFrame(records)
    print("--- Battery Empirical Evaluation Complete ---")
    print(f"Generated {len(df)} records")

    outdir = f"data/{experiment}/evaluation_results"
    os.makedirs(outdir, exist_ok=True)

    if output_file is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            f"battery_eval_{shift_type}_{distribution}"
            f"_alpha{len(alpha_values)}-{alpha_values[0]:.1f}-{alpha_values[-1]:.1f}"
            f"_noise{len(noise_levels)}-{noise_levels[0]:.1f}-{noise_levels[-1]:.1f}"
            f"_trials{num_trials}_zero{str(bool(True)).lower()}_{ts}.csv"
        )
        output_file = os.path.join(outdir, fname)

    # keep it inside the evaluation_results folder
    if not os.path.dirname(output_file):
        output_file = os.path.join(outdir, os.path.basename(output_file))
    elif not output_file.startswith(outdir):
        output_file = os.path.join(outdir, os.path.basename(output_file))

    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    return df

def main():
    p = argparse.ArgumentParser(description="Evaluate battery results (per-seed 90/10).")
    p.add_argument('--experiment', type=str, default='battery')
    p.add_argument('--results-dir', type=str, default=None,
                   help='Directory containing *_empirical_splits.pkl files (default: data/{experiment}/results_empirical_9010)')

    p.add_argument('--alpha_min', type=float, default=0.0)
    p.add_argument('--alpha_max', type=float, default=1.0)
    p.add_argument('--alpha_steps', type=int, default=10)

    p.add_argument('--noise_min', type=float, default=0.0)
    p.add_argument('--noise_max', type=float, default=5.0)
    p.add_argument('--noise_steps', type=int, default=20)

    p.add_argument('--trials', type=int, default=20)

    p.add_argument('--zero_mean', type=str, default='True')
    p.add_argument('--shift_type', type=str, default='additive', choices=['additive','multiplicative'])
    p.add_argument('--distribution', type=str, default='gaussian',
                   choices=['gaussian','exponential','student-t','translation','scaling'])

    p.add_argument('--output', type=str, default=None)

    args = p.parse_args()

    zero_mean = args.zero_mean.lower() in ['true','1','yes','y']
    alpha_vals = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    noise_vals = np.linspace(args.noise_min, args.noise_max, args.noise_steps)

    try:
        run_evaluation(
            experiment=args.experiment,
            results_dir=args.results_dir,
            alpha_values=alpha_vals,
            noise_levels=noise_vals,
            num_trials=args.trials,
            zero_mean=zero_mean,
            shift_type=args.shift_type,
            distribution=args.distribution,
            output_file=args.output
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()