#!/usr/bin/env python3
"""
Non-linear LUCAS — Empirical Evaluation

Evaluates learned abstractions T (3x6) from non_linear_optimization.py on the
non-linear LUCAS pack produced by lucas_nonlinear_generator.py.

What it expects:
  - Pack (joblib): --data-path (e.g., data/lucas/lucas_pack.pkl)
      {
        'T': (3,6),
        'omega': dict[iota->eta],
        'll': { iota: {'X','D','U'} arrays (N,6) },
        'hl': { eta:  {'X','D','U'} arrays (N,3) },
        ...
      }
  - Results directory: --results-dir (default data/{experiment}/results_nonlinear)
      Contains any of:
        diroca_cv_results_empirical.pkl
        gradca_cv_results_empirical.pkl
        baryca_cv_results_empirical.pkl
        abslingam_cv_results_empirical.pkl

What it does:
  - Loads result files and harmonizes into method->fold->{run_key}->{T_matrix, test_indices}
  - For each (alpha, noise_scale, trial, fold, method/run):
      * Takes D_ll[iota][test], D_hl[omega[iota]][test]
      * Optionally contaminates both with Huber model
      * Computes empirical error: || (Dll @ T^T) - Dhl ||_F / sqrt(N_test)
      * Averages across interventions

Saves a CSV under data/{experiment}/evaluation_results/ (or adjacent to data-path if experiment is None).

Usage example:
  python run_nonlinear_evaluation.py \
    --data-path data/lucas/lucas_pack.pkl \
    --results-dir data/lucas/results_nonlinear
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- contamination utils -----------------------------

def _ensure_cov(shape_def, d):
    """shape_def may be scalar, 1D diag, or full; return (d,d) ndarray."""
    arr = np.array(shape_def, dtype=float)
    if arr.ndim == 0:
        return np.eye(d) * float(arr)
    if arr.ndim == 1:
        return np.diag(arr.astype(float))
    return arr.astype(float)

def apply_shift(clean_data, shift_config, n_dims, model_level, seed=None):
    """
    Apply a parametric shift to clean_data (N,d).
    Supported distributions: gaussian, student-t, exponential, translation, scaling.
    shift_type: 'additive' or 'multiplicative'.
    """
    rng = np.random.default_rng(seed)
    shift_type = shift_config.get('type', 'additive')
    dist_type  = shift_config.get('distribution', 'gaussian')

    level_key = 'll_params' if model_level == 'L' else 'hl_params'
    params = shift_config.get(level_key, {})

    N, d = clean_data.shape
    if d != n_dims:
        # trust data; use its width to avoid mismatch surprises
        d = clean_data.shape[1]

    if dist_type == 'gaussian':
        mu    = np.array(params.get('mu', np.zeros(d)), dtype=float)
        Sigma = _ensure_cov(params.get('sigma', np.eye(d)), d)
        noise = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)
    elif dist_type == 'student-t':
        df    = params.get('df', 3)
        loc   = np.array(params.get('loc', np.zeros(d)), dtype=float)
        Shape = _ensure_cov(params.get('shape', np.eye(d)), d)
        # simple sampler via gaussian/chi2 for reproducibility
        z = rng.multivariate_normal(mean=np.zeros(d), cov=Shape, size=N)
        u = rng.chisquare(df, size=N) / df
        noise = loc + z / np.sqrt(u)[:, None]
    elif dist_type == 'exponential':
        scale = float(params.get('scale', 1.0))
        noise = rng.exponential(scale=scale, size=(N, d))
    elif dist_type == 'translation':
        c = float(params.get('c', 0.5))
        noise = np.full((N, d), c, dtype=float)
    elif dist_type == 'scaling':
        c = float(params.get('c', 1.5))
        noise = np.full((N, d), c, dtype=float)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

    if shift_type == 'additive':
        return clean_data + noise
    elif shift_type == 'multiplicative':
        return clean_data * noise
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

def apply_huber_contamination(clean_data, alpha, shift_config, n_dims, model_level, seed=None):
    """
    Huber contamination: with probability alpha, replace a sample with shifted version.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    if alpha == 0.0:
        return clean_data.copy()

    N = clean_data.shape[0]
    rng = np.random.default_rng(seed)

    noisy = apply_shift(clean_data, shift_config, n_dims, model_level, seed=seed)
    if alpha == 1.0:
        return noisy

    idx = rng.choice(N, size=int(alpha * N), replace=False)
    out = clean_data.copy()
    out[idx] = noisy[idx]
    return out

# ----------------------------- error metric -----------------------------

def empirical_error_fro(T_matrix, Dll_test, Dhl_test):
    """
    Frobenius error normalized by sqrt(N): ||Dll @ T^T - Dhl||_F / sqrt(N)
    Shapes: Dll (N,6), Dhl (N,3), T (3,6)
    """
    if Dll_test.size == 0 or Dhl_test.size == 0:
        return np.nan
    pred = Dll_test @ T_matrix.T
    diff = pred - Dhl_test
    N = Dll_test.shape[0]
    return np.linalg.norm(diff, ord='fro') / max(1.0, np.sqrt(N))

# ----------------------------- results loading -----------------------------

def collect_results(results_dir):
    """
    Gather all available CV result files and normalize to:
      methods[method_label]['fold_{i}'][run_key] = {'T_matrix': ..., 'test_indices': ...}
    """
    methods = {}

    # DiRoCA
    p = os.path.join(results_dir, "diroca_cv_results_empirical.pkl")
    if os.path.exists(p):
        data = joblib.load(p)
        # Flatten by run_key (eps_..._delta_...)
        if data:
            first_fold = next(iter(data))
            run_keys = list(data[first_fold].keys())
            for rk in run_keys:
                label = f"DiRoCA ({rk})"
                methods[label] = {}
                for fold_k, fold_dict in data.items():
                    if rk in fold_dict:
                        methods[label][fold_k] = {rk: fold_dict[rk]}

    # GradCA
    p = os.path.join(results_dir, "gradca_cv_results_empirical.pkl")
    if os.path.exists(p):
        data = joblib.load(p)
        if data:
            methods["GradCA"] = data

    # BaryCA
    p = os.path.join(results_dir, "baryca_cv_results_empirical.pkl")
    if os.path.exists(p):
        data = joblib.load(p)
        if data:
            methods["BaryCA"] = data

    # Abs-LiNGAM
    p = os.path.join(results_dir, "abslingam_cv_results_empirical.pkl")
    if os.path.exists(p):
        data = joblib.load(p)
        if data:
            first_fold = next(iter(data))
            styles = list(data[first_fold].keys())
            for st in styles:
                label = f"Abs-LiNGAM ({st})"
                methods[label] = {}
                for fold_k, fold_dict in data.items():
                    if st in fold_dict:
                        methods[label][fold_k] = {st: fold_dict[st]}

    if not methods:
        raise FileNotFoundError(f"No evaluation result files found in {results_dir}")

    return methods

# ----------------------------- data loading -----------------------------

def load_pack(data_path):
    pack = joblib.load(data_path)
    # build deterministic parts per intervention
    det_ll_dict = {iota: v['D'] for iota, v in pack['ll'].items()}
    det_hl_dict = {eta:  v['D'] for eta,  v in pack['hl'].items()}
    omega = pack['omega']
    # infer dimensions from obs
    iota0 = 'iota0'
    eta0  = omega.get('iota0', 'eta0') if 'omega' in pack else 'eta0'
    N_ll, l = pack['ll'][iota0]['D'].shape
    N_hl, h = pack['hl'][eta0]['D'].shape
    if N_ll != N_hl:
        raise ValueError("Inconsistent N between LL and HL observational splits.")
    return det_ll_dict, det_hl_dict, omega, N_ll, l, h

# ----------------------------- evaluation core -----------------------------

def run_evaluation(data_path,
                   results_dir,
                   experiment=None,
                   alpha_values=None,
                   noise_levels=None,
                   trials=2,
                   zero_mean=True,
                   shift_type='additive',
                   distribution='gaussian',
                   output_file=None):
    """
    Main evaluation loop. Returns a DataFrame with records.
    """
    if alpha_values is None:
        alpha_values = np.linspace(0.0, 1.0, 10)
    if noise_levels is None:
        noise_levels = np.linspace(0.0, 5.0, 20)

    det_ll_dict, det_hl_dict, omega, N, l, h = load_pack(data_path)
    methods = collect_results(results_dir)

    # base covariances (identity by default)
    base_sigma_L = np.eye(l)
    base_sigma_H = np.eye(h)

    # Pretty print
    print("[EVAL] Configuration")
    print(f"  - data_path    : {data_path}")
    print(f"  - results_dir  : {results_dir}")
    if experiment:
        print(f"  - experiment   : {experiment}")
    print(f"  - interventions: {len(omega)} (LL→HL)")
    print(f"  - dims (LL,HL) : ({l},{h})")
    print(f"  - N per int    : {N}")
    print(f"  - methods      : {len(methods)}")
    print(f"  - folds/sample : inferred from result files")

    print(f"  - alphas       : {alpha_values[0]}..{alpha_values[-1]} ({len(alpha_values)})")
    print(f"  - noise scales : {noise_levels[0]}..{noise_levels[-1]} ({len(noise_levels)})")
    print(f"  - trials       : {trials}")
    print(f"  - shift_type   : {shift_type}")
    print(f"  - distribution : {distribution}")
    print(f"  - zero_mean    : {zero_mean}")

    records = []

    for alpha in tqdm(alpha_values, desc="Alpha grid"):
        for scale in noise_levels:
            for trial in range(trials):
                # build shift config once per (alpha, scale, trial)
                if distribution == 'gaussian':
                    mu_L = np.zeros(l) if zero_mean else np.ones(l) * scale
                    mu_H = np.zeros(h) if zero_mean else np.ones(h) * scale
                    shift_cfg = {
                        'type': shift_type,
                        'distribution': 'gaussian',
                        'll_params': {'mu': mu_L, 'sigma': base_sigma_L * (scale**2)},
                        'hl_params': {'mu': mu_H, 'sigma': base_sigma_H * (scale**2)},
                    }
                elif distribution == 'student-t':
                    shift_cfg = {
                        'type': shift_type,
                        'distribution': 'student-t',
                        'll_params': {'df': 3, 'loc': np.zeros(l), 'shape': base_sigma_L * (scale**2)},
                        'hl_params': {'df': 3, 'loc': np.zeros(h), 'shape': base_sigma_H * (scale**2)},
                    }
                elif distribution == 'exponential':
                    shift_cfg = {
                        'type': shift_type,
                        'distribution': 'exponential',
                        'll_params': {'scale': scale},
                        'hl_params': {'scale': scale},
                    }
                elif distribution == 'translation':
                    shift_cfg = {
                        'type': 'additive',
                        'distribution': 'translation',
                        'll_params': {'c': scale},
                        'hl_params': {'c': scale},
                    }
                elif distribution == 'scaling':
                    shift_cfg = {
                        'type': 'multiplicative',
                        'distribution': 'scaling',
                        'll_params': {'c': scale},
                        'hl_params': {'c': scale},
                    }
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")

                for method_label, folds_dict in methods.items():
                    # figure out number of folds from keys
                    fold_keys = sorted(folds_dict.keys(),
                                       key=lambda k: int(k.split('_')[-1]) if k.startswith('fold_') else 1e9)

                    for fk in fold_keys:
                        run_dict = folds_dict[fk]  # dict with one or more run keys
                        for run_key, payload in run_dict.items():
                            T = payload['T_matrix']         # (3,6)
                            test_idx = payload['test_indices']

                            # Evaluate across interventions present in omega
                            errs = []
                            for iota, eta in omega.items():
                                if iota not in det_ll_dict or eta not in det_hl_dict:
                                    continue
                                Dll_test = det_ll_dict[iota][test_idx]  # (n_test,6)
                                Dhl_test = det_hl_dict[eta][test_idx]   # (n_test,3)

                                # apply Huber contamination on deterministic parts
                                Dll_cont = apply_huber_contamination(Dll_test, alpha, shift_cfg, l, 'L',
                                                                     seed=(trial*10007 + len(test_idx)))
                                Dhl_cont = apply_huber_contamination(Dhl_test, alpha, shift_cfg, h, 'H',
                                                                     seed=(trial*10007 + len(test_idx) + 1))
                                err = empirical_error_fro(T, Dll_cont, Dhl_cont)
                                if not np.isnan(err):
                                    errs.append(err)

                            avg_err = float(np.mean(errs)) if errs else np.nan
                            records.append({
                                'method': method_label,
                                'run': run_key,
                                'alpha': float(alpha),
                                'noise_scale': float(scale),
                                'trial': int(trial),
                                'fold': fk,
                                'error': avg_err
                            })

    df = pd.DataFrame(records)
    print("\n[EVAL] Completed")
    print(f"  - total records: {len(df)}")
    print(f"  - methods      : {df['method'].nunique() if len(df)>0 else 0}")
    if len(df) > 0:
        print(f"  - alpha uniq   : {df['alpha'].nunique()}")
        print(f"  - noise uniq   : {df['noise_scale'].nunique()}")

    # Output path
    if experiment is None:
        # put next to data-path
        root_dir = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'evaluation_results')
    else:
        root_dir = os.path.join('data', experiment, 'evaluation_results')
    os.makedirs(root_dir, exist_ok=True)

    if output_file is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            root_dir,
            f"nonlinear_empirical_eval_{distribution}_{shift_type}"
            f"_a{len(alpha_values)}_{alpha_values[0]:.2f}-{alpha_values[-1]:.2f}"
            f"_s{len(noise_levels)}_{noise_levels[0]:.2f}-{noise_levels[-1]:.2f}"
            f"_tr{trials}_{ts}.csv"
        )
    else:
        # normalize to our root_dir
        output_file = os.path.join(root_dir, os.path.basename(output_file))

    df.to_csv(output_file, index=False)
    print(f"[EVAL] Saved -> {output_file}")
    return df

# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate learned T on non-linear LUCAS pack.")
    p.add_argument('--data-path', type=str, required=True,
                   help='Path to lucas_pack.pkl')
    p.add_argument('--results-dir', type=str, default=None,
                   help='Directory with *_cv_results_empirical.pkl (defaults to data/{experiment}/results_nonlinear)')
    p.add_argument('--experiment', type=str, default='lucas',
                   help='Experiment name (for default paths & output dir).')

    # contamination grids
    p.add_argument('--alpha-min', type=float, default=0.0)
    p.add_argument('--alpha-max', type=float, default=1.0)
    p.add_argument('--alpha-steps', type=int, default=10)

    p.add_argument('--noise-min', type=float, default=0.0)
    p.add_argument('--noise-max', type=float, default=5.0)
    p.add_argument('--noise-steps', type=int, default=20)

    p.add_argument('--trials', type=int, default=2)

    p.add_argument('--zero-mean', type=str, default='True',
                   help='If True, use zero mean for contamination (case-insensitive).')

    p.add_argument('--shift-type', type=str, default='additive',
                   choices=['additive', 'multiplicative'])

    p.add_argument('--distribution', type=str, default='gaussian',
                   choices=['gaussian', 'exponential', 'student-t', 'translation', 'scaling'])

    p.add_argument('--output', type=str, default=None,
                   help='Optional output CSV filename (basename used).')
    return p.parse_args()

def main():
    args = parse_args()
    # defaults for results-dir
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = os.path.join('data', args.experiment, 'results_nonlinear')

    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    noise_levels = np.linspace(args.noise_min, args.noise_max, args.noise_steps)
    zero_mean = args.zero_mean.lower() in ['true', '1', 'yes', 'y']

    run_evaluation(
        data_path=args.data_path,
        results_dir=results_dir,
        experiment=args.experiment,
        alpha_values=alpha_values,
        noise_levels=noise_levels,
        trials=args.trials,
        zero_mean=zero_mean,
        shift_type=args.shift_type,
        distribution=args.distribution,
        output_file=args.output
    )

if __name__ == "__main__":
    main()
