#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys
from tqdm import tqdm
import warnings
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Contamination helpers
# ---------------------------------------------------------------------

def apply_shift(clean_data, shift_config, all_var_names, model_level, seed=None):
    rng = np.random.default_rng(seed)
    shift_type = shift_config.get('type')
    dist_type = shift_config.get('distribution', 'gaussian')
    n_samples, n_dims = clean_data.shape
    level_key = 'll_params' if model_level == 'L' else 'hl_params'
    params = shift_config.get(level_key, {})
    noise_matrix = np.zeros_like(clean_data)

    if dist_type == 'gaussian':
        mu = np.array(params.get('mu', np.zeros(n_dims)))
        sigma = params.get('sigma', np.eye(n_dims))
        if np.ndim(sigma) == 1: sigma = np.diag(sigma)
        noise_matrix = rng.multivariate_normal(mean=mu, cov=sigma, size=n_samples)
    elif dist_type == 'student-t':
        df = params.get('df', 3)
        loc = np.array(params.get('loc', np.zeros(n_dims)))
        shape = params.get('shape', np.eye(n_dims))
        if np.ndim(shape) == 1: shape = np.diag(shape)
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
        indices = [all_var_names.index(v) for v in vars_to_affect if v in all_var_names]
        final_noise[:, indices] = noise_matrix[:, indices]

    if shift_type == 'additive': return clean_data + final_noise
    elif shift_type == 'multiplicative': return clean_data * final_noise
    else: raise ValueError(f"Unknown shift type: {shift_type}")

def apply_huber_contamination(clean_data, alpha, shift_config, all_var_names, model_level, seed=None):
    if alpha == 0: return clean_data
    noisy_data = apply_shift(clean_data, shift_config, all_var_names, model_level, seed=seed)
    if alpha == 1: return noisy_data
    n_samples = clean_data.shape[0]
    n_cont = int(alpha * n_samples)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_samples, n_cont, replace=False)
    out = clean_data.copy()
    out[idx] = noisy_data[idx]
    return out

def calculate_empirical_error(T_matrix, Xll_test, Xhl_test):
    if Xll_test.shape[0] == 0: return np.nan
    try:
        Xhl_pred = Xll_test @ T_matrix.T
        diff = Xhl_pred - Xhl_test
        return float(np.linalg.norm(diff, ord="fro")**2 / (diff.shape[0] * diff.shape[1]))
    except: return np.nan

# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_pack(experiment):
    pack_path = os.path.join("data", experiment, "lucas_pack.pkl")
    pack = joblib.load(pack_path)
    omega = pack["omega"]
    
    # Observational anchors
    iota0 = 'iota0'
    eta0 = omega.get(iota0, 'eta0')
    if eta0 not in pack['hl']: eta0 = 'eta0'
    
    # Shared observational noise 
    U_ll_obs = pack["ll"][iota0]["U"]
    U_hl_obs = pack["hl"][eta0]["U"]
    
    N, d_l = U_ll_obs.shape
    d_h = U_hl_obs.shape[1]
    
    return pack, omega, N, d_l, d_h, pack_path, U_ll_obs, U_hl_obs

def load_results(experiment):
    base = os.path.join("data", experiment, "results_nonlinear")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Results dir not found: {base}")
        
    results = {}
    
    # 1. GradCA (Single Run)
    p = os.path.join(base, "gradca_cv_results_empirical.pkl")
    if os.path.exists(p):
        results["GradCA"] = joblib.load(p)
        
    # 2. BaryCA (Single Run)
    p = os.path.join(base, "baryca_cv_results_empirical.pkl")
    if os.path.exists(p):
        results["BaryCA"] = joblib.load(p)
        
    # 3. DiRoCA 
    p = os.path.join(base, "diroca_cv_results_empirical.pkl")
    if os.path.exists(p):
        data = joblib.load(p)
        if data:
            first_fold = next(iter(data.values()))
            for rk in list(first_fold.keys()):
                mname = f"DiRoCA ({rk})"
                sliced = {f: {rk: v[rk]} for f, v in data.items() if rk in v}
                results[mname] = sliced

    # 4. Abs-LiNGAM 
    p = os.path.join(base, "abslingam_cv_results_empirical.pkl")
    if os.path.exists(p):
        data = joblib.load(p)
        if data:
            first_fold = next(iter(data.values()))
            for style in list(first_fold.keys()):
                mname = f"Abs-LiNGAM ({style})"
                sliced = {f: {style: v[style]} for f, v in data.items() if style in v}
                results[mname] = sliced

    print(f"[LOAD] Found methods: {list(results.keys())}")
    return results, base

def rebuild_kfolds(N, k_folds=5, seed=23):
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = []
    fold_sizes = np.full(k_folds, N // k_folds, dtype=int)
    fold_sizes[: N % k_folds] += 1
    cur = 0
    for fs in fold_sizes:
        te = indices[cur : cur + fs]
        tr = np.setdiff1d(indices, te)
        folds.append({"train": tr, "test": te})
        cur += fs
    return folds

# ---------------------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------------------

def run_evaluation(experiment="lucas", alpha_values=None, noise_levels=None, num_trials=2, 
                   zero_mean=True, shift_type="additive", distribution="gaussian", 
                   k_folds=5, seed=23, output_file=None, mode="standard"):

    if alpha_values is None: alpha_values = np.linspace(0, 1.0, 10)
    if noise_levels is None: noise_levels = np.linspace(0, 5.0, 20)

    # Load Data
    pack, omega, N, d_l, d_h, _, U_ll_shared, U_hl_shared = load_pack(experiment)
    results_to_eval, _ = load_results(experiment)
    folds = rebuild_kfolds(N, k_folds, seed)

    ll_names = [f"ll_{i}" for i in range(d_l)]
    hl_names = [f"hl_{i}" for i in range(d_h)]
    base_sig_L = np.eye(d_l)
    base_sig_H = np.eye(d_h)

    print(f"[EVAL] Running MODE: {mode.upper()}")
    
    records = []

    for alpha in tqdm(alpha_values, desc=f"Alpha ({mode})"):
        for scale in noise_levels:
            for trial in range(num_trials):
                for fi, fold in enumerate(folds):
                    te = fold["test"]
                    
                    for m_name, res_dict in results_to_eval.items():
                        if f"fold_{fi}" not in res_dict: continue
                        
                        # Extract T
                        fold_data = res_dict[f"fold_{fi}"]
                        run_key = next(iter(fold_data))
                        T = np.array(fold_data[run_key]["T_matrix"], dtype=np.float32)

                        errs = []
                        for iota, eta in omega.items():
                            if iota not in pack["ll"] or eta not in pack["hl"]: continue
                            
                            # --- DATA SELECTION LOGIC ---
                            if mode == "standard":
                                # Protocol A: Generalization (Raw Samples with New Noise)
                                Xll_clean = pack["ll"][iota]["X"][te]
                                Xhl_clean = pack["hl"][eta]["X"][te]
                            
                            elif mode == "counterfactual":
                                # Protocol B: Consistency (Reconstructed with Shared Noise)
                                D_L = pack["ll"][iota]["D"][te]
                                D_H = pack["hl"][eta]["D"][te]
                                U_L = U_ll_shared[te]
                                U_H = U_hl_shared[te]
                                Xll_clean = D_L + U_L
                                Xhl_clean = D_H + U_H
                            # ----------------------------

                            # Shift Config
                            sigma_L = base_sig_L * (scale**2)
                            sigma_H = base_sig_H * (scale**2)
                            mu_L = np.zeros(d_l) if zero_mean else np.ones(d_l)*scale
                            mu_H = np.zeros(d_h) if zero_mean else np.ones(d_h)*scale
                            
                            shift_config = {
                                "type": shift_type, "distribution": distribution,
                                "ll_params": {"mu": mu_L, "sigma": sigma_L, "scale": scale, "df": 3, "loc": mu_L, "shape": sigma_L},
                                "hl_params": {"mu": mu_H, "sigma": sigma_H, "scale": scale, "df": 3, "loc": mu_H, "shape": sigma_H}
                            }

                            seed_base = hash((fi, m_name, alpha, scale, trial, iota, mode)) % (2**32)
                            Xll_final = apply_huber_contamination(Xll_clean, alpha, shift_config, ll_names, "L", seed=seed_base)
                            Xhl_final = apply_huber_contamination(Xhl_clean, alpha, shift_config, hl_names, "H", seed=seed_base+1)

                            e = calculate_empirical_error(T, Xll_final, Xhl_final)
                            if not np.isnan(e): errs.append(e)
                        
                        if errs:
                            records.append({
                                "method": m_name, "alpha": alpha, "noise_scale": scale,
                                "trial": trial, "fold": fi, "error": np.mean(errs), "mode": mode
                            })

    # Save
    out_dir = os.path.join("data", experiment, "evaluation_results")
    os.makedirs(out_dir, exist_ok=True)
    
    if output_file is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fname = (f"empirical_evaluation_{shift_type}_{distribution}_"
                 f"mode_{mode}_"  
                 f"alpha{len(alpha_values)}_noise{len(noise_levels)}_"
                 f"trials{num_trials}_{timestamp}.csv")
        output_file = os.path.join(out_dir, fname)
    else:
        base = os.path.basename(output_file)
        if mode not in base:
            name, ext = os.path.splitext(base)
            base = f"{name}_{mode}{ext}"
        output_file = os.path.join(out_dir, base)
    
    pd.DataFrame(records).to_csv(output_file, index=False)
    print(f"[SAVE] {output_file}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="lucas")
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_steps", type=int, default=10)
    parser.add_argument("--noise_min", type=float, default=0.0)
    parser.add_argument("--noise_max", type=float, default=5.0)
    parser.add_argument("--noise_steps", type=int, default=20)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--shift_type", default="additive")
    parser.add_argument("--distribution", default="gaussian")
    
    # Mode selection
    parser.add_argument("--eval_mode", type=str, default="both", 
                        choices=["standard", "counterfactual", "both"],
                        help="Evaluation protocol. 'standard' uses raw samples (generalization). 'counterfactual' uses shared noise (consistency).")
    
    args = parser.parse_args()
    
    modes_to_run = ["standard", "counterfactual"] if args.eval_mode == "both" else [args.eval_mode]
    
    for m in modes_to_run:
        run_evaluation(
            experiment=args.experiment,
            alpha_values=np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps),
            noise_levels=np.linspace(args.noise_min, args.noise_max, args.noise_steps),
            num_trials=args.trials,
            shift_type=args.shift_type,
            distribution=args.distribution,
            mode=m 
        )

if __name__ == "__main__":
    main()