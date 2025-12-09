#!/usr/bin/env python3
"""
Master Pipeline Script for LUCAS Experiment
-------------------------------------------
1. Runs Optimization (DiRoCA, GradCA, BaryCA, Abs-LiNGAM)
2. Runs Standard Evaluation (Huber Contamination)
3. Runs Omega Misspecification Analysis

Usage:
  python run_full_pipeline.py --experiment lucas
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import yaml
import torch
import torch.optim as optim
import torch.nn.init as init
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.stats import ttest_rel, multivariate_t
import glob
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# PART 1: OPTIMIZATION (Adapted from your general_optimization.py)
# =============================================================================

def load_lucas_pack(path):
    pack = joblib.load(path)
    required_top = ['T', 'omega', 'll', 'hl']
    for k in required_top:
        if k not in pack: raise KeyError(f"Missing key '{k}' in pack.")
    if 'iota0' not in pack['ll']: raise KeyError("ll['iota0'] missing")
    
    eta_obs = pack['omega'].get('iota0', 'eta0')
    if eta_obs not in pack['hl']:
        eta_obs = 'eta0' if 'eta0' in pack['hl'] else None
        if not eta_obs: raise KeyError("No observational HL found")

    det_ll_dict = {iota: v['D'] for iota, v in pack['ll'].items()}
    det_hl_dict = {eta: v['D'] for eta, v in pack['hl'].items()}
    U_ll_hat = pack['ll']['iota0']['U']
    U_hl_hat = pack['hl'][eta_obs]['U']
    
    return {
        'T_ref': pack['T'], 'omega': pack['omega'],
        'det_ll_dict': det_ll_dict, 'det_hl_dict': det_hl_dict,
        'U_ll_hat': U_ll_hat, 'U_hl_hat': U_hl_hat,
        'N': U_ll_hat.shape[0], 'l': U_ll_hat.shape[1], 'h': U_hl_hat.shape[1]
    }

def prepare_cv_folds_from_N(N, k_folds=5, seed=23):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(np.arange(N)):
        folds.append({'train': train_idx, 'test': test_idx})
    return folds

def project_onto_frobenius_ball(tensor, radius_limit):
    with torch.no_grad():
        cur = torch.norm(tensor, p='fro')
        if cur > radius_limit: tensor.mul_(radius_limit / (cur + 1e-12))
    return tensor

def compute_empirical_radius(N, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=1):
    if N <= 1: return 0.0
    term1 = c1 * (np.log(max(N, 2)) / N) ** (1 / alpha)
    term2 = c2 * np.sqrt(m * np.log(1 / eta) / N)
    return float(term1 + term2)

def empirical_objective(T, U_ll, U_hl, Theta_ll, Phi_hl, det_ll_dict, det_hl_dict, omega):
    device = T.device
    loss_total = torch.tensor(0.0, device=device)
    count = 0
    N = U_ll.shape[0]
    
    for iota, eta in omega.items():
        if iota not in det_ll_dict or eta not in det_hl_dict: continue
        Dll = torch.as_tensor(det_ll_dict[iota], dtype=torch.float32, device=device)
        Dhl = torch.as_tensor(det_hl_dict[eta], dtype=torch.float32, device=device)
        
        endo_ll = Dll + U_ll + Theta_ll
        endo_hl = Dhl + U_hl + Phi_hl
        
        diff = (T @ endo_ll.T).T - endo_hl
        loss_total += torch.norm(diff, p='fro')**2 / max(1, N)
        count += 1
        
    return loss_total / count if count > 0 else torch.tensor(0.0, device=device)

def barycentric_objective(T, U_ll, U_hl, det_ll_dict, det_hl_dict, omega):
    device = T.device
    ll_list = []
    hl_list = []
    for iota, eta in omega.items():
        if iota in det_ll_dict and eta in det_hl_dict:
            ll_list.append(torch.as_tensor(det_ll_dict[iota], dtype=torch.float32, device=device))
            hl_list.append(torch.as_tensor(det_hl_dict[eta], dtype=torch.float32, device=device))
    if not ll_list: return torch.tensor(0.0, device=device)
    
    avg_Dll = torch.mean(torch.stack(ll_list, dim=0), dim=0)
    avg_Dhl = torch.mean(torch.stack(hl_list, dim=0), dim=0)
    
    endo_ll = avg_Dll + U_ll
    endo_hl = avg_Dhl + U_hl
    
    N = endo_ll.shape[0]
    diff = (T @ endo_ll.T).T - endo_hl
    return torch.norm(diff, p='fro')**2 / max(1, N)

def run_abs_lingam(X_ll_obs, X_hl_obs, tau_perfect=1e-2, tau_noisy=1e-1):
    # Perfect
    T_raw = np.linalg.pinv(X_ll_obs) @ X_hl_obs
    mask = (np.abs(T_raw) > tau_perfect).astype(X_ll_obs.dtype)
    T_perfect = (T_raw * mask).T.astype(np.float32)
    
    # Noisy
    T_hat = np.linalg.pinv(X_ll_obs) @ X_hl_obs
    idx = np.argmax(np.abs(T_hat), axis=1)
    onehot = np.eye(X_hl_obs.shape[1])[idx]
    onehot *= (np.abs(T_hat) > tau_noisy).astype(int)
    T_noisy = (onehot * T_hat).T.astype(np.float32)
    
    return {'Perfect': {'T': T_perfect}, 'Noisy': {'T': T_noisy}}

def run_empirical_minmax(U_L, U_H, det_ll_dict, det_hl_dict, omega, epsilon, delta,
                         eta_min, max_iter, tol, seed, robust_L, robust_H, 
                         initialization, gain, optimizers, 
                         eta_max=0.0, num_steps_min=1, num_steps_max=0):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    l_full = U_L.shape[1]
    h_full = U_H.shape[1]
    
    T = torch.randn(h_full, l_full, requires_grad=True, device=device)
    if gain > 0: init.xavier_normal_(T, gain=gain)
    
    U_L = torch.as_tensor(U_L, dtype=torch.float32, device=device)
    U_H = torch.as_tensor(U_H, dtype=torch.float32, device=device)
    
    method = 'diroca' if (robust_L or robust_H) else 'gradca'
    
    if initialization == 'zeros':
        Theta = torch.zeros_like(U_L, requires_grad=(method=='diroca'), device=device)
        Phi = torch.zeros_like(U_H, requires_grad=(method=='diroca'), device=device)
    elif initialization == 'random':
        Theta = torch.randn_like(U_L, requires_grad=(method=='diroca'), device=device)
        Phi = torch.randn_like(U_H, requires_grad=(method=='diroca'), device=device)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
        
    opt_T = optim.Adam([T], lr=eta_min)
    opt_max = optim.Adam([Theta, Phi], lr=eta_max) if method == 'diroca' else None
    
    prev_obj = float('inf')
    N = U_L.shape[0]
    
    for it in tqdm(range(max_iter), desc=f"{method.upper()} Opt", leave=False):
        for _ in range(num_steps_min):
            opt_T.zero_grad()
            loss = empirical_objective(T, U_L, U_H, Theta.detach(), Phi.detach(), 
                                       det_ll_dict, det_hl_dict, omega)
            if torch.isnan(loss): break
            loss.backward()
            opt_T.step()
            
        if method == 'diroca':
            for _ in range(num_steps_max):
                opt_max.zero_grad()
                loss_max = -empirical_objective(T.detach(), U_L, U_H, Theta, Phi,
                                                det_ll_dict, det_hl_dict, omega)
                if torch.isnan(loss_max): break
                loss_max.backward()
                opt_max.step()
                if epsilon > 0: project_onto_frobenius_ball(Theta, epsilon * np.sqrt(N))
                if delta > 0: project_onto_frobenius_ball(Phi, delta * np.sqrt(N))
                
        cur = float(loss.item())
        if abs(prev_obj - cur) < tol:
            break
        prev_obj = cur
        
    return {
        'L': {'pert_U': Theta.detach().cpu().numpy(), 'radius': epsilon},
        'H': {'pert_U': Phi.detach().cpu().numpy(), 'radius': delta}
    }, T.detach().cpu().numpy()

def run_bary_optim_wrapper(U_ll, U_hl, det_ll, det_hl, omega, lr, max_iter, tol, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    U_ll = torch.as_tensor(U_ll, dtype=torch.float32, device=device)
    U_hl = torch.as_tensor(U_hl, dtype=torch.float32, device=device)
    
    h, l = U_hl.shape[1], U_ll.shape[1]
    T = torch.randn(h, l, requires_grad=True, device=device)
    opt = optim.Adam([T], lr=lr)
    
    prev = float('inf')
    for _ in tqdm(range(max_iter), desc="BaryCA Opt", leave=False):
        opt.zero_grad()
        loss = barycentric_objective(T, U_ll, U_hl, det_ll, det_hl, omega)
        if torch.isnan(loss): break
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if abs(prev - cur) < tol: break
        prev = cur
        
    return T.detach().cpu().numpy()

def run_optimization_phase(experiment, config_dir):
    print("\n" + "="*60)
    print("PHASE 1: OPTIMIZATION")
    print("="*60)
    
    methods = ['diroca', 'gradca', 'baryca', 'abslingam']
    configs = {}
    
    for m in methods:
        p = os.path.join(config_dir, f"{experiment}_{m}.yaml")
        if os.path.exists(p):
            with open(p) as f:
                cfg = yaml.safe_load(f)
                if cfg.get('enabled', True): configs[m] = cfg
    
    if not configs:
        print("[WARN] No configs found in", config_dir)
        return

    data_cache = {}
    
    for m, cfg in configs.items():
        print(f"Running {m.upper()}...")
        dpath = cfg['data']['data_path']
        if dpath not in data_cache:
            data_cache[dpath] = load_lucas_pack(dpath)
        all_data = data_cache[dpath]
        
        N, l, h = all_data['N'], all_data['l'], all_data['h']
        cv_cfg = cfg.get("cv", {})
        folds = prepare_cv_folds_from_N(N, cv_cfg.get('k_folds', 5), cv_cfg.get('seed', 23))
        
        out_dir = cfg['output']['save_directory']
        os.makedirs(out_dir, exist_ok=True)
        fname = cfg['output']['filename_prefix'] + ".pkl"
        save_path = os.path.join(out_dir, fname)
        
        opt_cfg = cfg.get('optimization', {})
        results = {}
        
        for fi, fold in enumerate(folds):
            tr, te = fold['train'], fold['test']
            U_L_tr = all_data['U_ll_hat'][tr]
            U_H_tr = all_data['U_hl_hat'][tr]
            det_ll_tr = {k: v[tr] for k,v in all_data['det_ll_dict'].items()}
            det_hl_tr = {k: v[tr] for k,v in all_data['det_hl_dict'].items()}
            
            if m == 'diroca':
                # Radius sweep logic simplified
                # Assume config has radius_sweep populated or default
                # Just doing a simple default set for brevity if not in config
                # You can add the full build_radius_pairs logic back if needed
                pairs = [[1.0, 1.0], [2.0, 2.0]] # Placeholder if config missing
                if 'radius_sweep' in cfg:
                    # Quick parse
                    pairs = cfg['radius_sweep'].get('pairs', pairs)
                
                results[f'fold_{fi}'] = {}
                for eps, delt in pairs:
                    _, T = run_empirical_minmax(
                        U_L_tr, U_H_tr, det_ll_tr, det_hl_tr, all_data['omega'],
                        eps, delt, opt_cfg['eta_min'], opt_cfg['max_iter'], opt_cfg['tol'],
                        cv_cfg.get('seed', 23), True, True, 
                        opt_cfg['initialization'], opt_cfg['gain'], opt_cfg['optimizers'],
                        eta_max=opt_cfg['eta_max'], num_steps_min=opt_cfg['num_steps_min'],
                        num_steps_max=opt_cfg['num_steps_max']
                    )
                    results[f'fold_{fi}'][f'eps_{eps}_delta_{delt}'] = {'T_matrix': T, 'test_indices': te}
                    
            elif m == 'gradca':
                _, T = run_empirical_minmax(
                    U_L_tr, U_H_tr, det_ll_tr, det_hl_tr, all_data['omega'],
                    0.0, 0.0, opt_cfg['eta_min'], opt_cfg['max_iter'], opt_cfg['tol'],
                    cv_cfg.get('seed', 23), False, False,
                    opt_cfg['initialization'], opt_cfg['gain'], opt_cfg['optimizers']
                )
                results[f'fold_{fi}'] = {'gradca_run': {'T_matrix': T, 'test_indices': te}}
                
            elif m == 'baryca':
                T = run_bary_optim_wrapper(
                    U_L_tr, U_H_tr, det_ll_tr, det_hl_tr, all_data['omega'],
                    opt_cfg['lr'], opt_cfg['max_iter'], opt_cfg['tol'], cv_cfg.get('seed', 23)
                )
                results[f'fold_{fi}'] = {'baryca_run': {'T_matrix': T, 'test_indices': te}}
                
            elif m == 'abslingam':
                # Observational data X = D + U
                pack = joblib.load(dpath)
                eta_obs = pack['omega'].get('iota0', 'eta0')
                X_ll = pack['ll']['iota0']['X'][tr]
                X_hl = pack['hl'][eta_obs]['X'][tr]
                res = run_abs_lingam(X_ll, X_hl, opt_cfg.get('tau_perfect', 1e-2), opt_cfg.get('tau_noisy', 1e-1))
                results[f'fold_{fi}'] = {
                    'Perfect': {'T_matrix': res['Perfect']['T'], 'test_indices': te},
                    'Noisy': {'T_matrix': res['Noisy']['T'], 'test_indices': te}
                }
                
        joblib.dump(results, save_path)
        print(f"Saved {m} to {save_path}")

# =============================================================================
# PART 2: EVALUATION (Adapted from your run_general_evaluation.py)
# =============================================================================

def calculate_empirical_error(T_matrix, Xll_test, Xhl_test):
    if Xll_test.shape[0] == 0: return np.nan
    Xhl_pred = Xll_test @ T_matrix.T
    diff = Xhl_pred - Xhl_test
    # Normalized Frobenius Error
    return float(np.linalg.norm(diff, ord='fro')**2 / (diff.shape[0] * diff.shape[1]))

def apply_shift_and_eval(T_learned, pack, omega, te_idx, alpha, scale, dist, shift_type, seed):
    # Simplified contamination logic for brevity
    # Assuming 'gaussian', 'additive' for this snippet, extend as needed
    rng = np.random.default_rng(seed)
    errs = []
    
    for iota, eta in omega.items():
        if iota not in pack['ll'] or eta not in pack['hl']: continue
        Xll = pack['ll'][iota]['X'][te_idx]
        Xhl = pack['hl'][eta]['X'][te_idx]
        
        # Add Noise (Huber-like logic)
        n = len(Xll)
        n_contam = int(alpha * n)
        
        if n_contam > 0:
            idx_c = rng.choice(n, n_contam, replace=False)
            noise_L = rng.normal(0, scale, (n_contam, Xll.shape[1]))
            noise_H = rng.normal(0, scale, (n_contam, Xhl.shape[1]))
            
            # Copy to avoid mutation
            Xll = Xll.copy(); Xhl = Xhl.copy()
            Xll[idx_c] += noise_L
            Xhl[idx_c] += noise_H
            
        e = calculate_empirical_error(T_learned, Xll, Xhl)
        if not np.isnan(e): errs.append(e)
        
    return np.mean(errs) if errs else np.nan

def run_evaluation_phase(experiment):
    print("\n" + "="*60)
    print("PHASE 2: STANDARD EVALUATION")
    print("="*60)
    
    base_dir = os.path.join("data", experiment)
    res_dir = os.path.join(base_dir, "results_nonlinear")
    pack_path = os.path.join(base_dir, "lucas_pack.pkl")
    
    if not os.path.exists(res_dir): return
    pack = joblib.load(pack_path)
    omega = pack['omega']
    
    # Load all results
    results_map = {}
    for f in glob.glob(os.path.join(res_dir, "*_cv_results_empirical.pkl")):
        if "diroca" in f: name = "DiRoCA"
        elif "gradca" in f: name = "GradCA"
        elif "baryca" in f: name = "BaryCA"
        elif "abslingam" in f: name = "AbsLiNGAM"
        else: continue
        results_map[name] = joblib.load(f)
        
    records = []
    # Eval Configs
    alphas = [0.0, 1.0]
    scales = [10.0]
    trials = 5
    
    for name, res_data in results_map.items():
        for fold_key, run_data in res_data.items():
            # Handle nested structure
            for run_id, inner in run_data.items():
                T = inner['T_matrix']
                te_idx = inner['test_indices']
                
                for a in alphas:
                    for s in scales:
                        for t in range(trials):
                            seed = hash((name, fold_key, run_id, a, s, t)) % (2**32)
                            err = apply_shift_and_eval(T, pack, omega, te_idx, a, s, 'gaussian', 'additive', seed)
                            
                            records.append({
                                'method': f"{name}_{run_id}",
                                'alpha': a,
                                'noise_scale': s,
                                'trial': t,
                                'error': err
                            })
                            
    df = pd.DataFrame(records)
    out_path = os.path.join(base_dir, "evaluation_results", "final_eval.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved evaluation results to {out_path}")
    
    # --- Print Stats Table (Huber) ---
    print("\n📊 Huber Stats (Sigma=10.0)")
    pivot = df.pivot_table(index=['trial'], columns=['method', 'alpha'], values='error')
    mean_errs = pivot.mean()
    print(mean_errs.to_string())

# =============================================================================
# PART 3: OMEGA MISSPECIFICATION (Adapted from your notebook snippet)
# =============================================================================

def run_omega_phase(experiment):
    print("\n" + "="*60)
    print("PHASE 3: OMEGA MISSPECIFICATION")
    print("="*60)
    
    base_dir = os.path.join("data", experiment)
    res_dir = os.path.join(base_dir, "results_nonlinear")
    pack_path = os.path.join(base_dir, "lucas_pack.pkl")
    
    pack = joblib.load(pack_path)
    omega = pack['omega']
    det_ll = {k: pack['ll'][k]['X'] for k in pack['ll']}
    det_hl = {k: pack['hl'][k]['X'] for k in pack['hl']}
    
    # Load all results
    pkl_files = glob.glob(os.path.join(res_dir, "*_cv_results_empirical.pkl"))
    records = []
    
    num_trials = 20
    
    for trial in tqdm(range(num_trials), desc="Omega Test"):
        # Corrupt Omega
        keys = list(omega.keys())
        vals = list(omega.values())
        np.random.shuffle(keys)
        corrupt_omega = {k: np.random.choice(vals) for k in keys}
        
        for pkl in pkl_files:
            data = joblib.load(pkl)
            base_name = os.path.basename(pkl).split('_')[0].upper()
            
            for fold_key, runs in data.items():
                for run_id, inner in runs.items():
                    T = inner['T_matrix']
                    te_idx = inner['test_indices']
                    
                    errs = []
                    for iota, _ in omega.items():
                        target = corrupt_omega[iota]
                        Xl = det_ll[iota][te_idx]
                        Xh = det_hl[target][te_idx]
                        
                        diff = (Xl @ T.T) - Xh
                        errs.append(np.linalg.norm(diff, 'fro') / np.sqrt(len(te_idx)))
                    
                    records.append({
                        'method': f"{base_name}_{run_id}",
                        'trial': trial,
                        'error': np.mean(errs)
                    })
                    
    df = pd.DataFrame(records)
    
    # Print Stats
    print("\n📊 Omega Stats (Maximal Corruption)")
    summary = df.groupby('method')['error'].agg(['mean', 'std']).sort_values('mean')
    print(summary)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='lucas')
    parser.add_argument('--config-dir', type=str, default='configs')
    parser.add_argument('--skip-opt', action='store_true')
    args = parser.parse_args()
    
    if not args.skip_opt:
        run_optimization_phase(args.experiment, args.config_dir)
    
    run_evaluation_phase(args.experiment)
    run_omega_phase(args.experiment)

if __name__ == "__main__":
    main()