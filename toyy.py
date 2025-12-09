#!/usr/bin/env python3
"""
YAML-Driven GradCA (Standard Causal Abstraction).
Loads parameters from a YAML config file and runs 5-Fold CV.
"""

import argparse
import joblib
import numpy as np
import torch
import torch.optim as optim
import os
import yaml
import sys

# ==========================================
# 1. SETUP & UTILS
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[CONFIG] Loaded configuration from {config_path}")
    return config

def load_lucas_data(path):
    print(f"[DATA] Loading from {path}...")
    # Robust path handling
    if not os.path.exists(path):
        # Try finding it relative to current dir if full path fails
        alt_path = os.path.join(os.getcwd(), path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            print(f"❌ Error: Data file not found at {path}")
            sys.exit(1)
        
    pack = joblib.load(path)
    eta_obs = pack['omega'].get('iota0', 'eta0')
    
    # Extract Matrices or Fallback (handles different pack versions)
    try:
        U_ll = pack['ll']['iota0']['U']
        U_hl = pack['hl'][eta_obs]['U']
        det_ll_dict = {k: v['D'] for k, v in pack['ll'].items()}
        det_hl_dict = {k: v['D'] for k, v in pack['hl'].items()}
    except KeyError:
        U_ll = pack['ll']['iota0']['X']
        U_hl = pack['hl'][eta_obs]['X']
        det_ll_dict = {k: np.zeros_like(v['X']) for k, v in pack['ll'].items()}
        det_hl_dict = {k: np.zeros_like(v['X']) for k, v in pack['hl'].items()}

    return {
        'U_ll': torch.tensor(U_ll, dtype=torch.float32, device=DEVICE),
        'U_hl': torch.tensor(U_hl, dtype=torch.float32, device=DEVICE),
        'det_ll': {k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in det_ll_dict.items()},
        'det_hl': {k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in det_hl_dict.items()},
        'omega': pack['omega']
    }

# ==========================================
# 2. LOSS FUNCTION
# ==========================================

def empirical_objective(T, idx, U_ll, U_hl, det_ll_dict, det_hl_dict, omega):
    U_L_sub = U_ll[idx]
    U_H_sub = U_hl[idx]
    N = U_L_sub.shape[0]
    
    loss_total = torch.tensor(0.0, device=DEVICE)
    count = 0
    
    # Standard GradCA implies zero perturbations (Theta/Phi = 0)
    # If you later want Robust GradCA, these would become learnable parameters.
    Theta = torch.zeros_like(U_L_sub)
    Phi   = torch.zeros_like(U_H_sub)
    
    for iota, eta in omega.items():
        if iota not in det_ll_dict or eta not in det_hl_dict:
            continue
        
        # Causal mechanism: Deterministic + Noise + Perturbation
        endo_ll = det_ll_dict[iota][idx] + U_L_sub + Theta
        endo_hl = det_hl_dict[eta][idx] + U_H_sub + Phi

        # Abstraction Error: || T(Endo_L) - Endo_H ||^2
        diff = (T @ endo_ll.T).T - endo_hl
        loss_total += torch.norm(diff, p='fro')**2 / max(1, N)
        count += 1
        
    return loss_total / max(1, count)

# ==========================================
# 3. TRAINING ROUTINE
# ==========================================

def train_gradca(train_idx, data, params, d_l, d_h):
    # Map YAML keys to training variables
    lr = params.get('eta_min', 0.001)  # using eta_min as learning rate
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 5000)
    init_type = params.get('initialization', 'zeros')

    # Initialization
    if init_type == 'zeros':
        T = torch.zeros(d_h, d_l, requires_grad=True, device=DEVICE)
    else:
        T = torch.randn(d_h, d_l, requires_grad=True, device=DEVICE) * 0.01

    optimizer = optim.Adam([T], lr=lr)
    
    prev_loss = float('inf')
    args = (data['U_ll'], data['U_hl'], data['det_ll'], data['det_hl'], data['omega'])
    
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = empirical_objective(T, train_idx, *args)
        loss.backward()
        optimizer.step()
        
        # Convergence Check
        current_loss = loss.item()
        if abs(prev_loss - current_loss) < tol:
            break
        prev_loss = current_loss

    return T.detach(), current_loss

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def run_from_config(config_path):
    # 1. Load Config
    cfg = load_config(config_path)
    opt_cfg = cfg['optimization']
    cv_cfg = cfg['cv']
    data_cfg = cfg['data']
    out_cfg = cfg['output']

    # 2. Load Data
    data = load_lucas_data(data_cfg['data_path'])
    d_l = data['U_ll'].shape[1]
    d_h = data['U_hl'].shape[1]
    N = data['U_ll'].shape[0]

    # 3. Setup Folds
    seed = cv_cfg.get('seed', 23)
    k_folds = cv_cfg.get('k_folds', 5)
    
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    
    fold_sizes = np.full(k_folds, N // k_folds, dtype=int)
    fold_sizes[: N % k_folds] += 1
    
    folds = []
    curr = 0
    for fs in fold_sizes:
        test_idx = indices[curr : curr + fs]
        train_idx = np.setdiff1d(indices, test_idx)
        folds.append({'train': train_idx, 'test': test_idx})
        curr += fs

    results = {'GradCA': {}}
    print(f"🚀 Running GradCA | LR: {opt_cfg['eta_min']} | Iter: {opt_cfg['max_iter']} | Tol: {opt_cfg['tol']}")

    # 4. Run Loop
    for f_idx, fold in enumerate(folds):
        print(f"--- Fold {f_idx} ---")
        
        T_final, final_train_loss = train_gradca(fold['train'], data, opt_cfg, d_l, d_h)
        
        # Evaluate on Test Set
        with torch.no_grad():
             test_err = empirical_objective(T_final, fold['test'], 
                                            data['U_ll'], data['U_hl'], 
                                            data['det_ll'], data['det_hl'], 
                                            data['omega']).item()

        print(f"   ✅ Train Loss: {final_train_loss:.4f} | Test Error: {test_err:.4f}")
        
        results['GradCA'][f"fold_{f_idx}"] = {}
        results['GradCA'][f"fold_{f_idx}"]['standard'] = {
            "T_matrix": T_final.cpu().numpy(),
            "config": cfg,
            "val_abs_error": test_err
        }

    # 5. Save Results
    out_dir = out_cfg['save_directory']
    fname = out_cfg.get('filename_prefix', 'gradca_results') + ".pkl"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, fname)
    
    joblib.dump(results['GradCA'], save_path)
    print(f"\n[SAVE] Results saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Set Global Seeds
    # We read the seed from the YAML inside run_from_config, 
    # but setting a default here for PyTorch/Numpy initialization is safe.
    torch.manual_seed(23)
    np.random.seed(23)
    
    run_from_config(args.config)