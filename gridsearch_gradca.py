#!/usr/bin/env python3
"""
Grid Search specifically for GradCA (Standard Causal Abstraction).

CRITICAL LOGIC:
1. Training Objective: Matches non_linear_optimization.py exactly.
2. Selection Metric: Matches run_general_evaluation.py exactly.
   (We select the params that minimize the Evaluation metric, not the Training loss).

Usage:
  python gridsearch_gradca.py --experiment lucas
"""

import argparse
import joblib
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import os
import sys
import itertools
from sklearn.model_selection import KFold

# ==========================================
# 1. SETUP & DATA LOADING (Matching Optimization Script)
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Renamed to match the call in main loop
def load_lucas_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pack not found at {path}")
        
    pack = joblib.load(path)
    eta_obs = pack['omega'].get('iota0', 'eta0')
    
    # Extract U and D (Training Components)
    try:
        U_ll = pack['ll']['iota0']['U']
        U_hl = pack['hl'][eta_obs]['U']
        det_ll_dict = {k: v['D'] for k, v in pack['ll'].items()}
        det_hl_dict = {k: v['D'] for k, v in pack['hl'].items()}
    except KeyError:
        # Fallback for pure observational packs
        U_ll = pack['ll']['iota0']['X']
        U_hl = pack['hl'][eta_obs]['X']
        det_ll_dict = {k: np.zeros_like(v['X']) for k, v in pack['ll'].items()}
        det_hl_dict = {k: np.zeros_like(v['X']) for k, v in pack['hl'].items()}
    
    # Extract X (Validation Components - Evaluation script uses X directly)
    # X = D + U. Note: The pack usually stores raw X.
    X_ll_full = pack['ll']['iota0']['X']
    X_hl_full = pack['hl'][eta_obs]['X']

    # Convert Training components to Torch
    U_ll_t = torch.tensor(U_ll, dtype=torch.float32, device=DEVICE)
    U_hl_t = torch.tensor(U_hl, dtype=torch.float32, device=DEVICE)
    det_ll_t = {k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in det_ll_dict.items()}
    det_hl_t = {k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in det_hl_dict.items()}
    
    return {
        'U_ll': U_ll_t, 'U_hl': U_hl_t,
        'det_ll': det_ll_t, 'det_hl': det_hl_t,
        'omega': pack['omega'],
        # Numpy arrays for Validation Metric
        'X_ll_np': X_ll_full, 
        'X_hl_np': X_hl_full
    }

# ==========================================
# 2. TRAINING OBJECTIVE (From Optimization Script)
# ==========================================

def empirical_objective(T, idx, U_ll, U_hl, det_ll_dict, det_hl_dict, omega):
    """
    Exact copy of logic from non_linear_optimization.py
    """
    U_L_sub = U_ll[idx]
    U_H_sub = U_hl[idx]
    N = U_L_sub.shape[0]
    loss_total = torch.tensor(0.0, device=DEVICE)
    count = 0
    
    # GradCA = Zero Perturbations
    Theta = torch.zeros_like(U_L_sub)
    Phi   = torch.zeros_like(U_H_sub)
    
    for iota, eta in omega.items():
        if iota not in det_ll_dict or eta not in det_hl_dict:
            continue
        
        Dll_sub = det_ll_dict[iota][idx]
        Dhl_sub = det_hl_dict[eta][idx]

        endo_ll = Dll_sub + U_L_sub + Theta
        endo_hl = Dhl_sub + U_H_sub + Phi

        # Optimization Logic: T maps Low -> High
        diff = (T @ endo_ll.T).T - endo_hl
        
        loss_total += torch.norm(diff, p='fro')**2 / max(1, N)
        count += 1
        
    return loss_total / max(1, count)

# ==========================================
# 3. EVALUATION METRIC (From Evaluation Script)
# ==========================================

def calculate_empirical_error(T_matrix, Xll_test, Xhl_test):
    """
    Exact copy of logic from run_general_evaluation.py
    Returns: || Xll_test @ T^T - Xhl_test ||_F^2 / (n * d_h)
    """
    if Xll_test.shape[0] == 0 or Xhl_test.shape[0] == 0:
        return np.nan
    
    # Ensure numpy
    if isinstance(T_matrix, torch.Tensor):
        T_matrix = T_matrix.detach().cpu().numpy()
        
    try:
        # Evaluation Logic: Project X_L -> X_H
        Xhl_pred = Xll_test @ T_matrix.T
        diff = Xhl_pred - Xhl_test
        # Metric: MSE normalized by dimensions
        err = np.linalg.norm(diff, ord="fro")**2 / (diff.shape[0] * diff.shape[1])
        return float(err)
    except Exception as e:
        return np.nan

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train_and_evaluate(
    train_idx, val_idx, 
    data, 
    lr, tol, max_iter, 
    d_l, d_h
):
    # 1. Initialize T (Zeros as standard for GradCA)
    T = torch.zeros(d_h, d_l, requires_grad=True, device=DEVICE)
    optimizer = optim.Adam([T], lr=lr)
    
    prev_train_loss = float('inf')
    
    # Unpack Data
    U_ll, U_hl = data['U_ll'], data['U_hl']
    det_ll, det_hl = data['det_ll'], data['det_hl']
    omega = data['omega']
    
    # Validation Data (Numpy)
    X_ll_val = data['X_ll_np'][val_idx]
    X_hl_val = data['X_hl_np'][val_idx]

    # Optimization Loop
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Train on Training Objective
        loss = empirical_objective(T, train_idx, U_ll, U_hl, det_ll, det_hl, omega)
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        
        # Check Convergence based on training loss stability
        if abs(prev_train_loss - curr_loss) < tol:
            break
        prev_train_loss = curr_loss

    # FINAL EVALUATION: Compute strict Abstraction Error on Validation Set
    final_metric = calculate_empirical_error(T, X_ll_val, X_hl_val)
    
    return final_metric

# ==========================================
# 5. GRID SEARCH MAIN
# ==========================================

def run_grid_search(experiment='lucas', k_folds=5, seed=23):
    
    print(f"🚀 Starting GradCA Grid Search (Target: Evaluation Metric)")
    
    # 1. Load Data
    path = os.path.join("data", experiment, f"{experiment}_pack.pkl")
    # Fallback path logic
    if not os.path.exists(path):
        path = os.path.join("data", experiment, "lucas_pack.pkl")
        
    data = load_lucas_data(path)
    N = data['U_ll'].shape[0]
    d_l = data['U_ll'].shape[1]
    d_h = data['U_hl'].shape[1]

    # 2. Setup Folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(np.arange(N)))

    # 3. Define Grid
    # We test typical learning rates and tolerances
    param_grid = {
        'lr': [1e-1, 1e-2, 1e-3, 5e-4],
        'tol': [1e-4, 1e-5, 1e-6],
        'max_iter': [5000] # Keep fixed high enough to allow convergence
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🔎 Testing {len(combinations)} configurations across {k_folds} folds...")
    
    results = []

    # 4. Iterate Grid
    for config in tqdm(combinations):
        fold_metrics = []
        
        # Run CV for this config
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            
            # Train and Get Validation Error
            val_error = train_and_evaluate(
                train_idx, val_idx,
                data,
                lr=config['lr'],
                tol=config['tol'],
                max_iter=config['max_iter'],
                d_l=d_l, d_h=d_h
            )
            fold_metrics.append(val_error)
        
        # Average across folds
        avg_error = np.mean(fold_metrics)
        std_error = np.std(fold_metrics)
        
        res_entry = config.copy()
        res_entry['avg_val_error'] = avg_error
        res_entry['std_val_error'] = std_error
        results.append(res_entry)

    # 5. Report Results
    df = pd.DataFrame(results)
    df = df.sort_values(by='avg_val_error')
    
    print("\n" + "="*60)
    print("🏆 TOP 5 CONFIGURATIONS (Lowest Evaluation Error)")
    print("="*60)
    print(df[['lr', 'tol', 'max_iter', 'avg_val_error', 'std_val_error']].head(5).to_string(index=False))
    
    best = df.iloc[0]
    print("\n✅ BEST CONFIGURATION:")
    print(f"   LR: {best['lr']}")
    print(f"   Tol: {best['tol']}")
    print(f"   Max Iter: {best['max_iter']}")
    print(f"   Error: {best['avg_val_error']:.6f}")
    
    # Save to CSV
    df.to_csv(f"gridsearch_gradca_{experiment}.csv", index=False)
    print(f"\n💾 Full results saved to gridsearch_gradca_{experiment}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="lucas")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    set_seed(args.seed)
    run_grid_search(args.experiment, seed=args.seed)