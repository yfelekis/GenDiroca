#!/usr/bin/env python3
"""
Config-driven optimization for LUCAS-style packs (general_optimization.py)
--------------------------------------------------------------------------
Works with the output of lucas_nonlinear_generator.py.
Reads hyperparameters from YAML configs in the 'configs/' directory.
Allows CLI overrides for quick testing.

Usage:
  python general_optimization.py --experiment lucas --config-dir configs
"""

import os
import argparse
import joblib
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

# ----------------------------- Data loading ------------------------------

def load_lucas_pack(path):
    print(f"[DATA] Loading {path}...")
    pack = joblib.load(path)
    required_top = ['T', 'omega', 'll', 'hl']
    for k in required_top:
        if k not in pack:
            raise KeyError(f"Missing key '{k}' in pack.")

    if 'iota0' not in pack['ll']:
        raise KeyError("ll['iota0'] (observational) missing in pack.ll")
    
    eta_obs = pack['omega'].get('iota0', 'eta0')
    if eta_obs not in pack['hl']:
        if 'eta0' in pack['hl']:
            eta_obs = 'eta0'
        else:
            raise KeyError("No observational HL found")

    det_ll_dict = {iota: v['D'] for iota, v in pack['ll'].items()}
    det_hl_dict = {eta: v['D'] for eta, v in pack['hl'].items()}

    U_ll_hat = pack['ll']['iota0']['U']
    U_hl_hat = pack['hl'][eta_obs]['U']

    N, l = U_ll_hat.shape
    _, h = U_hl_hat.shape
    
    return {
        'T_ref': pack['T'],
        'omega': pack['omega'],
        'det_ll_dict': det_ll_dict,
        'det_hl_dict': det_hl_dict,
        'U_ll_hat': U_ll_hat,
        'U_hl_hat': U_hl_hat,
        'N': N, 'l': l, 'h': h
    }

def prepare_cv_folds_from_N(N, k_folds=5, seed=23):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(np.arange(N)):
        folds.append({'train': train_idx, 'test': test_idx})
    return folds

# ----------------------------- Optimization Logic ------------------------

def project_onto_frobenius_ball(tensor, radius_limit):
    """Project tensor onto Frobenius norm ball ||X||_F <= radius_limit."""
    with torch.no_grad():
        cur = torch.norm(tensor, p='fro')
        if cur > radius_limit:
            tensor.mul_(radius_limit / (cur + 1e-12))
    return tensor

def compute_empirical_radius(N, eta=0.05, c1=10.0, c2=1.0, alpha=2.0, m=1):
    if N <= 1: return 0.0
    term1 = c1 * (np.log(max(N, 2)) / N) ** (1 / alpha)
    term2 = c2 * np.sqrt(m * np.log(1 / eta) / N)
    return float(term1 + term2)

def empirical_objective(T, U_ll, U_hl, Theta_ll, Phi_hl, det_ll_dict, det_hl_dict, omega):
    device = T.device
    # FIX: Explicit float32 casting
    U_ll = U_ll.to(device, dtype=torch.float32)
    U_hl = U_hl.to(device, dtype=torch.float32)
    Theta_ll = Theta_ll.to(device, dtype=torch.float32)
    Phi_hl = Phi_hl.to(device, dtype=torch.float32)
    
    loss_total = torch.tensor(0.0, device=device, dtype=torch.float32)
    count = 0
    N = U_ll.shape[0]

    for iota, eta in omega.items():
        if iota not in det_ll_dict or eta not in det_hl_dict: continue
        
        Dll = torch.as_tensor(det_ll_dict[iota], device=device, dtype=torch.float32)
        Dhl = torch.as_tensor(det_hl_dict[eta],  device=device, dtype=torch.float32)

        endo_ll = Dll + U_ll + Theta_ll
        endo_hl = Dhl + U_hl + Phi_hl

        diff = (T @ endo_ll.T).T - endo_hl
        loss_total += torch.norm(diff, p='fro')**2 / max(1, N)
        count += 1
        
    return loss_total / max(1, count)

def run_minmax_optimization(U_L, U_H, det_ll, det_hl, omega, config, epsilon, delta):
    seed = config['cv'].get('seed', 23)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    opt_cfg = config['optimization']
    l_full, h_full = U_L.shape[1], U_H.shape[1]
    
    # Init T
    # ALWAYS init random first to break symmetry
    T = torch.randn(h_full, l_full, requires_grad=True, device=device, dtype=torch.float32)
    
    # Apply Xavier gain if specified
    if opt_cfg.get('gain', 0.0) > 0:
        init.xavier_normal_(T, gain=opt_cfg['gain'])
    
    # Init Adversaries
    robust = (epsilon > 0 or delta > 0)
    method_label = 'DiRoCA' if robust else 'GradCA'
    
    # Initialization config strictly applies to Adversaries (Theta), NOT T
    init_type = opt_cfg.get('initialization', 'random')
    
    # GradCA (robust=False) -> Theta/Phi remain 0 (Nominal Objective)
    # DiRoCA (robust=True)  -> Theta/Phi can be random or 0
    if init_type == 'zeros':
        Theta = torch.zeros_like(torch.as_tensor(U_L), requires_grad=robust, device=device, dtype=torch.float32)
        Phi = torch.zeros_like(torch.as_tensor(U_H), requires_grad=robust, device=device, dtype=torch.float32)
    else:
        # Default to random for DiRoCA
        Theta = torch.randn_like(torch.as_tensor(U_L), requires_grad=robust, device=device, dtype=torch.float32)
        Phi = torch.randn_like(torch.as_tensor(U_H), requires_grad=robust, device=device, dtype=torch.float32)

    # Debug print for GradCA to confirm params
    if not robust:
        print(f"DEBUG: GradCA Running with eta={opt_cfg['eta_min']}, tol={opt_cfg['tol']}, gain={opt_cfg.get('gain', 0.0)}")

    # Optimizers
    opt_T = optim.Adam([T], lr=opt_cfg['eta_min'])
    opt_adv = None
    if robust:
        opt_adv = optim.Adam([Theta, Phi], lr=opt_cfg.get('eta_max', 0.001))

    # Loop
    U_L_t = torch.as_tensor(U_L, dtype=torch.float32, device=device)
    U_H_t = torch.as_tensor(U_H, dtype=torch.float32, device=device)
    N = U_L.shape[0]
    prev_obj = float('inf')

    desc = f"{method_label}"
    if robust:
        desc += f"(e={epsilon},d={delta})"
        
    iterator = tqdm(range(opt_cfg['max_iter']), desc=desc, leave=False)
    
    for it in iterator:
        # Min Step
        for _ in range(opt_cfg.get('num_steps_min', 1)):
            opt_T.zero_grad()
            loss = empirical_objective(T, U_L_t, U_H_t, Theta.detach(), Phi.detach(), det_ll, det_hl, omega)
            if torch.isnan(loss): break
            loss.backward()
            opt_T.step()
            
        # Max Step
        if robust:
            for _ in range(opt_cfg.get('num_steps_max', 1)):
                opt_adv.zero_grad()
                loss_adv = -empirical_objective(T.detach(), U_L_t, U_H_t, Theta, Phi, det_ll, det_hl, omega)
                loss_adv.backward()
                opt_adv.step()
                if epsilon > 0: project_onto_frobenius_ball(Theta, epsilon * np.sqrt(N))
                if delta > 0: project_onto_frobenius_ball(Phi, delta * np.sqrt(N))

        curr = float(loss.item())
        if abs(prev_obj - curr) < opt_cfg['tol']:
            iterator.set_postfix(status=f"Cvgd {it}")
            break
        prev_obj = curr
        
    return {
        'T_matrix': T.detach().cpu().numpy().astype(np.float32),
        'opt_params': {'eps': epsilon, 'delta': delta}
    }

# ----------------------------- Baselines --------------------------------

def run_baryca(U_L, U_H, det_ll, det_hl, omega, config):
    seed = config['cv'].get('seed', 23)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_cfg = config['optimization']
    
    U_L_t = torch.as_tensor(U_L, dtype=torch.float32, device=device)
    U_H_t = torch.as_tensor(U_H, dtype=torch.float32, device=device)
    
    ll_list, hl_list = [], []
    for iota, eta in omega.items():
        if iota in det_ll and eta in det_hl:
            ll_list.append(torch.as_tensor(det_ll[iota], dtype=torch.float32, device=device))
            hl_list.append(torch.as_tensor(det_hl[eta], dtype=torch.float32, device=device))
            
    avg_Dll = torch.mean(torch.stack(ll_list), dim=0)
    avg_Dhl = torch.mean(torch.stack(hl_list), dim=0)
    
    endo_ll = avg_Dll + U_L_t
    endo_hl = avg_Dhl + U_H_t
    N = endo_ll.shape[0]
    
    T = torch.randn(U_H.shape[1], U_L.shape[1], requires_grad=True, device=device)
    optimizer = optim.Adam([T], lr=opt_cfg.get('lr', 0.001))
    
    prev = float('inf')
    iterator = tqdm(range(opt_cfg['max_iter']), desc="BaryCA Optimization", leave=False)
    
    for it in iterator:
        optimizer.zero_grad()
        diff = (T @ endo_ll.T).T - endo_hl
        loss = torch.norm(diff, p='fro')**2 / max(1, N)
        loss.backward()
        optimizer.step()
        
        curr = float(loss.item())
        if abs(prev - curr) < opt_cfg['tol']:
            iterator.set_postfix(status=f"Converged iter {it}")
            break
        prev = curr
        
    return {'T_matrix': T.detach().cpu().numpy().astype(np.float32)}

def run_abs_lingam(U_ll_obs, U_hl_obs, config):
    opt_cfg = config['optimization']
    # Perfect
    T_raw = np.linalg.pinv(U_ll_obs) @ U_hl_obs
    mask = (np.abs(T_raw) > opt_cfg['tau_perfect']).astype(np.float32)
    T_perf = (T_raw * mask).T
    
    # Noisy
    T_hat = np.linalg.pinv(U_ll_obs) @ U_hl_obs
    idx = np.argmax(np.abs(T_hat), axis=1)
    onehot = np.eye(U_hl_obs.shape[1])[idx]
    onehot *= (np.abs(T_hat) > opt_cfg['tau_noisy']).astype(int)
    T_noisy = (onehot * T_hat).T

    return {
        'Perfect': {'T_matrix': T_perf.astype(np.float32)},
        'Noisy': {'T_matrix': T_noisy.astype(np.float32)} 
    }

# ----------------------------- Main -------------------------------------

def load_yaml(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='lucas')
    parser.add_argument('--config-dir', type=str, default='configs')
    
    # --- OVERRIDE FLAGS ---
    parser.add_argument('--gradca-max-iter', type=int, default=None)
    parser.add_argument('--gradca-eta-min', type=float, default=None)
    parser.add_argument('--gradca-tol', type=float, default=None)
    parser.add_argument('--gradca-gain', type=float, default=None)  # <--- ADDED THIS FLAG
    
    parser.add_argument('--diroca-max-iter', type=int, default=None)
    parser.add_argument('--baryca-max-iter', type=int, default=None)
    
    parser.add_argument('--skip-abslingam', action='store_true')
    parser.add_argument('--skip-diroca', action='store_true')
    parser.add_argument('--skip-gradca', action='store_true')
    parser.add_argument('--skip-baryca', action='store_true')

    args = parser.parse_args()
    
    methods = ['diroca', 'gradca', 'baryca', 'abslingam']
    configs = {}
    
    # Load configs
    for m in methods:
        p = os.path.join(args.config_dir, f"{args.experiment}_{m}.yaml")
        if os.path.exists(p):
            configs[m] = load_yaml(p)
            
    if not configs:
        raise ValueError(f"No configs found in {args.config_dir}")

    # --- APPLY CLI OVERRIDES ---
    if 'gradca' in configs:
        if args.gradca_max_iter is not None:
            configs['gradca']['optimization']['max_iter'] = args.gradca_max_iter
        if args.gradca_eta_min is not None:
            configs['gradca']['optimization']['eta_min'] = args.gradca_eta_min
        if args.gradca_tol is not None:
            configs['gradca']['optimization']['tol'] = args.gradca_tol
        if args.gradca_gain is not None:
            configs['gradca']['optimization']['gain'] = args.gradca_gain  # <--- LOGIC TO OVERRIDE
        if args.skip_gradca:
            configs['gradca']['enabled'] = False

    if 'diroca' in configs:
        if args.diroca_max_iter is not None:
            configs['diroca']['optimization']['max_iter'] = args.diroca_max_iter
        if args.skip_diroca:
            configs['diroca']['enabled'] = False

    if 'baryca' in configs:
        if args.baryca_max_iter is not None:
            configs['baryca']['optimization']['max_iter'] = args.baryca_max_iter
        if args.skip_baryca:
            configs['baryca']['enabled'] = False

    if 'abslingam' in configs:
        if args.skip_abslingam:
            configs['abslingam']['enabled'] = False


    # Load Data (once)
    data_cfg = configs[next(iter(configs))]['data']
    all_data = load_lucas_pack(data_cfg['data_path'])
    N, l, h = all_data['N'], all_data['l'], all_data['h']
    
    # Prepare Folds (once)
    seed = configs.get('diroca', list(configs.values())[0])['cv']['seed']
    folds = prepare_cv_folds_from_N(N, 5, seed)
    
    for m, cfg in configs.items():
        if not cfg.get('enabled', True): continue
        
        print(f"\n[{m.upper()}] Starting optimization...")
        out_dir = cfg['output']['save_directory']
        os.makedirs(out_dir, exist_ok=True)
        filename = cfg['output']['filename_prefix'] + ".pkl"
        
        results = {}
        
        for fi, fold in enumerate(tqdm(folds, desc=f"{m} folds")):
            tr, te = fold['train'], fold['test']
            
            # Slice Data
            U_ll_tr = all_data['U_ll_hat'][tr]
            U_hl_tr = all_data['U_hl_hat'][tr]
            det_ll_tr = {k: v[tr] for k, v in all_data['det_ll_dict'].items()}
            det_hl_tr = {k: v[tr] for k, v in all_data['det_hl_dict'].items()}
            
            results[f'fold_{fi}'] = {}
            
            if m == 'diroca':
                pairs = cfg.get('radius_sweep', {}).get('pairs', [])
                if not pairs:
                    be = compute_empirical_radius(N, m=l)
                    bd = compute_empirical_radius(N, m=h)
                    pairs = [(be, bd), (1.0,1.0), (2.0,2.0), (4.0,4.0), (8.0,8.0)]
                    
                for (eps, delt) in pairs:
                    print(f"  > Radius: eps={eps}, delta={delt}")
                    res = run_minmax_optimization(U_ll_tr, U_hl_tr, det_ll_tr, det_hl_tr, 
                                                all_data['omega'], cfg, eps, delt)
                    res['test_indices'] = te
                    results[f'fold_{fi}'][f'eps_{eps}_delta_{delt}'] = res
                    
            elif m == 'gradca':
                res = run_minmax_optimization(U_ll_tr, U_hl_tr, det_ll_tr, det_hl_tr,
                                            all_data['omega'], cfg, 0.0, 0.0)
                res['test_indices'] = te
                results[f'fold_{fi}']['gradca_run'] = res
                
            elif m == 'baryca':
                res = run_baryca(U_ll_tr, U_hl_tr, det_ll_tr, det_hl_tr, all_data['omega'], cfg)
                res['test_indices'] = te
                results[f'fold_{fi}']['baryca_run'] = res
                
            elif m == 'abslingam':
                res_dict = run_abs_lingam(U_ll_tr, U_hl_tr, cfg)
                for subk in res_dict:
                    res_dict[subk]['test_indices'] = te
                results[f'fold_{fi}'] = res_dict

        out_path = os.path.join(out_dir, filename)
        joblib.dump(results, out_path)
        print(f"[{m.upper()}] Saved to {out_path}")

    print("\n[DONE] All optimizations complete.")

if __name__ == '__main__':
    main()