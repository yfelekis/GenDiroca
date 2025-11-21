#!/usr/bin/env python3
"""
Config-driven optimization for LUCAS-style packs
------------------------------------------------

Works with the output of lucas_nonlinear_generator.py.

Pack format (joblib):
  {
    'T': (3x6) np.ndarray,
    'omega': dict[str->str],
    'll': { iota: {'X','D','U'} with arrays (N,6) },
    'hl': { eta: {'X','D','U'} with arrays (N,3) },
    'hl_model': {...},
    'graphs': {...}
  }

This script:
  * Loads per-method YAML configs based on the experiment name.
  * For each enabled method (DiRoCA, GradCA, BaryCA, Abs-LiNGAM):
      - Loads data (once per data_path).
      - Builds K-fold splits (per method, from config).
      - Runs the corresponding optimization / baseline.
      - Saves CV results to the configured output directory.

Usage:
  python non_linear_optimization.py --experiment lucas

Later, you can reuse the same script with a different experiment by
adding new configs `<experiment>_diroca.yaml`, etc., plus a loader
for that experiment in `load_pack_for_experiment`.
"""

import os
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn  # noqa: F401  (kept for possible extensions)
import torch.optim as optim
import torch.nn.init as init

import yaml


# ----------------------------- Data loading ------------------------------

def load_lucas_pack(path):
    """
    Loader specific to the LUCAS pack from lucas_nonlinear_generator.py.

    Returns a dict with:
      - T_ref: (h,l)
      - omega: dict
      - det_ll_dict: iota -> D_ll (N,l)
      - det_hl_dict: eta  -> D_hl (N,h)
      - U_ll_hat: (N,l) observational LL noise
      - U_hl_hat: (N,h) observational HL noise
      - N, l, h
    """
    pack = joblib.load(path)
    # Basic integrity checks
    required_top = ['T', 'omega', 'll', 'hl']
    for k in required_top:
        if k not in pack:
            raise KeyError(f"Missing key '{k}' in pack.")

    if 'iota0' not in pack['ll']:
        raise KeyError("ll['iota0'] (observational) missing in pack.ll")
    eta_obs = pack['omega'].get('iota0', 'eta0')
    if eta_obs not in pack['hl']:
        # fallback to 'eta0' if provided
        if 'eta0' in pack['hl']:
            eta_obs = 'eta0'
        else:
            raise KeyError("No observational HL found (expected omega['iota0'] or 'eta0')")

    # Deterministic parts
    det_ll_dict = {iota: v['D'] for iota, v in pack['ll'].items()}
    det_hl_dict = {eta: v['D'] for eta, v in pack['hl'].items()}

    # Noise anchors from observational splits
    U_ll_hat = pack['ll']['iota0']['U']        # (N,l)
    U_hl_hat = pack['hl'][eta_obs]['U']        # (N,h)

    N, l = U_ll_hat.shape
    _, h = U_hl_hat.shape

    omega = pack['omega']
    T_ref = pack['T']

    return {
        'T_ref': T_ref,
        'omega': omega,
        'det_ll_dict': det_ll_dict,
        'det_hl_dict': det_hl_dict,
        'U_ll_hat': U_ll_hat,
        'U_hl_hat': U_hl_hat,
        'N': N,
        'l': l,
        'h': h
    }


def load_pack_for_experiment(experiment: str, data_path: str):
    """
    Dispatcher for different experiments. Currently only 'lucas' is supported.
    Extend this with new loaders for other datasets.
    """
    if experiment.lower() == 'lucas':
        return load_lucas_pack(data_path)
    else:
        raise NotImplementedError(
            f"No loader implemented for experiment '{experiment}'. "
            "Please add a corresponding load_*_pack function."
        )


def prepare_cv_folds_from_N(N, k_folds=5, seed=23):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(np.arange(N)):
        folds.append({'train': train_idx, 'test': test_idx})
    return folds


# ----------------------------- helpers ----------------------------------

def project_onto_frobenius_ball(tensor, radius_limit):
    """Project tensor onto Frobenius norm ball ||X||_F <= radius_limit."""
    with torch.no_grad():
        cur = torch.norm(tensor, p='fro')
        if cur > radius_limit:
            tensor.mul_(radius_limit / (cur + 1e-12))
    return tensor


def compute_empirical_radius(N, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=1):
    """
    Simple (light-tail) empirical radius proxy, dimension-aware.
    """
    if N <= 1:
        return 0.0
    term1 = c1 * (np.log(max(N, 2)) / N) ** (1 / alpha)
    term2 = c2 * np.sqrt(m * np.log(1 / eta) / N)
    return float(term1 + term2)


# ----------------------------- objectives --------------------------------

def empirical_objective(T, U_ll, U_hl, Theta_ll, Phi_hl,
                        det_ll_dict, det_hl_dict, omega):
    """
    Average over interventions of Frobenius norm between mapped LL and HL:
    loss = (1/|I|) sum_i || T (D_ll^i + U_ll + Theta) - (D_hl^{omega(i)} + U_hl + Phi) ||_F^2 / N
    """
    device = T.device
    U_ll = U_ll.to(device)
    U_hl = U_hl.to(device)
    Theta_ll = Theta_ll.to(device)
    Phi_hl = Phi_hl.to(device)

    N = U_ll.shape[0]
    loss_total = torch.tensor(0.0, device=device)
    count = 0
    for iota, eta in omega.items():
        if iota not in det_ll_dict or eta not in det_hl_dict:
            continue
        Dll = torch.as_tensor(det_ll_dict[iota], dtype=torch.float32, device=device)  # (N,l)
        Dhl = torch.as_tensor(det_hl_dict[eta],  dtype=torch.float32, device=device)  # (N,h)

        endo_ll = Dll + U_ll + Theta_ll
        endo_hl = Dhl + U_hl + Phi_hl

        diff = (T @ endo_ll.T).T - endo_hl
        loss_total += torch.norm(diff, p='fro')**2 / max(1, N)
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=device)
    return loss_total / count


def barycentric_objective(T, U_ll, U_hl, det_ll_dict, det_hl_dict, omega):
    """
    BaryCA: compute average deterministic parts across interventions, then
    add shared U_ll, U_hl from the training subset, and fit T to map averages.
    """
    device = T.device
    U_ll = U_ll.to(device)
    U_hl = U_hl.to(device)

    ll_list = []
    hl_list = []
    for iota, eta in omega.items():
        if iota in det_ll_dict and eta in det_hl_dict:
            ll_list.append(torch.as_tensor(det_ll_dict[iota], dtype=torch.float32, device=device))
            hl_list.append(torch.as_tensor(det_hl_dict[eta],  dtype=torch.float32, device=device))
    if not ll_list:
        return torch.tensor(0.0, device=device)

    avg_Dll = torch.mean(torch.stack(ll_list, dim=0), dim=0)  # (N,l)
    avg_Dhl = torch.mean(torch.stack(hl_list, dim=0), dim=0)  # (N,h)

    endo_ll = avg_Dll + U_ll
    endo_hl = avg_Dhl + U_hl

    N = endo_ll.shape[0]
    diff = (T @ endo_ll.T).T - endo_hl
    return torch.norm(diff, p='fro')**2 / max(1, N)


# ----------------------------- Abs-LiNGAM (baseline) --------------------

def perfect_abstraction(px_samples, py_samples, tau_threshold=1e-2):
    T_raw = np.linalg.pinv(px_samples) @ py_samples  # (d_l, d_h)
    mask = (np.abs(T_raw) > tau_threshold).astype(px_samples.dtype)
    return T_raw * mask


def noisy_abstraction(px_samples, py_samples, tau_threshold=1e-1, refit_coeff=False):
    T_hat = np.linalg.pinv(px_samples) @ py_samples  # (d_l, d_h)
    idx = np.argmax(np.abs(T_hat), axis=1)           # (d_l,)
    onehot = np.eye(py_samples.shape[1])[idx]        # (d_l, d_h)
    onehot *= (np.abs(T_hat) > tau_threshold).astype(int)
    T_masked = onehot * T_hat
    # optional refit
    if refit_coeff:
        T_refit = T_masked.copy()
        for y in range(onehot.shape[1]):
            block = np.where(onehot[:, y] == 1)[0]
            if len(block) > 0:
                T_block = np.linalg.pinv(px_samples[:, block]) @ py_samples[:, y]
                T_refit[block, y] = T_block
        return T_refit
    return T_masked


def run_abs_lingam(X_ll_obs, X_hl_obs, tau_perfect=1e-2, tau_noisy=1e-1):
    # Shapes: (N, d_l) and (N, d_h)
    T_perfect = perfect_abstraction(X_ll_obs, X_hl_obs, tau_threshold=tau_perfect).T.astype(np.float32)
    T_noisy   = noisy_abstraction(X_ll_obs,   X_hl_obs, tau_threshold=tau_noisy, refit_coeff=False).T.astype(np.float32)
    return {
        'Perfect': {'T': T_perfect},
        'Noisy':   {'T': T_noisy}
    }


# ----------------------------- Core trainers -----------------------------

def run_empirical_minmax(U_L, U_H, det_ll_dict, det_hl_dict, omega,
                         epsilon, delta,
                         eta_min, eta_max,
                         num_steps_min, num_steps_max,
                         max_iter, tol, seed,
                         robust_L, robust_H,
                         initialization, gain, optimizers):
    """
    Core loop for DiRoCA (robust) / GradCA (non-robust) on a generic pack.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # T dimensions: map d_l -> d_h (so T is (d_h, d_l))
    l_full = U_L.shape[1]
    h_full = U_H.shape[1]
    T = torch.randn(h_full, l_full, requires_grad=True, device=device)
    if gain > 0:
        init.xavier_normal_(T, gain=gain)

    U_L = torch.as_tensor(U_L, dtype=torch.float32, device=device)
    U_H = torch.as_tensor(U_H, dtype=torch.float32, device=device)

    method = 'diroca' if (robust_L or robust_H) else 'gradca'

    # perturbations
    if initialization == 'zeros':
        Theta = torch.zeros_like(U_L, requires_grad=(method == 'diroca'), device=device)
        Phi   = torch.zeros_like(U_H, requires_grad=(method == 'diroca'), device=device)
    elif initialization == 'random':
        Theta = torch.randn_like(U_L, requires_grad=(method == 'diroca'), device=device)
        Phi   = torch.randn_like(U_H, requires_grad=(method == 'diroca'), device=device)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")

    # optimizers
    if optimizers == 'adam':
        opt_T = optim.Adam([T], lr=eta_min)
        opt_max = optim.Adam([Theta, Phi], lr=eta_max) if method == 'diroca' else None
    elif optimizers == 'adam_betas':
        opt_T = optim.Adam([T], lr=eta_min, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        opt_max = optim.Adam([Theta, Phi], lr=eta_max, betas=(0.9, 0.999),
                             eps=1e-8, amsgrad=True) if method == 'diroca' else None
    else:
        raise ValueError(f"Unknown optimizer: {optimizers}")

    prev_T_obj = float('inf')
    N = U_L.shape[0]

    for it in tqdm(range(max_iter), desc=f"{method.upper()} Optimization"):
        # Min step(s) on T (Θ,Φ detached)
        for _ in range(num_steps_min):
            opt_T.zero_grad()
            T_obj = empirical_objective(T, U_L, U_H, Theta.detach(), Phi.detach(),
                                        det_ll_dict, det_hl_dict, omega)
            if torch.isnan(T_obj):
                break
            T_obj.backward()
            opt_T.step()

        # Max step(s) on Theta,Phi if robust
        if method == 'diroca':
            for _ in range(num_steps_max):
                opt_max.zero_grad()
                max_obj = -empirical_objective(T.detach(), U_L, U_H, Theta, Phi,
                                               det_ll_dict, det_hl_dict, omega)
                if torch.isnan(max_obj):
                    break
                max_obj.backward()
                opt_max.step()

                # Project onto balls ||Theta||_F <= sqrt(N)*ε, ||Phi||_F <= sqrt(N)*δ
                if epsilon > 0:
                    project_onto_frobenius_ball(Theta, epsilon * np.sqrt(N))
                if delta > 0:
                    project_onto_frobenius_ball(Phi,   delta * np.sqrt(N))

        cur = float(T_obj.item())
        if abs(prev_T_obj - cur) < tol:
            print(f"{method.upper()} converged at iter {it+1} (Δobj<{tol}).")
            break
        prev_T_obj = cur

    T_final = T.detach().cpu().numpy().astype(np.float32)
    paramsL = {'pert_U': Theta.detach().cpu().numpy().astype(np.float32),
               'radius': epsilon}
    paramsH = {'pert_U': Phi.detach().cpu().numpy().astype(np.float32),
               'radius': delta}
    return {'L': paramsL, 'H': paramsH}, T_final


def run_bary_optim(U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega,
                   lr=1e-3, max_iter=3000, tol=1e-5, seed=42):
    """
    Optimize T on the barycentric objective.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    U_ll_hat = torch.as_tensor(U_ll_hat, dtype=torch.float32, device=device)
    U_hl_hat = torch.as_tensor(U_hl_hat, dtype=torch.float32, device=device)

    h_full, l_full = U_hl_hat.shape[1], U_ll_hat.shape[1]
    T = torch.randn(h_full, l_full, requires_grad=True, device=device)
    opt = optim.Adam([T], lr=lr)

    prev = float('inf')
    for it in tqdm(range(max_iter), desc="BaryCA Optimization"):
        opt.zero_grad()
        loss = barycentric_objective(T, U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega)
        if torch.isnan(loss):
            break
        loss.backward()
        opt.step()
        cur = float(loss.item())
        if abs(prev - cur) < tol:
            print(f"BaryCA converged at iter {it+1} (Δ<{tol}).")
            break
        prev = cur

    return T.detach().cpu().numpy().astype(np.float32)


# ----------------------------- Config helpers ---------------------------

def load_yaml_config(config_path: str):
    if not os.path.exists(config_path):
        print(f"[WARN] Config file not found: {config_path}. Skipping this method.")
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_radius_pairs(config, N, l, h):
    """
    Build list of (eps, delta) pairs from config.

    Supports two styles:
      radius_sweep:
        use_theoretical_bounds: true/false
        theoretical_params: {eta,c1,c2,alpha}
        additional_pairs: [ [eps,delta], ... ]

    or

      radius_sweep:
        use_theoretical_bounds: false
        pairs: [ [eps,delta], ... ]
    """
    rs_cfg = config.get("radius_sweep", {})
    pairs = []

    use_theoretical = rs_cfg.get("use_theoretical_bounds", False)
    if use_theoretical:
        tp = rs_cfg.get("theoretical_params", {})
        eta = tp.get("eta", 0.05)
        c1  = tp.get("c1", 1000.0)
        c2  = tp.get("c2", 1.0)
        alpha = tp.get("alpha", 2.0)
        base_eps = round(compute_empirical_radius(N, eta=eta, c1=c1, c2=c2, alpha=alpha, m=l), 3)
        base_del = round(compute_empirical_radius(N, eta=eta, c1=c1, c2=c2, alpha=alpha, m=h), 3)
        pairs.append([base_eps, base_del])

        addl = rs_cfg.get("additional_pairs", [])
        pairs.extend(addl)
    else:
        direct_pairs = rs_cfg.get("pairs", [])
        pairs.extend(direct_pairs)

    if not pairs:
        # Fallback
        pairs = [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]
    return pairs


# ----------------------------- Main orchestration -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='lucas',
        help='Experiment name (used to select configs and loader).'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='configs',
        help='Directory where <experiment>_<method>.yaml files live.'
    )
    args = parser.parse_args()

    experiment = args.experiment
    config_dir = args.config_dir

    methods = ['diroca', 'gradca', 'baryca', 'abslingam']
    configs = {}

    print(f"[SETUP] Experiment: {experiment}")
    print(f"[SETUP] Looking for configs in: {config_dir}")

    # Load configs per method (if they exist)
    for m in methods:
        cfg_path = os.path.join(config_dir, f"{experiment}_{m}.yaml")
        cfg = load_yaml_config(cfg_path)
        if cfg is not None:
            if not cfg.get("enabled", True):
                print(f"[CONFIG] {m} config found but 'enabled: false'. Skipping.")
                continue
            configs[m] = cfg

    if not configs:
        raise RuntimeError("No enabled configs found. Nothing to do.")

    # Data cache: avoid re-loading the same pack multiple times
    data_cache = {}

    # Run each method independently
    for method_name, cfg in configs.items():
        print("\n" + "="*70)
        print(f"[{method_name.upper()}] Starting for experiment '{experiment}'")
        print("="*70)

        data_cfg = cfg.get("data", {})
        data_path = data_cfg.get("data_path", None)
        if data_path is None:
            raise KeyError(f"[{method_name}] 'data_path' missing in config 'data' section.")

        exp_in_cfg = data_cfg.get("experiment", experiment)
        if exp_in_cfg != experiment:
            print(f"[WARN] experiment in config ('{exp_in_cfg}') "
                  f"differs from CLI experiment ('{experiment}'). Using CLI value.")

        # Load data (once per unique data_path)
        if data_path not in data_cache:
            print(f"[DATA] Loading pack from: {data_path}")
            all_data = load_pack_for_experiment(experiment, data_path)
            data_cache[data_path] = all_data
        else:
            all_data = data_cache[data_path]

        N = all_data['N']
        l = all_data['l']
        h = all_data['h']

        cv_cfg = cfg.get("cv", {})
        k_folds = cv_cfg.get("k_folds", 5)
        seed = cv_cfg.get("seed", 23)

        print(f"[CV] k_folds={k_folds}, seed={seed}, N={N}")
        folds = prepare_cv_folds_from_N(N, k_folds, seed)
        print(f"[CV] {len(folds)} folds, ~{len(folds[0]['train'])} train / fold")

        out_cfg = cfg.get("output", {})
        save_dir = out_cfg.get("save_directory", os.path.join('data', experiment, 'results_nonlinear'))
        filename_prefix = out_cfg.get("filename_prefix", f"{method_name}_cv_results")
        os.makedirs(save_dir, exist_ok=True)

        # Method-specific blocks
        if method_name == 'diroca':
            opt_cfg = cfg.get("optimization", {})
            radius_pairs = build_radius_pairs(cfg, N, l, h)

            print(f"[DIROCA] radius_pairs: {radius_pairs}")
            diroca_cv = {}
            for fi, fold in enumerate(folds):
                print(f"[DIROCA] Fold {fi+1}/{len(folds)}")
                tr = fold['train']; te = fold['test']
                U_ll_tr = all_data['U_ll_hat'][tr]
                U_hl_tr = all_data['U_hl_hat'][tr]
                det_ll_tr = {k: v[tr] for k, v in all_data['det_ll_dict'].items()}
                det_hl_tr = {k: v[tr] for k, v in all_data['det_hl_dict'].items()}

                diroca_cv[f'fold_{fi}'] = {}
                for (eps, delt) in radius_pairs:
                    print(f"  - eps={eps}, delta={delt}")
                    opt_params, T_learned = run_empirical_minmax(
                        U_L=U_ll_tr, U_H=U_hl_tr,
                        det_ll_dict=det_ll_tr, det_hl_dict=det_hl_tr,
                        omega=all_data['omega'],
                        epsilon=eps, delta=delt,
                        eta_min=opt_cfg.get('eta_min', 1e-3),
                        eta_max=opt_cfg.get('eta_max', 1e-3),
                        num_steps_min=opt_cfg.get('num_steps_min', 5),
                        num_steps_max=opt_cfg.get('num_steps_max', 2),
                        max_iter=opt_cfg.get('max_iter', 5000),
                        tol=opt_cfg.get('tol', 1e-4),
                        seed=seed,
                        robust_L=(eps > 0), robust_H=(delt > 0),
                        initialization=opt_cfg.get('initialization', 'random'),
                        gain=opt_cfg.get('gain', 0.0),
                        optimizers=opt_cfg.get('optimizers', 'adam')
                    )
                    key = f'eps_{eps}_delta_{delt}'
                    diroca_cv[f'fold_{fi}'][key] = {
                        'T_matrix': T_learned,
                        'optimization_params': opt_params,
                        'test_indices': te
                    }

            outp = os.path.join(save_dir, f"{filename_prefix}.pkl")
            joblib.dump(diroca_cv, outp)
            print(f"[DIROCA] saved -> {outp}")

        elif method_name == 'gradca':
            opt_cfg = cfg.get("optimization", {})
            gradca_cv = {}
            for fi, fold in enumerate(folds):
                print(f"[GRADCA] Fold {fi+1}/{len(folds)}")
                tr = fold['train']; te = fold['test']
                U_ll_tr = all_data['U_ll_hat'][tr]
                U_hl_tr = all_data['U_hl_hat'][tr]
                det_ll_tr = {k: v[tr] for k, v in all_data['det_ll_dict'].items()}
                det_hl_tr = {k: v[tr] for k, v in all_data['det_hl_dict'].items()}

                opt_params, T_learned = run_empirical_minmax(
                    U_L=U_ll_tr, U_H=U_hl_tr,
                    det_ll_dict=det_ll_tr, det_hl_dict=det_hl_tr,
                    omega=all_data['omega'],
                    epsilon=0.0, delta=0.0,
                    eta_min=opt_cfg.get('eta_min', 1e-3),
                    eta_max=0.0,
                    num_steps_min=opt_cfg.get('num_steps_min', 1),
                    num_steps_max=0,
                    max_iter=opt_cfg.get('max_iter', 5000),
                    tol=opt_cfg.get('tol', 1e-5),
                    seed=seed,
                    robust_L=False, robust_H=False,
                    initialization=opt_cfg.get('initialization', 'zeros'),
                    gain=opt_cfg.get('gain', 0.0),
                    optimizers=opt_cfg.get('optimizers', 'adam')
                )
                gradca_cv[f'fold_{fi}'] = {'gradca_run': {
                    'T_matrix': T_learned,
                    'optimization_params': opt_params,
                    'test_indices': te
                }}

            outp = os.path.join(save_dir, f"{filename_prefix}.pkl")
            joblib.dump(gradca_cv, outp)
            print(f"[GRADCA] saved -> {outp}")

        elif method_name == 'baryca':
            opt_cfg = cfg.get("optimization", {})
            baryca_cv = {}
            for fi, fold in enumerate(folds):
                print(f"[BARYCA] Fold {fi+1}/{len(folds)}")
                tr = fold['train']; te = fold['test']
                U_ll_tr = all_data['U_ll_hat'][tr]
                U_hl_tr = all_data['U_hl_hat'][tr]
                det_ll_tr = {k: v[tr] for k, v in all_data['det_ll_dict'].items()}
                det_hl_tr = {k: v[tr] for k, v in all_data['det_hl_dict'].items()}

                T_bary = run_bary_optim(
                    U_ll_hat=U_ll_tr, U_hl_hat=U_hl_tr,
                    det_ll_dict=det_ll_tr, det_hl_dict=det_hl_tr,
                    omega=all_data['omega'],
                    lr=opt_cfg.get('lr', 1e-3),
                    max_iter=opt_cfg.get('max_iter', 5000),
                    tol=opt_cfg.get('tol', 1e-5),
                    seed=seed
                )
                baryca_cv[f'fold_{fi}'] = {'baryca_run': {
                    'T_matrix': T_bary,
                    'test_indices': te
                }}

            outp = os.path.join(save_dir, f"{filename_prefix}.pkl")
            joblib.dump(baryca_cv, outp)
            print(f"[BARYCA] saved -> {outp}")

        elif method_name == 'abslingam':
            opt_cfg = cfg.get("optimization", {})
            tau_perfect = opt_cfg.get('tau_perfect', 1e-2)
            tau_noisy   = opt_cfg.get('tau_noisy', 1e-1)

            # reconstruct observational X from pack: X = D + U at obs
            pack = joblib.load(data_path)
            iota_obs = 'iota0'
            eta_obs  = pack['omega'].get('iota0', 'eta0') if 'omega' in pack else 'eta0'
            if iota_obs not in pack['ll']:
                raise KeyError("Missing ll['iota0'] to build Abs-LiNGAM inputs.")
            if eta_obs not in pack['hl']:
                raise KeyError(f"Missing hl['{eta_obs}'] to build Abs-LiNGAM inputs.")

            X_ll_obs = pack['ll'][iota_obs]['X']  # (N, d_l)
            X_hl_obs = pack['hl'][eta_obs]['X']   # (N, d_h)

            abs_cv = {}
            for fi, fold in enumerate(folds):
                print(f"[ABSLINGAM] Fold {fi+1}/{len(folds)}")
                tr = fold['train']; te = fold['test']
                res = run_abs_lingam(
                    X_ll_obs[tr], X_hl_obs[tr],
                    tau_perfect=tau_perfect,
                    tau_noisy=tau_noisy
                )
                abs_cv[f'fold_{fi}'] = {
                    'Perfect': {'T_matrix': res['Perfect']['T'], 'test_indices': te},
                    'Noisy':   {'T_matrix': res['Noisy']['T'],   'test_indices': te}
                }

            outp = os.path.join(save_dir, f"{filename_prefix}.pkl")
            joblib.dump(abs_cv, outp)
            print(f"[ABSLINGAM] saved -> {outp}")

        else:
            print(f"[WARN] Unknown method '{method_name}' - skipping.")

    # --------- Summary ----------
    print("\n" + "="*60)
    print("NON-LINEAR OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"Experiment: {experiment}")
    for m, cfg in configs.items():
        out_cfg = cfg.get("output", {})
        save_dir = out_cfg.get("save_directory", os.path.join('data', experiment, 'results_nonlinear'))
        filename_prefix = out_cfg.get("filename_prefix", f"{m}_cv_results")
        print(f"  - {m}: {os.path.join(save_dir, filename_prefix + '.pkl')}")
    print("="*60)


if __name__ == '__main__':
    main()
