#!/usr/bin/env python3
"""
Non-linear optimization for LUCAS-style packs
---------------------------------------------
Works with the output of lucas_nonlinear_generator.py.

Input pack (joblib):
  {
    'T': (3x6) np.ndarray,
    'omega': dict[str->str],
    'll': { iota: {'X','D','U'} with arrays (N,6) },
    'hl': { eta: {'X','D','U'} with arrays (N,3) },
    'hl_model': {...},
    'graphs': {...}
  }

This script:
  * Loads pack, builds:
      - det_ll_dict: iota -> D_ll (N,6)
      - det_hl_dict: eta  -> D_hl (N,3)
      - U_ll_hat := ll['iota0']['U'] (N,6)
      - U_hl_hat := hl['eta0']['U']  (N,3)
      - omega mapping
      - K-fold splits over N
  * Runs:
      - DiRoCA (empirical min-max in Frobenius balls)
      - GradCA (empirical min-only, Theta=Phi=0)
      - BaryCA (empirical barycentric averaging over interventions)
      - Abs-LiNGAM baselines (Perfect / Noisy) on observational X
  * Saves results as joblib pickles into out_dir.

Usage:
  python non_linear_optimization.py \
    --data-path data/lucas/lucas_pack.pkl \
    --experiment lucas \
    --k-folds 5 \
    --seed 23
"""

import os
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


# ----------------------------- I/O & setup ------------------------------

def load_lucas_pack(path):
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

    # Build det dicts
    det_ll_dict = {iota: v['D'] for iota, v in pack['ll'].items()}
    det_hl_dict = {eta: v['D'] for eta, v in pack['hl'].items()}

    # Noise hats from observational splits
    U_ll_hat = pack['ll']['iota0']['U']        # (N,6)
    U_hl_hat = pack['hl'][eta_obs]['U']        # (N,3)

    # Shapes sanity
    N, l = U_ll_hat.shape
    _, h = U_hl_hat.shape
    if l != 6 or h != 3:
        print(f"[WARN] Expected LL=6, HL=3; got l={l}, h={h}. Proceeding anyway.")

    omega = pack['omega']  # LL iota -> HL eta

    # Keep T from pack (for reference only)
    T_ref = pack['T']  # shape (3,6)

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
    term1 = c1 * (np.log(max(N,2)) / N) ** (1 / alpha)
    term2 = c2 * np.sqrt(m * np.log(1 / eta) / N)
    return float(term1 + term2)


# ----------------------------- objectives --------------------------------

def empirical_objective_lucas(T, U_ll, U_hl, Theta_ll, Phi_hl,
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
        Dll = torch.as_tensor(det_ll_dict[iota], dtype=torch.float32, device=device)  # (N,6)
        Dhl = torch.as_tensor(det_hl_dict[eta],  dtype=torch.float32, device=device)  # (N,3)

        endo_ll = Dll + U_ll + Theta_ll
        endo_hl = Dhl + U_hl + Phi_hl

        diff = (T @ endo_ll.T).T - endo_hl
        loss_total += torch.norm(diff, p='fro')**2 / max(1, N)
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=device)
    return loss_total / count


def barycentric_objective_lucas(T, U_ll, U_hl, det_ll_dict, det_hl_dict, omega):
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

    avg_Dll = torch.mean(torch.stack(ll_list, dim=0), dim=0)  # (N,6)
    avg_Dhl = torch.mean(torch.stack(hl_list, dim=0), dim=0)  # (N,3)

    endo_ll = avg_Dll + U_ll
    endo_hl = avg_Dhl + U_hl

    N = endo_ll.shape[0]
    diff = (T @ endo_ll.T).T - endo_hl
    return torch.norm(diff, p='fro')**2 / max(1, N)


# ----------------------------- Abs-LiNGAM (baseline) --------------------

def perfect_abstraction(px_samples, py_samples, tau_threshold=1e-2):
    T_raw = np.linalg.pinv(px_samples) @ py_samples  # (6,3)
    mask = (np.abs(T_raw) > tau_threshold).astype(px_samples.dtype)
    return T_raw * mask


def noisy_abstraction(px_samples, py_samples, tau_threshold=1e-1, refit_coeff=False):
    T_hat = np.linalg.pinv(px_samples) @ py_samples  # (6,3)
    idx = np.argmax(np.abs(T_hat), axis=1)           # (6,)
    onehot = np.eye(py_samples.shape[1])[idx]        # (6,3)
    onehot *= (np.abs(T_hat) > tau_threshold).astype(int)
    T_masked = onehot * T_hat
    # optional refit (usually not needed for 6->3 small case)
    if refit_coeff:
        T_refit = T_masked.copy()
        for y in range(onehot.shape[1]):
            block = np.where(onehot[:, y] == 1)[0]
            if len(block) > 0:
                T_block = np.linalg.pinv(px_samples[:, block]) @ py_samples[:, y]
                T_refit[block, y] = T_block
        return T_refit
    return T_masked


def run_abs_lingam_lucas(X_ll_obs, X_hl_obs, tau_perfect=1e-2, tau_noisy=1e-1):
    # Shapes: (N,6) and (N,3)
    T_perfect = perfect_abstraction(X_ll_obs, X_hl_obs, tau_threshold=tau_perfect).T.astype(np.float32)  # (3,6)
    T_noisy   = noisy_abstraction(X_ll_obs,   X_hl_obs, tau_threshold=tau_noisy, refit_coeff=False).T.astype(np.float32)
    return {
        'Perfect': {'T': T_perfect},
        'Noisy':   {'T': T_noisy}
    }


# ----------------------------- Core trainers -----------------------------

def run_empirical_erica_optimization_lucas(U_L, U_H, det_ll_dict, det_hl_dict, omega,
                                           epsilon, delta,
                                           eta_min, eta_max,
                                           num_steps_min, num_steps_max,
                                           max_iter, tol, seed,
                                           robust_L, robust_H,
                                           initialization, gain, optimizers):
    """
    Core loop for DiRoCA (robust) / GradCA (non-robust) on lucas pack.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # T dimensions: map 6->3 (so T is (3,6))
    l_full, h_full = U_L.shape[1], U_H.shape[1]
    T = torch.randn(h_full, l_full, requires_grad=True, device=device)
    if gain > 0:
        init.xavier_normal_(T, gain=gain)

    U_L = torch.as_tensor(U_L, dtype=torch.float32, device=device)
    U_H = torch.as_tensor(U_H, dtype=torch.float32, device=device)

    method = 'erica' if (robust_L or robust_H) else 'enrico'

    # perturbations
    if initialization == 'zeros':
        Theta = torch.zeros_like(U_L, requires_grad=(method == 'erica'), device=device)
        Phi   = torch.zeros_like(U_H, requires_grad=(method == 'erica'), device=device)
    elif initialization == 'random':
        Theta = torch.randn_like(U_L, requires_grad=(method == 'erica'), device=device)
        Phi   = torch.randn_like(U_H, requires_grad=(method == 'erica'), device=device)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")

    # optimizers
    if optimizers == 'adam':
        opt_T = optim.Adam([T], lr=eta_min)
        opt_max = optim.Adam([Theta, Phi], lr=eta_max) if method == 'erica' else None
    elif optimizers == 'adam_betas':
        opt_T = optim.Adam([T], lr=eta_min, betas=(0.9,0.999), eps=1e-8, amsgrad=True)
        opt_max = optim.Adam([Theta, Phi], lr=eta_max, betas=(0.9,0.999), eps=1e-8, amsgrad=True) if method == 'erica' else None
    else:
        raise ValueError(f"Unknown optimizer: {optimizers}")

    prev_T_obj = float('inf')
    N = U_L.shape[0]

    for it in tqdm(range(max_iter), desc=f"{method.upper()} Optimization"):
        # Min step(s) on T (Θ,Φ detached)
        for _ in range(num_steps_min):
            opt_T.zero_grad()
            T_obj = empirical_objective_lucas(T, U_L, U_H, Theta.detach(), Phi.detach(),
                                              det_ll_dict, det_hl_dict, omega)
            if torch.isnan(T_obj):
                break
            T_obj.backward()
            opt_T.step()

        # Max step(s) on Theta,Phi if robust
        if method == 'erica':
            for _ in range(num_steps_max):
                opt_max.zero_grad()
                max_obj = -empirical_objective_lucas(T.detach(), U_L, U_H, Theta, Phi,
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
            print(f"Converged at iter {it+1} (Δobj<{tol}).")
            break
        prev_T_obj = cur

    T_final = T.detach().cpu().numpy().astype(np.float32)
    paramsL = {'pert_U': Theta.detach().cpu().numpy().astype(np.float32),
               'radius': epsilon}
    paramsH = {'pert_U': Phi.detach().cpu().numpy().astype(np.float32),
               'radius': delta}
    return {'L': paramsL, 'H': paramsH}, T_final


def run_empirical_bary_optim_lucas(U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega,
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
        loss = barycentric_objective_lucas(T, U_ll_hat, U_hl_hat, det_ll_dict, det_hl_dict, omega)
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


# ----------------------------- Orchestration -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', type=str, required=True, help='Path to lucas_pack.pkl')
    ap.add_argument('--experiment', type=str, default='lucas')
    ap.add_argument('--out-dir', type=str, default=None)
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=23)

    # DiRoCA params
    ap.add_argument('--diroca-max-iter', type=int, default=5000)
    ap.add_argument('--diroca-tol', type=float, default=1e-4)
    ap.add_argument('--diroca-eta-min', type=float, default=1e-3)
    ap.add_argument('--diroca-eta-max', type=float, default=1e-3)
    ap.add_argument('--diroca-steps-min', type=int, default=5)
    ap.add_argument('--diroca-steps-max', type=int, default=2)
    ap.add_argument('--diroca-init', type=str, choices=['zeros','random'], default='random')
    ap.add_argument('--diroca-optim', type=str, choices=['adam','adam_betas'], default='adam')
    ap.add_argument('--diroca-gain', type=float, default=0.0)

    # GradCA params (reuses DiRoCA core with eps=delta=0, and no max step)
    ap.add_argument('--gradca-max-iter', type=int, default=5000)
    ap.add_argument('--gradca-tol', type=float, default=1e-5)
    ap.add_argument('--gradca-eta-min', type=float, default=1e-3)
    ap.add_argument('--gradca-steps-min', type=int, default=1)
    ap.add_argument('--gradca-init', type=str, choices=['zeros','random'], default='zeros')
    ap.add_argument('--gradca-optim', type=str, choices=['adam','adam_betas'], default='adam')
    ap.add_argument('--gradca-gain', type=float, default=0.0)

    # BaryCA params
    ap.add_argument('--baryca-max-iter', type=int, default=5000)
    ap.add_argument('--baryca-tol', type=float, default=1e-5)

    # switches
    ap.add_argument('--skip-diroca', action='store_true')
    ap.add_argument('--skip-gradca', action='store_true')
    ap.add_argument('--skip-baryca', action='store_true')
    ap.add_argument('--skip-abslingam', action='store_true')

    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join('data', args.experiment, 'results_nonlinear')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[SETUP] loading data: {args.data_path}")
    all_data = load_lucas_pack(args.data_path)
    N = all_data['N']
    l, h = all_data['l'], all_data['h']

    print("[SETUP] building CV folds...")
    folds = prepare_cv_folds_from_N(N, args.k_folds, args.seed)
    print(f"[SETUP] {len(folds)} folds, ~{len(folds[0]['train'])} train / fold")

    # Precompute radius baseline (dim-aware)
    base_eps = round(compute_empirical_radius(N, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=l), 3)
    base_del = round(compute_empirical_radius(N, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=h), 3)
    radius_pairs = [(base_eps, base_del), (1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]

    results = {}

    # --------- DiRoCA (empirical) ----------
    if not args.skip_diroca:
        print("[DIROCA] start")
        diroca_cv = {}
        for fi, fold in enumerate(folds):
            print(f"[DIROCA] Fold {fi+1}/{len(folds)}")
            tr = fold['train']; te = fold['test']
            U_ll_tr = all_data['U_ll_hat'][tr]
            U_hl_tr = all_data['U_hl_hat'][tr]
            # slice det dicts to train indices
            det_ll_tr = {k: v[tr] for k, v in all_data['det_ll_dict'].items()}
            det_hl_tr = {k: v[tr] for k, v in all_data['det_hl_dict'].items()}

            diroca_cv[f'fold_{fi}'] = {}
            for (eps, delt) in radius_pairs:
                print(f"  - eps={eps}, delta={delt}")
                opt_params, T_learned = run_empirical_erica_optimization_lucas(
                    U_L=U_ll_tr, U_H=U_hl_tr,
                    det_ll_dict=det_ll_tr, det_hl_dict=det_hl_tr,
                    omega=all_data['omega'],
                    epsilon=eps, delta=delt,
                    eta_min=args.diroca_eta_min, eta_max=args.diroca_eta_max,
                    num_steps_min=args.diroca_steps_min, num_steps_max=args.diroca_steps_max,
                    max_iter=args.diroca_max_iter, tol=args.diroca_tol, seed=args.seed,
                    robust_L=(eps>0), robust_H=(delt>0),
                    initialization=args.diroca_init, gain=args.diroca_gain,
                    optimizers=args.diroca_optim
                )
                key = f'eps_{eps}_delta_{delt}'
                diroca_cv[f'fold_{fi}'][key] = {
                    'T_matrix': T_learned,
                    'optimization_params': opt_params,
                    'test_indices': te
                }
        outp = os.path.join(args.out_dir, "diroca_cv_results_empirical.pkl")
        joblib.dump(diroca_cv, outp)
        print(f"[DIROCA] saved -> {outp}")
        results['diroca'] = outp

    # --------- GradCA (empirical) ----------
    if not args.skip_gradca:
        print("[GRADCA] start")
        gradca_cv = {}
        for fi, fold in enumerate(folds):
            print(f"[GRADCA] Fold {fi+1}/{len(folds)}")
            tr = fold['train']; te = fold['test']
            U_ll_tr = all_data['U_ll_hat'][tr]
            U_hl_tr = all_data['U_hl_hat'][tr]
            det_ll_tr = {k: v[tr] for k, v in all_data['det_ll_dict'].items()}
            det_hl_tr = {k: v[tr] for k, v in all_data['det_hl_dict'].items()}

            opt_params, T_learned = run_empirical_erica_optimization_lucas(
                U_L=U_ll_tr, U_H=U_hl_tr,
                det_ll_dict=det_ll_tr, det_hl_dict=det_hl_tr,
                omega=all_data['omega'],
                epsilon=0.0, delta=0.0,
                eta_min=args.gradca_eta_min, eta_max=0.0,
                num_steps_min=args.gradca_steps_min, num_steps_max=0,
                max_iter=args.gradca_max_iter, tol=args.gradca_tol, seed=args.seed,
                robust_L=False, robust_H=False,
                initialization=args.gradca_init, gain=args.gradca_gain,
                optimizers=args.gradca_optim
            )
            gradca_cv[f'fold_{fi}'] = {'gradca_run': {
                'T_matrix': T_learned,
                'optimization_params': opt_params,
                'test_indices': te
            }}
        outp = os.path.join(args.out_dir, "gradca_cv_results_empirical.pkl")
        joblib.dump(gradca_cv, outp)
        print(f"[GRADCA] saved -> {outp}")
        results['gradca'] = outp

    # --------- BaryCA (empirical) ----------
    if not args.skip_baryca:
        print("[BARYCA] start")
        baryca_cv = {}
        for fi, fold in enumerate(folds):
            print(f"[BARYCA] Fold {fi+1}/{len(folds)}")
            tr = fold['train']; te = fold['test']
            U_ll_tr = all_data['U_ll_hat'][tr]
            U_hl_tr = all_data['U_hl_hat'][tr]
            det_ll_tr = {k: v[tr] for k, v in all_data['det_ll_dict'].items()}
            det_hl_tr = {k: v[tr] for k, v in all_data['det_hl_dict'].items()}

            T_bary = run_empirical_bary_optim_lucas(
                U_ll_hat=U_ll_tr, U_hl_hat=U_hl_tr,
                det_ll_dict=det_ll_tr, det_hl_dict=det_hl_tr,
                omega=all_data['omega'],
                lr=1e-3, max_iter=args.baryca_max_iter, tol=args.baryca_tol, seed=args.seed
            )
            baryca_cv[f'fold_{fi}'] = {'baryca_run': {
                'T_matrix': T_bary,
                'test_indices': te
            }}
        outp = os.path.join(args.out_dir, "baryca_cv_results_empirical.pkl")
        joblib.dump(baryca_cv, outp)
        print(f"[BARYCA] saved -> {outp}")
        results['baryca'] = outp

    # --------- Abs-LiNGAM baseline ----------
    if not args.skip_abslingam:
        print("[ABSLINGAM] start")
        # observational X
        # We can reconstruct X from pack: X = D + U at obs.
        # Use iota0 and corresponding eta_obs
        pack = joblib.load(args.data_path)
        iota_obs = 'iota0'
        eta_obs  = pack['omega'].get('iota0', 'eta0') if 'omega' in pack else 'eta0'
        if iota_obs not in pack['ll']:
            raise KeyError("Missing ll['iota0'] to build Abs-LiNGAM inputs.")
        if eta_obs not in pack['hl']:
            raise KeyError(f"Missing hl['{eta_obs}'] to build Abs-LiNGAM inputs.")

        X_ll_obs = pack['ll'][iota_obs]['X']  # (N,6)
        X_hl_obs = pack['hl'][eta_obs]['X']   # (N,3)

        abslingam = {}
        for fi, fold in enumerate(folds):
            te = fold['test']
            # Fit on train portion of observational data for fairness
            tr = fold['train']
            res = run_abs_lingam_lucas(X_ll_obs[tr], X_hl_obs[tr],
                                       tau_perfect=1e-2, tau_noisy=1e-1)
            abslingam[f'fold_{fi}'] = {
                'Perfect': {'T_matrix': res['Perfect']['T'], 'test_indices': te},
                'Noisy':   {'T_matrix': res['Noisy']['T'],   'test_indices': te}
            }
        outp = os.path.join(args.out_dir, "abslingam_cv_results_empirical.pkl")
        joblib.dump(abslingam, outp)
        print(f"[ABSLINGAM] saved -> {outp}")
        results['abslingam'] = outp

    # --------- Summary ----------
    print("\n" + "="*60)
    print("NON-LINEAR OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Output dir: {args.out_dir}")
    for k, v in results.items():
        print(f"  - {k}: {v}")
    print("="*60)


if __name__ == '__main__':
    main()
