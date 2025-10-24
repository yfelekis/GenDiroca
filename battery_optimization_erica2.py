#!/usr/bin/env python3
"""
Battery Empirical Optimization (Affine ANM, ERICA-style)
- Per-intervention buckets with deterministic + noise
- Robustness via Frobenius-ball radii (epsilon, delta)
- Sweeps multiple (epsilon, delta) pairs including empirical lower bounds
- Repeated 90/10 splits used only as different seeds/runs
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any, Optional, List

import joblib
import numpy as np
import torch
import torch.nn.init as init
from tqdm import tqdm

import utilities as ut        # must provide: load_all_data, compute_empirical_radius
import opt_tools as optools   # Abs-LiNGAM helper (abs_lingam_reconstruction_v2)

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("battery_erica_affine")


# ----------------- Helpers -----------------
def project_onto_frobenius_ball(X: torch.Tensor, radius_per_sample: float) -> torch.Tensor:
    """
    Project onto ||X||_F <= sqrt(N)*radius_per_sample  (empirical ERICA ball).
    """
    if X.numel() == 0:
        return X
    N = X.shape[0]
    bound = (N ** 0.5) * radius_per_sample
    fro = torch.norm(X, p='fro')
    if fro > 0 and fro > bound:
        X = X * (bound / fro)
    return X


def make_stratified_9010_splits(Dhl_obs: np.ndarray, test_size: float, seeds: List[int]) -> List[dict]:
    """
    Build 'folds' dicts with 'train' and 'test' indices.
    Here, we mainly use seeds for repeated runs; we do not sub-index buckets.
    Stratify by HL CG (assumed to be column 0 of Dhl_obs).
    """
    assert 0.0 < test_size < 1.0, "test_size must be in (0,1)"
    N = Dhl_obs.shape[0]
    y = Dhl_obs[:, 0]  # CG values

    label_to_indices = {}
    for idx, lab in enumerate(y):
        label_to_indices.setdefault(lab, []).append(idx)

    saved_folds = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        test_idx = []
        for _, idxs in label_to_indices.items():
            n = len(idxs)
            n_test = max(1, int(round(test_size * n)))
            chosen = rng.choice(idxs, size=n_test, replace=False)
            test_idx.append(chosen)
        test_idx = np.unique(np.concatenate(test_idx))
        train_idx = np.setdiff1d(np.arange(N), test_idx)
        saved_folds.append({"train": train_idx, "test": test_idx, "seed": seed})
    return saved_folds


# ----------------- Core affine objective (per intervention) -----------------
def affine_empirical_objective_per_bucket(
    det_ll: Dict[Any, np.ndarray],
    det_hl: Dict[Any, np.ndarray],
    noise_ll: Dict[Any, np.ndarray],
    noise_hl: Dict[Any, np.ndarray],
    omega: Dict[Any, Any],
    T: torch.Tensor,
    Theta_dict: Dict[Any, torch.Tensor],
    Phi_dict: Dict[Any, torch.Tensor],
) -> torch.Tensor:
    """
    Loss = average over iota!=None of
           (1/(d_h*N_i)) * || T (D_l(i) + U_l(i) + Θ_i)ᵀ  - (D_h(ω(i)) + U_h(ω(i)) + Φ_{ω(i)})ᵀ ||_F^2
    """
    losses = []
    for iota, D_l_np in det_ll.items():
        if iota is None:
            continue
        eta = omega.get(iota, None)
        if eta is None:
            continue
        D_h_np = det_hl.get(eta, None)
        if D_h_np is None:
            continue

        U_l_np = noise_ll.get(iota, None)
        U_h_np = noise_hl.get(eta, None)
        if U_l_np is None or U_h_np is None:
            continue

        D_l = torch.from_numpy(D_l_np).float()
        D_h = torch.from_numpy(D_h_np).float()
        U_l = torch.from_numpy(U_l_np).float()
        U_h = torch.from_numpy(U_h_np).float()

        N_l = min(D_l.shape[0], U_l.shape[0])
        N_h = min(D_h.shape[0], U_h.shape[0])
        N_min = min(N_l, N_h)
        if N_min == 0:
            continue

        Theta_i = Theta_dict[iota][:N_min]
        Phi_i   = Phi_dict[eta][:N_min]

        L_part = D_l[:N_min] + U_l[:N_min] + Theta_i   # (N_min, d_l)
        H_part = D_h[:N_min] + U_h[:N_min] + Phi_i     # (N_min, d_h)

        diff = T @ L_part.T - H_part.T                 # (d_h, N_min)
        losses.append(torch.norm(diff, p='fro')**2 / (diff.shape[0] * diff.shape[1]))

    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    return sum(losses) / len(losses)


def run_one_erica_affine(
    all_data: Dict[str, Any],
    saved_folds: List[dict],
    epsilon: float,
    delta: float,
    eta_min: float = 1e-3,
    eta_max: float = 1e-3,
    num_steps_min: int = 5,
    num_steps_max: int = 2,
    max_iter: int = 5000,
    tol: float = 1e-4,
    seed: int = 23,
    initialization: str = "zeros",
    gain: float = 0.0,
    optimizers: str = "adam",
) -> Dict[str, Any]:
    """
    Run ERICA-style min–max for a *single* (epsilon, delta) pair,
    using per-intervention buckets and per-bucket perturbations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    det_ll = all_data["LLmodel"]["deterministic"]   # {iota: (N_i, d_l)}
    det_hl = all_data["HLmodel"]["deterministic"]   # {eta:  (M_i, d_h)}
    noise_ll_all = all_data["LLmodel"]["noise"]     # contains buckets + 'U_ll_hat'
    noise_hl_all = all_data["HLmodel"]["noise"]     # contains buckets + 'U_hl_hat'
    omega = all_data["abstraction_data"]["omega"]

    # Keep true buckets only (avoid 'U_ll_hat'/'U_hl_hat' extras)
    def is_bucket(k): return (k is not None) and hasattr(k, "vv")
    noise_ll = {k: v for k, v in noise_ll_all.items() if is_bucket(k)}
    noise_hl = {k: v for k, v in noise_hl_all.items() if is_bucket(k)}

    ll_keys = [k for k in det_ll.keys() if is_bucket(k)]
    hl_keys = [k for k in det_hl.keys() if is_bucket(k)]

    # Dimensions from any det bucket
    d_l = next(iter(det_ll.values())).shape[1]
    d_h = next(iter(det_hl.values())).shape[1]

    results = {}

    # init per-bucket Θ/Φ
    def init_theta_phi(init_mode: str):
        Theta_dict: Dict[Any, torch.Tensor] = {}
        Phi_dict: Dict[Any, torch.Tensor] = {}
        for iota in ll_keys:
            N_i = det_ll[iota].shape[0]
            Theta_dict[iota] = (torch.zeros if init_mode == "zeros" else torch.randn)(N_i, d_l, requires_grad=True)
        for eta in hl_keys:
            M_i = det_hl[eta].shape[0]
            Phi_dict[eta] = (torch.zeros if init_mode == "zeros" else torch.randn)(M_i, d_h, requires_grad=True)
        return Theta_dict, Phi_dict

    for fold_id, fold in enumerate(saved_folds):
        seed_fold = int(fold.get("seed", seed))
        torch.manual_seed(seed_fold)
        np.random.seed(seed_fold)

        T = torch.randn(d_h, d_l, requires_grad=True)
        if gain > 0:
            init.xavier_normal_(T, gain=gain)

        Theta_dict, Phi_dict = init_theta_phi(initialization)

        if optimizers == "adam":
            optimizer_T = torch.optim.Adam([T], lr=eta_min)
            optimizer_max = torch.optim.Adam(list(Theta_dict.values()) + list(Phi_dict.values()), lr=eta_max)
        elif optimizers == "adam_betas":
            optimizer_T = torch.optim.Adam([T], lr=eta_min, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
            optimizer_max = torch.optim.Adam(list(Theta_dict.values()) + list(Phi_dict.values()),
                                             lr=eta_max, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizers}")

        prev_obj = float("inf")

        for it in tqdm(range(max_iter), desc=f"(ε={epsilon}, δ={delta}) | Fold {fold_id+1}/{len(saved_folds)}"):
            # ----- Min over T -----
            for _ in range(num_steps_min):
                optimizer_T.zero_grad()
                obj_T = affine_empirical_objective_per_bucket(
                    det_ll, det_hl, noise_ll, noise_hl, omega, T, Theta_dict, Phi_dict
                )
                obj_T.backward()
                optimizer_T.step()

            # ----- Max over {Θ_i}, {Φ_η} -----
            for _ in range(num_steps_max):
                optimizer_max.zero_grad()
                obj_max = -affine_empirical_objective_per_bucket(
                    det_ll, det_hl, noise_ll, noise_hl, omega, T, Theta_dict, Phi_dict
                )
                obj_max.backward()
                optimizer_max.step()

                # Project each bucket’s Θ/Φ
                with torch.no_grad():
                    for iota in ll_keys:
                        Theta_dict[iota].data = project_onto_frobenius_ball(Theta_dict[iota].data, epsilon)
                    for eta in hl_keys:
                        Phi_dict[eta].data = project_onto_frobenius_ball(Phi_dict[eta].data, delta)

            # Convergence check
            with torch.no_grad():
                cur = obj_T.item()
                if abs(prev_obj - cur) < tol:
                    print(f"Converged at iter {it+1} (Δ={abs(prev_obj-cur):.2e})")
                    break
                prev_obj = cur

        results[f"fold_{fold_id}"] = {
            "seed": seed_fold,
            "T": T.detach().numpy(),
            "train_N": len(fold["train"]),
            "test_N": len(fold["test"]),
            "epsilon": float(epsilon),
            "delta": float(delta),
        }

    return results


def run_diroca_radius_sweep(
    all_data: Dict[str, Any],
    saved_folds: List[dict],
    radius_pairs: List[tuple],
    outdir: str
) -> Dict[str, Any]:
    """
    Sweep multiple (epsilon, delta) pairs and save per-pair results.
    """
    os.makedirs(outdir, exist_ok=True)
    all_runs = {}
    for (eps, delt) in radius_pairs:
        logger.info(f"Running DiRoCA affine for ε={eps}, δ={delt}")
        res = run_one_erica_affine(
            all_data, saved_folds,
            epsilon=eps, delta=delt,
            eta_min=1e-3, eta_max=1e-3,
            num_steps_min=5, num_steps_max=2,
            max_iter=5000, tol=1e-4,
            seed=saved_folds[0].get("seed", 23),
            initialization="zeros", gain=0.0, optimizers="adam"
        )
        key = f"epsilon_{eps}_delta_{delt}"
        all_runs[key] = res

    joblib.dump(all_runs, os.path.join(outdir, "diroca_cv_results_empirical.pkl"))
    logger.info(f"Saved DiRoCA sweep to {outdir}")
    return all_runs

def run_gradca_affine(
    all_data: Dict[str, Any],
    saved_folds: List[dict],
    outdir: str,
    eta_min: float = 1e-3,
    num_steps_min: int = 1,
    max_iter: int = 2000,
    tol: float = 1e-4,
    seed: int = 23,
    initialization: str = "zeros",
    gain: float = 0.0,
    optimizers: str = "adam",
) -> Dict[str, Any]:
    """
    GRADCA = ERICA without robustness (no adversarial max step).
    Minimizes the same per-bucket affine objective with Θ=Φ=0 (fixed).
    Runs across the same 90/10 folds (seeds).
    """
    torch.manual_seed(seed); np.random.seed(seed)

    det_ll = all_data["LLmodel"]["deterministic"]
    det_hl = all_data["HLmodel"]["deterministic"]
    noise_ll_all = all_data["LLmodel"]["noise"]
    noise_hl_all = all_data["HLmodel"]["noise"]
    omega = all_data["abstraction_data"]["omega"]

    def is_bucket(k): return (k is not None) and hasattr(k, "vv")
    noise_ll = {k: v for k, v in noise_ll_all.items() if is_bucket(k)}
    noise_hl = {k: v for k, v in noise_hl_all.items() if is_bucket(k)}

    d_l = next(iter(det_ll.values())).shape[1]
    d_h = next(iter(det_hl.values())).shape[1]

    results = {}

    # Fixed zero perturbations (no max step)
    Theta0 = {k: torch.zeros(det_ll[k].shape[0], d_l, requires_grad=False) for k in det_ll if is_bucket(k)}
    Phi0   = {k: torch.zeros(det_hl[k].shape[0], d_h, requires_grad=False) for k in det_hl if is_bucket(k)}

    for fold_id, fold in enumerate(saved_folds):
        seed_fold = int(fold.get("seed", seed))
        torch.manual_seed(seed_fold); np.random.seed(seed_fold)

        T = torch.randn(d_h, d_l, requires_grad=True)
        if gain > 0:
            init.xavier_normal_(T, gain=gain)

        if optimizers == "adam":
            optimizer_T = torch.optim.Adam([T], lr=eta_min)
        elif optimizers == "adam_betas":
            optimizer_T = torch.optim.Adam([T], lr=eta_min, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizers}")

        prev_obj = float("inf")
        for it in tqdm(range(max_iter), desc=f"GRADCA | Fold {fold_id+1}/{len(saved_folds)}"):
            # ----- Min over T only (Θ,Φ fixed zeros) -----
            for _ in range(num_steps_min):
                optimizer_T.zero_grad()
                obj_T = affine_empirical_objective_per_bucket(
                    det_ll, det_hl, noise_ll, noise_hl, omega, T, Theta0, Phi0
                )
                obj_T.backward()
                optimizer_T.step()

            with torch.no_grad():
                cur = obj_T.item()
                if abs(prev_obj - cur) < tol:
                    break
                prev_obj = cur

        results[f"fold_{fold_id}"] = {
            "seed": seed_fold,
            "T": T.detach().numpy(),
            "train_N": len(fold["train"]),
            "test_N": len(fold["test"]),
        }

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(results, os.path.join(outdir, "gradca_cv_results_empirical.pkl"))
    logger.info(f"GRADCA results saved to {outdir}")
    return results

# def run_baryca_empirical(all_data, saved_folds, outdir):
#     """
#     BARYCA (empirical) per-fold baseline:
#       1) Compute barycentric mixing matrices A_L^bary, A_H^bary by averaging
#          the structural (mixing) matrices across interventions.
#       2) For each fold, take TRAIN indices of the pooled exogenous noises U^ℓ, U^h.
#       3) Form barycentric endogenous samples:
#              X_L^bary = A_L^bary @ U^ℓ_train.T
#              X_H^bary = A_H^bary @ U^h_train.T
#       4) Fit T by least squares:  T = X_H^bary @ pinv(X_L^bary)
#     Saves: {fold_k: {"seed", "T", "train_N", "test_N"}} → baryca_cv_results_empirical.pkl
#     """
#     import os, joblib, numpy as np
#     import logging
#     logger = logging.getLogger("battery_erica_affine")

#     # 1) Structural matrices per intervention
#     LLmodels = all_data["LLmodel"].get("scm_instances")
#     HLmodels = all_data["HLmodel"].get("scm_instances")
#     Ill      = all_data["LLmodel"]["intervention_set"]
#     Ihl      = all_data["HLmodel"]["intervention_set"]

#     # Expect ut.compute_struc_matrices(models, I) → list of torch/numpy matrices (mixing)
#     L_mats = ut.compute_struc_matrices(LLmodels, Ill)
#     H_mats = ut.compute_struc_matrices(HLmodels, Ihl)

#     # Convert to numpy and average (barycentric mixing)
#     def _to_np_stack(mats):
#         arrs = []
#         for M in mats:
#             if hasattr(M, "detach"):  # torch tensor
#                 arrs.append(M.detach().cpu().numpy())
#             elif hasattr(M, "numpy"):  # numpy-like
#                 arrs.append(np.asarray(M))
#             else:
#                 arrs.append(np.array(M))
#         return np.stack(arrs, axis=0)

#     L_stack = _to_np_stack(L_mats)     # shape: (#interventions, d_l, d_l)
#     H_stack = _to_np_stack(H_mats)     # shape: (#interventions, d_h, d_h)
#     A_L_bary = L_stack.mean(axis=0)    # (d_l, d_l)
#     A_H_bary = H_stack.mean(axis=0)    # (d_h, d_h)

#     # 2) Pooled exogenous noises (aligned with train/test indices)
#     U_ll_pool = all_data["LLmodel"]["noise"]["U_ll_hat"]  # (N, d_l)
#     U_hl_pool = all_data["HLmodel"]["noise"]["U_hl_hat"]  # (N, d_h)

#     results = {}
#     for k, fold in enumerate(saved_folds):
#         train_idx = fold["train"]
#         test_idx  = fold["test"]

#         U_ll_train = U_ll_pool[train_idx]   # (N_train, d_l)
#         U_hl_train = U_hl_pool[train_idx]   # (N_train, d_h)

#         # 3) Barycentric endogenous samples
#         # Shapes: A_* (d_*, d_*), U_*_train.T (d_*, N_train) → X_* (d_*, N_train)
#         X_L_bary = A_L_bary @ U_ll_train.T
#         X_H_bary = A_H_bary @ U_hl_train.T

#         # 4) Closed-form least squares on TRAIN split
#         try:
#             Tmat = X_H_bary @ np.linalg.pinv(X_L_bary)
#         except np.linalg.LinAlgError:
#             logger.warning(f"[BARYCA] Pseudoinverse failed on fold {k}; using zeros.")
#             Tmat = np.zeros((X_H_bary.shape[0], X_L_bary.shape[0]))

#         results[f"fold_{k}"] = {
#             "seed": int(fold.get("seed", 0)),
#             "T": Tmat,                       # (d_h, d_l)
#             "train_N": len(train_idx),
#             "test_N": len(test_idx),
#         }
#         logger.info(f"[BARYCA] Fold {k+1}/{len(saved_folds)} done "
#                     f"(train_N={len(train_idx)}, test_N={len(test_idx)}).")

#     os.makedirs(outdir, exist_ok=True)
#     joblib.dump(results, os.path.join(outdir, "baryca_cv_results_empirical.pkl"))
#     logger.info(f"BARYCA results saved to {outdir}")
#     return results
def run_baryca_empirical(all_data, saved_folds, outdir):
    """
    BARYCA (empirical) per-fold:
      • Build barycentric mixing matrices A_L^bary, A_H^bary.
        - Prefer SCM structural matrices if present.
        - Else, per-intervention LS:  D_i^T ≈ A_i @ U_i^T  ⇒  A_i = D_i^T @ pinv(U_i^T).
        - Else, single global LS on pooled data.
      • For each fold (train split):
          X_L^bary = A_L^bary @ U_ll_train.T
          X_H^bary = A_H^bary @ U_hl_train.T
          T = X_H^bary @ pinv(X_L^bary)
    Saves baryca_cv_results_empirical.pkl
    """
    import os, joblib, numpy as np
    import logging
    logger = logging.getLogger("battery_erica_affine")

    det_ll = all_data["LLmodel"].get("deterministic", {})
    det_hl = all_data["HLmodel"].get("deterministic", {})
    noise_ll_all = all_data["LLmodel"].get("noise", {})
    noise_hl_all = all_data["HLmodel"].get("noise", {})

    # Buckets (skip pooled 'U_ll_hat'/'U_hl_hat')
    def is_bucket(k): return (k is not None) and hasattr(k, "vv")
    ll_keys = [k for k in det_ll.keys() if is_bucket(k)]
    hl_keys = [k for k in det_hl.keys() if is_bucket(k)]

    U_ll_pool = noise_ll_all.get("U_ll_hat")  # (N, d_l)
    U_hl_pool = noise_hl_all.get("U_hl_hat")  # (N, d_h)
    d_l = U_ll_pool.shape[1]
    d_h = U_hl_pool.shape[1]

    # ---------- try SCM structural matrices first ----------
    LLmodels = all_data["LLmodel"].get("scm_instances")
    HLmodels = all_data["HLmodel"].get("scm_instances")
    Ill = all_data["LLmodel"].get("intervention_set")
    Ihl = all_data["HLmodel"].get("intervention_set")

    A_L_list, A_H_list = [], []

    if LLmodels is not None and HLmodels is not None and Ill and Ihl:
        try:
            L_mats = ut.compute_struc_matrices(LLmodels, Ill)  # list of (d_l,d_l)
            H_mats = ut.compute_struc_matrices(HLmodels, Ihl)  # list of (d_h,d_h)
            # to numpy
            L_arr = np.stack([m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m) for m in L_mats], 0)
            H_arr = np.stack([m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m) for m in H_mats], 0)
            A_L_bary = L_arr.mean(axis=0)
            A_H_bary = H_arr.mean(axis=0)
            logger.info("[BARYCA] Using SCM structural matrices (barycentric).")
        except Exception as e:
            logger.warning(f"[BARYCA] SCM matrices unavailable ({e}); falling back to LS mixing per intervention.")
            L_mats = H_mats = None
            A_L_bary = None
            A_H_bary = None
    else:
        A_L_bary = None
        A_H_bary = None

    # ---------- fallback: learn per-intervention mixing by LS ----------
    if A_L_bary is None or A_H_bary is None:
        # Collect A_i from buckets where both D_i and U_i exist
        for iota in ll_keys:
            U_i = noise_ll_all.get(iota)
            D_i = det_ll.get(iota)
            if U_i is None or D_i is None or len(U_i) == 0 or len(D_i) == 0:
                continue
            # Fit D_i^T ≈ A_i @ U_i^T
            try:
                A_i = D_i.T @ np.linalg.pinv(U_i.T)
                if A_i.shape == (d_l, d_l):
                    A_L_list.append(A_i)
            except np.linalg.LinAlgError:
                pass

        for eta in hl_keys:
            U_i = noise_hl_all.get(eta)
            D_i = det_hl.get(eta)
            if U_i is None or D_i is None or len(U_i) == 0 or len(D_i) == 0:
                continue
            try:
                A_i = D_i.T @ np.linalg.pinv(U_i.T)
                if A_i.shape == (d_h, d_h):
                    A_H_list.append(A_i)
            except np.linalg.LinAlgError:
                pass

        if A_L_list and A_H_list:
            A_L_bary = np.mean(np.stack(A_L_list, 0), axis=0)
            A_H_bary = np.mean(np.stack(A_H_list, 0), axis=0)
            logger.info("[BARYCA] Using LS-derived per-intervention mixing (barycentric).")
        else:
            A_L_bary = None
            A_H_bary = None

    # ---------- last resort: global LS on pooled data ----------
    if A_L_bary is None or A_H_bary is None:
        logger.warning("[BARYCA] Falling back to global LS mixing from pooled data.")
        # Fit D ≈ U @ A^T using pooled deterministic+noise if available; otherwise identity.
        # We approximate D with endogenous (deterministic + noise) across all buckets.
        # If deterministic unavailable, just use identity.
        try:
            # Build pooled endogenous X from buckets if present
            Xl_list, Xh_list = [], []
            for iota in ll_keys:
                U_i = noise_ll_all.get(iota)
                D_i = det_ll.get(iota)
                if U_i is not None and D_i is not None and len(U_i) == len(D_i):
                    Xl_list.append(D_i)  # endogenous approx
            for eta in hl_keys:
                U_i = noise_hl_all.get(eta)
                D_i = det_hl.get(eta)
                if U_i is not None and D_i is not None and len(U_i) == len(D_i):
                    Xh_list.append(D_i)

            if Xl_list:
                Xl = np.vstack(Xl_list)
                Ul = np.vstack([noise_ll_all[k] for k in ll_keys if noise_ll_all.get(k) is not None])
                A_L_bary = (Xl.T @ np.linalg.pinv(Ul.T)) if Ul.size and Xl.size else np.eye(d_l)
            else:
                A_L_bary = np.eye(d_l)

            if Xh_list:
                Xh = np.vstack(Xh_list)
                Uh = np.vstack([noise_hl_all[k] for k in hl_keys if noise_hl_all.get(k) is not None])
                A_H_bary = (Xh.T @ np.linalg.pinv(Uh.T)) if Uh.size and Xh.size else np.eye(d_h)
            else:
                A_H_bary = np.eye(d_h)
        except Exception as e:
            logger.warning(f"[BARYCA] Global LS fallback failed ({e}); using identities.")
            A_L_bary = np.eye(d_l)
            A_H_bary = np.eye(d_h)

    # ---------- per-fold fit of T on TRAIN split ----------
    results = {}
    for k, fold in enumerate(saved_folds):
        train_idx = fold["train"]
        test_idx  = fold["test"]

        U_ll_train = U_ll_pool[train_idx]   # (N_train, d_l)
        U_hl_train = U_hl_pool[train_idx]   # (N_train, d_h)

        X_L_bary = A_L_bary @ U_ll_train.T  # (d_l, N_train)
        X_H_bary = A_H_bary @ U_hl_train.T  # (d_h, N_train)

        try:
            Tmat = X_H_bary @ np.linalg.pinv(X_L_bary)  # (d_h, d_l)
        except np.linalg.LinAlgError:
            logger.warning(f"[BARYCA] pinv failed on fold {k}; using zeros.")
            Tmat = np.zeros((d_h, d_l))

        results[f"fold_{k}"] = {
            "seed": int(fold.get("seed", 0)),
            "T": Tmat,
            "train_N": len(train_idx),
            "test_N": len(test_idx),
        }
        logger.info(f"[BARYCA] Fold {k+1}/{len(saved_folds)} done "
                    f"(train_N={len(train_idx)}, test_N={len(test_idx)}).")

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(results, os.path.join(outdir, "baryca_cv_results_empirical.pkl"))
    logger.info(f"BARYCA results saved to {outdir}")
    return results



def run_abslingam(all_data: Dict[str, Any], saved_folds: List[dict], outdir: str):
    logger.info("Running Abs-LiNGAM baseline (per-fold, train-split)…")
    Dll_obs = all_data["LLmodel"]["data"][None]  # (N, d_l)
    Dhl_obs = all_data["HLmodel"]["data"][None]  # (N, d_h)

    results = {}
    for k, fold in enumerate(saved_folds):
        train_idx = fold["train"]
        Dll_train = Dll_obs[train_idx]
        Dhl_train = Dhl_obs[train_idx]

        fold_results = {}
        for style in ["Perfect", "Noisy"]:
            T_matrix, error = optools.abs_lingam_reconstruction_v2(
                Dll_train, Dhl_train,
                n_paired_samples=min(len(Dll_train), len(Dhl_train)),
                style=style, tau_threshold=1e-2
            )
            fold_results[style] = {
                "T": T_matrix,
                "error": error,
                "train_N": len(train_idx),
                "test_N": len(fold["test"]),
            }

        results[f"fold_{k}"] = fold_results
        logger.info(f"[Abs-LiNGAM] Fold {k+1}/{len(saved_folds)} done "
                    f"(train_N={len(train_idx)}, test_N={len(fold['test'])}).")

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(results, os.path.join(outdir, "abslingam_cv_results_empirical.pkl"))
    logger.info(f"Abs-LiNGAM results saved to {outdir}")
    return results


# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Battery ERICA-style (Affine ANM) — Robust Radius Sweep")
    parser.add_argument("--experiment", type=str, default="battery")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-diroca", action="store_true")
    parser.add_argument("--skip-gradca", action="store_true")
    parser.add_argument("--skip-baryca", action="store_true")
    parser.add_argument("--skip-abslingam", action="store_true")
    args = parser.parse_args()

    exp = args.experiment
    outdir = args.output_dir or f"data/{exp}/results_empirical_9010"
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"Output directory: {outdir}")

    # Load data bundle produced by the prep script
    all_data = ut.load_all_data(exp)
    all_data["experiment_name"] = exp

    # Repeated 90/10 splits (used for seeds/repeats)
    Dhl_obs = all_data["HLmodel"]["data"][None]
    saved_folds = make_stratified_9010_splits(Dhl_obs, args.test_size, args.seeds)
    logger.info(f"Built {len(saved_folds)} stratified 90/10 splits over seeds: {args.seeds}")

    # Empirical lower bounds (like original)
    l = all_data["LLmodel"]["noise"]["U_ll_hat"].shape[1]
    h = all_data["HLmodel"]["noise"]["U_hl_hat"].shape[1]
    train_n = len(saved_folds[0]["train"])
    ll_bound = round(ut.compute_empirical_radius(N=train_n, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=l), 3)
    hl_bound = round(ut.compute_empirical_radius(N=train_n, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=h), 3)
    radius_pairs = [
        (ll_bound, hl_bound),  # theoretical lower bound
        (1.0, 1.0),
        (2.0, 2.0),
        (4.0, 4.0),
        (8.0, 8.0),
    ]
    logger.info(f"Radius sweep: {radius_pairs}")

    results = {}

    if not args.skip_diroca:
        results["diroca"] = run_diroca_radius_sweep(all_data, saved_folds, radius_pairs, outdir)

    if not args.skip_gradca:
        results["gradca"] = run_gradca_affine(all_data, saved_folds, outdir, eta_min=1e-3, num_steps_min=1, max_iter=2000, tol=1e-4, seed=23, initialization="zeros", gain=0.0, optimizers="adam")


    if not args.skip_baryca:
        results["baryca"] = run_baryca_empirical(all_data, saved_folds, outdir)

    if not args.skip_abslingam:
        results["abslingam"] = run_abslingam(all_data, saved_folds, outdir)

    logger.info("All optimizations completed.")
    print("\n" + "=" * 60)
    print("BATTERY EMPIRICAL OPTIMIZATION (Affine ANM + Radius Sweep) — COMPLETED")
    print("=" * 60)
    print(f"Experiment: {exp}")
    print(f"Output dir: {outdir}")
    print(f"Methods run: {list(results.keys())}")
    print(f"Number of folds: {len(saved_folds)}")


if __name__ == "__main__":
    main()
