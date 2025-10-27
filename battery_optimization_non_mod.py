#!/usr/bin/env python3
import argparse
import os
import sys
import logging
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn.init as init
from tqdm import tqdm

import utilities as ut      # must provide: load_all_data, compute_empirical_radius
import opt_tools as optools # Abs-LiNGAM helper (abs_lingam_reconstruction_v2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("battery_erica_affine_global")


# ----------------- Splits -----------------
def make_stratified_9010_splits(Dhl_obs: np.ndarray, test_size: float, seeds: List[int]) -> List[dict]:
    """
    Build 'folds' dicts with 'train' and 'test' indices from HL observations (N, d_h).
    We stratify by the first column (assumed CG / group label).
    """
    assert Dhl_obs.ndim == 2, "Dhl_obs must be (N, d_h)"
    assert 0.0 < test_size < 1.0, "test_size must be in (0,1)"
    N = Dhl_obs.shape[0]
    y = Dhl_obs[:, 0]

    label_to_indices: Dict[Any, List[int]] = {}
    for idx, lab in enumerate(y):
        label_to_indices.setdefault(lab, []).append(idx)

    folds = []
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
        folds.append({"train": train_idx, "test": test_idx, "seed": seed})
    return folds


# ----------------- Global Frobenius projection -----------------
def project_onto_frobenius_ball_global(X: torch.Tensor, radius_per_sample: float) -> torch.Tensor:
    """
    Project onto ||X||_F <= sqrt(N_total) * radius_per_sample.
    Operates in-place style (returns the projected tensor).
    """
    if X.numel() == 0:
        return X
    N_total = X.shape[0]
    bound = (N_total ** 0.5) * radius_per_sample
    fro = torch.norm(X, p='fro')
    if fro > 0 and fro > bound:
        X.mul_(bound / fro)
    return X


# ----------------- Bucket utilities: fold slicing + alignment -----------------
def _is_bucket_key(k) -> bool:
    # mirror your convention: real buckets have a .vv attribute and are not None
    return (k is not None) and hasattr(k, "vv")


def build_fold_aligned_buckets(
    det_ll: Dict[Any, np.ndarray],
    det_hl: Dict[Any, np.ndarray],
    noise_ll: Dict[Any, np.ndarray],
    noise_hl: Dict[Any, np.ndarray],
    row_idx_ll: Dict[Any, np.ndarray],
    row_idx_hl: Dict[Any, np.ndarray],
    omega: Dict[Any, Any],
    train_idx: np.ndarray,
) -> Tuple[
    Dict[Any, torch.Tensor], Dict[Any, torch.Tensor],
    Dict[Any, torch.Tensor], Dict[Any, torch.Tensor],
    Dict[Any, np.ndarray], Dict[Any, np.ndarray], int, int
]:
    """
    For each intervention pair (iota, eta=omega(iota)):
      - restrict to TRAIN indices using provided per-bucket global row indices,
      - align by global row indices (intersection + consistent ordering),
      - return per-bucket torch tensors (on CPU) and the local-to-global slice maps
        into the future global Theta/Phi tensors.
    Returns:
      det_ll_t, det_hl_t, noise_ll_t, noise_hl_t: dicts {iota/eta -> torch.FloatTensor (n_ij, d_*)}
      slice_map_ll, slice_map_hl: dicts {iota/eta -> np.ndarray (n_ij,) } positions in the concatenated order
      total_n_ll, total_n_hl: totals (should be equal if alignment is 1:1 per-pair)
    """
    det_ll_t, det_hl_t = {}, {}
    noise_ll_t, noise_hl_t = {}, {}
    slice_map_ll, slice_map_hl = {}, {}

    concat_cursor_ll = 0
    concat_cursor_hl = 0
    total_n_ll = 0
    total_n_hl = 0

    # Precompute train sets for fast membership
    train_set = set(train_idx.tolist())

    for iota, Dl_np in det_ll.items():
        if not _is_bucket_key(iota):
            continue
        eta = omega.get(iota, None)
        if eta is None:
            continue
        Dh_np = det_hl.get(eta, None)
        if Dh_np is None:
            continue

        Ul_np = noise_ll.get(iota, None)
        Uh_np = noise_hl.get(eta, None)
        if Ul_np is None or Uh_np is None:
            continue

        idx_l_global = row_idx_ll.get(iota, None)
        idx_h_global = row_idx_hl.get(eta, None)
        if idx_l_global is None or idx_h_global is None:
            raise ValueError(
                "Missing row_idx for fold slicing/alignment. "
                "Please provide all_data['LLmodel']['row_idx'][iota] and "
                "all_data['HLmodel']['row_idx'][eta]."
            )

        # restrict to TRAIN
        idx_l_train_mask = np.array([g in train_set for g in idx_l_global], dtype=bool)
        idx_h_train_mask = np.array([g in train_set for g in idx_h_global], dtype=bool)

        if not idx_l_train_mask.any() or not idx_h_train_mask.any():
            continue

        Dl_tr = Dl_np[idx_l_train_mask]
        Ul_tr = Ul_np[idx_l_train_mask]
        il_tr = idx_l_global[idx_l_train_mask]

        Dh_tr = Dh_np[idx_h_train_mask]
        Uh_tr = Uh_np[idx_h_train_mask]
        ih_tr = idx_h_global[idx_h_train_mask]

        # align by intersection of global indices
        common = np.intersect1d(il_tr, ih_tr)
        if common.size == 0:
            continue

        # maps from global id -> position
        pos_l = {g: p for p, g in enumerate(il_tr)}
        pos_h = {g: p for p, g in enumerate(ih_tr)}
        sel_l = np.array([pos_l[g] for g in common], dtype=int)
        sel_h = np.array([pos_h[g] for g in common], dtype=int)

        Dl_al = Dl_tr[sel_l]
        Ul_al = Ul_tr[sel_l]
        Dh_al = Dh_tr[sel_h]
        Uh_al = Uh_tr[sel_h]

        # ensure same length
        n = min(len(Dl_al), len(Dh_al))
        if n == 0:
            continue
        if len(Dl_al) != len(Dh_al):
            # should not happen after intersection, but guard anyway
            n = min(len(Dl_al), len(Dh_al))
            Dl_al, Ul_al = Dl_al[:n], Ul_al[:n]
            Dh_al, Uh_al = Dh_al[:n], Uh_al[:n]
            common = common[:n]

        # cache tensors
        det_ll_t[iota] = torch.from_numpy(Dl_al).float()
        noise_ll_t[iota] = torch.from_numpy(Ul_al).float()
        det_hl_t[iota] = torch.from_numpy(Dh_al).float()  # key by iota to avoid many→one overwrite
        noise_hl_t[iota] = torch.from_numpy(Uh_al).float()  # key by iota to avoid many→one overwrite

        # record slice maps into concatenated order (global thetas)
        slice_map_ll[iota] = np.arange(concat_cursor_ll, concat_cursor_ll + n, dtype=int)
        slice_map_hl[iota] = np.arange(concat_cursor_hl, concat_cursor_hl + n, dtype=int)  # key by iota
        concat_cursor_ll += n
        concat_cursor_hl += n
        total_n_ll += n
        total_n_hl += n

    return (det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
            slice_map_ll, slice_map_hl, total_n_ll, total_n_hl)


# ----------------- Objective with GLOBAL Theta/Phi -----------------
def affine_empirical_objective_global(
    det_ll_t: Dict[Any, torch.Tensor],
    det_hl_t: Dict[Any, torch.Tensor],
    noise_ll_t: Dict[Any, torch.Tensor],
    noise_hl_t: Dict[Any, torch.Tensor],
    omega: Dict[Any, Any],
    T: torch.Tensor,
    Theta_L_global: torch.Tensor,
    Phi_H_global: torch.Tensor,
    slice_map_ll: Dict[Any, np.ndarray],
    slice_map_hl: Dict[Any, np.ndarray],
) -> torch.Tensor:
    """
    F(T) = E_{iota ~ uniform} || T(D_l(i) + U_l(i) + Θ_global[slice_i])^T
                         - (D_h(ω(i)) + U_h(ω(i)) + Φ_global[slice_ω(i)])^T ||_F^2

    Uniform weighting over available intervention pairs (buckets with n>0).
    """
    losses = []
    for iota, Dl in det_ll_t.items():
        eta = omega.get(iota, None)
        if eta is None:
            continue
        Dh = det_hl_t.get(iota, None)  # read HL by iota to avoid many→one overwrite
        Ul = noise_ll_t.get(iota, None)
        Uh = noise_hl_t.get(iota, None)  # read HL by iota to avoid many→one overwrite
        if Dh is None or Ul is None or Uh is None:
            continue

        sl = slice_map_ll[iota]
        sh = slice_map_hl[iota]  # read HL by iota
        # index the global thetas (same n on both sides by construction)
        Theta_i = Theta_L_global[sl, :]
        Phi_i   = Phi_H_global[sh, :]

        L_part = Dl + Ul + Theta_i   # (n, d_l)
        H_part = Dh + Uh + Phi_i     # (n, d_h)

        diff = T @ L_part.T - H_part.T   # (d_h, n)
        # normalize by d_h * n for scale invariance across buckets
        losses.append(torch.norm(diff, p='fro')**2 / (diff.shape[0] * diff.shape[1]))

    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    # uniform q over interventions -> simple mean
    return sum(losses) / len(losses)


# ----------------- Core optimizer (global adversary) -----------------
def run_one_erica_affine_global(
    all_data: Dict[str, Any],
    saved_folds: List[dict],
    epsilon: float,   # low-level per-sample radius
    delta: float,     # high-level per-sample radius
    eta_min: float = 1e-3,
    eta_max: float = 1e-3,
    num_steps_min: int = 5,
    num_steps_max: int = 2,
    max_iter: int = 5000,
    tol: float = 1e-4,
    seed: int = 23,
    initialization: str = "random",
    gain: float = 0.0,
    optimizers: str = "adam",
) -> Dict[str, Any]:

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    det_ll_all = all_data["LLmodel"]["deterministic"]
    det_hl_all = all_data["HLmodel"]["deterministic"]
    noise_ll_all = all_data["LLmodel"]["noise"]
    noise_hl_all = all_data["HLmodel"]["noise"]
    row_idx_ll = all_data["LLmodel"]["row_idx"]
    row_idx_hl = all_data["HLmodel"]["row_idx"]
    omega = all_data["abstraction_data"]["omega"]

    # Keys we consider as buckets:
    ll_keys = [k for k in det_ll_all.keys() if _is_bucket_key(k)]

    # Dimensions:
    d_l = next(iter(det_ll_all.values())).shape[1]
    d_h = next(iter(det_hl_all.values())).shape[1]

    results = {}

    for fold_id, fold in enumerate(saved_folds):
        seed_fold = int(fold.get("seed", seed))
        torch.manual_seed(seed_fold)
        np.random.seed(seed_fold)

        # Build fold-sliced + aligned bucket tensors and slice maps
        (det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
         slice_map_ll, slice_map_hl, tot_n_ll, tot_n_hl) = build_fold_aligned_buckets(
            det_ll=det_ll_all, det_hl=det_hl_all,
            noise_ll=noise_ll_all, noise_hl=noise_hl_all,
            row_idx_ll=row_idx_ll, row_idx_hl=row_idx_hl,
            omega=omega, train_idx=fold["train"]
        )

        if tot_n_ll == 0 or tot_n_hl == 0:
            logger.warning(f"[Fold {fold_id}] No aligned train samples; skipping.")
            results[f"fold_{fold_id}"] = {
                "seed": seed_fold, "T": np.zeros((d_h, d_l)),
                "train_N": 0, "test_N": len(fold["test"]),
                "epsilon": float(epsilon), "delta": float(delta),
            }
            continue

        # Global adversary variables (one tensor per side)
        Theta_L_global = (torch.zeros if initialization == "zeros" else torch.randn)(
            (tot_n_ll, d_l), requires_grad=True
        )
        Phi_H_global = (torch.zeros if initialization == "zeros" else torch.randn)(
            (tot_n_hl, d_h), requires_grad=True
        )

        # Learnable abstraction matrix
        T = torch.randn(d_h, d_l, requires_grad=True)
        if gain > 0:
            init.xavier_normal_(T, gain=gain)

        # Optimizers
        if optimizers == "adam":
            optimizer_T = torch.optim.Adam([T], lr=eta_min)
            optimizer_max = torch.optim.Adam([Theta_L_global, Phi_H_global], lr=eta_max)
        elif optimizers == "adam_betas":
            optimizer_T = torch.optim.Adam([T], lr=eta_min, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
            optimizer_max = torch.optim.Adam([Theta_L_global, Phi_H_global],
                                             lr=eta_max, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        else:
            raise ValueError(f"Unknown optimizer: {optimizers}")

        prev_cycle_obj = float("inf")

        for it in tqdm(range(max_iter),
                       desc=f"(ε={epsilon}, δ={delta}) | Fold {fold_id+1}/{len(saved_folds)}"):
            # ----- MIN over T -----
            for _ in range(num_steps_min):
                optimizer_T.zero_grad()
                obj_min = affine_empirical_objective_global(
                    det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
                    omega, T, Theta_L_global, Phi_H_global, slice_map_ll, slice_map_hl
                )
                obj_min.backward()
                optimizer_T.step()

            # ----- MAX over (Theta_L_global, Phi_H_global) -----
            for _ in range(num_steps_max):
                optimizer_max.zero_grad()
                obj_max = -affine_empirical_objective_global(
                    det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
                    omega, T, Theta_L_global, Phi_H_global, slice_map_ll, slice_map_hl
                )
                obj_max.backward()
                optimizer_max.step()

                # Project global thetas once each
                with torch.no_grad():
                    project_onto_frobenius_ball_global(Theta_L_global, epsilon)
                    project_onto_frobenius_ball_global(Phi_H_global,   delta)

            # Convergence check AFTER a full min–max cycle
            with torch.no_grad():
                cur = obj_min.item()
                if abs(prev_cycle_obj - cur) < tol:
                    logger.info(f"Converged at iter {it+1} (Δ={abs(prev_cycle_obj-cur):.2e})")
                    break
                prev_cycle_obj = cur

        results[f"fold_{fold_id}"] = {
            "seed": seed_fold,
            "T": T.detach().cpu().numpy(),
            "train_N": tot_n_ll,  # aligned train count per side (equal)
            "test_N": len(fold["test"]),
            "epsilon": float(epsilon),
            "delta": float(delta),
        }

    return results


# ----------------- Sweeps -----------------
def run_diroca_radius_sweep(
    all_data: Dict[str, Any],
    saved_folds: List[dict],
    radius_pairs: List[tuple],
    outdir: str
) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    all_runs = {}
    for (eps, delt) in radius_pairs:
        logger.info(f"Running DiRoCA (global adversary) for ε={eps}, δ={delt}")
        res = run_one_erica_affine_global(
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


# ----------------- Non-robust GRADCA (Θ=Φ=0) -----------------
def run_gradca_affine(
    all_data: Dict[str, Any],
    saved_folds: List[dict],
    outdir: str,
    eta_min: float = 1e-3,
    num_steps_min: int = 1,
    max_iter: int = 5000,
    tol: float = 1e-4,
    seed: int = 23,
    initialization: str = "zeros",
    gain: float = 0.0,
    optimizers: str = "adam",
) -> Dict[str, Any]:

    torch.manual_seed(seed); np.random.seed(seed)

    det_ll_all = all_data["LLmodel"]["deterministic"]
    det_hl_all = all_data["HLmodel"]["deterministic"]
    noise_ll_all = all_data["LLmodel"]["noise"]
    noise_hl_all = all_data["HLmodel"]["noise"]
    row_idx_ll = all_data["LLmodel"]["row_idx"]
    row_idx_hl = all_data["HLmodel"]["row_idx"]
    omega = all_data["abstraction_data"]["omega"]

    d_l = next(iter(det_ll_all.values())).shape[1]
    d_h = next(iter(det_hl_all.values())).shape[1]

    results = {}

    for fold_id, fold in enumerate(saved_folds):
        (det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
         slice_map_ll, slice_map_hl, tot_n_ll, tot_n_hl) = build_fold_aligned_buckets(
            det_ll=det_ll_all, det_hl=det_hl_all,
            noise_ll=noise_ll_all, noise_hl=noise_hl_all,
            row_idx_ll=row_idx_ll, row_idx_hl=row_idx_hl,
            omega=omega, train_idx=fold["train"]
        )

        # fixed zero perturbations (global)
        Theta_L0 = torch.zeros((tot_n_ll, d_l))
        Phi_H0   = torch.zeros((tot_n_hl, d_h))

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
            for _ in range(num_steps_min):
                optimizer_T.zero_grad()
                obj_T = affine_empirical_objective_global(
                    det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
                    omega, T, Theta_L0, Phi_H0, slice_map_ll, slice_map_hl
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
            "T": T.detach().cpu().numpy(),
            "train_N": tot_n_ll,
            "test_N": len(fold["test"]),
        }

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(results, os.path.join(outdir, "gradca_cv_results_empirical.pkl"))
    logger.info(f"GRADCA results saved to {outdir}")
    return results


# ----------------- BARYCA (empirical) with affine intercept -----------------
def run_baryca_empirical(all_data: Dict[str, Any], saved_folds: List[dict], outdir: str):
    """
    BARYCA (empirical) per-fold:
      1) Estimate per-intervention mixing with LS **with intercept**:
            D_i^T ≈ A_i @ [U_i^T; 1^T]  => A_i ∈ R^{d × (d+1)}
         average them to A_bary, split to (W, b).
      2) For each fold (train split), form barycentric endogenous:
            X_L^bary = W_L @ U_ll_train.T + b_L[:,None]
            X_H^bary = W_H @ U_hl_train.T + b_H[:,None]
         then fit T by LS: T = X_H^bary @ pinv(X_L^bary).
    Requires: 'row_idx' dicts for proper train slicing and alignment.
    """
    det_ll = all_data["LLmodel"].get("deterministic", {})
    det_hl = all_data["HLmodel"].get("deterministic", {})
    noise_ll_all = all_data["LLmodel"].get("noise", {})
    noise_hl_all = all_data["HLmodel"].get("noise", {})
    row_idx_ll = all_data["LLmodel"]["row_idx"]
    row_idx_hl = all_data["HLmodel"]["row_idx"]
    omega = all_data["abstraction_data"]["omega"]

    # pooled noises (for quick dim checks)
    d_l = next(iter(det_ll.values())).shape[1]
    d_h = next(iter(det_hl.values())).shape[1]

    # Estimate per-intervention mixing with bias, then average
    def fit_with_bias(U: np.ndarray, D: np.ndarray) -> np.ndarray:
        U_aug = np.concatenate([U, np.ones((U.shape[0], 1))], axis=1)  # (N, d+1)
        # D^T ≈ A @ U_aug^T  => A ∈ R^{d × (d+1)}
        return D.T @ np.linalg.pinv(U_aug.T)

    A_L_list, A_H_list = [], []
    for iota, Dl in det_ll.items():
        if not _is_bucket_key(iota): continue
        Ul = noise_ll_all.get(iota, None)
        if Ul is None or len(Ul) == 0 or len(Dl) == 0: continue
        try:
            A_i = fit_with_bias(Ul, Dl)
            if A_i.shape == (d_l, d_l + 1):
                A_L_list.append(A_i)
        except np.linalg.LinAlgError:
            pass

    for eta, Dh in det_hl.items():
        if not _is_bucket_key(eta): continue
        Uh = noise_hl_all.get(eta, None)
        if Uh is None or len(Uh) == 0 or len(Dh) == 0: continue
        try:
            A_i = fit_with_bias(Uh, Dh)
            if A_i.shape == (d_h, d_h + 1):
                A_H_list.append(A_i)
        except np.linalg.LinAlgError:
            pass

    if not A_L_list or not A_H_list:
        logger.warning("[BARYCA] Not enough per-intervention fits; falling back to identity+zero bias.")
        W_L_bary, b_L_bary = np.eye(d_l), np.zeros((d_l, 1))
        W_H_bary, b_H_bary = np.eye(d_h), np.zeros((d_h, 1))
    else:
        A_L_bary = np.mean(np.stack(A_L_list, 0), axis=0)  # (d_l, d_l+1)
        A_H_bary = np.mean(np.stack(A_H_list, 0), axis=0)  # (d_h, d_h+1)
        W_L_bary, b_L_bary = A_L_bary[:, :d_l], A_L_bary[:, d_l:d_l+1]
        W_H_bary, b_H_bary = A_H_bary[:, :d_h], A_H_bary[:, d_h:d_h+1]

    results = {}
    for k, fold in enumerate(saved_folds):
        # Reuse aligned build to get TRAIN-aligned U’s (order matters)
        (det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
         slice_map_ll, slice_map_hl, tot_n_ll, _) = build_fold_aligned_buckets(
            det_ll=det_ll, det_hl=det_hl,
            noise_ll=noise_ll_all, noise_hl=noise_hl_all,
            row_idx_ll=row_idx_ll, row_idx_hl=row_idx_hl,
            omega=omega, train_idx=fold["train"]
        )

        # Build barycentric X from aligned U's (preserving per-pair alignment)
        X_L_list, X_H_list = [], []
        for iota, Ul_t in noise_ll_t.items():
            eta = omega.get(iota, None)
            if eta is None: continue
            Uh_t = noise_hl_t.get(iota, None)  # read HL by iota to match new keying
            if Uh_t is None: continue

            # Convert tensors to numpy before LS math
            Ul_np = Ul_t.detach().cpu().numpy()
            Uh_np = Uh_t.detach().cpu().numpy()

            # shapes: W_* (d_*, d_*), b_* (d_*,1), U_*^T (d_*, n)
            Xl = (W_L_bary @ Ul_np.T + b_L_bary)     # (d_l, n)
            Xh = (W_H_bary @ Uh_np.T + b_H_bary)     # (d_h, n)
            X_L_list.append(Xl)
            X_H_list.append(Xh)

        if not X_L_list or not X_H_list:
            logger.warning(f"[BARYCA] Fold {k}: no aligned train samples; returning zeros.")
            Tmat = np.zeros((d_h, d_l))
        else:
            X_L_bary = np.concatenate(X_L_list, axis=1)  # (d_l, N_train_aligned)
            X_H_bary = np.concatenate(X_H_list, axis=1)  # (d_h, N_train_aligned)
            try:
                Tmat = (X_H_bary @ np.linalg.pinv(X_L_bary)).astype(np.float64)
            except np.linalg.LinAlgError:
                logger.warning(f"[BARYCA] pinv failed on fold {k}; using zeros.")
                Tmat = np.zeros((d_h, d_l))

        results[f"fold_{k}"] = {
            "seed": int(fold.get("seed", 0)),
            "T": Tmat,
            "train_N": tot_n_ll,
            "test_N": len(fold["test"]),
        }
        logger.info(f"[BARYCA] Fold {k+1}/{len(saved_folds)} done "
                    f"(train_N={tot_n_ll}, test_N={len(fold['test'])}).")

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(results, os.path.join(outdir, "baryca_cv_results_empirical.pkl"))
    logger.info(f"BARYCA results saved to {outdir}")
    return results


# ----------------- Abs-LiNGAM baseline (per-fold, train split) -----------------
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
    parser = argparse.ArgumentParser(description="Battery ERICA-style (Affine ANM) — Global Adversary + Fold Handling")
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

    # Load data bundle
    all_data = ut.load_all_data(exp)
    all_data["experiment_name"] = exp

    # Build stratified 90/10 splits on HL observations (NO [None])
    Dhl_obs = all_data["HLmodel"]["data"][None]  # (N, d_h)
    saved_folds = make_stratified_9010_splits(Dhl_obs, args.test_size, args.seeds)
    logger.info(f"Built {len(saved_folds)} stratified 90/10 splits over seeds: {args.seeds}")

    # Empirical lower bounds (like original)
    l = next(iter(all_data["LLmodel"]["deterministic"].values())).shape[1]
    h = next(iter(all_data["HLmodel"]["deterministic"].values())).shape[1]
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
        results["gradca"] = run_gradca_affine(
            all_data, saved_folds, outdir,
            eta_min=1e-3, num_steps_min=1, max_iter=5000, tol=1e-4,
            seed=23, initialization="zeros", gain=0.0, optimizers="adam"
        )

    if not args.skip_baryca:
        results["baryca"] = run_baryca_empirical(all_data, saved_folds, outdir)

    if not args.skip_abslingam:
        results["abslingam"] = run_abslingam(all_data, saved_folds, outdir)

    logger.info("All optimizations completed.")
    print("\n" + "=" * 60)
    print("BATTERY EMPIRICAL OPTIMIZATION (Affine ANM + Global Adversary + Folds) — COMPLETED")
    print("=" * 60)
    print(f"Experiment: {exp}")
    print(f"Output dir: {outdir}")
    print(f"Methods run: {list(results.keys())}")
    print(f"Number of folds: {len(saved_folds)}")


if __name__ == "__main__":
    main()
