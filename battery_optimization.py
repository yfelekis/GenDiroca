#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import yaml
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn.init as init
from tqdm import tqdm

import utilities as ut      
import opt_tools as optools 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("battery_modular_optimization")


# ----------------- Config Loading -----------------
def load_method_config(method: str, experiment: str = "battery") -> Dict[str, Any]:
    """Load configuration for a specific method."""
    config_path = f"configs/{method}_opt_config_empirical_{experiment}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure numeric values are properly typed
    config = _ensure_numeric_types(config)
    
    logger.info(f"Loaded config for {method}: {config_path}")
    return config


def _ensure_numeric_types(obj):
    """Recursively ensure numeric values in config are proper types."""
    if isinstance(obj, dict):
        return {k: _ensure_numeric_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ensure_numeric_types(item) for item in obj]
    elif isinstance(obj, str):
        # Try to convert string numbers to float/int
        try:
            if '.' in obj or 'e' in obj.lower():
                return float(obj)
            else:
                return int(obj)
        except ValueError:
            return obj
    else:
        return obj


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


# # ----------------- Bucket utilities: fold slicing + alignment -----------------

def _is_bucket_key(k) -> bool:
    return (k is None) or ((k is not None) and hasattr(k, "vv"))


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
      - return per-bucket torch tensors and the local-to-global slice maps
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
            n = min(len(Dl_al), len(Dh_al))
            Dl_al, Ul_al = Dl_al[:n], Ul_al[:n]
            Dh_al, Uh_al = Dh_al[:n], Uh_al[:n]
            common = common[:n]

        # cache tensors
        det_ll_t[iota] = torch.from_numpy(Dl_al).float()
        noise_ll_t[iota] = torch.from_numpy(Ul_al).float()
        det_hl_t[iota] = torch.from_numpy(Dh_al).float()  
        noise_hl_t[iota] = torch.from_numpy(Uh_al).float()  

        # record slice maps into concatenated order (global thetas)
        slice_map_ll[iota] = np.arange(concat_cursor_ll, concat_cursor_ll + n, dtype=int)
        slice_map_hl[iota] = np.arange(concat_cursor_hl, concat_cursor_hl + n, dtype=int)  # key by iota
        concat_cursor_ll += n
        concat_cursor_hl += n
        total_n_ll += n
        total_n_hl += n

    return (det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
            slice_map_ll, slice_map_hl, total_n_ll, total_n_hl)


# ----------------- Objective -----------------
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
    
    losses = []
    for iota, Dl in det_ll_t.items():
        eta = omega.get(iota, None)
        if eta is None:
            continue
        Dh = det_hl_t.get(iota, None)  
        Ul = noise_ll_t.get(iota, None)
        Uh = noise_hl_t.get(iota, None)  
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


# ----------------- Implementations -----------------
def run_diroca(all_data: Dict[str, Any], config: Dict[str, Any], saved_folds: List[dict]) -> Dict[str, Any]:
    """Run DiRoCA optimization with config-driven parameters."""
    opt_config = config["optimization"]
    radius_config = config["radius_sweep"]
    output_config = config["output"]
    
    # Build radius pairs
    radius_pairs = []
    
    if radius_config.get("use_theoretical_bounds", True):
        # Compute theoretical bounds
        l = next(iter(all_data["LLmodel"]["deterministic"].values())).shape[1]
        h = next(iter(all_data["HLmodel"]["deterministic"].values())).shape[1]
        train_n = len(saved_folds[0]["train"])
        
        theoretical_params = radius_config.get("theoretical_params", {})
        ll_bound = round(ut.compute_empirical_radius(
            N=train_n, 
            eta=theoretical_params.get("eta", 0.05),
            c1=theoretical_params.get("c1", 1000.0),
            c2=theoretical_params.get("c2", 1.0),
            alpha=theoretical_params.get("alpha", 2.0),
            m=l
        ), 3)
        hl_bound = round(ut.compute_empirical_radius(
            N=train_n,
            eta=theoretical_params.get("eta", 0.05),
            c1=theoretical_params.get("c1", 1000.0),
            c2=theoretical_params.get("c2", 1.0),
            alpha=theoretical_params.get("alpha", 2.0),
            m=h
        ), 3)
        
        radius_pairs.append((ll_bound, hl_bound))
        logger.info(f"Theoretical bounds: ε={ll_bound}, δ={hl_bound}")
    
    # Add additional radius pairs
    for pair in radius_config.get("additional_pairs", []):
        radius_pairs.append(tuple(pair))
    
    logger.info(f"DiRoCA radius sweep: {radius_pairs}")
    
    # Run optimization
    all_runs = {}
    for (eps, delt) in radius_pairs:
        logger.info(f"Running DiRoCA (global adversary) for ε={eps}, δ={delt}")
        res = run_one_erica_affine_global(
            all_data, saved_folds,
            epsilon=eps, delta=delt,
            eta_min=opt_config["eta_min"],
            eta_max=opt_config["eta_max"],
            num_steps_min=opt_config["num_steps_min"],
            num_steps_max=opt_config["num_steps_max"],
            max_iter=opt_config["max_iter"],
            tol=opt_config["tol"],
            seed=saved_folds[0].get("seed", 23),
            initialization=opt_config["initialization"],
            gain=opt_config["gain"],
            optimizers=opt_config["optimizers"]
        )
        key = f"epsilon_{eps}_delta_{delt}"
        all_runs[key] = res

    # Save results
    os.makedirs(output_config["save_directory"], exist_ok=True)
    output_path = os.path.join(output_config["save_directory"], f"{output_config['filename_prefix']}.pkl")
    joblib.dump(all_runs, output_path)
    logger.info(f"Saved DiRoCA sweep to {output_path}")
    return all_runs


def run_gradca(all_data: Dict[str, Any], config: Dict[str, Any], saved_folds: List[dict]) -> Dict[str, Any]:
    """Run GradCA optimization with config-driven parameters."""
    opt_config = config["optimization"]
    output_config = config["output"]
    
    torch.manual_seed(23)
    np.random.seed(23)

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

        seed_fold = int(fold.get("seed", 23))
        torch.manual_seed(seed_fold)
        np.random.seed(seed_fold)

        T = torch.randn(d_h, d_l, requires_grad=True)
        if opt_config["gain"] > 0:
            init.xavier_normal_(T, gain=opt_config["gain"])

        if opt_config["optimizers"] == "adam":
            optimizer_T = torch.optim.Adam([T], lr=opt_config["eta_min"])
        elif opt_config["optimizers"] == "adam_betas":
            optimizer_T = torch.optim.Adam([T], lr=opt_config["eta_min"], betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizers']}")

        prev_obj = float("inf")
        for it in tqdm(range(opt_config["max_iter"]), desc=f"GRADCA | Fold {fold_id+1}/{len(saved_folds)}"):
            for _ in range(opt_config["num_steps_min"]):
                optimizer_T.zero_grad()
                obj_T = affine_empirical_objective_global(
                    det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
                    omega, T, Theta_L0, Phi_H0, slice_map_ll, slice_map_hl
                )
                obj_T.backward()
                optimizer_T.step()

            with torch.no_grad():
                cur = obj_T.item()
                if abs(prev_obj - cur) < opt_config["tol"]:
                    break
                prev_obj = cur

        results[f"fold_{fold_id}"] = {
            "seed": seed_fold,
            "T": T.detach().cpu().numpy(),
            "train_N": tot_n_ll,
            "test_N": len(fold["test"]),
        }

    # Save results
    os.makedirs(output_config["save_directory"], exist_ok=True)
    output_path = os.path.join(output_config["save_directory"], f"{output_config['filename_prefix']}.pkl")
    joblib.dump(results, output_path)
    logger.info(f"GradCA results saved to {output_path}")
    return results

# ----------------------------- BaryCA -----------------------------

def barycentric_objective_battery_lucas_style(
    T: torch.Tensor,
    det_ll_t: Dict[Any, torch.Tensor],
    det_hl_t: Dict[Any, torch.Tensor],
    noise_ll_t: Dict[Any, torch.Tensor],
    noise_hl_t: Dict[Any, torch.Tensor],
) -> torch.Tensor:
    """
    Steps:
      1) Concatenate ALL aligned TRAIN deterministic samples across interventions.
         (No stacking => no equal-N requirement.)
      2) Take the global mean deterministic vector on each side.
      3) Add TRAIN aligned noises (concatenated).
      4) Fit T by minimizing Frobenius mismatch.

    loss = || T (mean_Dll + U_ll_concat) - (mean_Dhl + U_hl_concat) ||_F^2 / N
    """
    device = T.device

    # collect aligned train buckets (already aligned by build_fold_aligned_buckets)
    ll_det_list, hl_det_list = [], []
    ll_noise_list, hl_noise_list = [], []

    for iota, Dl in det_ll_t.items():
        Dh = det_hl_t.get(iota, None)
        Ul = noise_ll_t.get(iota, None)
        Uh = noise_hl_t.get(iota, None)
        if Dh is None or Ul is None or Uh is None:
            continue

        ll_det_list.append(Dl.to(device))
        hl_det_list.append(Dh.to(device))
        ll_noise_list.append(Ul.to(device))
        hl_noise_list.append(Uh.to(device))

    if not ll_det_list:
        return torch.tensor(0.0, device=device)

    # 1) concatenate (N_total, d_*)
    Dll_all = torch.cat(ll_det_list, dim=0)
    Dhl_all = torch.cat(hl_det_list, dim=0)
    Ull_all = torch.cat(ll_noise_list, dim=0)
    Uhl_all = torch.cat(hl_noise_list, dim=0)

    # 2) global deterministic barycenters (1, d_*)
    mean_Dll = Dll_all.mean(dim=0, keepdim=True)
    mean_Dhl = Dhl_all.mean(dim=0, keepdim=True)

    # 3) barycentric endogenous samples (broadcast mean over all train rows)
    endo_ll = mean_Dll + Ull_all   # (N_total, d_l)
    endo_hl = mean_Dhl + Uhl_all   # (N_total, d_h)

    # 4) Frobenius loss
    diff = (T @ endo_ll.T).T - endo_hl
    N_total = endo_ll.shape[0]
    return torch.norm(diff, p="fro")**2 / max(1, N_total)


def run_baryca(all_data: Dict[str, Any], config: Dict[str, Any], saved_folds: List[dict]) -> Dict[str, Any]:
    """
      per fold:
        - build aligned TRAIN buckets (det+noise) using build_fold_aligned_buckets
        - compute barycentric objective as:
             mean_det + train_noise
        - optimize T with Adam GD on that barycentric objective

    This is the same computational philosophy as the LUCAS script,
    except barycenter uses concatenation so unequal bucket sizes are OK.
    """
    opt_config = config["optimization"]
    output_config = config["output"]

    det_ll_all = all_data["LLmodel"]["deterministic"]
    det_hl_all = all_data["HLmodel"]["deterministic"]
    noise_ll_all = all_data["LLmodel"]["noise"]
    noise_hl_all = all_data["HLmodel"]["noise"]
    row_idx_ll = all_data["LLmodel"]["row_idx"]
    row_idx_hl = all_data["HLmodel"]["row_idx"]
    omega = all_data["abstraction_data"]["omega"]

    d_l = next(iter(det_ll_all.values())).shape[1]
    d_h = next(iter(det_hl_all.values())).shape[1]

    lr = opt_config.get("lr", 1e-3)
    max_iter = opt_config.get("max_iter", 5000)
    tol = opt_config.get("tol", 1e-5)

    results = {}

    for fold_id, fold in enumerate(saved_folds):
        train_idx = np.asarray(fold["train"], dtype=int)
        seed_fold = int(fold.get("seed", 23))
        torch.manual_seed(seed_fold)
        np.random.seed(seed_fold)

        # Build aligned TRAIN buckets (keyed by iota, HL stored under same iota)
        (det_ll_t, det_hl_t, noise_ll_t, noise_hl_t,
         slice_map_ll, slice_map_hl, tot_n_ll, tot_n_hl) = build_fold_aligned_buckets(
            det_ll=det_ll_all, det_hl=det_hl_all,
            noise_ll=noise_ll_all, noise_hl=noise_hl_all,
            row_idx_ll=row_idx_ll, row_idx_hl=row_idx_hl,
            omega=omega, train_idx=train_idx
        )

        if tot_n_ll == 0 or tot_n_hl == 0:
            logger.warning(f"[BaryCA] Fold {fold_id}: no aligned TRAIN samples; using zero T.")
            Tmat = np.zeros((d_h, d_l))
            results[f"fold_{fold_id}"] = {
                "seed": seed_fold, "T": Tmat,
                "train_N": 0, "test_N": len(fold["test"]),
            }
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LUCAS-style: optimize T on barycentric objective
        T = torch.randn(d_h, d_l, requires_grad=True, device=device)
        opt = torch.optim.Adam([T], lr=lr)

        prev = float("inf")
        for it in tqdm(range(max_iter), desc="BaryCA Optimization"):
            opt.zero_grad()
            loss = barycentric_objective_battery_lucas_style(
                T, det_ll_t, det_hl_t, noise_ll_t, noise_hl_t
            )
            if torch.isnan(loss):
                logger.warning(f"[BaryCA] Fold {fold_id}: NaN loss, aborting.")
                break
            loss.backward()
            opt.step()

            cur = float(loss.item())
            if abs(prev - cur) < tol:
                logger.info(f"[BaryCA] Fold {fold_id}: converged at iter {it+1} (Δ<{tol}).")
                break
            prev = cur

        Tmat = T.detach().cpu().numpy()

        results[f"fold_{fold_id}"] = {
            "seed": seed_fold,
            "T": Tmat,
            "train_N": int(tot_n_ll),
            "test_N": int(len(fold["test"])),
        }

        logger.info(f"[BaryCA] Fold {fold_id+1}/{len(saved_folds)} done "
                    f"(train_N={tot_n_ll}, test_N={len(fold['test'])}).")

    # Save results
    os.makedirs(output_config["save_directory"], exist_ok=True)
    output_path = os.path.join(
        output_config["save_directory"],
        f"{output_config['filename_prefix']}.pkl"
    )
    joblib.dump(results, output_path)
    logger.info(f"BaryCA results saved to {output_path}")
    return results

def run_abslingam(all_data: Dict[str, Any], config: Dict[str, Any], saved_folds: List[dict]) -> Dict[str, Any]:
    """Run Abs-LiNGAM optimization with config-driven parameters."""
    opt_config = config["optimization"]
    output_config = config["output"]
    
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
            tau_threshold = opt_config.get("tau_threshold", 1e-2) if style == "Perfect" else opt_config.get("tau_threshold_noisy", 1e-1)
            refit_coeff = opt_config.get("refit_coeff", False)
            
            T_matrix, error = optools.abs_lingam_reconstruction_v2(
                Dll_train, Dhl_train,
                n_paired_samples=min(len(Dll_train), len(Dhl_train)),
                style=style, tau_threshold=tau_threshold
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

    # Save results
    os.makedirs(output_config["save_directory"], exist_ok=True)
    output_path = os.path.join(output_config["save_directory"], f"{output_config['filename_prefix']}.pkl")
    joblib.dump(results, output_path)
    logger.info(f"Abs-LiNGAM results saved to {output_path}")
    return results


# ----------------- Core optimizer -----------------
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


# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Modular Battery Optimization")
    parser.add_argument("--experiment", type=str, default="battery")
    parser.add_argument("--methods", type=str, nargs="+", 
                       choices=["diroca", "gradca", "baryca", "abslingam"],
                       default=["diroca", "gradca", "baryca", "abslingam"],
                       help="Methods to run")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    exp = args.experiment
    logger.info(f"Starting modular optimization for experiment: {exp}")
    logger.info(f"Methods to run: {args.methods}")

    # Load data bundle
    all_data = ut.load_all_data(exp)
    all_data["experiment_name"] = exp

    results = {}

    for method in args.methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Load method-specific config
            config = load_method_config(method, exp)
            
            # Build stratified splits using config
            cv_config = config["cv"]
            Dhl_obs = all_data["HLmodel"]["data"][None]  # (N, d_h)
            saved_folds = make_stratified_9010_splits(Dhl_obs, cv_config["test_size"], cv_config["seeds"])
            logger.info(f"Built {len(saved_folds)} stratified splits over seeds: {cv_config['seeds']}")
            
            # Run method-specific optimization
            if method == "diroca":
                results[method] = run_diroca(all_data, config, saved_folds)
            elif method == "gradca":
                results[method] = run_gradca(all_data, config, saved_folds)
            elif method == "baryca":
                results[method] = run_baryca(all_data, config, saved_folds)
            elif method == "abslingam":
                results[method] = run_abslingam(all_data, config, saved_folds)
            else:
                logger.warning(f"Unknown method: {method}")
                continue
                
        except Exception as e:
            logger.error(f"Error running {method}: {e}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info("BATTERY OPTIMIZATION — COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Experiment: {exp}")
    logger.info(f"Methods completed: {list(results.keys())}")
    logger.info(f"Number of folds: {len(saved_folds) if 'saved_folds' in locals() else 'N/A'}")


if __name__ == "__main__":
    main()
