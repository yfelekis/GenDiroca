#!/usr/bin/env python3
import argparse
import os
import pickle
import gc
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torchvision.transforms as T


# ============================================================
# 1. Utility: load evaluation state
# ============================================================

def load_eval_state(eval_dir):
    print(f"Loading CMNIST eval state from: {eval_dir}")
    
    with open(os.path.join(eval_dir, "cmnist_all_results.pkl"), "rb") as f:
        all_results = pickle.load(f)
    with open(os.path.join(eval_dir, "cmnist_cv_folds.pkl"), "rb") as f:
        cv_folds = pickle.load(f)
    with open(os.path.join(eval_dir, "cmnist_omega.pkl"), "rb") as f:
        omega = pickle.load(f)
    with open(os.path.join(eval_dir, "cmnist_Dll_samples.pkl"), "rb") as f:
        Dll_samples = pickle.load(f)
    with open(os.path.join(eval_dir, "cmnist_Dhl_samples.pkl"), "rb") as f:
        Dhl_samples = pickle.load(f)

    # Optional dicts
    det_ll_dict_opt, det_hl_dict_opt = None, None
    path_det_ll = os.path.join(eval_dir, "cmnist_det_ll_dict_opt.pt")
    if os.path.exists(path_det_ll):
        det_ll_dict_opt = torch.load(path_det_ll, map_location="cpu")

    print("All eval state loaded.\n")
    return all_results, cv_folds, omega, Dll_samples, Dhl_samples, det_ll_dict_opt, det_hl_dict_opt, None, None


# ============================================================
# 2. Low-level transforms
# ============================================================

def apply_huber_contamination_cmnist(X_flat, alpha, noise_scale, noise_dims, seed, loc=0.0):
    if alpha <= 0 or noise_scale <= 0: return X_flat
    device = X_flat.device
    N, D = X_flat.shape
    torch.manual_seed(int(seed) % (2**32))
    
    X_corrupt = X_flat.clone()
    X_sub = X_corrupt[:, noise_dims]
    mask = torch.rand(N, 1, device=device) < alpha
    noise = torch.randn_like(X_sub) * noise_scale + loc
    X_sub_noisy = torch.where(mask, X_sub + noise, X_sub)
    X_corrupt[:, noise_dims] = X_sub_noisy
    return X_corrupt

def build_camera_transform(rotation_deg, blur_sigma, perspective_distortion):
    transforms = []
    # Rotation
    if rotation_deg != 0:
        transforms.append(T.RandomAffine(degrees=rotation_deg, translate=(0.1, 0.1), scale=None))
    # Blur
    if blur_sigma > 0:
        transforms.append(T.GaussianBlur(kernel_size=3, sigma=(blur_sigma, blur_sigma)))
    # Angle (Perspective)
    if perspective_distortion > 0:
        transforms.append(T.RandomPerspective(distortion_scale=perspective_distortion, p=1.0))
        
    if len(transforms) == 0: return None
    return T.Compose(transforms)

def apply_camera_transform_cmnist(X_flat, transform, seed):
    if transform is None: return X_flat
    N, D = X_flat.shape
    X_img = X_flat.view(N, 3, 32, 32)
    X_shifted = torch.empty_like(X_img)
    base_seed = int(seed) % (2**32)
    
    for i in range(N):
        s = (base_seed + i) % (2**32)
        torch.manual_seed(s)
        X_shifted[i] = transform(X_img[i])
    return X_shifted.view(N, -1)


# ============================================================
# 3. Metrics
# ============================================================

def extract_z_pred_and_true(T_matrix, Dll_full, Dhl_full):
    T = torch.tensor(T_matrix, dtype=torch.float32) if not isinstance(T_matrix, torch.Tensor) else T_matrix
    X = torch.tensor(Dll_full, dtype=torch.float32) if not isinstance(Dll_full, torch.Tensor) else Dll_full
    Y = torch.tensor(Dhl_full, dtype=torch.float32) if not isinstance(Dhl_full, torch.Tensor) else Dhl_full
    
    device = torch.device("cpu")
    T = T.to(device)
    X = X.to(device)
    Y = Y.to(device)

    # Use robust slicing
    if T.shape[1] >= 3072:
        # Check if it has extra label rows (common issue)
        if T.shape[0] > 64 and T.shape[0] < 100: 
             # Likely (84, 3092) or similar -> slice off labels
             T_pix = T[20:, :3072]
        else:
             T_pix = T[:, :3072]
    else:
        T_pix = T

    d_z = T_pix.shape[0]

    # Handle X shape
    if X.shape[1] >= 3072:
        X_pix = X[:, :3072]
    else:
        raise ValueError(f"Unexpected Dll shape {tuple(X.shape)}")

    if Y.shape[1] == d_z:
        Z_true = Y
    elif Y.shape[1] >= 20 + d_z:
        Z_true = Y[:, -d_z:]
    else:
        raise ValueError(f"Unexpected Dhl shape {tuple(Y.shape)}")

    Z_pred = X_pix @ T_pix.T
    return Z_pred, Z_true

def compute_all_metrics(Z_pred, Z_true, eps=1e-9):
    diff = Z_pred - Z_true
    sq_l2 = diff.pow(2).sum(dim=1)
    sq_true = Z_true.pow(2).sum(dim=1)
    rel_l2 = (sq_l2 / (sq_true + eps)).mean().item() * 100.0
    return {"rel_l2": float(rel_l2)}


# ============================================================
# 4. Evaluation Loops
# ============================================================

def method_display_name(g, r):
    if g.startswith("diroca"): return f"DiRoCA ({r})"
    if g == "abslingam": return r
    return g.replace("_", " ").title()

def evaluate_huber_noshift(
    all_results, omega, Dll_samples, Dhl_samples, N_TRIALS=5, noise_scale=0.5
):
    shift_type = "huber"
    alphas = [0.0]
    records = []
    
    count = 0
    for mg, folds in all_results.items():
        if isinstance(folds, dict):
            for fk, fd in folds.items():
                if isinstance(fd, dict) and "error" not in fd: count += len(fd)
    
    pbar = tqdm(total=count * len(alphas) * N_TRIALS, desc="Evaluating (No Shift)")
    LL_PIXELS = slice(0, 3072)

    for method_group_key, method_results_inner in all_results.items():
        if not isinstance(method_results_inner, dict): continue
        for fold_key, fold_data in method_results_inner.items():
            if not fold_key.startswith("fold_") or not isinstance(fold_data, dict) or "error" in fold_data: continue
            fold_idx = int(fold_key.split("_")[-1])

            for run_key, run_result in fold_data.items():
                if "error" in run_result or run_result.get("T_matrix") is None:
                    pbar.update(len(alphas) * N_TRIALS); continue

                T_matrix = run_result["T_matrix"]
                test_idx = run_result["test_indices"]
                method_name = method_display_name(method_group_key, run_key)

                for alpha in alphas:
                    ns = 0.0 
                    for trial in range(N_TRIALS):
                        metrics_accumulator = {"rel_l2": []}
                        for iota, eta in omega.items():
                            try:
                                if iota not in Dll_samples or eta not in Dhl_samples: continue
                                ll_imgs, _, ll_digits, ll_colors = Dll_samples[iota]
                                Dhl_clean = Dhl_samples[eta]

                                ll_imgs_01 = ll_imgs[test_idx]
                                ll_digits_test = ll_digits[test_idx]
                                ll_colors_test = ll_colors[test_idx]
                                Dhl_clean_test = Dhl_clean[test_idx]

                                ll_imgs_flat = (ll_imgs_01 * 2 - 1).view(ll_imgs_01.shape[0], -1)
                                seed = hash((fold_idx, run_key, float(alpha), trial, iota, "huber")) % (2**32)
                                
                                ll_imgs_shift = apply_huber_contamination_cmnist(
                                    ll_imgs_flat, alpha=float(alpha), noise_scale=ns,
                                    noise_dims=LL_PIXELS, seed=seed
                                )

                                Dll_full = torch.cat([ll_imgs_shift, F.one_hot(ll_digits_test, 10).float(), F.one_hot(ll_colors_test, 10).float()], dim=1)
                                Z_pred, Z_true = extract_z_pred_and_true(T_matrix, Dll_full, Dhl_clean_test)
                                val = compute_all_metrics(Z_pred, Z_true)["rel_l2"]
                                metrics_accumulator["rel_l2"].append(val)
                            except: pass

                        avg_val = float(np.nanmean(metrics_accumulator["rel_l2"])) if metrics_accumulator["rel_l2"] else np.nan
                        records.append({
                            "shift_type": shift_type, "metric": "rel_l2", "method": method_name,
                            "fold": fold_idx, "alpha": float(alpha), "trial": trial, "value": avg_val
                        })
                        pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)


def evaluate_camera_param_sweep(
    all_results, omega, Dll_samples, Dhl_samples, N_TRIALS=5, camera_levels=None
):
    shift_type = "camera"
    param_names = ["rotation", "blur", "angle"]
    levels = ["low", "high"]

    count = 0
    for mg, folds in all_results.items():
        if isinstance(folds, dict):
            for fk, fd in folds.items():
                if isinstance(fd, dict) and "error" not in fd: count += len(fd)
    
    pbar = tqdm(total=count * len(param_names) * len(levels) * N_TRIALS, desc="Evaluating (camera)")
    records = []

    for method_group_key, method_results_inner in all_results.items():
        if not isinstance(method_results_inner, dict): continue
        for fold_key, fold_data in method_results_inner.items():
            if not fold_key.startswith("fold_") or not isinstance(fold_data, dict) or "error" in fold_data: continue
            fold_idx = int(fold_key.split("_")[-1])

            for run_key, run_result in fold_data.items():
                if "error" in run_result or run_result.get("T_matrix") is None:
                    pbar.update(len(param_names) * len(levels) * N_TRIALS); continue

                T_matrix = run_result["T_matrix"]
                test_idx = run_result["test_indices"]
                method_name = method_display_name(method_group_key, run_key)

                for param_name in param_names:
                    for level in levels:
                        for trial in range(N_TRIALS):
                            # Defaults
                            rot_deg = 0; blur_sigma = 0; angle_distortion = 0
                            
                            if param_name == "rotation":
                                rot_deg = camera_levels["rotation"][level]
                            elif param_name == "blur":
                                blur_sigma = camera_levels["blur"][level]
                            elif param_name == "angle":
                                angle_distortion = camera_levels["angle"][level]

                            # Construct transform - FIXED CALL HERE
                            transform = build_camera_transform(
                                rotation_deg=rot_deg,
                                blur_sigma=blur_sigma,
                                perspective_distortion=angle_distortion,
                            )

                            metrics_accumulator = {"rel_l2": []}
                            for iota, eta in omega.items():
                                try:
                                    if iota not in Dll_samples or eta not in Dhl_samples: continue
                                    ll_imgs, _, ll_digits, ll_colors = Dll_samples[iota]
                                    Dhl_clean = Dhl_samples[eta]

                                    ll_imgs_flat = (ll_imgs[test_idx] * 2 - 1).view(ll_imgs[test_idx].shape[0], -1)
                                    seed = hash((fold_idx, run_key, param_name, level, trial, iota, "camera")) % (2**32)
                                    
                                    ll_imgs_shift = apply_camera_transform_cmnist(ll_imgs_flat, transform=transform, seed=seed)
                                    Dll_full = torch.cat([ll_imgs_shift, F.one_hot(ll_digits[test_idx], 10).float(), F.one_hot(ll_colors[test_idx], 10).float()], dim=1)
                                    
                                    Z_pred, Z_true = extract_z_pred_and_true(T_matrix, Dll_full, Dhl_clean[test_idx])
                                    val = compute_all_metrics(Z_pred, Z_true)["rel_l2"]
                                    metrics_accumulator["rel_l2"].append(val)
                                except: pass

                            avg_val = float(np.nanmean(metrics_accumulator["rel_l2"])) if metrics_accumulator["rel_l2"] else np.nan
                            records.append({
                                "shift_type": shift_type, "metric": "rel_l2", "method": method_name,
                                "fold": fold_idx, "camera_param": param_name, "camera_level": level, "value": avg_val
                            })
                            pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)


# ============================================================
# 5. PDF Reporting
# ============================================================

def make_pdf_report(df_h, df_c, out_pdf):
    with PdfPages(out_pdf) as pdf:
        # Huber Table
        if not df_h.empty:
            summ = df_h.groupby(["method"])["value"].agg(["mean", "std"]).reset_index()
            fig, ax = plt.subplots(figsize=(10, len(summ)/2+2)); ax.axis("off")
            ax.set_title("No Shift (Huber Alpha=0) - Relative L2 (%)")
            cols = ["Method", "Mean", "Std"]
            txt = []
            for _, row in summ.iterrows():
                txt.append([row["method"], f"{row['mean']:.2f}", f"{row['std']:.2f}" if not np.isnan(row["std"]) else "-"])
            t = ax.table(cellText=txt, colLabels=cols, loc="center"); t.scale(1, 1.5)
            pdf.savefig(fig); plt.close(fig)

        # Camera Tables
        if not df_c.empty:
            summ = df_c
            params = sorted(summ["camera_param"].unique())
            for p in params:
                # Group by method and level
                grp = summ[summ["camera_param"]==p].groupby(["method", "camera_level"])["value"].agg(["mean", "std"]).reset_index()
                
                fig, ax = plt.subplots(figsize=(10, len(grp)/2+2)); ax.axis("off")
                ax.set_title(f"Camera: {p.capitalize()} (Rel L2 %)")
                cols = ["Method", "Low", "High"]
                txt = []
                methods = sorted(grp["method"].unique())
                for m in methods:
                    row = [m]
                    for l in ["low", "high"]:
                        s = grp[(grp["method"]==m) & (grp["camera_level"]==l)]
                        if s.empty: row.append("-")
                        else: row.append(f"{s['mean'].item():.2f} ± {s['std'].item():.2f}")
                    txt.append(row)
                t = ax.table(cellText=txt, colLabels=cols, loc="center"); t.scale(1, 1.5)
                pdf.savefig(fig); plt.close(fig)
    print(f"Report saved: {out_pdf}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="data/cmnist/results_empirical")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--out-pdf", default="cmnist_final_corrected.pdf")
    
    # Hardcoded Parameters as requested
    p.add_argument("--camera-rotation-degrees", type=float, nargs=2, default=[10.0, 90.0])
    p.add_argument("--camera-blur-sigma", type=float, nargs=2, default=[0.1, 2.0])
    p.add_argument("--camera-angle-distortion", type=float, nargs=2, default=[0.05, 0.6])
    
    a = p.parse_args()
    
    res, folds, om, dll, dhl, _,_,_,_ = load_eval_state(a.eval_dir)
    
    lvls = ["low", "high"]
    cam_lvls = {
        "rotation": {l: a.camera_rotation_degrees[i] for i,l in enumerate(lvls)},
        "blur": {l: a.camera_blur_sigma[i] for i,l in enumerate(lvls)},
        "angle": {l: a.camera_angle_distortion[i] for i,l in enumerate(lvls)},
    }
    
    print("Evaluating No Shift...")
    df_h = evaluate_huber_noshift(res, om, dll, dhl, a.n_trials, 0.5)
    
    print("Evaluating Camera...")
    df_c = evaluate_camera_param_sweep(res, om, dll, dhl, a.n_trials, cam_lvls)
    
    make_pdf_report(pd.DataFrame(df_h), pd.DataFrame(df_c), a.out_pdf)

if __name__ == "__main__":
    main()