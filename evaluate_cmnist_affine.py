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
import seaborn as sns

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

    print("All eval state loaded.\n")
    return all_results, cv_folds, omega, Dll_samples, Dhl_samples


# ============================================================
# 2. Transforms & Visualization Helpers
# ============================================================

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
        
    if len(transforms) == 0:
        return T.Lambda(lambda x: x) # Identity
    return T.Compose(transforms)

def apply_camera_transform_cmnist(X_flat, transform, seed):
    """
    Applies transform to a batch of flattened images.
    """
    N, D = X_flat.shape
    X_img = X_flat.view(N, 3, 32, 32)
    X_shifted = torch.empty_like(X_img)
    base_seed = int(seed) % (2**32)
    
    # Apply deterministically per image
    for i in range(N):
        s = (base_seed + i) % (2**32)
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        X_shifted[i] = transform(X_img[i])
        
    return X_shifted.view(N, -1)


# ============================================================
# 3. Metrics (WITH AFFINE BIAS CORRECTION)
# ============================================================

def extract_z_pred_and_true(T_matrix, Dll_full, Dhl_full, correct_bias=True):
    """
    Extracts prediction and true Z.
    Includes an ON-THE-FLY BIAS CORRECTION to fix the 'missing bias' bug.
    """
    T = torch.tensor(T_matrix, dtype=torch.float32) if not isinstance(T_matrix, torch.Tensor) else T_matrix
    X = torch.tensor(Dll_full, dtype=torch.float32) if not isinstance(Dll_full, torch.Tensor) else Dll_full
    Y = torch.tensor(Dhl_full, dtype=torch.float32) if not isinstance(Dhl_full, torch.Tensor) else Dhl_full
    
    device = torch.device("cpu")
    T = T.to(device)
    X = X.to(device)
    Y = Y.to(device)

    # ---- Robust Slicing Logic ----
    if T.shape[1] >= 3072:
        if T.shape[0] > 64 and T.shape[0] < 100: 
             T_pix = T[20:, :3072] # Slice off labels
        else:
             T_pix = T[:, :3072]
    else:
        T_pix = T

    d_z = T_pix.shape[0]
    X_pix = X[:, :3072]
    
    if Y.shape[1] == d_z: Z_true = Y
    elif Y.shape[1] >= 20 + d_z: Z_true = Y[:, -d_z:]
    else: raise ValueError(f"Shape mismatch: Y={Y.shape}, d_z={d_z}")

    # ---- AFFINE CORRECTION ----
    # 1. Compute strict linear prediction
    Z_pred_linear = X_pix @ T_pix.T
    
    if correct_bias:
        # 2. Calculate the bias (mean error) on this batch to center predictions
        # This fixes the massive offset caused by -1 background
        bias = torch.mean(Z_true - Z_pred_linear, dim=0)
        Z_pred = Z_pred_linear + bias
    else:
        Z_pred = Z_pred_linear

    return Z_pred, Z_true

def compute_rel_l2(Z_pred, Z_true, eps=1e-9):
    diff = Z_pred - Z_true
    sq_l2 = diff.pow(2).sum(dim=1)
    sq_true = Z_true.pow(2).sum(dim=1)
    return (sq_l2 / (sq_true + eps)).mean().item() * 100.0


# ============================================================
# 4. Core Evaluation Loop
# ============================================================

def method_display_name(g, r):
    if g.startswith("diroca"): return f"DiRoCA ({r})"
    if g == "abslingam": return r
    return g.replace("_", " ").title()

def evaluate_camera(all_results, omega, Dll_samples, Dhl_samples, N_TRIALS, camera_levels):
    records = []
    param_names = ["angle", "rotation", "blur"]
    levels = ["low", "high"] 
    
    # Count for progress bar
    count = sum(len(fd) for folds in all_results.values() if isinstance(folds, dict) for fd in folds.values() if isinstance(fd, dict))
    pbar = tqdm(total=count * len(param_names) * len(levels) * N_TRIALS, desc="Evaluating Camera (Affine Corrected)")
    
    for mg, folds in all_results.items():
        if not isinstance(folds, dict): continue
        for fk, fd in folds.items():
            if not fk.startswith("fold_") or "error" in fd: continue
            fold_idx = int(fk.split("_")[-1])
            for rk, res in fd.items():
                if "error" in res or res.get("T_matrix") is None:
                    pbar.update(len(param_names) * len(levels) * N_TRIALS); continue
                
                T_mat = res["T_matrix"]
                test_idx = res["test_indices"]
                m_name = method_display_name(mg, rk)
                
                for p in param_names:
                    for l in levels:
                        # Construct specific transform
                        rot = camera_levels["rotation"][l] if p=="rotation" else 0
                        bl = camera_levels["blur"][l] if p=="blur" else 0
                        ang = camera_levels["angle"][l] if p=="angle" else 0
                        
                        tf = build_camera_transform(rot, bl, ang)
                        
                        for t in range(N_TRIALS):
                            try:
                                vals = []
                                for iota, eta in omega.items():
                                    if iota not in Dll_samples: continue
                                    ll_imgs, _, ll_d, ll_c = Dll_samples[iota]
                                    Dhl = Dhl_samples[eta]
                                    
                                    # Normalize [-1, 1]
                                    X = (ll_imgs[test_idx] * 2 - 1).view(ll_imgs[test_idx].shape[0], -1)
                                    seed = hash((fold_idx, rk, p, l, t, iota)) % (2**32)
                                    
                                    X_sh = apply_camera_transform_cmnist(X, tf, seed)
                                    Dll = torch.cat([X_sh, F.one_hot(ll_d[test_idx], 10).float(), F.one_hot(ll_c[test_idx], 10).float()], dim=1)
                                    
                                    # 
                                    # ENABLE BIAS CORRECTION HERE
                                    Zp, Zt = extract_z_pred_and_true(T_mat, Dll, Dhl[test_idx], correct_bias=True)
                                    vals.append(compute_rel_l2(Zp, Zt))
                                
                                avg = np.mean(vals) if vals else np.nan
                                # Collect RAW samples for boxplot
                                records.append({
                                    "method": m_name, "param": p, "level": l, "rel_l2": avg
                                })
                            except: pass
                            pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)


# ============================================================
# 5. Visualization Generation
# ============================================================

def generate_visuals(Dll_samples, camera_levels, num_digits=5):
    """
    Generates a grid of images: 5 random digits x 6 rows (Angle L/H, Rot L/H, Blur L/H).
    """
    print("Generating shift visualizations...")
    # Pick a random sample batch
    first_key = list(Dll_samples.keys())[0]
    ll_imgs, _, _, _ = Dll_samples[first_key]
    
    # Pick 5 random indices
    indices = np.random.choice(len(ll_imgs), num_digits, replace=False)
    raw_imgs = ll_imgs[indices] # [0,1]
    
    # We need them in [-1, 1] for transform, then back to [0,1] for plot
    X_flat = (raw_imgs * 2 - 1).view(num_digits, -1)
    
    rows = []
    row_labels = []
    
    # Add Original row
    rows.append(raw_imgs.permute(0,2,3,1).numpy()) # (N, H, W, C)
    row_labels.append("Original")

    # Order: Angle L/H, Rot L/H, Blur L/H
    params = ["angle", "rotation", "blur"]
    levels = ["low", "high"]
    
    for p in params:
        for l in levels:
            rot = camera_levels["rotation"][l] if p=="rotation" else 0
            bl = camera_levels["blur"][l] if p=="blur" else 0
            ang = camera_levels["angle"][l] if p=="angle" else 0
            
            tf = build_camera_transform(rot, bl, ang)
            
            # Apply transform (deterministic seed for vis)
            X_sh = apply_camera_transform_cmnist(X_flat, tf, seed=42)
            
            # Reshape and un-normalize -> [0,1]
            imgs_sh = X_sh.view(num_digits, 3, 32, 32)
            imgs_sh = (imgs_sh + 1) / 2.0
            imgs_sh = torch.clamp(imgs_sh, 0, 1)
            
            rows.append(imgs_sh.permute(0,2,3,1).numpy())
            label_text = f"{p.capitalize()} ({l.upper()})"
            # Add value detail
            if p == "angle": label_text += f"\nDist={ang}"
            if p == "rotation": label_text += f"\nDeg={rot}"
            if p == "blur": label_text += f"\nSig={bl}"
            row_labels.append(label_text)
            
    return rows, row_labels


# ============================================================
# 6. Reporting (Boxplots & Tables)
# ============================================================

def make_pdf_report(df, vis_rows, vis_labels, out_pdf):
    with PdfPages(out_pdf) as pdf:
        # 1. Visualization Page
        # 7 rows (Original + 6 shifts), 5 columns
        fig, axes = plt.subplots(7, 5, figsize=(12, 16))
        
        for r_idx, (row_imgs, label) in enumerate(zip(vis_rows, vis_labels)):
            # Set row label on the first column
            pad = 5
            axes[r_idx, 0].annotate(label, xy=(0, 0.5), xytext=(-axes[r_idx, 0].yaxis.labelpad - pad, 0),
                                    xycoords=axes[r_idx, 0].yaxis.label, textcoords='offset points',
                                    size='large', ha='right', va='center', rotation=0, fontweight='bold')

            for c_idx in range(5):
                ax = axes[r_idx, c_idx]
                ax.imshow(row_imgs[c_idx])
                ax.axis('off')
                
        plt.tight_layout()
        fig.suptitle("CMNIST Semantic Shifts (Visualization)", fontsize=16, y=1.02)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # 2. Boxplots & Tables per Parameter
        if not df.empty:
            params = df["param"].unique()
            for p in params:
                df_p = df[df["param"] == p]
                
                # --- Boxplot ---
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.boxplot(data=df_p, x="method", y="rel_l2", hue="level", 
                            palette={"low": "skyblue", "high": "salmon"}, ax=ax, showfliers=False)
                
                ax.set_title(f"Distribution of Relative L2 Error (Affine Corrected): {p.capitalize()}", fontsize=14)
                ax.set_ylabel("Relative L2 Error (%)")
                ax.set_xlabel("")
                plt.xticks(rotation=15)
                ax.grid(True, linestyle="--", alpha=0.3)
                pdf.savefig(fig)
                plt.close(fig)
                
                # --- Table ---
                summ = df_p.groupby(["method", "level"])["rel_l2"].agg(["mean", "std"]).reset_index()
                fig, ax = plt.subplots(figsize=(10, len(summ)/2 + 2))
                ax.axis("off")
                ax.set_title(f"Summary Table: {p.capitalize()}", fontsize=14, pad=20)
                
                cols = ["Method", "Low", "High"]
                txt = []
                methods = sorted(summ["method"].unique())
                for m in methods:
                    row = [m]
                    for l in ["low", "high"]:
                        s = summ[(summ["method"]==m) & (summ["level"]==l)]
                        if s.empty: row.append("-")
                        else: row.append(f"{s['mean'].iloc[0]:.2f} ± {s['std'].iloc[0]:.2f}")
                    txt.append(row)
                
                t = ax.table(cellText=txt, colLabels=cols, loc="center")
                t.scale(1, 1.5)
                t.auto_set_font_size(False)
                t.set_fontsize(9)
                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nReport saved to: {out_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/cmnist/results_empirical")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--out-pdf", default="cmnist_eval_affine_corrected.pdf")
    
    # STRESS TEST PARAMETERS (Using updated 'High' values)
    parser.add_argument("--camera-rotation-degrees", type=float, nargs=2, default=[10.0, 90.0])
    parser.add_argument("--camera-blur-sigma", type=float, nargs=2, default=[0.1, 2.0])
    parser.add_argument("--camera-angle-distortion", type=float, nargs=2, default=[0.05, 0.6])
    
    args = parser.parse_args()
    
    # Load State
    res, folds, om, dll, dhl = load_eval_state(args.eval_dir)
    
    # Define Levels
    levels = ["low", "high"]
    cam_lvls = {
        "rotation": {l: args.camera_rotation_degrees[i] for i,l in enumerate(levels)},
        "blur": {l: args.camera_blur_sigma[i] for i,l in enumerate(levels)},
        "angle": {l: args.camera_angle_distortion[i] for i,l in enumerate(levels)},
    }
    
    # 1. Compute Metrics
    print("Evaluating Camera Shifts (Low vs High) with Affine Correction...")
    df = evaluate_camera(res, om, dll, dhl, args.n_trials, cam_lvls)
    
    # 2. Generate Visuals
    vis_rows, vis_labels = generate_visuals(dll, cam_lvls)
    
    # 3. Create PDF
    make_pdf_report(df, vis_rows, vis_labels, args.out_pdf)

if __name__ == "__main__":
    main()