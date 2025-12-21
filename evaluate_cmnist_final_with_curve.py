#!/usr/bin/env python3
import argparse
import os
import pickle
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
import torchvision.transforms.functional as TF
from scipy.stats import ttest_rel

# ============================================================
# 1. Utility: Load State
# ============================================================
def load_eval_state(eval_dir):
    print(f"Loading CMNIST eval state from: {eval_dir}")
    with open(os.path.join(eval_dir, "cmnist_all_results.pkl"), "rb") as f:
        all_results = pickle.load(f)
    with open(os.path.join(eval_dir, "cmnist_omega.pkl"), "rb") as f:
        omega = pickle.load(f)
    with open(os.path.join(eval_dir, "cmnist_Dll_samples.pkl"), "rb") as f:
        Dll_samples = pickle.load(f)
    with open(os.path.join(eval_dir, "cmnist_Dhl_samples.pkl"), "rb") as f:
        Dhl_samples = pickle.load(f)
    print("State loaded.\n")
    return all_results, omega, Dll_samples, Dhl_samples

# ============================================================
# 2. Transforms
# ============================================================
def build_camera_transform(rotation_deg, brightness_factor):
    transforms = []
    if rotation_deg != 0:
        transforms.append(T.RandomAffine(
            degrees=(rotation_deg, rotation_deg), 
            translate=(0.1, 0.1), 
            scale=None, 
            fill=0 
        ))
    if brightness_factor != 1.0:
        transforms.append(T.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))
    if len(transforms) == 0:
        return T.Lambda(lambda x: x)
    return T.Compose(transforms)

def apply_transform(X_flat, transform, seed):
    N, D = X_flat.shape
    X_img = X_flat.view(N, 3, 32, 32)
    X_01 = (X_img + 1) * 0.5
    X_shifted = torch.empty_like(X_01)
    base_seed = int(seed) % (2**32)
    for i in range(N):
        s = (base_seed + i) % (2**32)
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        X_shifted[i] = transform(X_01[i])
    X_out = X_shifted * 2 - 1
    return torch.clamp(X_out, -1, 1).view(N, -1)

def apply_huber_contamination_cmnist(X_flat, alpha, noise_scale, seed):
    if alpha <= 0: return X_flat
    N, D = X_flat.shape
    base_seed = int(seed) % (2**32)
    torch.manual_seed(base_seed); np.random.seed(base_seed); random.seed(base_seed)
    mask = torch.rand(N, 1) < alpha
    noise = torch.randn_like(X_flat) * noise_scale
    X_corrupt = torch.where(mask, X_flat + noise, X_flat)
    return torch.clamp(X_corrupt, -1, 1)

# ============================================================
# 3. Metrics (Affine Bias Corrected)
# ============================================================
def extract_z_pred_and_true(T_matrix, Dll_full, Dhl_full, correct_bias=True):
    T = torch.tensor(T_matrix, dtype=torch.float32) if not isinstance(T_matrix, torch.Tensor) else T_matrix
    X = torch.tensor(Dll_full, dtype=torch.float32) if not isinstance(Dll_full, torch.Tensor) else Dll_full
    Y = torch.tensor(Dhl_full, dtype=torch.float32) if not isinstance(Dhl_full, torch.Tensor) else Dhl_full
    
    device = torch.device("cpu")
    T = T.to(device)
    X = X.to(device)
    Y = Y.to(device)

    if T.shape[1] >= 3072:
        if T.shape[0] > 64 and T.shape[0] < 100: T_pix = T[20:, :3072]
        else: T_pix = T[:, :3072]
    else: T_pix = T

    d_z = T_pix.shape[0]
    X_pix = X[:, :3072]
    if Y.shape[1] == d_z: Z_true = Y
    elif Y.shape[1] >= 20 + d_z: Z_true = Y[:, -d_z:]
    else: raise ValueError("Shape mismatch")

    Z_pred_linear = X_pix @ T_pix.T
    if correct_bias:
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
# 4. Statistical Analysis
# ============================================================
def print_statistical_summary(df, title="Statistical Summary"):
    print(f"\n{'='*20} {title} {'='*20}")
    
    # Handle different grouping keys
    if "level" in df.columns:
        df["sample_id"] = df["fold"].astype(str) + "_" + df["trial"].astype(str)
        groups = df.groupby(["param", "level"])
    elif "sigma" in df.columns:
        df["sample_id"] = df["fold"].astype(str) + "_" + df["trial"].astype(str)
        groups = df.groupby(["sigma"])
    elif "alpha" in df.columns:
        df["sample_id"] = df["fold"].astype(str) + "_" + df["trial"].astype(str)
        groups = df.groupby(["alpha"])
    else:
        return

    for group_key, group in groups:
        # Format header
        if isinstance(group_key, tuple):
            print(f"\n>>> {group_key[0].upper()} | {str(group_key[1]).upper()}")
        elif "sigma" in df.columns:
            print(f"\n>>> Sigma: {group_key:.2f}")
        elif "alpha" in df.columns:
            print(f"\n>>> Alpha: {group_key:.2f}")
            
        print("-" * 80)
        print(f"{'Method':<35} | {'Mean ± Std':<20} | {'vs Ref (p-value)':<15}")
        print("-" * 80)
        
        try:
            pivot_df = group.pivot(index="sample_id", columns="method", values="rel_l2")
            means = pivot_df.mean()
            stds = pivot_df.std()
            best_method = means.idxmin()
            
            for method in means.sort_values().index:
                m_val = means[method]
                s_val = stds[method]
                
                if method == best_method:
                    p_str = "(REF)"
                else:
                    try:
                        valid = pivot_df[[best_method, method]].dropna()
                        if len(valid) > 1:
                            _, p_val = ttest_rel(valid[best_method], valid[method])
                            p_str = f"{p_val:.4e}{' *' if p_val < 0.05 else ''}"
                        else:
                            p_str = "N/A"
                    except: p_str = "Err"

                prefix = ">> " if method == best_method else "   "
                print(f"{prefix}{method:<32} | {m_val:.2f} ± {s_val:.2f}   | {p_str}")
        except: pass

# ============================================================
# 5. Method Display Helper
# ============================================================
def method_display_name(g, r):
    if g.startswith("diroca"): return f"DiRoCA ({r})"
    if g == "abslingam": return r
    return g.replace("_", " ").title()

# ============================================================
# 6. Evaluation Loops
# ============================================================
def evaluate_camera(all_results, omega, Dll_samples, Dhl_samples, N_TRIALS, levels_dict):
    records = []
    param_names = ["rotation", "lighting"]
    levels = ["low", "high"]
    
    count = sum(len(fd) for folds in all_results.values() if isinstance(folds, dict) for fd in folds.values() if isinstance(fd, dict))
    pbar = tqdm(total=count * len(param_names) * len(levels) * N_TRIALS, desc="Eval Camera")
    
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
                        rot = levels_dict["rotation"][l] if p=="rotation" else 0
                        bright = levels_dict["lighting"][l] if p=="lighting" else 1.0
                        tf = build_camera_transform(rot, bright)
                        
                        for t in range(N_TRIALS):
                            try:
                                vals = []
                                for iota, eta in omega.items():
                                    if iota not in Dll_samples: continue
                                    ll_imgs, _, ll_d, ll_c = Dll_samples[iota]
                                    Dhl = Dhl_samples[eta]
                                    X = (ll_imgs[test_idx] * 2 - 1).view(ll_imgs[test_idx].shape[0], -1)
                                    seed = hash((fold_idx, p, l, t, iota)) % (2**32)
                                    X_sh = apply_transform(X, tf, seed)
                                    Dll = torch.cat([X_sh, F.one_hot(ll_d[test_idx], 10).float(), F.one_hot(ll_c[test_idx], 10).float()], dim=1)
                                    Zp, Zt = extract_z_pred_and_true(T_mat, Dll, Dhl[test_idx], correct_bias=True)
                                    vals.append(compute_rel_l2(Zp, Zt))
                                avg = np.mean(vals) if vals else np.nan
                                records.append({"method": m_name, "param": p, "level": l, "trial": t, "fold": fold_idx, "rel_l2": avg})
                            except: pass
                            pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)

def evaluate_huber_points(all_results, omega, Dll_samples, Dhl_samples, N_TRIALS):
    records = []
    configs = [(0.0, 0.0, "clean"), (1.0, 0.2, "noisy_0.2"), (1.0, 0.5, "noisy_0.5")]
    count = sum(len(fd) for folds in all_results.values() if isinstance(folds, dict) for fd in folds.values() if isinstance(fd, dict))
    pbar = tqdm(total=count * len(configs) * N_TRIALS, desc="Eval Huber Points")
    
    for mg, folds in all_results.items():
        if not isinstance(folds, dict): continue
        for fk, fd in folds.items():
            if not fk.startswith("fold_") or "error" in fd: continue
            fold_idx = int(fk.split("_")[-1])
            for rk, res in fd.items():
                if "error" in res or res.get("T_matrix") is None:
                    pbar.update(len(configs) * N_TRIALS); continue
                
                T_mat = res["T_matrix"]
                test_idx = res["test_indices"]
                m_name = method_display_name(mg, rk)
                
                for alpha, scale, level_name in configs:
                    for t in range(N_TRIALS):
                        try:
                            vals = []
                            for iota, eta in omega.items():
                                if iota not in Dll_samples: continue
                                ll_imgs, _, ll_d, ll_c = Dll_samples[iota]
                                Dhl = Dhl_samples[eta]
                                X = (ll_imgs[test_idx] * 2 - 1).view(ll_imgs[test_idx].shape[0], -1)
                                seed = hash((fold_idx, "huber_pt", level_name, t, iota)) % (2**32)
                                X_sh = apply_huber_contamination_cmnist(X, alpha=alpha, noise_scale=scale, seed=seed)
                                Dll = torch.cat([X_sh, F.one_hot(ll_d[test_idx], 10).float(), F.one_hot(ll_c[test_idx], 10).float()], dim=1)
                                Zp, Zt = extract_z_pred_and_true(T_mat, Dll, Dhl[test_idx], correct_bias=True)
                                vals.append(compute_rel_l2(Zp, Zt))
                            avg = np.mean(vals) if vals else np.nan
                            records.append({"method": m_name, "param": "huber", "level": level_name, "trial": t, "fold": fold_idx, "rel_l2": avg})
                        except: pass
                        pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)

def evaluate_sigma_curve(all_results, omega, Dll_samples, Dhl_samples, N_TRIALS):
    """Vary Sigma (0.0->1.0), Fix Alpha=1.0"""
    records = []
    sigmas = np.linspace(0.0, 1.0, 10)
    count = sum(len(fd) for folds in all_results.values() if isinstance(folds, dict) for fd in folds.values() if isinstance(fd, dict))
    pbar = tqdm(total=count * len(sigmas) * N_TRIALS, desc="Eval Sigma Curve")
    
    for mg, folds in all_results.items():
        if not isinstance(folds, dict): continue
        for fk, fd in folds.items():
            if not fk.startswith("fold_") or "error" in fd: continue
            fold_idx = int(fk.split("_")[-1])
            for rk, res in fd.items():
                if "error" in res or res.get("T_matrix") is None:
                    pbar.update(len(sigmas) * N_TRIALS); continue
                
                T_mat = res["T_matrix"]
                test_idx = res["test_indices"]
                m_name = method_display_name(mg, rk)
                
                for sigma in sigmas:
                    for t in range(N_TRIALS):
                        try:
                            vals = []
                            for iota, eta in omega.items():
                                if iota not in Dll_samples: continue
                                ll_imgs, _, ll_d, ll_c = Dll_samples[iota]
                                Dhl = Dhl_samples[eta]
                                X = (ll_imgs[test_idx] * 2 - 1).view(ll_imgs[test_idx].shape[0], -1)
                                seed = hash((fold_idx, "sigma_curve", sigma, t, iota)) % (2**32)
                                # Alpha=1.0, Vary Sigma
                                X_sh = apply_huber_contamination_cmnist(X, alpha=1.0, noise_scale=sigma, seed=seed)
                                Dll = torch.cat([X_sh, F.one_hot(ll_d[test_idx], 10).float(), F.one_hot(ll_c[test_idx], 10).float()], dim=1)
                                Zp, Zt = extract_z_pred_and_true(T_mat, Dll, Dhl[test_idx], correct_bias=True)
                                vals.append(compute_rel_l2(Zp, Zt))
                            avg = np.mean(vals) if vals else np.nan
                            records.append({"method": m_name, "sigma": sigma, "trial": t, "fold": fold_idx, "rel_l2": avg})
                        except: pass
                        pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)

def evaluate_alpha_curve(all_results, omega, Dll_samples, Dhl_samples, N_TRIALS):
    """Vary Alpha (0.0->1.0), Fix Sigma=0.5"""
    records = []
    alphas = np.linspace(0.0, 1.0, 10)
    count = sum(len(fd) for folds in all_results.values() if isinstance(folds, dict) for fd in folds.values() if isinstance(fd, dict))
    pbar = tqdm(total=count * len(alphas) * N_TRIALS, desc="Eval Alpha Curve")
    
    for mg, folds in all_results.items():
        if not isinstance(folds, dict): continue
        for fk, fd in folds.items():
            if not fk.startswith("fold_") or "error" in fd: continue
            fold_idx = int(fk.split("_")[-1])
            for rk, res in fd.items():
                if "error" in res or res.get("T_matrix") is None:
                    pbar.update(len(alphas) * N_TRIALS); continue
                
                T_mat = res["T_matrix"]
                test_idx = res["test_indices"]
                m_name = method_display_name(mg, rk)
                
                for alpha in alphas:
                    for t in range(N_TRIALS):
                        try:
                            vals = []
                            for iota, eta in omega.items():
                                if iota not in Dll_samples: continue
                                ll_imgs, _, ll_d, ll_c = Dll_samples[iota]
                                Dhl = Dhl_samples[eta]
                                X = (ll_imgs[test_idx] * 2 - 1).view(ll_imgs[test_idx].shape[0], -1)
                                seed = hash((fold_idx, "alpha_curve", alpha, t, iota)) % (2**32)
                                # Vary Alpha, Sigma=0.5
                                X_sh = apply_huber_contamination_cmnist(X, alpha=alpha, noise_scale=0.5, seed=seed)
                                Dll = torch.cat([X_sh, F.one_hot(ll_d[test_idx], 10).float(), F.one_hot(ll_c[test_idx], 10).float()], dim=1)
                                Zp, Zt = extract_z_pred_and_true(T_mat, Dll, Dhl[test_idx], correct_bias=True)
                                vals.append(compute_rel_l2(Zp, Zt))
                            avg = np.mean(vals) if vals else np.nan
                            records.append({"method": m_name, "alpha": alpha, "trial": t, "fold": fold_idx, "rel_l2": avg})
                        except: pass
                        pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)

# ============================================================
# 7. Visualization & Reporting
# ============================================================
def generate_visuals(Dll_samples, levels_dict, num_digits=5):
    print("Generating visuals...")
    first_key = list(Dll_samples.keys())[0]
    ll_imgs, _, _, _ = Dll_samples[first_key]
    indices = np.random.choice(len(ll_imgs), num_digits, replace=False)
    raw_imgs = ll_imgs[indices]
    X_flat = (raw_imgs * 2 - 1).view(num_digits, -1)
    
    rows = []
    row_labels = []
    
    rows.append(raw_imgs.permute(0,2,3,1).numpy())
    row_labels.append("Original\n(Huber 0)")

    params = ["rotation", "lighting"]
    levels = ["low", "high"]
    
    for p in params:
        for l in levels:
            rot = levels_dict["rotation"][l] if p=="rotation" else 0
            bright = levels_dict["lighting"][l] if p=="lighting" else 1.0
            tf = build_camera_transform(rot, bright)
            X_sh = apply_transform(X_flat, tf, seed=42)
            imgs_sh = X_sh.view(num_digits, 3, 32, 32)
            imgs_sh = (imgs_sh + 1) / 2.0
            imgs_sh = torch.clamp(imgs_sh, 0, 1)
            rows.append(imgs_sh.permute(0,2,3,1).numpy())
            label = f"{p.capitalize()} ({l.upper()})"
            if p=="rotation": label += f"\n{rot} deg"
            if p=="lighting": label += f"\nBright={bright}"
            row_labels.append(label)

    # Huber 0.2
    X_hub_low = apply_huber_contamination_cmnist(X_flat, alpha=1.0, noise_scale=0.2, seed=42)
    imgs_hub_low = X_hub_low.view(num_digits, 3, 32, 32)
    imgs_hub_low = (imgs_hub_low + 1) / 2.0
    imgs_hub_low = torch.clamp(imgs_hub_low, 0, 1)
    rows.append(imgs_hub_low.permute(0,2,3,1).numpy())
    row_labels.append("Huber\n(σ=0.2)")

    # Huber 0.5
    X_hub_high = apply_huber_contamination_cmnist(X_flat, alpha=1.0, noise_scale=0.5, seed=42)
    imgs_hub_high = X_hub_high.view(num_digits, 3, 32, 32)
    imgs_hub_high = (imgs_hub_high + 1) / 2.0
    imgs_hub_high = torch.clamp(imgs_hub_high, 0, 1)
    rows.append(imgs_hub_high.permute(0,2,3,1).numpy())
    row_labels.append("Huber\n(σ=0.5)")
            
    return rows, row_labels

def make_pdf_report(df_cam, df_hub, df_sigma, df_alpha, vis_rows, vis_labels, out_pdf):
    with PdfPages(out_pdf) as pdf:
        # 1. Visuals
        num_rows = len(vis_rows)
        fig, axes = plt.subplots(num_rows, 5, figsize=(12, 2 * num_rows))
        for r, (imgs, lbl) in enumerate(zip(vis_rows, vis_labels)):
            axes[r,0].annotate(lbl, xy=(0, 0.5), xytext=(-5, 0), xycoords="axes fraction", textcoords="offset points", ha='right', va='center', fontweight='bold')
            for c in range(5):
                axes[r,c].imshow(imgs[c]); axes[r,c].axis('off')
        plt.tight_layout()
        fig.suptitle("CMNIST Shift Visuals", y=1.01, fontsize=16)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        
        # 2. Camera Boxplots
        if not df_cam.empty:
            for p in df_cam["param"].unique():
                df_p = df_cam[df_cam["param"] == p]
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df_p, x="method", y="rel_l2", hue="level", palette="pastel", showfliers=False, ax=ax)
                ax.set_title(f"Relative Error: {p.capitalize()}")
                plt.xticks(rotation=15); ax.grid(True, alpha=0.3)
                pdf.savefig(fig); plt.close(fig)

        # 3. Huber Boxplots
        if not df_hub.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            hue_order = sorted(df_hub["level"].unique())
            sns.boxplot(data=df_hub, x="method", y="rel_l2", hue="level", hue_order=hue_order, palette="pastel", showfliers=False, ax=ax)
            ax.set_title(f"Relative Error: Huber (Points)")
            plt.xticks(rotation=15); ax.grid(True, alpha=0.3)
            pdf.savefig(fig); plt.close(fig)

        # 4. Curve: Error vs Sigma (All Methods)
        if not df_sigma.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df_sigma, x="sigma", y="rel_l2", hue="method", marker="o", ax=ax)
            ax.set_title("Robustness: Error vs Noise Scale (α=1.0)")
            ax.set_xlabel("Noise Scale (σ)"); ax.set_ylabel("Relative Error (%)")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # 5. Curve: Error vs Alpha (All Methods)
        if not df_alpha.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df_alpha, x="alpha", y="rel_l2", hue="method", marker="o", ax=ax)
            ax.set_title("Robustness: Error vs Contamination (σ=0.5)")
            ax.set_xlabel("Contamination Probability (α)"); ax.set_ylabel("Relative Error (%)")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # 6. Curve: Error vs Sigma (DiRoCA Only)
        if not df_sigma.empty:
            df_diroca = df_sigma[df_sigma['method'].str.contains("DiRoCA", case=False)]
            if not df_diroca.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df_diroca, x="sigma", y="rel_l2", hue="method", marker="o", ax=ax)
                ax.set_title("DiRoCA Robustness: Error vs Noise Scale (α=1.0)")
                ax.set_xlabel("Noise Scale (σ)"); ax.set_ylabel("Relative Error (%)")
                ax.grid(True, linestyle="--", alpha=0.5)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig); plt.close(fig)

        # 7. Curve: Error vs Alpha (DiRoCA Only)
        if not df_alpha.empty:
            df_diroca = df_alpha[df_alpha['method'].str.contains("DiRoCA", case=False)]
            if not df_diroca.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df_diroca, x="alpha", y="rel_l2", hue="method", marker="o", ax=ax)
                ax.set_title("DiRoCA Robustness: Error vs Contamination (σ=0.5)")
                ax.set_xlabel("Contamination Probability (α)"); ax.set_ylabel("Relative Error (%)")
                ax.grid(True, linestyle="--", alpha=0.5)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig); plt.close(fig)

    print(f"Saved PDF to {out_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="data/cmnist/results_empirical")
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()
    
    res, om, dll, dhl = load_eval_state(args.eval_dir)[:4]
    
    levels_dict = {
        "rotation": {"low": 30.0, "high": 90.0},
        "lighting": {"low": 0.6, "high": 0.2} 
    }
    
    print("Evaluating Fixed Shifts...")
    df_cam = evaluate_camera(res, om, dll, dhl, args.n_trials, levels_dict)
    
    print("Evaluating Huber Points...")
    df_hub = evaluate_huber_points(res, om, dll, dhl, args.n_trials)
    
    print("Evaluating Noise Curve (Vary Sigma)...")
    df_sigma = evaluate_sigma_curve(res, om, dll, dhl, args.n_trials)
    
    print("Evaluating Alpha Curve (Vary Alpha)...")
    df_alpha = evaluate_alpha_curve(res, om, dll, dhl, args.n_trials)
    
    print_statistical_summary(df_cam, "Camera Stats")
    print_statistical_summary(df_hub, "Huber Points Stats")
    
    vis_rows, vis_lbls = generate_visuals(dll, levels_dict)
    make_pdf_report(df_cam, df_hub, df_sigma, df_alpha, vis_rows, vis_lbls, "cmnist_results_robustness_curves.pdf")

if __name__ == "__main__":
    main()