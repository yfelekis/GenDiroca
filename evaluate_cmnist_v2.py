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
from scipy.stats import ttest_rel

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

    det_ll_dict_opt, det_hl_dict_opt = None, None
    U_ll_hat_opt, U_hl_hat_opt = None, None
    
    print("All eval state loaded.\n")
    return all_results, cv_folds, omega, Dll_samples, Dhl_samples, det_ll_dict_opt, det_hl_dict_opt, U_ll_hat_opt, U_hl_hat_opt


# ============================================================
# 2. Transforms
# ============================================================

def apply_huber_contamination_cmnist(X_flat, alpha, noise_scale, noise_dims, seed, loc=0.0):
    if alpha <= 0 or noise_scale <= 0:
        return X_flat
    device = X_flat.device
    N, D = X_flat.shape
    
    torch.manual_seed(int(seed) % (2**32))
    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed) % (2**32))

    X_corrupt = X_flat.clone()
    X_sub = X_corrupt[:, noise_dims]
    mask = torch.rand(N, 1, device=device) < alpha
    noise = torch.randn_like(X_sub) * noise_scale + loc
    X_sub_noisy = torch.where(mask, X_sub + noise, X_sub)
    X_corrupt[:, noise_dims] = X_sub_noisy
    return X_corrupt

def build_camera_transform(rotation_deg, zoom_range, brightness, contrast, saturation, blur_sigma, perspective_distortion):
    transforms = []
    if rotation_deg != 0 or zoom_range is not None:
        if zoom_range is None: zoom_range = (1.0, 1.0)
        transforms.append(T.RandomAffine(degrees=rotation_deg, translate=(0.1, 0.1), scale=zoom_range))
    if any([brightness, contrast, saturation]):
        transforms.append(T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation))
    if blur_sigma > 0:
        transforms.append(T.GaussianBlur(kernel_size=3, sigma=(blur_sigma, blur_sigma)))
    if perspective_distortion > 0:
        transforms.append(T.RandomPerspective(distortion_scale=perspective_distortion, p=1.0))
    return T.Compose(transforms) if transforms else None

def apply_camera_transform_cmnist(X_flat, transform, seed):
    if transform is None: return X_flat
    N, D = X_flat.shape
    X_img = X_flat.view(N, 3, 32, 32)
    X_shifted = torch.empty_like(X_img)
    base_seed = int(seed) % (2**32)
    
    for i in range(N):
        s = (base_seed + i) % (2**32)
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        X_shifted[i] = transform(X_img[i])
    return X_shifted.view(N, -1)


# ============================================================
# 3. Metrics
# ============================================================

def extract_z_pred_and_true(T_matrix, Dll_full, Dhl_full):
    T = torch.tensor(T_matrix, dtype=torch.float32) if not isinstance(T_matrix, torch.Tensor) else T_matrix
    X = torch.tensor(Dll_full, dtype=torch.float32) if not isinstance(Dll_full, torch.Tensor) else Dll_full
    Y = torch.tensor(Dhl_full, dtype=torch.float32) if not isinstance(Dhl_full, torch.Tensor) else Dhl_full
    
    if T.shape[1] >= 3072: T_pix = T[:, :3072]
    else: T_pix = T
    
    d_z = T_pix.shape[0]
    X_pix = X[:, :3072]
    
    if Y.shape[1] == d_z: Z_true = Y
    elif Y.shape[1] >= 20 + d_z: Z_true = Y[:, -d_z:]
    else: raise ValueError(f"Shape mismatch: Y={Y.shape}, d_z={d_z}")

    Z_pred = X_pix @ T_pix.T
    return Z_pred, Z_true

def compute_rel_l2(Z_pred, Z_true, eps=1e-9):
    diff = Z_pred - Z_true
    sq_l2 = diff.pow(2).sum(dim=1)
    sq_true = Z_true.pow(2).sum(dim=1)
    return (sq_l2 / (sq_true + eps)).mean().item() * 100.0


# ============================================================
# 4. Evaluation
# ============================================================

def method_display_name(g, r):
    if g.startswith("diroca"): return f"DiRoCA ({r})"
    if g == "abslingam": return r
    return g.replace("_", " ").title()

def evaluate_huber(all_results, omega, Dll_samples, Dhl_samples, N_TRIALS, ALPHA_VALUES, noise_scale):
    records = []
    count = sum(len(fd) for folds in all_results.values() if isinstance(folds, dict) for fd in folds.values() if isinstance(fd, dict))
    pbar = tqdm(total=count * len(ALPHA_VALUES) * N_TRIALS, desc="Evaluating (huber)")
    
    for mg, folds in all_results.items():
        if not isinstance(folds, dict): continue
        for fk, fd in folds.items():
            if not fk.startswith("fold_") or "error" in fd: continue
            fold_idx = int(fk.split("_")[-1])
            for rk, res in fd.items():
                if "error" in res or res.get("T_matrix") is None:
                    pbar.update(len(ALPHA_VALUES) * N_TRIALS); continue
                
                T_mat = res["T_matrix"]
                test_idx = res["test_indices"]
                m_name = method_display_name(mg, rk)
                
                for alpha in ALPHA_VALUES:
                    ns = 0.0 if np.isclose(alpha, 0.0) else noise_scale
                    for t in range(N_TRIALS):
                        vals = []
                        for iota, eta in omega.items():
                            try:
                                if iota not in Dll_samples: continue
                                ll_imgs, _, ll_d, ll_c = Dll_samples[iota]
                                Dhl = Dhl_samples[eta]
                                
                                X = ll_imgs[test_idx] * 2 - 1
                                X_flat = X.view(X.shape[0], -1)
                                seed = hash((fold_idx, rk, alpha, t, iota)) % (2**32)
                                
                                X_corr = apply_huber_contamination_cmnist(X_flat, alpha, ns, slice(0, 3072), seed)
                                Dll = torch.cat([X_corr, F.one_hot(ll_d[test_idx], 10).float(), F.one_hot(ll_c[test_idx], 10).float()], dim=1)
                                
                                Zp, Zt = extract_z_pred_and_true(T_mat, Dll, Dhl[test_idx])
                                vals.append(compute_rel_l2(Zp, Zt))
                            except: pass
                        
                        avg = np.mean(vals) if vals else np.nan
                        records.append({"method": m_name, "fold": fold_idx, "alpha": alpha, "rel_l2": avg})
                        pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)

def evaluate_camera(all_results, omega, Dll_samples, Dhl_samples, N_TRIALS, camera_levels, param_names):
    records = []
    # ONLY Low and High levels
    levels = ["low", "high"] 
    count = sum(len(fd) for folds in all_results.values() if isinstance(folds, dict) for fd in folds.values() if isinstance(fd, dict))
    pbar = tqdm(total=count * len(param_names) * len(levels) * N_TRIALS, desc="Evaluating (camera)")
    
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
                        rot = camera_levels["rotation"][l] if p=="rotation" else 0
                        zm = camera_levels["zoom"][l] if p=="zoom" else None
                        bl = camera_levels["blur"][l] if p=="blur" else 0
                        ang = camera_levels["angle"][l] if p=="angle" else 0
                        tf = build_camera_transform(rot, zm, 0, 0, 0, bl, ang)
                        
                        for t in range(N_TRIALS):
                            vals = []
                            for iota, eta in omega.items():
                                try:
                                    if iota not in Dll_samples: continue
                                    ll_imgs, _, ll_d, ll_c = Dll_samples[iota]
                                    Dhl = Dhl_samples[eta]
                                    
                                    X = ll_imgs[test_idx] * 2 - 1
                                    X_flat = X.view(X.shape[0], -1)
                                    seed = hash((fold_idx, rk, p, l, t, iota)) % (2**32)
                                    
                                    X_sh = apply_camera_transform_cmnist(X_flat, tf, seed)
                                    Dll = torch.cat([X_sh, F.one_hot(ll_d[test_idx], 10).float(), F.one_hot(ll_c[test_idx], 10).float()], dim=1)
                                    
                                    Zp, Zt = extract_z_pred_and_true(T_mat, Dll, Dhl[test_idx])
                                    vals.append(compute_rel_l2(Zp, Zt))
                                except: pass
                            
                            avg = np.mean(vals) if vals else np.nan
                            records.append({"method": m_name, "fold": fold_idx, "param": p, "level": l, "rel_l2": avg})
                            pbar.update(1)
    pbar.close()
    return pd.DataFrame(records)


# ============================================================
# 5. Reporting
# ============================================================

def get_sig_map(df, methods, best):
    if not best or best == "N/A": return {}
    sig_map = {}
    best_folds = df[df["method"] == best].groupby("fold")["rel_l2"].mean()
    
    for m in methods:
        if m == best: 
            sig_map[m] = False; continue
        curr = df[df["method"] == m].groupby("fold")["rel_l2"].mean()
        common = best_folds.index.intersection(curr.index)
        if len(common) < 2: 
            sig_map[m] = False; continue
        
        try:
            stat, p = ttest_rel(curr.loc[common], best_folds.loc[common])
            sig_map[m] = (p < 0.05 and stat > 0)
        except: sig_map[m] = False
    return sig_map

def make_pdf(df_h, df_c, out_path, Dll_samples, cam_lvls, ns, params):
    iota = list(Dll_samples.keys())[0]
    imgs = Dll_samples[iota][0][:5]
    imgs_flat = (imgs * 2 - 1).view(5, -1)
    
    # Huber Vis
    h_flat = apply_huber_contamination_cmnist(imgs_flat, 1.0, ns, slice(0, 3072), 42)
    h_imgs = (h_flat.view(5, 3, 32, 32) + 1) / 2
    
    fig1, ax1 = plt.subplots(2, 5, figsize=(10, 4))
    fig1.suptitle("Visual Check: Huber (Alpha=1.0)", y=1.02)
    for i in range(5):
        ax1[0, i].imshow(imgs[i].permute(1,2,0).clamp(0,1)); ax1[0, i].axis("off")
        ax1[1, i].imshow(h_imgs[i].permute(1,2,0).clamp(0,1)); ax1[1, i].axis("off")
    ax1[0,0].set_title("Original"); ax1[1,0].set_title("Huber")
    plt.tight_layout()

    # Camera Vis
    # ONLY Low and High
    lvls = ["low", "high"] 
    rows = 1 + len(params) * 2
    fig2, ax2 = plt.subplots(rows, 5, figsize=(10, rows*1.5))
    fig2.suptitle("Visual Check: Camera Shifts (Low vs High)", y=1.005)
    for i in range(5):
        ax2[0, i].imshow(imgs[i].permute(1,2,0).clamp(0,1)); ax2[0, i].axis("off")
    ax2[0,0].text(-0.5, 0.5, "Original", transform=ax2[0,0].transAxes, ha="right")
    
    r = 1
    for p in params:
        for l in lvls:
            rot = cam_lvls["rotation"][l] if p=="rotation" else 0
            zm = cam_lvls["zoom"][l] if p=="zoom" else None
            bl = cam_lvls["blur"][l] if p=="blur" else 0
            ang = cam_lvls["angle"][l] if p=="angle" else 0
            tf = build_camera_transform(rot, zm, 0, 0, 0, bl, ang)
            
            c_flat = apply_camera_transform_cmnist(imgs_flat, tf, 42)
            c_imgs = (c_flat.view(5, 3, 32, 32) + 1) / 2
            
            for i in range(5):
                ax2[r, i].imshow(c_imgs[i].permute(1,2,0).clamp(0,1)); ax2[r, i].axis("off")
            ax2[r, 0].text(-0.5, 0.5, f"{p} ({l})", transform=ax2[r,0].transAxes, ha="right")
            r += 1
    plt.tight_layout()

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig1); pdf.savefig(fig2)
        plt.close(fig1); plt.close(fig2)
        
        if not df_h.empty:
            summ = df_h.groupby(["method", "alpha"])["rel_l2"].agg(["mean", "std"]).reset_index()
            alphas = sorted(summ["alpha"].unique())
            fig, ax = plt.subplots(figsize=(10, len(summ)/2 + 2)); ax.axis("off")
            ax.set_title("Huber Noise Results")
            cols = ["Method"] + [f"A={a}" for a in alphas]
            txt = []
            
            methods = sorted(summ["method"].unique())
            best_map = {}
            for a in alphas:
                sub = summ[summ["alpha"] == a]
                if sub.empty or sub["mean"].isna().all(): continue
                b = sub.loc[sub["mean"].idxmin(), "method"]
                best_map[a] = (b, get_sig_map(df_h[df_h["alpha"]==a], methods, b))
                
            for m in methods:
                row = [m]
                for a in alphas:
                    s = summ[(summ["method"]==m) & (summ["alpha"]==a)]
                    if s.empty or pd.isna(s["mean"].item()): row.append("nan"); continue
                    val = f"{s['mean'].item():.2f}"
                    bm, sm = best_map.get(a, (None, {}))
                    if m == bm: val = f"** {val} **"
                    if sm.get(m, False): val += " *"
                    row.append(val)
                txt.append(row)
            t = ax.table(cellText=txt, colLabels=cols, loc="center")
            t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.5)
            pdf.savefig(fig); plt.close(fig)

        if not df_c.empty:
            actual = [p for p in params if p in df_c["param"].unique()]
            for p in actual:
                summ = df_c[df_c["param"]==p].groupby(["method", "level"])["rel_l2"].agg(["mean", "std"]).reset_index()
                fig, ax = plt.subplots(figsize=(10, len(summ)/2 + 2)); ax.axis("off")
                ax.set_title(f"Camera: {p}")
                cols = ["Method", "Low", "High"] # Only Low/High
                txt = []
                
                best_map = {}
                for l in ["low", "high"]:
                    sub = summ[summ["level"]==l]
                    if sub.empty or sub["mean"].isna().all(): continue
                    b = sub.loc[sub["mean"].idxmin(), "method"]
                    best_map[l] = (b, get_sig_map(df_c[(df_c["param"]==p) & (df_c["level"]==l)], methods, b))
                
                for m in methods:
                    row = [m]
                    for l in ["low", "high"]:
                        s = summ[(summ["method"]==m) & (summ["level"]==l)]
                        if s.empty or pd.isna(s["mean"].item()): row.append("nan"); continue
                        val = f"{s['mean'].item():.2f}"
                        bm, sm = best_map.get(l, (None, {}))
                        if m == bm: val = f"** {val} **"
                        if sm.get(m, False): val += " *"
                        row.append(val)
                    txt.append(row)
                t = ax.table(cellText=txt, colLabels=cols, loc="center")
                t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.5)
                pdf.savefig(fig); plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="data/cmnist/results_empirical")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--noise-scale-alpha1", type=float, default=0.5)
    p.add_argument("--out-pdf", default="data/cmnist/results_empirical/cmnist_final_report.pdf")
    p.add_argument("--camera-params", nargs="+", default=["angle", "rotation", "blur", "zoom"])
    
    # 2 values per param: Low, High
    p.add_argument("--camera-rotation-degrees", type=float, nargs=2, default=[10, 60])
    p.add_argument("--camera-zoom-min", type=float, nargs=2, default=[0.95, 0.80])
    p.add_argument("--camera-zoom-max", type=float, nargs=2, default=[1.05, 1.20])
    p.add_argument("--camera-blur-sigma", type=float, nargs=2, default=[0.5, 2.0])
    p.add_argument("--camera-angle-distortion", type=float, nargs=2, default=[0.1, 0.4])
    
    a = p.parse_args()
    
    res, folds, om, dll, dhl, _,_,_,_ = load_eval_state(a.eval_dir)
    
    # Map to only 2 levels
    lvls = ["low", "high"]
    cam_lvls = {
        "rotation": {l: a.camera_rotation_degrees[i] for i,l in enumerate(lvls)},
        "zoom": {l: (a.camera_zoom_min[i], a.camera_zoom_max[i]) for i,l in enumerate(lvls)},
        "blur": {l: a.camera_blur_sigma[i] for i,l in enumerate(lvls)},
        "angle": {l: a.camera_angle_distortion[i] for i,l in enumerate(lvls)},
    }
    
    print("Evaluating Huber...")
    df_h = evaluate_huber(res, om, dll, dhl, a.n_trials, [0.0, 1.0], a.noise_scale_alpha1)
    
    print(f"Evaluating Camera ({a.camera_params})...")
    df_c = evaluate_camera(res, om, dll, dhl, a.n_trials, cam_lvls, a.camera_params)
    
    print("Generating PDF...")
    make_pdf(df_h, df_c, a.out_pdf, dll, cam_lvls, a.noise_scale_alpha1, a.camera_params)

if __name__ == "__main__":
    main()