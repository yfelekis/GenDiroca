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
# 1. Utility: load evaluation state saved by cmnist_opt script
# ============================================================

def load_eval_state(eval_dir):
    print(f"Loading CMNIST eval state from: {eval_dir}")

    # 1) all_results
    all_results_path = os.path.join(eval_dir, "cmnist_all_results.pkl")
    with open(all_results_path, "rb") as f:
        all_results = pickle.load(f)
    print(f"  ✓ Loaded all_results from {all_results_path}")

    # 2) cv_folds (not strictly needed for eval; useful for sanity)
    cv_folds_path = os.path.join(eval_dir, "cmnist_cv_folds.pkl")
    with open(cv_folds_path, "rb") as f:
        cv_folds = pickle.load(f)
    print(f"  ✓ Loaded cv_folds from {cv_folds_path}")

    # 3) omega
    omega_path = os.path.join(eval_dir, "cmnist_omega.pkl")
    with open(omega_path, "rb") as f:
        omega = pickle.load(f)
    print(f"  ✓ Loaded omega from {omega_path}")

    # 4) deterministic opt-view dicts (optional)
    det_ll_opt_path = os.path.join(eval_dir, "cmnist_det_ll_dict_opt.pt")
    det_hl_opt_path = os.path.join(eval_dir, "cmnist_det_hl_dict_opt.pt")
    if os.path.exists(det_ll_opt_path) and os.path.exists(det_hl_opt_path):
        det_ll_dict_opt = torch.load(det_ll_opt_path, map_location="cpu")
        det_hl_dict_opt = torch.load(det_hl_opt_path, map_location="cpu")
        print(f"  ✓ Loaded det_ll_dict_opt from {det_ll_opt_path}")
        print(f"  ✓ Loaded det_hl_dict_opt from {det_hl_opt_path}")
    else:
        det_ll_dict_opt, det_hl_dict_opt = None, None
        print("  ⚠ det_ll_dict_opt / det_hl_dict_opt not found; continuing without them.")

    # 5) opt-view noise (optional)
    U_ll_opt_path = os.path.join(eval_dir, "cmnist_U_ll_hat_opt.pt")
    U_hl_opt_path = os.path.join(eval_dir, "cmnist_U_hl_hat_opt.pt")
    if os.path.exists(U_ll_opt_path) and os.path.exists(U_hl_opt_path):
        U_ll_hat_opt = torch.load(U_ll_opt_path, map_location="cpu")
        U_hl_hat_opt = torch.load(U_hl_opt_path, map_location="cpu")
        print(f"  ✓ Loaded U_ll_hat_opt from {U_ll_opt_path}")
        print(f"  ✓ Loaded U_hl_hat_opt from {U_hl_opt_path}")
    else:
        U_ll_hat_opt, U_hl_hat_opt = None, None
        print("  ⚠ U_ll_hat_opt / U_hl_hat_opt not found; continuing without them.")

    # 6) Dll_samples / Dhl_samples
    dll_samples_path = os.path.join(eval_dir, "cmnist_Dll_samples.pkl")
    with open(dll_samples_path, "rb") as f:
        Dll_samples = pickle.load(f)
    print(f"  ✓ Loaded Dll_samples from {dll_samples_path}")

    dhl_samples_path = os.path.join(eval_dir, "cmnist_Dhl_samples.pkl")
    with open(dhl_samples_path, "rb") as f:
        Dhl_samples = pickle.load(f)
    print(f"  ✓ Loaded Dhl_samples from {dhl_samples_path}")

    print("All eval state loaded.\n")

    return (
        all_results,
        cv_folds,
        omega,
        Dll_samples,
        Dhl_samples,
        det_ll_dict_opt,
        det_hl_dict_opt,
        U_ll_hat_opt,
        U_hl_hat_opt,
    )


# ============================================================
# 2. Low-level transforms: Huber noise & Camera shift
# ============================================================

def apply_huber_contamination_cmnist(
    X_flat,
    alpha,
    noise_scale,
    noise_dims,
    seed,
    loc=0.0,
):
    """
    Huber-type contamination on a subset of dimensions (noise_dims).

    X_flat: (N, D) tensor in [-1, 1]
    alpha: probability that a given sample is corrupted
    noise_scale: std of Gaussian noise added to corrupted samples
    noise_dims: slice or index tensor for dims to be corrupted
    seed: random seed
    loc: mean of Gaussian noise (default 0.0)

    Returns: X_corrupted (N, D)
    """
    if alpha <= 0 or noise_scale <= 0:
        return X_flat

    device = X_flat.device
    N, D = X_flat.shape

    # Version-agnostic global seeding
    torch.manual_seed(int(seed) % (2**32))
    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed) % (2**32))

    X_corrupt = X_flat.clone()
    X_sub = X_corrupt[:, noise_dims]  # (N, d_sub)

    # Sample mask and noise
    mask = torch.rand(N, 1, device=device) < alpha
    noise = torch.randn_like(X_sub) * noise_scale + loc

    X_sub_noisy = torch.where(mask, X_sub + noise, X_sub)
    X_corrupt[:, noise_dims] = X_sub_noisy
    return X_corrupt


def build_camera_transform(rotation_deg,
                           zoom_range,
                           brightness,
                           contrast,
                           saturation,
                           blur_sigma,
                           perspective_distortion):
    """
    Build a torchvision transform pipeline for camera-like perturbations.
    We assume inputs are tensors in [-1, 1] with shape (3, 32, 32).
    """
    transforms = []

    # RandomAffine: rotation + zoom, with a fixed small translation
    if rotation_deg != 0 or zoom_range is not None:
        if zoom_range is None:
            zoom_range = (1.0, 1.0)
        transforms.append(
            T.RandomAffine(
                degrees=rotation_deg,
                translate=(0.1, 0.1),
                scale=zoom_range,
            )
        )

    # ColorJitter: lighting (brightness, contrast) + color intensity (saturation)
    if brightness != 0.0 or contrast != 0.0 or saturation != 0.0:
        transforms.append(
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            )
        )

    # GaussianBlur: blur strength
    if blur_sigma > 0:
        transforms.append(
            T.GaussianBlur(kernel_size=3, sigma=(blur_sigma, blur_sigma))
        )

    # RandomPerspective: camera angle / perspective distortion
    if perspective_distortion > 0:
        transforms.append(
            T.RandomPerspective(distortion_scale=perspective_distortion, p=1.0)
        )

    if len(transforms) == 0:
        return None
    return T.Compose(transforms)


def apply_camera_transform_cmnist(
    X_flat,
    transform,
    seed,
):
    """
    Apply a camera-like transform to every image in the batch.

    X_flat: (N, 3072) flattened [-1,1] RGB 32x32
    transform: torchvision transform or None
    seed: base seed for reproducibility
    """
    if transform is None:
        return X_flat

    device = X_flat.device
    N, D = X_flat.shape
    assert D == 3 * 32 * 32, f"Expected 3072 dims, got {D}"

    X_img = X_flat.view(N, 3, 32, 32)
    X_shifted = torch.empty_like(X_img)

    base_seed = int(seed) % (2**32)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)

    # Apply transform per image for deterministic variation
    for i in range(N):
        # Offset seed for each image so they don't all get identical transforms
        img_seed = (base_seed + i) % (2**32)
        torch.manual_seed(img_seed)
        np.random.seed(img_seed)
        random.seed(img_seed)

        X_shifted[i] = transform(X_img[i])

    return X_shifted.view(N, -1)


# ============================================================
# 3. Core: extract pixels→z mapping & compute metrics
# ============================================================

def extract_z_pred_and_true(T_matrix, Dll_full, Dhl_full):
    """
    Given:
      - T_matrix : various shapes, but conceptually d_z x 3072 (pixels->z)
      - Dll_full : (N, 3072 + 20) = [pixels | digit10 | color10]
      - Dhl_full : (N, 20 + d_z)   = [digit10 | color10 | z]

    Returns:
      - Z_pred : (N, d_z)
      - Z_true : (N, d_z)
    """
    T = T_matrix if isinstance(T_matrix, torch.Tensor) else torch.tensor(T_matrix, dtype=torch.float32)
    X = Dll_full if isinstance(Dll_full, torch.Tensor) else torch.tensor(Dll_full, dtype=torch.float32)
    Y = Dhl_full if isinstance(Dhl_full, torch.Tensor) else torch.tensor(Dhl_full, dtype=torch.float32)

    device = torch.device("cpu")
    T = T.to(device)
    X = X.to(device)
    Y = Y.to(device)

    # ---- extract pixel->z block from T ----
    if T.shape[1] == 3072 and T.shape[0] <= 64:
        # already (d_z, 3072)
        T_pix = T
    elif T.shape == (84, 3092):
        # old-style [Xd|Xc|z] x [pixels|Xd|Xc]
        T_pix = T[20:, :3072]
    else:
        # fallback: assume first 3072 columns are pixels, all rows for z
        if T.shape[1] >= 3072:
            T_pix = T[:, :3072]
        else:
            raise ValueError(f"Unexpected T shape {tuple(T.shape)} for pixel->z mapping.")

    d_z = T_pix.shape[0]

    # ---- extract pixels from X ----
    if X.shape[1] == 3072:
        X_pix = X
    elif X.shape[1] >= 3092:
        X_pix = X[:, :3072]
    else:
        raise ValueError(f"Unexpected Dll shape {tuple(X.shape)}; cannot find 3072 pixel dims.")

    # ---- extract z from Y ----
    if Y.shape[1] == d_z:
        Z_true = Y
    elif Y.shape[1] >= 20 + d_z:
        Z_true = Y[:, -d_z:]
    else:
        raise ValueError(f"Unexpected Dhl shape {tuple(Y.shape)}; cannot find z block of size {d_z}.")

    # ---- forward: z_pred = x_pix @ T_pix^T ----
    Z_pred = X_pix @ T_pix.T  # (N, d_z)

    return Z_pred, Z_true


def compute_all_metrics(Z_pred, Z_true, eps=1e-9):
    """
    Compute:
      - mse_sq   : mean squared L2 (E ||diff||^2)
      - rmse     : mean L2 (E ||diff||)
      - rel_l2   : 100 * E[ ||diff||^2 / (||z_true||^2 + eps) ]
      - norm_l2  : rmse / d_z
      - snr_db   : 10 log10( E||z_true||^2 / E||diff||^2 )

    NOTE: Relative L2 is the primary/default metric.
    """
    diff = Z_pred - Z_true
    d_z = Z_true.shape[1]

    sq_l2 = diff.pow(2).sum(dim=1)       # (N,)
    sq_true = Z_true.pow(2).sum(dim=1)   # (N,)

    mse_sq = sq_l2.mean().item()                     # E||diff||^2
    rmse = diff.norm(p=2, dim=1).mean().item()       # E||diff||

    rel_l2 = (sq_l2 / (sq_true + eps)).mean().item() * 100.0

    norm_l2 = rmse / float(d_z)

    signal_power = sq_true.mean().item() + eps
    noise_power = sq_l2.mean().item() + eps
    snr_db = 10.0 * math.log10(signal_power / noise_power)

    return {
        "mse_sq": float(mse_sq),
        "rmse": float(rmse),
        "rel_l2": float(rel_l2),
        "norm_l2": float(norm_l2),
        "snr_db": float(snr_db),
    }


# ============================================================
# 4. Main evaluation loops (Huber: α-grid, Camera: param sweep)
# ============================================================

def method_display_name(method_group_key, run_key):
    if method_group_key.startswith("diroca_"):
        return f"DiRoCA ({run_key})"
    elif method_group_key == "gradca":
        return "GradCA"
    elif method_group_key == "baryca":
        return "BaryCA"
    elif method_group_key == "abslingam":
        return run_key
    else:
        return f"{method_group_key}_{run_key}"


def evaluate_huber(
    all_results,
    omega,
    Dll_samples,
    Dhl_samples,
    N_TRIALS=5,
    ALPHA_VALUES=None,
    noise_scale_for_alpha1=0.5,
):
    """
    Evaluate Huber contamination for a grid of alpha values in [0,1].
    Returns a DataFrame with columns:
      ['shift_type', 'metric', 'method', 'fold', 'alpha', 'trial', 'value']
    """
    shift_type = "huber"
    if ALPHA_VALUES is None:
        ALPHA_VALUES = np.linspace(0.0, 1.0, 10)

    records = []

    # rough count for tqdm
    total_configs = 0
    for mg, folds in all_results.items():
        if isinstance(folds, dict):
            for fk, fd in folds.items():
                if isinstance(fd, dict) and "error" not in fd:
                    total_configs += len(fd)
    total_configs *= len(ALPHA_VALUES) * N_TRIALS

    pbar = tqdm(total=total_configs, desc="Evaluating (huber)")

    LL_PIXELS = slice(0, 3072)

    for method_group_key, method_results_inner in all_results.items():
        if not isinstance(method_results_inner, dict):
            continue

        for fold_key, fold_data in method_results_inner.items():
            if not fold_key.startswith("fold_"):
                continue
            if not isinstance(fold_data, dict) or "error" in fold_data:
                continue

            fold_idx = int(fold_key.split("_")[-1])

            for run_key, run_result in fold_data.items():
                if "error" in run_result or run_result.get("T_matrix") is None:
                    pbar.update(len(ALPHA_VALUES) * N_TRIALS)
                    continue

                T_matrix = run_result["T_matrix"]
                test_idx = run_result["test_indices"]

                method_name = method_display_name(method_group_key, run_key)

                for alpha in ALPHA_VALUES:
                    # interpret noise scale: alpha=0 → no noise; alpha>0 → fixed sigma
                    if np.isclose(alpha, 0.0):
                        noise_scale = 0.0
                    else:
                        noise_scale = noise_scale_for_alpha1

                    for trial in range(N_TRIALS):
                        metrics_accumulator = {
                            "mse_sq": [],
                            "rmse": [],
                            "rel_l2": [],
                            "norm_l2": [],
                            "snr_db": [],
                        }

                        for iota, eta in omega.items():
                            try:
                                if iota not in Dll_samples or eta not in Dhl_samples:
                                    continue

                                ll_imgs, ll_shapes, ll_digits, ll_colors = Dll_samples[iota]
                                Dhl_clean = Dhl_samples[eta]

                                # test subset
                                ll_imgs_01 = ll_imgs[test_idx]      # (N,3,32,32) in [0,1]
                                ll_digits_test = ll_digits[test_idx]
                                ll_colors_test = ll_colors[test_idx]
                                Dhl_clean_test = Dhl_clean[test_idx]

                                # [0,1] → [-1,1]
                                ll_imgs_tanh = ll_imgs_01 * 2 - 1
                                ll_imgs_flat = ll_imgs_tanh.view(ll_imgs_tanh.shape[0], -1)

                                seed = hash((fold_idx, run_key, float(alpha), trial, iota, "huber")) % (2**32)
                                ll_imgs_shift = apply_huber_contamination_cmnist(
                                    ll_imgs_flat,
                                    alpha=float(alpha),
                                    noise_scale=noise_scale,
                                    noise_dims=LL_PIXELS,
                                    seed=seed,
                                    loc=0.0,
                                )

                                # assemble LL full [pixels | digit10 | color10]
                                ll_digits_onehot = F.one_hot(ll_digits_test, 10).float()
                                ll_colors_onehot = F.one_hot(ll_colors_test, 10).float()

                                Dll_full = torch.cat(
                                    [ll_imgs_shift, ll_digits_onehot, ll_colors_onehot],
                                    dim=1,
                                )

                                Z_pred, Z_true = extract_z_pred_and_true(
                                    T_matrix,
                                    Dll_full,
                                    Dhl_clean_test,
                                )
                                metric_vals = compute_all_metrics(Z_pred, Z_true)

                                for k in metrics_accumulator.keys():
                                    metrics_accumulator[k].append(metric_vals[k])

                            except Exception as e:
                                print(f"[HUBER ERROR] {e} | Context {method_name}, fold={fold_idx}, alpha={alpha}")
                                for k in metrics_accumulator.keys():
                                    metrics_accumulator[k].append(np.nan)

                        # average over interventions for this trial
                        for metric_name, vals in metrics_accumulator.items():
                            avg_val = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
                            records.append(
                                {
                                    "shift_type": shift_type,
                                    "metric": metric_name,
                                    "method": method_name,
                                    "fold": fold_idx,
                                    "alpha": float(alpha),
                                    "trial": trial,
                                    "value": avg_val,
                                    "camera_param": None,
                                    "camera_level": None,
                                    "camera_value": None,
                                }
                            )

                        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(records)
    return df


def evaluate_camera_param_sweep(
    all_results,
    omega,
    Dll_samples,
    Dhl_samples,
    N_TRIALS=5,
    camera_levels=None,
):
    """
    Evaluate camera-like shifts for α=1, sweeping over:
      - rotation
      - zoom
      - lighting (brightness+contrast)
      - color (saturation)
      - blur sigma
      - camera angle (perspective)

    For each parameter, we test low/medium/high while keeping the others at medium.

    Returns a DataFrame with columns:
      ['shift_type', 'metric', 'method', 'fold',
       'camera_param', 'camera_level', 'camera_value',
       'alpha', 'trial', 'value']
    """
    shift_type = "camera"

    if camera_levels is None:
        raise ValueError("camera_levels must be provided.")

    param_names = ["rotation", "zoom", "lighting", "color", "blur", "angle"]
    levels = ["low", "med", "high"]

    # rough count for tqdm
    total_configs = 0
    for mg, folds in all_results.items():
        if isinstance(folds, dict):
            for fk, fd in folds.items():
                if isinstance(fd, dict) and "error" not in fd:
                    total_configs += len(fd)
    total_configs *= len(param_names) * len(levels) * N_TRIALS

    pbar = tqdm(total=total_configs, desc="Evaluating (camera param sweep)")

    records = []

    for method_group_key, method_results_inner in all_results.items():
        if not isinstance(method_results_inner, dict):
            continue

        for fold_key, fold_data in method_results_inner.items():
            if not fold_key.startswith("fold_"):
                continue
            if not isinstance(fold_data, dict) or "error" in fold_data:
                continue

            fold_idx = int(fold_key.split("_")[-1])

            for run_key, run_result in fold_data.items():
                if "error" in run_result or run_result.get("T_matrix") is None:
                    pbar.update(len(param_names) * len(levels) * N_TRIALS)
                    continue

                T_matrix = run_result["T_matrix"]
                test_idx = run_result["test_indices"]
                method_name = method_display_name(method_group_key, run_key)

                for param_name in param_names:
                    for level in levels:
                        for trial in range(N_TRIALS):
                            metrics_accumulator = {
                                "mse_sq": [],
                                "rmse": [],
                                "rel_l2": [],
                                "norm_l2": [],
                                "snr_db": [],
                            }

                            # Build param settings: chosen param at given level; others fixed at 'med'
                            rot_deg = camera_levels["rotation"][level if param_name == "rotation" else "med"]
                            zoom_range = camera_levels["zoom"][level if param_name == "zoom" else "med"]
                            brightness = camera_levels["lighting"][level if param_name == "lighting" else "med"]["brightness"]
                            contrast = camera_levels["lighting"][level if param_name == "lighting" else "med"]["contrast"]
                            saturation = camera_levels["color"][level if param_name == "color" else "med"]
                            blur_sigma = camera_levels["blur"][level if param_name == "blur" else "med"]
                            angle_distortion = camera_levels["angle"][level if param_name == "angle" else "med"]

                            # A compact textual representation of the active value
                            if param_name == "rotation":
                                param_value_repr = f"{rot_deg:.2f} deg"
                            elif param_name == "zoom":
                                param_value_repr = f"[{zoom_range[0]:.2f}, {zoom_range[1]:.2f}]"
                            elif param_name == "lighting":
                                param_value_repr = f"b={brightness:.2f}, c={contrast:.2f}"
                            elif param_name == "color":
                                param_value_repr = f"sat={saturation:.2f}"
                            elif param_name == "blur":
                                param_value_repr = f"sigma={blur_sigma:.2f}"
                            elif param_name == "angle":
                                param_value_repr = f"dist={angle_distortion:.2f}"
                            else:
                                param_value_repr = "unknown"

                            transform = build_camera_transform(
                                rotation_deg=rot_deg,
                                zoom_range=zoom_range,
                                brightness=brightness,
                                contrast=contrast,
                                saturation=saturation,
                                blur_sigma=blur_sigma,
                                perspective_distortion=angle_distortion,
                            )

                            for iota, eta in omega.items():
                                try:
                                    if iota not in Dll_samples or eta not in Dhl_samples:
                                        continue

                                    ll_imgs, ll_shapes, ll_digits, ll_colors = Dll_samples[iota]
                                    Dhl_clean = Dhl_samples[eta]

                                    ll_imgs_01 = ll_imgs[test_idx]
                                    ll_digits_test = ll_digits[test_idx]
                                    ll_colors_test = ll_colors[test_idx]
                                    Dhl_clean_test = Dhl_clean[test_idx]

                                    # [0,1] → [-1,1]
                                    ll_imgs_tanh = ll_imgs_01 * 2 - 1
                                    ll_imgs_flat = ll_imgs_tanh.view(ll_imgs_tanh.shape[0], -1)

                                    seed = hash((fold_idx, run_key, param_name, level, trial, iota, "camera")) % (2**32)
                                    ll_imgs_shift = apply_camera_transform_cmnist(
                                        ll_imgs_flat,
                                        transform=transform,
                                        seed=seed,
                                    )

                                    # assemble LL full [pixels | digit10 | color10]
                                    ll_digits_onehot = F.one_hot(ll_digits_test, 10).float()
                                    ll_colors_onehot = F.one_hot(ll_colors_test, 10).float()

                                    Dll_full = torch.cat(
                                        [ll_imgs_shift, ll_digits_onehot, ll_colors_onehot],
                                        dim=1,
                                    )

                                    Z_pred, Z_true = extract_z_pred_and_true(
                                        T_matrix,
                                        Dll_full,
                                        Dhl_clean_test,
                                    )
                                    metric_vals = compute_all_metrics(Z_pred, Z_true)

                                    for k in metrics_accumulator.keys():
                                        metrics_accumulator[k].append(metric_vals[k])

                                except Exception as e:
                                    print(f"[CAMERA ERROR] {e} | Context {method_name}, fold={fold_idx}, param={param_name}, level={level}")
                                    for k in metrics_accumulator.keys():
                                        metrics_accumulator[k].append(np.nan)

                            # average over interventions for this trial
                            for metric_name, vals in metrics_accumulator.items():
                                avg_val = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
                                records.append(
                                    {
                                        "shift_type": shift_type,
                                        "metric": metric_name,
                                        "method": method_name,
                                        "fold": fold_idx,
                                        "alpha": 1.0,  # always fully shifted for camera
                                        "trial": trial,
                                        "value": avg_val,
                                        "camera_param": param_name,
                                        "camera_level": level,
                                        "camera_value": param_value_repr,
                                    }
                                )

                            pbar.update(1)

    pbar.close()
    df = pd.DataFrame(records)
    return df


# ============================================================
# 5. Aggregation & PDF reporting
# ============================================================

def summarize_huber(df_huber):
    """
    Aggregates mean ± std across folds and trials per:
      (shift_type, metric, alpha, method)
    """
    if df_huber.empty:
        return pd.DataFrame()
    df_summary = (
        df_huber
        .groupby(["shift_type", "metric", "alpha", "method"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return df_summary


def summarize_camera(df_camera):
    """
    Aggregates mean ± std across folds and trials per:
      (shift_type, metric, camera_param, camera_level, method)
    """
    if df_camera.empty:
        return pd.DataFrame()
    df_summary = (
        df_camera
        .groupby(["shift_type", "metric", "camera_param", "camera_level", "method"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return df_summary


def make_pdf_report(df_huber_summary, df_camera_summary, out_pdf):
    """
    Create a multipage PDF with:
      - Huber: tables for all metrics × α and a plot of rel_l2 vs α
      - Camera: tables for rel_l2 per parameter (low/med/high) per method
    """
    metrics_order = ["rel_l2", "mse_sq", "rmse", "norm_l2", "snr_db"]
    metric_labels = {
        "mse_sq": "Squared L2 (MSE)",
        "rmse": "Raw L2 (RMSE)",
        "rel_l2": "Relative L2 (%)",
        "norm_l2": "Normalized L2 (per-dim)",
        "snr_db": "SNR (dB)",
    }

    with PdfPages(out_pdf) as pdf:
        # ---------------- Huber: rel_l2 vs alpha plot ----------------
        if not df_huber_summary.empty:
            df_huber_rel = df_huber_summary[df_huber_summary["metric"] == "rel_l2"]
            if not df_huber_rel.empty:
                alphas = sorted(df_huber_rel["alpha"].unique())
                methods = sorted(df_huber_rel["method"].unique())

                fig, ax = plt.subplots(figsize=(10, 6))
                for method in methods:
                    df_m = df_huber_rel[df_huber_rel["method"] == method]
                    df_m = df_m.sort_values("alpha")
                    ax.plot(df_m["alpha"], df_m["mean"], marker="o", label=method)

                ax.set_title("Huber contamination: Relative L2 vs α")
                ax.set_xlabel("α (corruption probability)")
                ax.set_ylabel("Relative L2 (%)")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend(fontsize=6, loc="best")

                pdf.savefig(fig)
                plt.close(fig)

        # ---------------- Huber: tables for all metrics ----------------
        if not df_huber_summary.empty:
            for metric in metrics_order:
                df_metric = df_huber_summary[df_huber_summary["metric"] == metric]
                if df_metric.empty:
                    continue

                for alpha in sorted(df_metric["alpha"].unique()):
                    df_alpha = df_metric[df_metric["alpha"] == alpha].sort_values("mean")

                    fig, ax = plt.subplots(figsize=(11, 6))
                    ax.axis("off")

                    title = f"Huber Noise – {metric_labels.get(metric, metric)} (α={alpha:.2f})"
                    ax.set_title(title, fontsize=14, pad=20)

                    table_data = [["Method", "Mean", "Std"]]
                    for _, row in df_alpha.iterrows():
                        table_data.append(
                            [
                                row["method"],
                                f"{row['mean']:.4f}",
                                f"{row['std']:.4f}" if not np.isnan(row["std"]) else "nan",
                            ]
                        )

                    table = ax.table(
                        cellText=table_data,
                        loc="center",
                        cellLoc="left",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1.0, 1.3)

                    pdf.savefig(fig)
                    plt.close(fig)

        # ---------------- Camera: rel_l2 tables per parameter ----------------
        if not df_camera_summary.empty:
            df_cam_rel = df_camera_summary[df_camera_summary["metric"] == "rel_l2"]
            if not df_cam_rel.empty:
                param_names = sorted(df_cam_rel["camera_param"].unique())
                levels = ["low", "med", "high"]

                for param in param_names:
                    df_p = df_cam_rel[df_cam_rel["camera_param"] == param]

                    # Build table: one row per method, columns for low/med/high mean/std
                    methods = sorted(df_p["method"].unique())
                    fig, ax = plt.subplots(figsize=(11, 6))
                    ax.axis("off")

                    title = f"Camera shift – Relative L2 (%) for parameter: {param}"
                    ax.set_title(title, fontsize=14, pad=20)

                    header = ["Method"]
                    for level in levels:
                        header.extend([f"{level}_mean", f"{level}_std"])
                    table_data = [header]

                    for m in methods:
                        row = [m]
                        for level in levels:
                            df_ml = df_p[(df_p["method"] == m) & (df_p["camera_level"] == level)]
                            if df_ml.empty:
                                row.extend(["-", "-"])
                            else:
                                mean_val = df_ml["mean"].iloc[0]
                                std_val = df_ml["std"].iloc[0]
                                row.append(f"{mean_val:.4f}")
                                row.append(f"{std_val:.4f}" if not np.isnan(std_val) else "nan")
                        table_data.append(row)

                    table = ax.table(
                        cellText=table_data,
                        loc="center",
                        cellLoc="left",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1.0, 1.3)

                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"\nPDF report saved to: {out_pdf}")


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CMNIST evaluation (v2): Huber (α-grid) & Camera param-sweep with relative L2 as default metric."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=os.path.join("data", "cmnist", "results_empirical"),
        help="Directory where cmnist_opt saved eval state.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of random trials per configuration.",
    )
    parser.add_argument(
        "--noise-scale-alpha1",
        type=float,
        default=0.5,
        help="Noise std (σ) used for alpha>0 in Huber contamination.",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default=os.path.join("data", "cmnist", "results_empirical", "cmnist_eval_metrics_v2.pdf"),
        help="Output PDF path for summary tables and plots.",
    )

    # ---------------- Camera parameter levels (low, med, high) ----------------
    parser.add_argument(
        "--camera-rotation-degrees",
        type=float,
        nargs=3,
        default=[10.0, 30.0, 60.0],  # low, med, high
        help="Rotation degrees (low, med, high).",
    )
    parser.add_argument(
        "--camera-zoom-min",
        type=float,
        nargs=3,
        default=[0.95, 0.90, 0.80],
        help="Zoom min (low, med, high).",
    )
    parser.add_argument(
        "--camera-zoom-max",
        type=float,
        nargs=3,
        default=[1.05, 1.10, 1.20],
        help="Zoom max (low, med, high).",
    )
    parser.add_argument(
        "--camera-brightness",
        type=float,
        nargs=3,
        default=[0.10, 0.20, 0.40],
        help="Brightness jitter amplitude (low, med, high).",
    )
    parser.add_argument(
        "--camera-contrast",
        type=float,
        nargs=3,
        default=[0.10, 0.20, 0.40],
        help="Contrast jitter amplitude (low, med, high).",
    )
    parser.add_argument(
        "--camera-saturation",
        type=float,
        nargs=3,
        default=[0.05, 0.10, 0.20],
        help="Saturation jitter amplitude (low, med, high).",
    )
    parser.add_argument(
        "--camera-blur-sigma",
        type=float,
        nargs=3,
        default=[0.10, 0.50, 1.00],
        help="Gaussian blur sigma (low, med, high).",
    )
    parser.add_argument(
        "--camera-angle-distortion",
        type=float,
        nargs=3,
        default=[0.05, 0.20, 0.40],
        help="Perspective distortion scale (low, med, high).",
    )

    args = parser.parse_args()

    (
        all_results,
        cv_folds,
        omega,
        Dll_samples,
        Dhl_samples,
        det_ll_dict_opt,
        det_hl_dict_opt,
        U_ll_hat_opt,
        U_hl_hat_opt,
    ) = load_eval_state(args.eval_dir)

    # Build camera_levels dict
    levels = ["low", "med", "high"]
    camera_levels = {
        "rotation": {lvl: args.camera_rotation_degrees[i] for i, lvl in enumerate(levels)},
        "zoom": {
            lvl: (args.camera_zoom_min[i], args.camera_zoom_max[i])
            for i, lvl in enumerate(levels)
        },
        "lighting": {
            lvl: {
                "brightness": args.camera_brightness[i],
                "contrast": args.camera_contrast[i],
            }
            for i, lvl in enumerate(levels)
        },
        "color": {lvl: args.camera_saturation[i] for i, lvl in enumerate(levels)},
        "blur": {lvl: args.camera_blur_sigma[i] for i, lvl in enumerate(levels)},
        "angle": {lvl: args.camera_angle_distortion[i] for i, lvl in enumerate(levels)},
    }

    # ---------------- Huber evaluation ----------------
    print("\n================ Huber evaluation (α-grid, all metrics) ================")
    alpha_grid = np.linspace(0.0, 1.0, 10)
    df_huber = evaluate_huber(
        all_results=all_results,
        omega=omega,
        Dll_samples=Dll_samples,
        Dhl_samples=Dhl_samples,
        N_TRIALS=args.n_trials,
        ALPHA_VALUES=alpha_grid,
        noise_scale_for_alpha1=args.noise_scale_alpha1,
    )
    print("\nHuber evaluation raw head:")
    print(df_huber.head())

    # ---------------- Camera param-sweep evaluation ----------------
    print("\n================ Camera-shift evaluation (param sweep, α=1) ================")
    df_camera = evaluate_camera_param_sweep(
        all_results=all_results,
        omega=omega,
        Dll_samples=Dll_samples,
        Dhl_samples=Dhl_samples,
        N_TRIALS=args.n_trials,
        camera_levels=camera_levels,
    )
    print("\nCamera evaluation raw head:")
    print(df_camera.head())

    # ---------------- Save raw results (combined) ----------------
    # For backward compatibility: one big df with extra columns for camera.
    df_all = pd.concat([df_huber, df_camera], ignore_index=True)
    out_pkl = os.path.join(args.eval_dir, "cmnist_eval_all_metrics.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(df_all, f)
    print(f"\n✓ Saved raw metric evaluations to {out_pkl}")

    # ---------------- Summaries ----------------
    df_huber_summary = summarize_huber(df_huber)
    df_camera_summary = summarize_camera(df_camera)

    print("\n=== Huber summary (head) ===")
    print(df_huber_summary.head(20))

    print("\n=== Camera summary (head) ===")
    print(df_camera_summary.head(20))

    # ---------------- PDF report ----------------
    make_pdf_report(df_huber_summary, df_camera_summary, args.out_pdf)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
