#!/usr/bin/env python3
"""
CMNIST Evaluation Script with Metric Selection and PDF Output.

Usage:
    python cmnist_evaluate.py --metric {mse,cosine,znorm} --output-pdf OUTPUT.pdf

Metric meanings:
  - mse    : per-dimension MSE on z
  - cosine : cosine error = 1 - mean_cos_sim
  - znorm  : z-normalized MSE on z
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ===============================================================
# 0. LOADING STATE (YOU NEED TO ADAPT THIS PART)
# ===============================================================

def load_cmnist_state():
    """
    TODO: Adapt this to your setup.
    It must return:
      - all_results: dict (what you saved from cmnist_optimization)
      - Dll_samples: dict[iota] -> (images_01, ..., digits, colors)
      - Dhl_samples: dict[eta]  -> HL vectors (N,84)
      - omega: dict mapping iota -> eta
      - det_ll_dict_opt: dict[iota] -> (N,3072) deterministic LL pixels (opt view)
      - det_hl_dict_opt: dict[eta]  -> (N,64) deterministic HL z (opt view)
      - U_ll_hat_run: (N,3072) LL noise (test-time, used in IV eval)
      - U_hl_hat_run: (N,64)   HL noise
      - cv_folds: cross-validation folds if needed (optional here)

    For example, if you have a single pickle with everything:
        state = torch.load("data/cmnist/cmnist_state.pt")
        return (state['all_results'], state['Dll_samples'], ...)

    For now, this is just a placeholder.
    """
    raise NotImplementedError(
        "Please implement load_cmnist_state() to return the CMNIST state "
        "(all_results, Dll_samples, Dhl_samples, omega, det_ll_dict_opt, "
        " det_hl_dict_opt, U_ll_hat_run, U_hl_hat_run, cv_folds)."
    )


# ===============================================================
# 1. METRIC HELPERS ON z
# ===============================================================

def compute_z_metrics(z_pred, z_true, mu_z=None, std_z=None):
    """
    z_pred, z_true: (N, d_z)
    mu_z, std_z   : (d_z,) global mean/std for z-normalized MSE (optional)

    Returns:
      dict with:
        - mse_per_dim     : float
        - mean_cos_sim    : float
        - cos_err         : float (1 - mean_cos_sim)
        - mse_z_norm      : float or NaN
    """
    if not isinstance(z_pred, torch.Tensor):
        z_pred = torch.tensor(z_pred, dtype=torch.float32)
    if not isinstance(z_true, torch.Tensor):
        z_true = torch.tensor(z_true, dtype=torch.float32)

    device = z_pred.device
    z_true = z_true.to(device)

    N, d = z_true.shape
    assert z_pred.shape == z_true.shape, f"Shape mismatch: {z_pred.shape} vs {z_true.shape}"

    # 1) per-dim MSE
    diff = z_pred - z_true
    mse_per_dim = (diff.pow(2).sum() / (N * d)).item()

    # 2) cosine similarity
    num = (z_pred * z_true).sum(dim=1)                        # (N,)
    den = z_pred.norm(dim=1) * z_true.norm(dim=1) + 1e-8      # (N,)
    cos_sim = num / den                                       # (N,)
    mean_cos_sim = cos_sim.mean().item()
    cos_err = 1.0 - mean_cos_sim

    # 3) z-normalized MSE
    if (mu_z is not None) and (std_z is not None):
        mu_z  = mu_z.to(device)
        std_z = std_z.to(device)
        z_true_std = (z_true - mu_z) / std_z
        z_pred_std = (z_pred - mu_z) / std_z
        diff_std   = z_pred_std - z_true_std
        mse_z_norm = (diff_std.pow(2).sum() / (N * d)).item()
    else:
        mse_z_norm = float('nan')

    return {
        "mse_per_dim": mse_per_dim,
        "mean_cos_sim": mean_cos_sim,
        "cos_err": cos_err,
        "mse_z_norm": mse_z_norm,
    }


def scalar_from_metrics(metrics, metric_name):
    """
    Map metrics dict → single scalar according to metric_name.
    Smaller is always better.
    """
    if metrics is None:
        return np.nan

    if metric_name == "mse":
        return metrics["mse_per_dim"]
    elif metric_name == "cosine":
        # treat error as (1 - cos), so smaller is better
        return metrics["cos_err"]
    elif metric_name == "znorm":
        return metrics["mse_z_norm"]
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")


# ===============================================================
# 2. PIXELS-ONLY HUBER CONTAMINATION & CAMERA SHIFTS
# ===============================================================

def apply_huber_contamination_cmnist(clean_data, alpha, noise_scale, noise_dims, seed=None, loc=0.0):
    """
    Contaminate ONLY the specified dimensions (columns) of clean_data with Gaussian noise.
    Args:
        clean_data : (N, D) tensor or array
        alpha      : fraction of samples to contaminate in [0,1]
        noise_scale: std of Gaussian noise
        noise_dims : slice / list / np.ndarray / torch.Tensor of column indices
        seed       : RNG seed
        loc        : mean shift of noise (0.0 = zero-mean)
    """
    import numpy as _np

    if alpha is None or noise_scale is None or _np.isclose(alpha, 0.0) or _np.isclose(noise_scale, 0.0):
        if isinstance(clean_data, torch.Tensor):
            return clean_data.clone().to(torch.float32)
        return torch.tensor(clean_data, dtype=torch.float32)

    if isinstance(clean_data, torch.Tensor):
        data_cont = clean_data.to(torch.float32).clone()
    else:
        data_cont = torch.tensor(clean_data, dtype=torch.float32).clone()
    device = data_cont.device
    N, D = data_cont.shape

    # Build index tensor
    if isinstance(noise_dims, slice):
        start = 0 if noise_dims.start is None else noise_dims.start
        stop  = D if noise_dims.stop is None else noise_dims.stop
        step  = 1 if noise_dims.step is None else noise_dims.step
        noise_idx = torch.arange(start, stop, step, device=device)
    elif isinstance(noise_dims, (list, tuple, _np.ndarray, torch.Tensor)):
        noise_idx = torch.as_tensor(noise_dims, dtype=torch.long, device=device)
    else:
        raise TypeError(f"Unsupported type for noise_dims: {type(noise_dims)}")

    noise_idx = noise_idx[(noise_idx >= 0) & (noise_idx < D)]
    if noise_idx.numel() == 0:
        return data_cont

    data_to_noise = data_cont.index_select(dim=1, index=noise_idx)  # (N, |noise_idx|)

    rng = _np.random.default_rng(seed)
    noise = rng.normal(loc=loc, scale=noise_scale, size=tuple(data_to_noise.shape)).astype(_np.float32)
    noise_tensor = torch.tensor(noise, dtype=torch.float32, device=device)

    noisy_slice = data_to_noise + noise_tensor

    if alpha >= 1.0:
        data_cont[:, noise_idx] = noisy_slice
        return data_cont

    n_contaminate = int(alpha * N)
    if n_contaminate <= 0:
        return data_cont

    idx_to_contaminate_np = rng.choice(N, size=n_contaminate, replace=False)
    idx_to_contaminate = torch.as_tensor(idx_to_contaminate_np, dtype=torch.long, device=device)

    data_cont.index_copy_(
        dim=0,
        index=idx_to_contaminate,
        source=data_cont.index_select(0, idx_to_contaminate).scatter(
            1,
            noise_idx.view(1, -1).expand(n_contaminate, -1),
            noisy_slice.index_select(0, idx_to_contaminate)
        )
    )
    return data_cont


import torchvision.transforms as TV

def apply_camera_shifts_cmnist(images_01, alpha, transform, seed=None):
    """
    Apply a camera/augmentation transform to a fraction alpha of images.
    Input:  images_01 : (N, C, H, W) in [0,1]
    Output: images_tanh_aug : (N, C, H, W) in [-1,1]
    """
    import numpy as _np

    if alpha <= 0.0:
        return images_01 * 2.0 - 1.0

    images_aug = images_01.clone()
    N = images_01.shape[0]
    rng = _np.random.default_rng(seed)
    n_aug = int(alpha * N)
    if n_aug == 0:
        return images_01 * 2.0 - 1.0

    idx_all = _np.arange(N)
    idx_aug = rng.choice(idx_all, size=n_aug, replace=False)

    for j in idx_aug:
        images_aug[j] = transform(images_aug[j])

    images_tanh_aug = images_aug * 2.0 - 1.0
    return images_tanh_aug


# ===============================================================
# 3. CORE: FROM T & DATA → SCALAR ERROR (SELECT METRIC)
# ===============================================================

def calculate_empirical_error_flat(
    T_matrix,
    Dll_test_flat,
    Dhl_test,
    metric_name,
    mu_z=None,
    std_z=None
):
    """
    Compute z-based metrics and return ONE scalar according to metric_name.
    Supports:
      - T: (64,3072)      (pixels -> z)
      - T: (84,3092) full (we slice z<-pixels)
    Dll_test_flat:
      - (N,3072) or (N,3092) [pixels | labels]
    Dhl_test:
      - (N,64) or (N,84) [labels | z]
    """
    try:
        T = T_matrix if isinstance(T_matrix, torch.Tensor) else torch.tensor(T_matrix, dtype=torch.float32)
        X = Dll_test_flat if isinstance(Dll_test_flat, torch.Tensor) else torch.tensor(Dll_test_flat, dtype=torch.float32)
        Y = Dhl_test if isinstance(Dhl_test, torch.Tensor) else torch.tensor(Dhl_test, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T = T.to(device)
        X = X.to(device)
        Y = Y.to(device)

        # Case A: T is (d_z, 3072)
        if T.shape[1] == 3072:
            d_z = T.shape[0]

            if X.shape[1] == 3072:
                ll_pixels = X
            elif X.shape[1] == 3092:
                ll_pixels = X[:, :3072]
            else:
                return np.nan

            if Y.shape[1] == d_z:
                hl_z = Y
            elif Y.shape[1] == 20 + d_z:
                hl_z = Y[:, 20:]
            else:
                return np.nan

            with torch.no_grad():
                z_pred = ll_pixels @ T.T
            metrics = compute_z_metrics(z_pred, hl_z, mu_z=mu_z, std_z=std_z)
            return scalar_from_metrics(metrics, metric_name)

        # Case B: full old T (84,3092) and Y (N,84)
        if T.shape == (84, 3092) and Y.shape[1] == 84:
            with torch.no_grad():
                T_pix_to_z = T[20:, :3072]      # (64,3072)
                if X.shape[1] == 3092:
                    ll_pixels = X[:, :3072]
                else:
                    ll_pixels = X
                z_pred = ll_pixels @ T_pix_to_z.T
                hl_z = Y[:, 20:]
            metrics = compute_z_metrics(z_pred, hl_z, mu_z=mu_z, std_z=std_z)
            return scalar_from_metrics(metrics, metric_name)

        return np.nan
    except Exception as e:
        print(f"Error in calculate_empirical_error_flat: {e}")
        return np.nan


# ===============================================================
# 4. SEMANTIC (IV) EVALUATION
# ===============================================================

@torch.no_grad()
def semantic_errors_pixels_to_z(
    T_matrix,
    det_ll_dict_opt,
    det_hl_dict_opt,
    U_ll_hat_opt,
    U_hl_hat_opt,
    omega,
    metric_name,
    mu_z,
    std_z,
    test_idx=None,
    device=None
):
    """
    e_{iota,eta}(T) = E||T(D_ll[iota]+U_ll) - (D_hl[eta]+U_hl)||^2, but
    scalar error is chosen via metric_name on z only.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def as_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32, device=device)

    T = as_tensor(T_matrix).to(device)
    U_ll = as_tensor(U_ll_hat_opt).to(device)
    U_hl = as_tensor(U_hl_hat_opt).to(device)

    if test_idx is not None:
        test_idx = torch.as_tensor(test_idx, dtype=torch.long, device=device)
        U_ll = U_ll.index_select(0, test_idx)
        U_hl = U_hl.index_select(0, test_idx)

    pair_errors = {}

    for iota, eta in omega.items():
        if iota not in det_ll_dict_opt or eta not in det_hl_dict_opt:
            continue

        Dll = as_tensor(det_ll_dict_opt[iota]).to(device)
        Dhl = as_tensor(det_hl_dict_opt[eta]).to(device)

        if test_idx is not None:
            Dll = Dll.index_select(0, test_idx)
            Dhl = Dhl.index_select(0, test_idx)

        X_ll = Dll + U_ll           # (N_test,3072)
        Z_hl = Dhl + U_hl           # (N_test,64)

        Z_pred = X_ll @ T.T         # (N_test,64)
        metrics = compute_z_metrics(Z_pred, Z_hl, mu_z=mu_z, std_z=std_z)
        pair_errors[(iota, eta)] = scalar_from_metrics(metrics, metric_name)

    vals = np.array(list(pair_errors.values()), dtype=np.float32)
    out = {
        "pair_errors": pair_errors,
        "mean_iv_error": float(np.mean(vals)) if len(vals) else np.nan,
        "max_iv_error": float(np.max(vals)) if len(vals) else np.nan,
        "var_iv_error": float(np.var(vals)) if len(vals) else np.nan,
    }
    return out


def run_semantic_eval_all_methods(
    all_results,
    det_ll_dict_opt, det_hl_dict_opt,
    U_ll_hat_opt, U_hl_hat_opt,
    omega,
    metric_name,
    mu_z,
    std_z
):
    records = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for method_group_key, folds in all_results.items():
        if not isinstance(folds, dict):
            continue

        for fold_key, fold_data in folds.items():
            if not fold_key.startswith("fold_"):
                continue
            if not isinstance(fold_data, dict) or "error" in fold_data:
                continue
            fold_idx = int(fold_key.split("_")[-1])

            for run_key, run_result in fold_data.items():
                if not isinstance(run_result, dict) or "error" in run_result:
                    continue
                if run_result.get("T_matrix") is None:
                    continue

                T = run_result["T_matrix"]
                test_idx = run_result.get("test_indices", None)

                if method_group_key.startswith("diroca_"):
                    method_name = f"DiRoCA ({run_key})"
                elif method_group_key == "gradca":
                    method_name = "GradCA"
                elif method_group_key == "baryca":
                    method_name = "BaryCA"
                elif method_group_key == "abslingam":
                    method_name = run_key
                else:
                    method_name = f"{method_group_key}_{run_key}"

                sem = semantic_errors_pixels_to_z(
                    T, det_ll_dict_opt, det_hl_dict_opt,
                    U_ll_hat_opt, U_hl_hat_opt,
                    omega,
                    metric_name=metric_name,
                    mu_z=mu_z,
                    std_z=std_z,
                    test_idx=test_idx,
                    device=device
                )

                records.append({
                    "method_group": method_group_key,
                    "run_key": run_key,
                    "fold": fold_idx,
                    "method": method_name,
                    "mean_iv_error": sem["mean_iv_error"],
                    "max_iv_error": sem["max_iv_error"],
                    "var_iv_error": sem["var_iv_error"],
                })

    return pd.DataFrame(records)


# ===============================================================
# 5. HUBER & CAMERA EVALUATION (USING CHOSEN METRIC)
# ===============================================================

def run_huber_eval(
    all_results, Dll_samples, Dhl_samples, omega,
    mu_z, std_z, metric_name,
    n_trials=5,
    noise_scale_for_alpha1=0.5,
    alpha_values=(0.0, 1.0),
):
    LL_PIXEL_DIMS = slice(0, 3072)
    evaluation_records = []

    # count total methods
    total_methods_trained = 0
    for method_group_key, method_data_inner in all_results.items():
        if isinstance(method_data_inner, dict):
            for fold_key, fold_data in method_data_inner.items():
                if isinstance(fold_data, dict) and 'error' not in fold_data:
                    total_methods_trained += len(fold_data)

    total_configs = total_methods_trained * len(alpha_values) * n_trials
    if total_configs == 0:
        print("No valid training results in all_results.")
        return pd.DataFrame()

    pbar = tqdm(total=total_configs, desc="Evaluating Huber (clean + noisy)")

    for method_group_key, method_results_inner in all_results.items():
        if not isinstance(method_results_inner, dict):
            continue

        for fold_key, fold_data in method_results_inner.items():
            if not fold_key.startswith('fold_'):
                continue
            if 'error' in fold_data:
                continue

            fold_idx = int(fold_key.split('_')[-1])

            for run_key, run_result in fold_data.items():
                if 'error' in run_result or run_result.get('T_matrix') is None:
                    pbar.update(len(alpha_values) * n_trials)
                    continue

                T_matrix = run_result['T_matrix']
                test_idx = run_result['test_indices']
                if test_idx is None:
                    pbar.update(len(alpha_values) * n_trials)
                    continue

                # pretty name
                if method_group_key.startswith('diroca_'):
                    eval_method_name = f"DiRoCA ({run_key})"
                elif method_group_key == 'gradca':
                    eval_method_name = "GradCA"
                elif method_group_key == 'baryca':
                    eval_method_name = "BaryCA"
                elif method_group_key == 'abslingam':
                    eval_method_name = run_key
                else:
                    eval_method_name = f"{method_group_key}_{run_key}"

                for alpha in alpha_values:
                    noise_scale = 0.0 if np.isclose(alpha, 0.0) else noise_scale_for_alpha1
                    loc_ll = 0.0

                    for trial in range(n_trials):
                        trial_errors = []

                        for iota, eta in list(omega.items()):
                            try:
                                if iota not in Dll_samples or eta not in Dhl_samples:
                                    continue

                                ll_images_01, _, ll_digits, ll_colors = Dll_samples[iota]
                                max_idx = max(test_idx) if len(test_idx) > 0 else -1
                                if max_idx >= len(ll_images_01):
                                    continue

                                ll_images_test_01 = ll_images_01[test_idx]
                                ll_digits_test    = ll_digits[test_idx]
                                ll_colors_test    = ll_colors[test_idx]
                                Dhl_test_clean    = Dhl_samples[eta][test_idx]

                                seed = hash((fold_idx, run_key, float(alpha),
                                             float(noise_scale), trial, str(iota))) % (2**32)

                                ll_images_test_tanh = ll_images_test_01 * 2.0 - 1.0
                                ll_images_test_flat = ll_images_test_tanh.view(ll_images_test_tanh.shape[0], -1)

                                ll_images_cont_flat = apply_huber_contamination_cmnist(
                                    ll_images_test_flat, alpha, noise_scale,
                                    noise_dims=LL_PIXEL_DIMS, seed=seed, loc=loc_ll
                                )

                                ll_digits_onehot = F.one_hot(ll_digits_test, num_classes=10).float()
                                ll_colors_onehot = F.one_hot(ll_colors_test, num_classes=10).float()

                                device = ll_images_cont_flat.device
                                Dll_test_cont_flat_full = torch.cat(
                                    [ll_images_cont_flat,
                                     ll_digits_onehot.to(device),
                                     ll_colors_onehot.to(device)],
                                    dim=1
                                )

                                Dhl_test = torch.as_tensor(Dhl_test_clean, dtype=torch.float32, device=device)

                                err_scalar = calculate_empirical_error_flat(
                                    T_matrix,
                                    Dll_test_cont_flat_full,
                                    Dhl_test,
                                    metric_name=metric_name,
                                    mu_z=mu_z,
                                    std_z=std_z
                                )
                                if not np.isnan(err_scalar):
                                    trial_errors.append(err_scalar)

                            except Exception as e:
                                print(f"[Huber ERROR] {e} | "
                                      f"M={eval_method_name}, F={fold_idx}, "
                                      f"R={run_key}, A={alpha}, T={trial}, Iota={iota}")
                                trial_errors.append(np.nan)

                        record = {
                            'shift_type': 'huber_noise',
                            'method': eval_method_name,
                            'fold': fold_idx,
                            'alpha': float(alpha),
                            'noise_scale': float(noise_scale),
                            'trial': trial,
                            'error': float(np.nanmean(trial_errors)) if trial_errors else np.nan,
                        }
                        if method_group_key.startswith('diroca_'):
                            record['train_epsilon'] = run_result.get('epsilon', np.nan)
                            record['train_delta']   = run_result.get('delta', np.nan)

                        evaluation_records.append(record)
                        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(evaluation_records)
    return df


def run_camera_eval(
    all_results, Dll_samples, Dhl_samples, omega,
    mu_z, std_z, metric_name,
    n_trials=5,
    alpha_values=(0.0, 1.0),
):
    camera_transform = TV.Compose([
        TV.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        TV.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1
        ),
    ])

    eval_records = []

    total_methods_trained = 0
    for method_group_key, method_data_inner in all_results.items():
        if isinstance(method_data_inner, dict):
            for fold_key, fold_data in method_data_inner.items():
                if isinstance(fold_data, dict) and 'error' not in fold_data:
                    total_methods_trained += len(fold_data)

    total_configs = total_methods_trained * len(alpha_values) * n_trials
    if total_configs == 0:
        print("No valid training results in all_results.")
        return pd.DataFrame()

    pbar = tqdm(total=total_configs, desc="Evaluating camera shifts")

    for method_group_key, method_results_inner in all_results.items():
        if not isinstance(method_results_inner, dict):
            continue

        for fold_key, fold_data in method_results_inner.items():
            if not fold_key.startswith('fold_'):
                continue
            if 'error' in fold_data:
                continue

            fold_idx = int(fold_key.split('_')[-1])

            for run_key, run_result in fold_data.items():
                if 'error' in run_result or run_result.get('T_matrix') is None:
                    pbar.update(len(alpha_values) * n_trials)
                    continue

                T_matrix = run_result['T_matrix']
                test_idx = run_result['test_indices']
                if test_idx is None:
                    pbar.update(len(alpha_values) * n_trials)
                    continue

                if method_group_key.startswith('diroca_'):
                    eval_method_name = f"DiRoCA ({run_key})"
                elif method_group_key == 'gradca':
                    eval_method_name = "GradCA"
                elif method_group_key == 'baryca':
                    eval_method_name = "BaryCA"
                elif method_group_key == 'abslingam':
                    eval_method_name = run_key
                else:
                    eval_method_name = f"{method_group_key}_{run_key}"

                # Determine view type
                T_arr = T_matrix if isinstance(T_matrix, torch.Tensor) else torch.tensor(T_matrix)
                T_shape = tuple(T_arr.shape)
                is_opt_view  = (T_shape == (64, 3072))
                is_full_view = (T_shape == (84, 3092))

                if not (is_opt_view or is_full_view):
                    print(f"[Warning] Unexpected T shape {T_shape} for {eval_method_name}. Skipping camera eval.")
                    pbar.update(len(alpha_values) * n_trials)
                    continue

                for alpha_cam in alpha_values:
                    for trial in range(n_trials):
                        trial_errors = []

                        for iota, eta in list(omega.items()):
                            try:
                                if iota not in Dll_samples or eta not in Dhl_samples:
                                    continue

                                ll_images, _, ll_digits, ll_colors = Dll_samples[iota]
                                max_idx = max(test_idx) if len(test_idx) > 0 else -1
                                if max_idx >= len(ll_images):
                                    continue

                                ll_images_test = ll_images[test_idx]          # (N,C,H,W) in [0,1]
                                ll_digits_test = ll_digits[test_idx]
                                ll_colors_test = ll_colors[test_idx]

                                Dhl_test_full = Dhl_samples[eta][test_idx]   # (N,84)
                                Dhl_test_z    = Dhl_test_full[:, 20:]        # (N,64)

                                seed = hash((fold_idx, run_key, float(alpha_cam),
                                             trial, str(iota))) % (2**32)

                                ll_images_shifted_tanh = apply_camera_shifts_cmnist(
                                    ll_images_test, alpha_cam, camera_transform, seed=seed
                                )
                                ll_pixels_shifted_flat = ll_images_shifted_tanh.view(
                                    ll_images_shifted_tanh.shape[0], -1
                                )
                                device = ll_pixels_shifted_flat.device

                                if is_opt_view:
                                    Dll_input = ll_pixels_shifted_flat
                                    T_use     = T_arr.to(device)
                                    Dhl_target = torch.as_tensor(Dhl_test_z, dtype=torch.float32, device=device)
                                else:
                                    T_tensor = T_arr.to(device)
                                    T_use    = T_tensor[20:, :3072]
                                    Dll_input  = ll_pixels_shifted_flat
                                    Dhl_target = torch.as_tensor(Dhl_test_z, dtype=torch.float32, device=device)

                                err_scalar = calculate_empirical_error_flat(
                                    T_use,
                                    Dll_input,
                                    Dhl_target,
                                    metric_name=metric_name,
                                    mu_z=mu_z,
                                    std_z=std_z
                                )
                                if not np.isnan(err_scalar):
                                    trial_errors.append(err_scalar)

                            except Exception as e:
                                print(f"[Camera ERROR] {e} | "
                                      f"M={eval_method_name}, F={fold_idx}, "
                                      f"R={run_key}, A={alpha_cam}, T={trial}, Iota={iota}")
                                trial_errors.append(np.nan)

                        record = {
                            'shift_type': 'camera_aug',
                            'method': eval_method_name,
                            'fold': fold_idx,
                            'alpha': float(alpha_cam),
                            'trial': trial,
                            'error': float(np.nanmean(trial_errors)) if trial_errors else np.nan,
                        }
                        if method_group_key.startswith('diroca_'):
                            record['train_epsilon'] = run_result.get('epsilon', np.nan)
                            record['train_delta']   = run_result.get('delta', np.nan)

                        eval_records.append(record)
                        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(eval_records)
    return df


# ===============================================================
# 6. NORMS + CONDITION NUMBER
# ===============================================================

def compute_T_norms_and_condition_numbers(all_results):
    records = []

    def extract_T_pix_to_z(T):
        T = T if isinstance(T, torch.Tensor) else torch.tensor(T, dtype=torch.float32)
        if T.ndim != 2:
            raise ValueError(f"T must be 2D, got {tuple(T.shape)}")

        if T.shape[1] == 3072:
            return T
        if T.shape == (84, 3092):
            return T[20:, :3072]
        if T.shape[1] > 3072:
            return T[:, :3072]
        return T

    for method_group_key, folds in all_results.items():
        if not isinstance(folds, dict):
            continue

        for fold_key, fold_data in folds.items():
            if not fold_key.startswith("fold_"):
                continue
            if not isinstance(fold_data, dict) or "error" in fold_data:
                continue

            fold_idx = int(fold_key.split("_")[-1])

            for run_key, run_result in fold_data.items():
                if not isinstance(run_result, dict):
                    continue
                if "error" in run_result or run_result.get("T_matrix") is None:
                    continue

                T_raw = run_result["T_matrix"]
                try:
                    T = extract_T_pix_to_z(T_raw)
                except Exception as e:
                    print(f"[skip] could not parse T for {method_group_key}/{fold_key}/{run_key}: {e}")
                    continue

                if method_group_key.startswith("diroca_"):
                    method_name = f"DiRoCA ({run_key})"
                elif method_group_key == "gradca":
                    method_name = "GradCA"
                elif method_group_key == "baryca":
                    method_name = "BaryCA"
                elif method_group_key == "abslingam":
                    method_name = run_key
                else:
                    method_name = f"{method_group_key}_{run_key}"

                with torch.no_grad():
                    fro_norm = torch.linalg.norm(T, ord='fro').item()
                    try:
                        spec_norm = torch.linalg.norm(T, ord=2).item()
                    except RuntimeError:
                        spec_norm = float("nan")

                    try:
                        s = torch.linalg.svdvals(T)
                        s_max = s.max().item()
                        s_min = s.min().item()
                        if s_min <= 0:
                            cond_number = np.inf
                        else:
                            cond_number = float(s_max / s_min)
                    except RuntimeError:
                        cond_number = float("nan")

                eps_train   = run_result.get("epsilon", np.nan)
                delta_train = run_result.get("delta", np.nan)

                records.append({
                    "method_group": method_group_key,
                    "run_key": run_key,
                    "fold": fold_idx,
                    "method": method_name,
                    "T_shape": tuple(T.shape),
                    "fro_norm": fro_norm,
                    "spec_norm": spec_norm,
                    "cond_number": cond_number,
                    "epsilon_train": float(eps_train) if eps_train is not None else np.nan,
                    "delta_train": float(delta_train) if delta_train is not None else np.nan,
                })

    df = pd.DataFrame(records)
    return df


# ===============================================================
# 7. PLOTTING TO PDF
# ===============================================================

def _plot_bar(ax, df_summary, title, metric_label):
    methods = df_summary.index.tolist()
    means = df_summary["mean"].values
    stds  = df_summary["std"].values

    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def save_results_to_pdf(
    pdf_path,
    metric_name,
    df_iv,
    df_huber,
    df_camera,
    df_norms
):
    metric_label = {
        "mse": "Per-dim MSE (z)",
        "cosine": "Cosine error (1 - mean cos)",
        "znorm": "z-normalized MSE"
    }.get(metric_name, metric_name)

    with PdfPages(pdf_path) as pdf:
        # 1) Semantic IV summary
        if len(df_iv) > 0:
            df_iv_agg = df_iv.groupby("method")[["mean_iv_error"]].agg(["mean", "std"]).sort_values(("mean_iv_error","mean"))
            fig, ax = plt.subplots(figsize=(8,4))
            tmp = df_iv_agg["mean_iv_error"]
            _plot_bar(ax, tmp, f"Semantic IV error ({metric_label})", metric_label)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 2) Huber clean / noisy
        if len(df_huber) > 0:
            for alpha_val, title_suffix in [(0.0, "Clean (alpha=0)"), (1.0, "Fully Noisy (alpha=1)")]:
                df_sub = df_huber[np.isclose(df_huber["alpha"], alpha_val)]
                if len(df_sub) == 0:
                    continue
                df_agg = df_sub.groupby("method")["error"].agg(["mean","std"]).sort_values("mean")

                fig, ax = plt.subplots(figsize=(8,4))
                _plot_bar(ax, df_agg, f"Huber noise – {title_suffix} ({metric_label})", metric_label)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # 3) Camera clean / shifted
        if len(df_camera) > 0:
            for alpha_val, title_suffix in [(0.0, "Clean (alpha=0)"), (1.0, "Shifted (alpha=1)")]:
                df_sub = df_camera[np.isclose(df_camera["alpha"], alpha_val)]
                if len(df_sub) == 0:
                    continue
                df_agg = df_sub.groupby("method")["error"].agg(["mean","std"]).sort_values("mean")

                fig, ax = plt.subplots(figsize=(8,4))
                _plot_bar(ax, df_agg, f"Camera shifts – {title_suffix} ({metric_label})", metric_label)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # 4) Norms & condition numbers
        if len(df_norms) > 0:
            df_norm_agg = df_norms.groupby("method")[["fro_norm","spec_norm","cond_number"]].agg(["mean","std"])

            # Frobenius
            fig, ax = plt.subplots(figsize=(8,4))
            tmp = df_norm_agg["fro_norm"]
            _plot_bar(ax, tmp, "Frobenius norm of T (pixels→z)", "fro_norm")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Spectral
            fig, ax = plt.subplots(figsize=(8,4))
            tmp = df_norm_agg["spec_norm"]
            _plot_bar(ax, tmp, "Spectral norm of T (pixels→z)", "spec_norm")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Condition number (log-scale)
            fig, ax = plt.subplots(figsize=(8,4))
            tmp = df_norm_agg["cond_number"]
            methods = tmp.index.tolist()
            means = tmp["mean"].values
            stds  = tmp["std"].values

            x = np.arange(len(methods))
            ax.bar(x, means)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_yscale("log")
            ax.set_ylabel("cond_number (log scale)")
            ax.set_title("Condition number of T (pixels→z)")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ===============================================================
# 8. MAIN
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="CMNIST evaluation with metric selection + PDF output.")
    parser.add_argument(
        "--metric",
        type=str,
        default="mse",
        choices=["mse", "cosine", "znorm"],
        help="Metric to use: mse (per-dim), cosine (1-mean_cos), znorm (z-normalized MSE)"
    )
    parser.add_argument(
        "--output-pdf",
        type=str,
        required=True,
        help="Path for PDF with summary plots."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of trials per config for Huber/Camera."
    )
    args = parser.parse_args()

    metric_name = args.metric
    output_pdf = args.output_pdf

    print(f"=== CMNIST Evaluation ===")
    print(f"Metric: {metric_name}")
    print(f"Output PDF: {output_pdf}")

    # 1) Load state (YOU must implement load_cmnist_state)
    (
        all_results,
        Dll_samples,
        Dhl_samples,
        omega,
        det_ll_dict_opt,
        det_hl_dict_opt,
        U_ll_hat_run,
        U_hl_hat_run,
        cv_folds
    ) = load_cmnist_state()

    # 2) Global z mean/std
    all_z_list = []
    for eta, Dhl_eta in Dhl_samples.items():
        Dhl_eta = torch.as_tensor(Dhl_eta, dtype=torch.float32)
        z_eta = Dhl_eta[:, 20:]   # (N_eta,64)
        all_z_list.append(z_eta)
    Z_all = torch.cat(all_z_list, dim=0)
    mu_z_global = Z_all.mean(dim=0)
    std_z_global = Z_all.std(dim=0) + 1e-8

    # 3) Semantic IV
    print("\nRunning semantic (IV) evaluation...")
    df_iv = run_semantic_eval_all_methods(
        all_results,
        det_ll_dict_opt,
        det_hl_dict_opt,
        U_ll_hat_run,
        U_hl_hat_run,
        omega,
        metric_name=metric_name,
        mu_z=mu_z_global,
        std_z=std_z_global
    )
    print(df_iv.sort_values("mean_iv_error").head(20))
    print(df_iv.groupby("method")[["mean_iv_error","max_iv_error","var_iv_error"]].agg(["mean","std"]))

    # 4) Huber eval
    print("\nRunning Huber noise evaluation...")
    df_huber = run_huber_eval(
        all_results,
        Dll_samples,
        Dhl_samples,
        omega,
        mu_z_global,
        std_z_global,
        metric_name=metric_name,
        n_trials=args.n_trials,
        noise_scale_for_alpha1=0.5,
        alpha_values=(0.0, 1.0)
    )

    # 5) Camera eval
    print("\nRunning camera-shift evaluation...")
    df_camera = run_camera_eval(
        all_results,
        Dll_samples,
        Dhl_samples,
        omega,
        mu_z_global,
        std_z_global,
        metric_name=metric_name,
        n_trials=args.n_trials,
        alpha_values=(0.0, 1.0)
    )

    # 6) Norms + cond numbers
    print("\nComputing norms and condition numbers...")
    df_norms = compute_T_norms_and_condition_numbers(all_results)

    # 7) Save to PDF
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    print(f"\nSaving summary PDF to {output_pdf} ...")
    save_results_to_pdf(
        output_pdf,
        metric_name,
        df_iv,
        df_huber,
        df_camera,
        df_norms
    )
    print("Done.")


if __name__ == "__main__":
    main()
