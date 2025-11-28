#!/usr/bin/env python3
import argparse
import os
import pickle
import gc

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Utility: load evaluation state saved by cmnist_opt script
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

    # 4) deterministic opt-view dicts (not directly needed here,
    #    we work with Dll_samples / Dhl_samples instead)
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

    # 5) opt-view noise (not needed for this eval)
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
# Low-level transforms: Huber noise & Camera shift
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

    # Use global seeding (version-agnostic)
    torch.manual_seed(int(seed) % (2**32))

    X_corrupt = X_flat.clone()
    X_sub = X_corrupt[:, noise_dims]  # (N, d_sub)

    # Sample mask and noise without generator=
    mask = torch.rand(N, 1, device=device) < alpha
    noise = torch.randn_like(X_sub) * noise_scale + loc

    X_sub_noisy = torch.where(mask, X_sub + noise, X_sub)
    X_corrupt[:, noise_dims] = X_sub_noisy
    return X_corrupt



def apply_camera_shift_cmnist(
    X_flat,
    alpha,
    max_shift=3,
    seed=0,
):
    """
    Simple camera shift: roll the image by up to max_shift pixels
    in both H and W, and interpolate between original and shifted
    according to alpha.

    X_flat: (N, 3072) flattened [-1,1] RGB 32x32
    alpha: 0.0 → no shift; 1.0 → fully shifted
    max_shift: maximum pixel shift for both H and W
    seed: random seed per trial/env
    """
    if alpha <= 0:
        return X_flat

    device = X_flat.device
    N, D = X_flat.shape
    assert D == 3 * 32 * 32, f"Expected 3072 dims, got {D}"

    # Global seeding again
    torch.manual_seed(int(seed) % (2**32))

    X_img = X_flat.view(N, 3, 32, 32)

    # Random shifts in [-max_shift, max_shift]
    shift_h = torch.randint(
        low=-max_shift,
        high=max_shift + 1,
        size=(N,),
        device=device,
    )
    shift_w = torch.randint(
        low=-max_shift,
        high=max_shift + 1,
        size=(N,),
        device=device,
    )

    X_shifted = torch.empty_like(X_img)
    for i in range(N):
        X_shifted[i] = torch.roll(
            X_img[i],
            shifts=(int(shift_h[i].item()), int(shift_w[i].item())),
            dims=(1, 2),
        )

    X_mix = (1.0 - alpha) * X_img + alpha * X_shifted
    return X_mix.view(N, -1)



# ============================================================
# Core: extract pixels→z mapping & compute metrics
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
    # to tensors
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

    # ---- forward: z_pred = T_pix @ x_pix ----
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
    """
    diff = Z_pred - Z_true
    d_z = Z_true.shape[1]

    # per-sample squared L2
    sq_l2 = diff.pow(2).sum(dim=1)        # (N,)
    sq_true = Z_true.pow(2).sum(dim=1)    # (N,)

    mse_sq = sq_l2.mean().item()          # E||diff||^2
    rmse = diff.norm(p=2, dim=1).mean().item()   # E||diff|| (raw L2)

    rel_l2 = (sq_l2 / (sq_true + eps)).mean().item() * 100.0

    norm_l2 = rmse / float(d_z)

    signal_power = sq_true.mean().item() + eps
    noise_power = sq_l2.mean().item() + eps
    snr_db = 10.0 * np.log10(signal_power / noise_power)

    return {
        "mse_sq": float(mse_sq),
        "rmse": float(rmse),
        "rel_l2": float(rel_l2),
        "norm_l2": float(norm_l2),
        "snr_db": float(snr_db),
    }


# ============================================================
# Main evaluation loops (Huber & Camera)
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


def evaluate_shift(
    all_results,
    omega,
    Dll_samples,
    Dhl_samples,
    shift_type="huber",
    N_TRIALS=5,
    ALPHA_VALUES=(0.0, 1.0),
    noise_scale_for_alpha1=0.5,
    max_camera_shift=3,
):
    """
    Generic evaluation over all methods for a given shift_type ∈ {"huber", "camera"}.

    Returns a DataFrame with columns:
      ['shift_type', 'metric', 'method', 'fold', 'alpha', 'trial', 'value']
    """
    assert shift_type in {"huber", "camera"}

    records = []

    # Count configs (for tqdm)
    total_configs = 0
    for mg, folds in all_results.items():
        if isinstance(folds, dict):
            for fk, fd in folds.items():
                if isinstance(fd, dict) and "error" not in fd:
                    total_configs += len(fd)
    total_configs *= len(ALPHA_VALUES) * N_TRIALS

    desc = f"Evaluating ({shift_type})"
    pbar = tqdm(total=total_configs, desc=desc)

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
                    # still progress the bar for consistency
                    pbar.update(len(ALPHA_VALUES) * N_TRIALS)
                    continue

                T_matrix = run_result["T_matrix"]
                test_idx = run_result["test_indices"]

                method_name = method_display_name(method_group_key, run_key)

                for alpha in ALPHA_VALUES:
                    if np.isclose(alpha, 0.0):
                        noise_scale = 0.0
                    else:
                        noise_scale = noise_scale_for_alpha1

                    for trial in range(N_TRIALS):
                        # one set of metrics aggregated over interventions
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

                                # pick test subset
                                ll_imgs_01 = ll_imgs[test_idx]      # (N, 3, 32, 32) in [0,1]
                                ll_digits_test = ll_digits[test_idx]
                                ll_colors_test = ll_colors[test_idx]
                                Dhl_clean_test = Dhl_clean[test_idx]  # (N, 20 + d_z)

                                # convert [0,1] → [-1,1]
                                ll_imgs_tanh = ll_imgs_01 * 2 - 1
                                ll_imgs_flat = ll_imgs_tanh.view(ll_imgs_tanh.shape[0], -1)  # (N,3072)

                                # Apply shift
                                if shift_type == "huber":
                                    seed = hash((fold_idx, run_key, alpha, trial, iota)) % (2**32)
                                    ll_imgs_shift = apply_huber_contamination_cmnist(
                                        ll_imgs_flat,
                                        alpha=alpha,
                                        noise_scale=noise_scale,
                                        noise_dims=LL_PIXELS,
                                        seed=seed,
                                        loc=0.0,
                                    )
                                else:  # camera
                                    seed = hash((fold_idx, run_key, alpha, trial, iota, "camera")) % (2**32)
                                    ll_imgs_shift = apply_camera_shift_cmnist(
                                        ll_imgs_flat,
                                        alpha=alpha,
                                        max_shift=max_camera_shift,
                                        seed=seed,
                                    )

                                # assemble LL full input [pixels | digit10 | color10]
                                ll_digits_onehot = F.one_hot(ll_digits_test, 10).float()
                                ll_colors_onehot = F.one_hot(ll_colors_test, 10).float()

                                Dll_full = torch.cat(
                                    [ll_imgs_shift, ll_digits_onehot, ll_colors_onehot],
                                    dim=1,
                                )

                                # compute z_pred / z_true and all metrics
                                Z_pred, Z_true = extract_z_pred_and_true(
                                    T_matrix,
                                    Dll_full,
                                    Dhl_clean_test,
                                )
                                metric_vals = compute_all_metrics(Z_pred, Z_true)

                                for k in metrics_accumulator.keys():
                                    metrics_accumulator[k].append(metric_vals[k])

                            except Exception as e:
                                print(f"[{shift_type.upper()} ERROR] {e} | Context {method_name}, fold={fold_idx}, alpha={alpha}")
                                for k in metrics_accumulator.keys():
                                    metrics_accumulator[k].append(np.nan)

                        # average over all interventions for this trial
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
                                }
                            )

                        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(records)
    return df


# ============================================================
# Aggregation & PDF reporting
# ============================================================

def summarize_metrics(df_all):
    """
    Aggregates mean ± std across folds and trials per:
      (shift_type, metric, alpha, method)
    """
    df_summary = (
        df_all
        .groupby(["shift_type", "metric", "alpha", "method"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return df_summary


def make_pdf_report(df_summary, out_pdf):
    """
    Create a multipage PDF with one table per:
       (shift_type, metric, alpha)
    """
    metrics_order = ["mse_sq", "rmse", "rel_l2", "norm_l2", "snr_db"]
    metric_labels = {
        "mse_sq": "Squared L2 (MSE)",
        "rmse": "Raw L2 (RMSE)",
        "rel_l2": "Relative L2 (%)",
        "norm_l2": "Normalized L2 (per-dim)",
        "snr_db": "SNR (dB)",
    }

    shift_labels = {
        "huber": "Huber Noise",
        "camera": "Camera Shift",
    }

    with PdfPages(out_pdf) as pdf:
        for shift_type in sorted(df_summary["shift_type"].unique()):
            for metric in metrics_order:
                df_metric = df_summary[df_summary["metric"] == metric]
                if df_metric.empty:
                    continue

                df_shift = df_metric[df_metric["shift_type"] == shift_type]
                if df_shift.empty:
                    continue

                for alpha in sorted(df_shift["alpha"].unique()):
                    df_alpha = df_shift[df_shift["alpha"] == alpha].sort_values("mean")

                    fig, ax = plt.subplots(figsize=(11, 6))
                    ax.axis("off")

                    title = f"{shift_labels.get(shift_type, shift_type.capitalize())} – {metric_labels.get(metric, metric)} (alpha={alpha})"
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

    print(f"\nPDF report saved to: {out_pdf}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CMNIST evaluation: Huber & Camera shifts with multiple metrics.")
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
        help="Number of random trials per (method, fold, alpha, shift_type).",
    )
    parser.add_argument(
        "--noise-scale-alpha1",
        type=float,
        default=0.5,
        help="Noise std for alpha=1.0 in Huber contamination.",
    )
    parser.add_argument(
        "--max-camera-shift",
        type=int,
        default=3,
        help="Maximum pixel shift (up/down/left/right) for camera shift.",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default=os.path.join("data", "cmnist", "results_empirical", "cmnist_eval_metrics.pdf"),
        help="Output PDF path for summary tables.",
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

    # ---------------- Huber evaluation ----------------
    print("\n================ Huber evaluation (all metrics) ================")
    df_huber = evaluate_shift(
        all_results=all_results,
        omega=omega,
        Dll_samples=Dll_samples,
        Dhl_samples=Dhl_samples,
        shift_type="huber",
        N_TRIALS=args.n_trials,
        ALPHA_VALUES=(0.0, 1.0),
        noise_scale_for_alpha1=args.noise_scale_alpha1,
        max_camera_shift=args.max_camera_shift,
    )
    print("\nHuber evaluation raw head:")
    print(df_huber.head())

    # ---------------- Camera evaluation ----------------
    print("\n================ Camera-shift evaluation (all metrics) ================")
    df_camera = evaluate_shift(
        all_results=all_results,
        omega=omega,
        Dll_samples=Dll_samples,
        Dhl_samples=Dhl_samples,
        shift_type="camera",
        N_TRIALS=args.n_trials,
        ALPHA_VALUES=(0.0, 1.0),
        noise_scale_for_alpha1=args.noise_scale_alpha1,  # unused in camera
        max_camera_shift=args.max_camera_shift,
    )
    print("\nCamera evaluation raw head:")
    print(df_camera.head())

    # Combine and summarize
    df_all = pd.concat([df_huber, df_camera], ignore_index=True)

    # Save raw results
    out_pkl = os.path.join(args.eval_dir, "cmnist_eval_all_metrics.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(df_all, f)
    print(f"\n✓ Saved raw metric evaluations to {out_pkl}")

    df_summary = summarize_metrics(df_all)

    print("\n=== Aggregated summary (mean ± std) ===")
    print(df_summary.head(30))

    # PDF report
    make_pdf_report(df_summary, args.out_pdf)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
