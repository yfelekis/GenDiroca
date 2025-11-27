#!/usr/bin/env python3
import argparse
import os
import gc
import pickle
import torch

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb  # unused in script mode but harmless

# --- Vision / models ---
import torchvision
from torchvision import transforms

# --- sklearn ---
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- Project modules ---
from cmnist_train_new import ImageColorizerUNet, HLCellMeansShrunkVec
from operations import Intervention

# --- Plotting ---
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import joblib
import pandas as pd


# ============================================================
# Helper: empirical Wasserstein-type radius
# ============================================================
def compute_empirical_radius(N, eta=0.05, c1=1000.0, c2=1.0, alpha=2.0, m=1):
    """
    Compute an empirical Wasserstein-type radius (heuristic formula).
    """
    if N <= 1:
        return 0.0
    term1 = c1 * (np.log(N) / N) ** (1.0 / alpha)
    term2 = c2 * np.sqrt(m * np.log(1.0 / eta) / N)
    return term1 + term2


# ============================================================
# Data loading and basic setup
# ============================================================
def load_cmnist_data(data_dir):
    print(f"Loading CMNIST data and models from: {data_dir}")

    # -----------------------------
    # Load intervention mapping
    # -----------------------------
    # omega: dict[label_str -> label_str], e.g. "obs", "D=6", "C=7", ...
    omega = torch.load(os.path.join(data_dir, "intervention_mapping.pkl"))
    Ill_relevant = list(omega.keys())
    Ihl_relevant = list(omega.values())

    print("Interventions loaded:")
    print("  Ill_relevant:", Ill_relevant)
    print("  Ihl_relevant:", Ihl_relevant)

    # -----------------------------
    # Load LL / HL samples
    # -----------------------------
    # Dll_samples[label] = (imgs_01, shapes_norm, digits, colors)
    Dll_samples = torch.load(os.path.join(data_dir, "dll_samples.pkl"))

    # Dhl_samples[label] = (N, 20 + d_z)
    Dhl_samples = torch.load(os.path.join(data_dir, "dhl_samples.pkl"))

    obs_key = "obs"
    assert obs_key in Dll_samples, f"'obs' not found in Dll_samples keys: {Dll_samples.keys()}"
    assert obs_key in Dhl_samples, f"'obs' not found in Dhl_samples keys: {Dhl_samples.keys()}"

    d_z = Dhl_samples[obs_key].shape[1] - 20

    # -----------------------------
    # Load low-level U-Net model
    # -----------------------------
    ll_model_state = torch.load(os.path.join(data_dir, "ll_model_unet.pth"), map_location="cpu")
    ll_model = ImageColorizerUNet()
    ll_model.load_state_dict(ll_model_state)
    ll_model.eval()

    # Abduced low-level noise U_ll_hat (in tanh / [-1,1] space, shape (N, 3, 32, 32))
    U_ll_hat = torch.load(os.path.join(data_dir, "U_ll_hat.pkl"))

    # -----------------------------
    # Load high-level cell-means model on z
    # -----------------------------
    hl_model = joblib.load(os.path.join(data_dir, "hl_model_cellmeans_z.joblib"))

    # Abduced high-level noise U_hl_hat in latent space z, shape (N, d_z)
    U_hl_hat = torch.load(os.path.join(data_dir, "U_hl_hat_z.pkl"))

    print("\nData & models loaded successfully!")
    print(f"  - Observational LL images shape : {Dll_samples[obs_key][0].shape}   # imgs_01")
    print(f"  - Observational LL shapes shape : {Dll_samples[obs_key][1].shape}   # shapes_norm")
    print(f"  - Observational HL tensor shape : {Dhl_samples[obs_key].shape}      # 20 + d_z (d_z={d_z})")
    print(f"  - Low-level noise U_ll_hat      : {U_ll_hat.shape}")
    print(f"  - High-level noise U_hl_hat     : {U_hl_hat.shape}")

    return (
        omega,
        Ill_relevant,
        Ihl_relevant,
        Dll_samples,
        Dhl_samples,
        ll_model,
        hl_model,
        U_ll_hat,
        U_hl_hat,
        d_z,
    )


# Variable names (LL / HL)
D_LL, C_LL, P_LL = "Digit", "Color", "Pixels"
D_HL, C_HL, I_HL = "Digit_", "Color_", "Image_"  # I_HL stands for the latent z block


# =============================================================================
# Build Intervention OBJECTS from string labels
# =============================================================================
def parse_intervention_str(label_str, D_name, C_name):
    """
    Convert label strings like:
      "obs", "D=6", "C=7", "D=6,C=7"
    into Intervention objects (or None for obs).
    """
    if label_str is None or label_str == "obs":
        return None
    iv = {}
    for part in label_str.split(","):
        part = part.strip()
        if part.startswith("D="):
            iv[D_name] = int(part.split("=")[1])
        elif part.startswith("C="):
            iv[C_name] = int(part.split("=")[1])
    return Intervention(iv)


def build_intervention_objects(Ill_relevant, Ihl_relevant):
    ll_interventions = {lab: parse_intervention_str(lab, D_LL, C_LL) for lab in Ill_relevant}
    hl_interventions = {lab: parse_intervention_str(lab, D_HL, C_HL) for lab in Ihl_relevant}

    print("\nIntervention objects constructed.")
    print(
        "  Example LL:",
        {k: (v.vv() if v is not None else {}) for k, v in list(ll_interventions.items())[:3]},
    )
    print(
        "  Example HL:",
        {k: (v.vv() if v is not None else {}) for k, v in list(hl_interventions.items())[:3]},
    )

    return ll_interventions, hl_interventions


# ============================================================
# Parent info helpers
# ============================================================
def get_obs_key(sample_dict, name=""):
    # prefer explicit "obs" if present
    if "obs" in sample_dict:
        return "obs"
    if None in sample_dict:
        return None
    k0 = next(iter(sample_dict.keys()))
    print(f"[Warning] No explicit obs key for {name}. Using first key: {k0}")
    return k0


def rebuild_parent_infos(Dll_samples, Dhl_samples):
    iota_obs = get_obs_key(Dll_samples, "Dll_samples")
    eta_obs = get_obs_key(Dhl_samples, "Dhl_samples")

    final_images_obs, img_shapes_obs, digits_obs, colors_obs = Dll_samples[iota_obs]

    parent_info_ll_tuple = (
        final_images_obs,
        img_shapes_obs,
        digits_obs,
        colors_obs,
    )

    parent_info_hl = Dhl_samples[eta_obs][:, :20].cpu().numpy()

    print("Rebuilt parent infos:")
    print(
        f"  iota_obs={iota_obs}, eta_obs={eta_obs}"
    )
    print(
        "  parent_info_ll_tuple: images={}, shapes={}, digits={}, colors={}".format(
            final_images_obs.shape, img_shapes_obs.shape, digits_obs.shape, colors_obs.shape
        )
    )
    print(f"  parent_info_hl shape: {parent_info_hl.shape}")

    return iota_obs, eta_obs, parent_info_ll_tuple, parent_info_hl


# ============================================================
# Deterministic functions (FIXED)
# ============================================================
def det_hl_func_FIXED(hl_model, parent_info_hl, intervention):
    """
    High-level deterministic function (FIXED).

    parent_info_hl: numpy array (N,20) = [Xd(10) | Xc(10)]
    intervention: Intervention object (or None)

    Returns torch.Tensor (N, 20 + d_z):
        [Xd | Xc | z_pred]
    """
    features = parent_info_hl.copy()

    if intervention is not None and hasattr(intervention, "vv"):
        iv_dict = intervention.vv()

        # Digit_ do()
        if D_HL in iv_dict:
            new_digits = np.zeros_like(features[:, :10])
            new_digits[:, iv_dict[D_HL]] = 1
            features[:, :10] = new_digits

        # Color_ do()
        if C_HL in iv_dict:
            new_colors = np.zeros_like(features[:, 10:20])
            new_colors[:, iv_dict[C_HL]] = 1
            features[:, 10:20] = new_colors

    Xd = features[:, :10]
    Xc = features[:, 10:20]
    z_pred = hl_model.predict(Xd, Xc)

    if z_pred.ndim == 1:
        z_pred = z_pred.reshape(-1, 1)

    full_vector = np.concatenate([Xd, Xc, z_pred], axis=1)
    return torch.tensor(full_vector, dtype=torch.float32)


def det_ll_func_FIXED(ll_model, parent_info_ll_tuple, intervention):
    """
    Low-level deterministic function (FIXED).

    parent_info_ll_tuple:
        (final_images_obs, img_shapes_obs, digits_obs, colors_obs)

    Returns torch.Tensor (N, 3092):
        [pixels_flat(3072) | digit_onehot(10) | color_onehot(10)]
    """
    _, img_shapes, digits, colors = parent_info_ll_tuple
    n_samples = digits.shape[0]

    if intervention is not None and hasattr(intervention, "vv"):
        iv_dict = intervention.vv()
        if D_LL in iv_dict:
            digits = torch.full_like(digits, iv_dict[D_LL])
        if C_LL in iv_dict:
            colors = torch.full_like(colors, iv_dict[C_LL])

    ll_model.eval()
    device = next(ll_model.parameters()).device
    img_shapes = img_shapes.to(device)
    digits_dev = digits.to(device).long()
    colors_dev = colors.to(device).long()

    with torch.no_grad():
        predicted_images = ll_model(img_shapes, digits_dev, colors_dev)  # [-1,1]

    flattened_images = predicted_images.cpu().view(n_samples, -1)  # (N,3072)
    digits_onehot = F.one_hot(digits, num_classes=10).float()
    colors_onehot = F.one_hot(colors, num_classes=10).float()

    full_vector = torch.cat([flattened_images, digits_onehot, colors_onehot], dim=1)
    return full_vector


def frob_diff(A, B):
    return torch.norm(A - B, p="fro").item()


def rebuild_deterministic_dicts(
    omega,
    Ill_relevant,
    Ihl_relevant,
    ll_interventions,
    hl_interventions,
    ll_model,
    hl_model,
    parent_info_ll_tuple,
    parent_info_hl,
):
    print("\n" + "=" * 80)
    print("REBUILD + VERIFY DETERMINISTIC DICTIONARIES (LL + HL) — FULL SCRIPT")
    print("=" * 80)

    assert set(ll_interventions.keys()) == set(Ill_relevant), "LL intervention labels mismatch vs omega keys."
    assert set(hl_interventions.keys()) == set(Ihl_relevant), "HL intervention labels mismatch vs omega values."

    print("✅ Using ll_interventions / hl_interventions from construction.")

    det_hl_dict = {}
    for lab in tqdm(Ihl_relevant, desc="HL deterministic (FIXED)"):
        det_hl_dict[lab] = det_hl_func_FIXED(
            hl_model,
            parent_info_hl,
            hl_interventions[lab],
        )

    det_ll_dict = {}
    for lab in tqdm(Ill_relevant, desc="LL deterministic (FIXED)"):
        det_ll_dict[lab] = det_ll_func_FIXED(
            ll_model,
            parent_info_ll_tuple,
            ll_interventions[lab],
        )

    print(f"det_ll_dict['obs'] shape: {det_ll_dict['obs'].shape}")
    print(f"det_hl_dict['obs'] shape: {det_hl_dict['obs'].shape}")

    # Quick verification
    print("\n" + "=" * 80)
    print("VERIFYING DETERMINISTIC DICTIONARIES")
    print("=" * 80)

    print("\n[1] Checking if interventions are objects or strings...")
    ll_key_types = {type(k) for k in Ill_relevant}
    hl_key_types = {type(k) for k in Ihl_relevant}
    print("  LL key types:", ll_key_types)
    print("  HL key types:", hl_key_types)

    ll_any_vv = any(hasattr(ll_interventions[k], "vv") for k in Ill_relevant if ll_interventions[k] is not None)
    hl_any_vv = any(hasattr(hl_interventions[k], "vv") for k in Ihl_relevant if hl_interventions[k] is not None)
    print("  Any LL intervention has .vv()? ", ll_any_vv)
    print("  Any HL intervention has .vv()? ", hl_any_vv)

    base_key = "obs"
    base_ll = det_ll_dict[base_key]
    ll_diffs = []
    print(f"\n[det_ll_dict] Pairwise diffs vs base key = '{base_key}'")
    for k in Ill_relevant:
        diff_val = frob_diff(base_ll, det_ll_dict[k])
        ll_diffs.append(diff_val)
        if k != base_key:
            print(f"  diff('{base_key}', '{k}') = {diff_val:.6e}")
    print(
        f"\n  det_ll_dict max diff across all envs: min={np.min(ll_diffs):.3e}, "
        f"median={np.median(ll_diffs):.3e}, max={np.max(ll_diffs):.3e}"
    )

    base_hl = det_hl_dict[base_key]
    hl_diffs = []
    print(f"\n[det_hl_dict] Pairwise diffs vs base key = '{base_key}'")
    for k in Ihl_relevant:
        diff_val = frob_diff(base_hl, det_hl_dict[k])
        hl_diffs.append(diff_val)
        if k != base_key:
            print(f"  diff('{base_key}', '{k}') = {diff_val:.6e}")
    print(
        f"\n  det_hl_dict max diff across all envs: min={np.min(hl_diffs):.3e}, "
        f"median={np.median(hl_diffs):.3e}, max={np.max(hl_diffs):.3e}"
    )

    print("\n[4] Checking HL predict() contract (Xd,Xc) vs old features usage...")
    features = parent_info_hl.copy()
    Xd = features[:, :10]
    Xc = features[:, 10:20]
    z_correct = hl_model.predict(Xd, Xc)
    z_wrong = hl_model.predict(features, features)  # old buggy call

    mean_err = np.mean(np.abs(z_correct - z_wrong))
    max_err = np.max(np.abs(z_correct - z_wrong))
    print(f"  mean |z_correct - z_wrong| = {mean_err:.6e}")
    print(f"  max  |z_correct - z_wrong| = {max_err:.6e}")

    print("\n[5] Checking which env pairs would be used by empirical_objective...")
    used_pairs = []
    for iota, eta in omega.items():
        ok = (iota in det_ll_dict) and (eta in det_hl_dict)
        if ok:
            used_pairs.append((iota, eta))
    print(f"  Used env pairs: {len(used_pairs)} / {len(omega)}")

    print("\n" + "=" * 80)
    print("DONE.")
    print("=" * 80)

    return det_ll_dict, det_hl_dict


# ============================================================
# Subsampling
# ============================================================
def subsample_data(
    Dll_samples,
    Dhl_samples,
    U_ll_hat,
    U_hl_hat,
    det_ll_dict,
    det_hl_dict,
    N_SUB=2000,
):
    print("\n" + "=" * 20 + f" Subsampling to N_SUB = {N_SUB} " + "=" * 20)

    iota_obs = get_obs_key(Dll_samples, name="Dll_samples")
    eta_obs = get_obs_key(Dhl_samples, name="Dhl_samples")
    print(f"Using observational keys: iota_obs = {iota_obs}, eta_obs = {eta_obs}")

    N_total = Dll_samples[iota_obs][0].shape[0]
    print(f"Total available samples per intervention: N_total = {N_total}")

    if N_SUB >= N_total:
        print("N_SUB >= N_total: no subsampling will be applied.")
        sub_idx = torch.arange(N_total, dtype=torch.long)
    else:
        rng = np.random.default_rng(23)
        perm = rng.permutation(N_total)
        sub_idx = torch.tensor(perm[:N_SUB], dtype=torch.long)
        print(f"Using a random subset of size {N_SUB} from {N_total} samples.")

    Dll_samples_sub = {}
    for iota, tpl in Dll_samples.items():
        final_images, img_shapes, digits, colors = tpl
        Dll_samples_sub[iota] = (
            final_images[sub_idx],
            img_shapes[sub_idx],
            digits[sub_idx],
            colors[sub_idx],
        )

    print(f"Dll_samples: reduced to N_SUB={Dll_samples_sub[iota_obs][0].shape[0]} for all interventions.")

    Dhl_samples_sub = {}
    for eta, tensor in Dhl_samples.items():
        Dhl_samples_sub[eta] = tensor[sub_idx]

    print(f"Dhl_samples: reduced to N_SUB={Dhl_samples_sub[eta_obs].shape[0]} for all interventions.")

    U_ll_hat_sub = U_ll_hat[sub_idx]
    U_hl_hat_sub = U_hl_hat[sub_idx]
    print(f"U_ll_hat: {U_ll_hat.shape} → {U_ll_hat_sub.shape}")
    print(f"U_hl_hat: {U_hl_hat.shape} → {U_hl_hat_sub.shape}")

    det_ll_dict_sub = {k: v[sub_idx] for k, v in det_ll_dict.items()}
    det_hl_dict_sub = {k: v[sub_idx] for k, v in det_hl_dict.items()}

    some_iota = next(iter(det_ll_dict_sub.keys()))
    some_eta = next(iter(det_hl_dict_sub.keys()))
    print(f"det_ll_dict['{some_iota}']: {det_ll_dict[some_iota].shape} → {det_ll_dict_sub[some_iota].shape}")
    print(f"det_hl_dict['{some_eta}']: {det_hl_dict[some_eta].shape} → {det_hl_dict_sub[some_eta].shape}")

    print("\n✅ Subsampling complete. All subsequent training will use the reduced dataset.")

    # Optimization-only views
    det_ll_dict_opt = {k: v[:, :3072].contiguous() for k, v in det_ll_dict_sub.items()}
    det_hl_dict_opt = {k: v[:, 20:].contiguous() for k, v in det_hl_dict_sub.items()}

    print("\nCreated optimization-only deterministic dictionaries:")
    print(f"  det_ll_dict_opt['obs'] shape: {det_ll_dict_opt['obs'].shape}   # pixels only")
    print(f"  det_hl_dict_opt['obs'] shape: {det_hl_dict_opt['obs'].shape}   # z only")

    return (
        Dll_samples_sub,
        Dhl_samples_sub,
        U_ll_hat_sub,
        U_hl_hat_sub,
        det_ll_dict_sub,
        det_hl_dict_sub,
        det_ll_dict_opt,
        det_hl_dict_opt,
        iota_obs,
    )


# ============================================================
# CV folds
# ============================================================
def setup_cv_folds(data_tuple, n_splits=5, seed=23, iota_obs="obs"):
    print("\n" + "=" * 20 + " Setting up cross-validation " + "=" * 20)
    print(f"Observational LL key: {iota_obs}")

    print(f"Setting up {n_splits}-fold cross-validation...")
    num_samples = data_tuple[0].shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(np.arange(num_samples)):
        folds.append({"train": train_idx, "test": test_idx})
    print(f"✓ CV folds created. Each fold will have ~{len(folds[0]['train'])} training samples.")
    print(f"Created {len(folds)} CV folds (using observational key = {iota_obs})")
    return folds


# ============================================================
# Noise flattening
# ============================================================
def fix_noise_data(U_ll_hat, U_hl_hat):
    """
    Prepare noise data for optimization:
      - LL noise: flatten from (N, 3, 32, 32) → (N, 3072)
      - HL noise: leave as is (already (N, d_z))

    Also returns LL-OPT noise (pixels only), which is what optimization uses.
    """
    N = U_ll_hat.shape[0]

    if U_ll_hat.ndim == 4 and U_ll_hat.shape[1:] == (3, 32, 32):
        U_ll_hat_fixed = U_ll_hat.reshape(N, -1)  # (N, 3072)
        print(f"Flattened U_ll_hat from {tuple(U_ll_hat.shape)} → {tuple(U_ll_hat_fixed.shape)}")
    else:
        raise ValueError(f"Unexpected U_ll_hat shape: {U_ll_hat.shape}, expected (N,3,32,32)")

    if U_hl_hat.ndim == 2:
        U_hl_hat_fixed = U_hl_hat
        print(f"U_hl_hat retained as {tuple(U_hl_hat_fixed.shape)} (latent z noise)")
    else:
        raise ValueError(f"Unexpected U_hl_hat shape: {U_hl_hat.shape}, expected (N, d_z) 2D tensor")

    U_ll_hat_opt = U_ll_hat_fixed
    U_hl_hat_opt = U_hl_hat_fixed

    print(f"\nFinal corrected shapes:")
    print(f"  U_ll_hat_fixed: {tuple(U_ll_hat_fixed.shape)}")
    print(f"  U_hl_hat_fixed: {tuple(U_hl_hat_fixed.shape)}")
    print(f"  U_ll_hat_opt  : {tuple(U_ll_hat_opt.shape)}   # used in optimization")
    print(f"  U_hl_hat_opt  : {tuple(U_hl_hat_opt.shape)}   # ignored if delta=0")

    return U_ll_hat_fixed, U_hl_hat_fixed, U_ll_hat_opt, U_hl_hat_opt


# ============================================================
# Utility: projection onto Frobenius ball
# ============================================================
def project_onto_frobenius_ball(tensor, radius_limit):
    """
    Project tensor onto a Frobenius norm ball:

        ||X||_F <= radius_limit
    """
    with torch.no_grad():
        current_norm = torch.norm(tensor, p="fro")
        if current_norm > radius_limit:
            tensor.mul_(radius_limit / (current_norm + 1e-12))
    return tensor


# ============================================================
# Abs-LiNGAM: perfect & noisy (pixels → z)
# ============================================================
def perfect_abstraction_opt(px_samples, py_samples, tau_threshold=1e-2):
    T_raw = np.linalg.pinv(px_samples) @ py_samples  # (3072, d_z)
    T_raw_mask = np.abs(T_raw) > tau_threshold
    T_raw = T_raw * T_raw_mask
    return T_raw


def noisy_abstraction_opt(px_samples, py_samples, tau_threshold=1e-1, refit_coeff=False):
    try:
        T_raw_hat = np.linalg.pinv(px_samples) @ py_samples  # (3072, d_z)

        tau_mask_indices = np.argmax(np.abs(T_raw_hat), axis=1)  # (3072,)

        d_z = py_samples.shape[1]
        tau_mask_hat = np.eye(d_z)[tau_mask_indices]  # (3072, d_z)

        tau_mask_hat *= (np.abs(T_raw_hat) > tau_threshold).astype(np.int32)
        T_masked = tau_mask_hat * T_raw_hat

        if not refit_coeff:
            return T_masked

        print("Warning: refit_coeff=True in noisy_abstraction_opt not fully verified.")
        T_refit = T_masked.copy()

        for j in range(d_z):
            block = np.where(tau_mask_hat[:, j] == 1)[0]
            if len(block) == 0:
                continue
            T_block = np.linalg.pinv(px_samples[:, block]) @ py_samples[:, j]
            T_refit[block, j] = T_block

        return T_refit

    except Exception as e:
        print(f"Error in noisy_abstraction_opt: {e}")
        return np.zeros((px_samples.shape[1], py_samples.shape[1]), dtype=px_samples.dtype)


def run_abs_lingam_cmnist(
    Dll_obs_full,
    Dhl_obs_full,
    tau_perfect: float = 1e-2,
    tau_noisy: float = 1e-1,
):
    """
    Abs-LiNGAM in the *OPT view*:

        pixels (LL)  →  latent z (HL)

    We IGNORE digit/color one-hot vectors on both sides.
    """
    if isinstance(Dll_obs_full, torch.Tensor):
        Dll_obs_full = Dll_obs_full.detach().cpu().numpy()
    if isinstance(Dhl_obs_full, torch.Tensor):
        Dhl_obs_full = Dhl_obs_full.detach().cpu().numpy()

    try:
        N_ll, D_ll_full = Dll_obs_full.shape
        N_hl, D_hl_full = Dhl_obs_full.shape
        if N_ll != N_hl:
            raise ValueError(f"N mismatch between LL ({N_ll}) and HL ({N_hl})")

        if D_ll_full < 3072:
            raise ValueError(f"Expected at least 3072 LL dims (pixels), got {D_ll_full}")
        X_pixels = Dll_obs_full[:, :3072]

        if D_hl_full == 84:
            Y_z = Dhl_obs_full[:, 20:]
        elif D_hl_full == 64:
            Y_z = Dhl_obs_full
        else:
            raise ValueError(
                f"Unexpected HL dim {D_hl_full}. Expected 84 ([Xd|Xc|z]) or 64 (z only)."
            )

        T_perfect_raw = perfect_abstraction_opt(X_pixels, Y_z, tau_threshold=tau_perfect)
        T_noisy_raw = noisy_abstraction_opt(X_pixels, Y_z, tau_threshold=tau_noisy, refit_coeff=False)

        T_perfect = T_perfect_raw.T.astype(np.float32)
        T_noisy = T_noisy_raw.T.astype(np.float32)

        return {
            "Perfect": {"T": T_perfect},
            "Noisy": {"T": T_noisy},
        }

    except Exception as e:
        print(f"ERROR during Abs-LiNGAM (pixels→z): {e}")
        return {
            "Perfect": {"T": None},
            "Noisy": {"T": None},
        }


# ============================================================
# Empirical objectives (OPT view)
# ============================================================
def empirical_objective_cmnist_opt(
    T_opt,
    U_ll,
    U_hl,
    Theta_ll,
    det_ll_dict,
    det_hl_dict,
    omega,
):
    """
    OPT-view objective (DiRoCA/GradCA):

      - LL side: pixels only
      - HL side: latent z only
    """
    total_loss = 0.0
    device = T_opt.device

    U_ll = U_ll.to(device)
    U_hl = U_hl.to(device)
    Theta_ll = Theta_ll.to(device)

    num_envs = 0
    N = U_ll.shape[0]

    for iota, eta in omega.items():
        if iota not in det_ll_dict or eta not in det_hl_dict:
            continue

        det_ll_full = det_ll_dict[iota].to(device)  # (N, 3092)
        det_hl_full = det_hl_dict[eta].to(device)  # (N, 84)

        if det_ll_full.shape[0] != N or det_hl_full.shape[0] != N:
            continue

        ll_pixels = det_ll_full[:, :U_ll.shape[1]]  # (N,3072)
        ll_pixels_noisy = ll_pixels + U_ll + Theta_ll

        z_part = det_hl_full[:, 20:]  # (N,d_z)
        z_noisy = z_part + U_hl

        mapped_z = (T_opt @ ll_pixels_noisy.T).T  # (N,d_z)
        diff = mapped_z - z_noisy
        loss = torch.norm(diff, p="fro") ** 2 / N

        total_loss += loss
        num_envs += 1

    return total_loss / max(1, num_envs)


def barycentric_objective_cmnist_opt(
    T,
    U_ll_hat,
    U_hl_hat,
    det_ll_dict_opt,
    det_hl_dict_opt,
    omega,
):
    """
    BaryCA objective in the *opt view*.
    """
    device = T.device

    U_ll_hat = torch.as_tensor(U_ll_hat, dtype=torch.float32, device=device)
    U_hl_hat = torch.as_tensor(U_hl_hat, dtype=torch.float32, device=device)

    if U_ll_hat.ndim > 2:
        U_ll_hat = U_ll_hat.view(U_ll_hat.shape[0], -1)
    if U_hl_hat.ndim > 2:
        U_hl_hat = U_hl_hat.view(U_hl_hat.shape[0], -1)

    ll_list = []
    hl_list = []
    num_envs = 0

    for iota, eta in omega.items():
        if iota in det_ll_dict_opt and eta in det_hl_dict_opt:
            X_ll = det_ll_dict_opt[iota].to(device)
            X_hl = det_hl_dict_opt[eta].to(device)

            if X_ll.shape[0] != U_ll_hat.shape[0] or X_hl.shape[0] != U_hl_hat.shape[0]:
                continue

            ll_list.append(X_ll)
            hl_list.append(X_hl)
            num_envs += 1

    if num_envs == 0:
        return torch.tensor(0.0, device=device)

    avg_ll_pixels = torch.mean(torch.stack(ll_list, dim=0), dim=0)
    avg_hl_z = torch.mean(torch.stack(hl_list, dim=0), dim=0)

    N = avg_ll_pixels.shape[0]

    noisy_ll = avg_ll_pixels + U_ll_hat
    noisy_hl = avg_hl_z + U_hl_hat

    mapped_ll = (T @ noisy_ll.T).T
    diff = mapped_ll - noisy_hl
    loss = torch.norm(diff, p="fro") ** 2 / max(1, N)

    return loss


# ============================================================
# Monitor
# ============================================================
class EmpiricalMonitor:
    """
    Tracks metrics during empirical optimization (DiRoCA / GradCA / BaryCA)
    in the OPT view.
    """

    def __init__(self):
        self.iteration_history = []

    def track_iteration(
        self,
        iteration,
        T_objective,
        T_matrix,
        T_matrix_prev,
        adv_objective=None,
        Theta=None,
        Phi=None,
        epsilon=None,
        delta=None,
    ):
        metrics = {
            "iteration": iteration,
            "T_objective": (
                T_objective.item()
                if isinstance(T_objective, torch.Tensor)
                else float(T_objective)
            ),
            "delta_T_norm": (
                torch.norm(T_matrix - T_matrix_prev, p="fro").item()
                if T_matrix_prev is not None
                else 0.0
            ),
        }

        if adv_objective is not None and not (
            isinstance(adv_objective, float) and np.isnan(adv_objective)
        ):
            metrics["adv_objective"] = (
                adv_objective.item()
                if isinstance(adv_objective, torch.Tensor)
                else float(adv_objective)
            )

        if Theta is not None:
            metrics["theta_norm"] = torch.norm(Theta, p="fro").item()
            if epsilon is not None and Theta.shape[0] > 0:
                metrics["epsilon_boundary"] = epsilon * np.sqrt(Theta.shape[0])
            else:
                metrics["epsilon_boundary"] = np.nan

        if Phi is not None:
            metrics["phi_norm"] = torch.norm(Phi, p="fro").item()
            if delta is not None and Phi.shape[0] > 0:
                metrics["delta_boundary"] = delta * np.sqrt(Phi.shape[0])
            else:
                metrics["delta_boundary"] = np.nan

        self.iteration_history.append(metrics)

    def plot_summary(self, title_prefix: str = ""):
        if not self.iteration_history:
            print("No history to plot.")
            return

        df = pd.DataFrame(self.iteration_history).set_index("iteration")

        has_adv = "adv_objective" in df.columns and not df["adv_objective"].isnull().all()
        has_theta = "theta_norm" in df.columns and not df["theta_norm"].isnull().all()
        has_phi = "phi_norm" in df.columns and not df["phi_norm"].isnull().all()

        num_plots = 2 + int(has_adv) + int(has_theta or has_phi)

        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        fig.suptitle(
            f"{title_prefix} Optimization Convergence Summary (OPT view)",
            fontsize=16,
            y=1.02,
        )

        plot_idx = 0

        axes[plot_idx].plot(df.index, df["T_objective"], marker=".", linestyle="-", label="T Objective (minimize)")
        if has_adv:
            axes[plot_idx].plot(df.index, df["adv_objective"], marker=".", linestyle="-", label="Adv Objective (maximize)")
            axes[plot_idx].legend()
            plot_idx += 1

            axes[plot_idx].plot(df.index, df["adv_objective"], marker=".", linestyle="-")
            axes[plot_idx].set_ylabel("Objective")
            axes[plot_idx].set_title("Adversary Objective vs. Iteration")
            axes[plot_idx].grid(True, alpha=0.5)
        else:
            axes[plot_idx].legend()

        axes[plot_idx].set_ylabel("Objective")
        axes[plot_idx].set_title("Objective vs. Iteration")
        axes[plot_idx].grid(True, alpha=0.5)

        plot_idx += 1
        axes[plot_idx].plot(df.index, df["delta_T_norm"], marker=".", linestyle="-")
        axes[plot_idx].set_ylabel("||ΔT||_F")
        axes[plot_idx].set_title("Change in T (||T_t - T_{t-1}||_F)")
        axes[plot_idx].set_yscale("log")
        axes[plot_idx].grid(True, alpha=0.5)

        if has_theta or has_phi:
            plot_idx += 1
            if has_theta:
                axes[plot_idx].plot(df.index, df["theta_norm"], marker=".", linestyle="-", label="||Theta||_F (LL)")
                if "epsilon_boundary" in df.columns and not df["epsilon_boundary"].isnull().all():
                    eps_val = df["epsilon_boundary"].dropna().iloc[0]
                    axes[plot_idx].axhline(eps_val, linestyle="--", alpha=0.7, label=f"Epsilon boundary ({eps_val:.3f})")

            if has_phi:
                axes[plot_idx].plot(df.index, df["phi_norm"], marker=".", linestyle="-", label="||Phi||_F (HL)")
                if "delta_boundary" in df.columns and not df["delta_boundary"].isnull().all():
                    del_val = df["delta_boundary"].dropna().iloc[0]
                    axes[plot_idx].axhline(del_val, linestyle="--", alpha=0.7, label=f"Delta boundary ({del_val:.3f})")

            axes[plot_idx].set_ylabel("Frobenius norm")
            axes[plot_idx].set_title("Perturbation Norms vs. Iteration")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.5)

        axes[-1].set_xlabel("Iteration")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()


# ============================================================
# Core DiRoCA / GradCA / BaryCA optimizers
# ============================================================
def run_empirical_erica_optimization_fixed(
    U_L,
    U_H,
    det_ll_dict,
    det_hl_dict,
    omega,
    epsilon,
    delta,  # ignored (delta=0)
    eta_min,
    eta_max,
    num_steps_min,
    num_steps_max,
    max_iter,
    tol,
    seed,
    robust_L,
    robust_H,  # ignored
    initialization,
    experiment,
    gain,
    optimizers,
    monitor=None,
    clip_T=10.0,
    clip_Theta=10.0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    method = "erica" if robust_L else "enrico"

    U_L = torch.as_tensor(U_L, dtype=torch.float32).to(device)
    U_H = torch.as_tensor(U_H, dtype=torch.float32).to(device)
    if U_L.ndim > 2:
        U_L = U_L.view(U_L.shape[0], -1)
    if U_H.ndim > 2:
        U_H = U_H.view(U_H.shape[0], -1)

    N, l = U_L.shape
    d_z = U_H.shape[1]

    det_ll_dict_dev = {k: v.to(device) for k, v in det_ll_dict.items()}
    det_hl_dict_dev = {k: v.to(device) for k, v in det_hl_dict.items()}

    T_opt = torch.randn(d_z, l, requires_grad=True, device=device)
    if gain > 0:
        init.xavier_normal_(T_opt, gain=gain)

    requires_grad_adv = (method == "erica")
    if initialization == "zeros":
        Theta = torch.zeros(N, l, requires_grad=requires_grad_adv, device=device)
    elif initialization == "random":
        Theta = torch.randn(N, l, requires_grad=requires_grad_adv, device=device)
    else:
        raise ValueError(f"Unknown initialization: {initialization}")

    Phi = torch.zeros(N, d_z, requires_grad=False, device=device)

    if optimizers == "adam":
        optimizer_T = optim.Adam([T_opt], lr=eta_min)
        optimizer_max = optim.Adam([Theta], lr=eta_max) if robust_L else None
    elif optimizers == "adam_betas":
        optimizer_T = optim.Adam(
            [T_opt],
            lr=eta_min,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True,
        )
        optimizer_max = (
            optim.Adam(
                [Theta],
                lr=eta_max,
                betas=(0.9, 0.999),
                eps=1e-8,
                amsgrad=True,
            )
            if robust_L
            else None
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizers}")

    prev_T_objective = float("inf")
    if monitor is None:
        monitor = EmpiricalMonitor()

    for iteration in tqdm(range(max_iter), desc=f"{method.upper()} Optimization (OPT view)"):
        T_prev = T_opt.detach().clone()
        final_T_objective_in_iter = torch.tensor(np.nan, device=device)
        final_adv_objective_in_iter = torch.tensor(np.nan, device=device)

        for _ in range(num_steps_min):
            optimizer_T.zero_grad()
            T_objective = empirical_objective_cmnist_opt(
                T_opt,
                U_L,
                U_H,
                Theta.detach(),
                det_ll_dict_dev,
                det_hl_dict_dev,
                omega=omega,
            )
            if torch.isnan(T_objective):
                print(f"\nNaN in T objective at iter {iteration + 1}. Stopping.")
                break
            T_objective.backward()
            torch.nn.utils.clip_grad_norm_([T_opt], max_norm=clip_T)
            optimizer_T.step()
            final_T_objective_in_iter = T_objective

        if robust_L:
            for _ in range(num_steps_max):
                optimizer_max.zero_grad()
                max_objective = -empirical_objective_cmnist_opt(
                    T_opt.detach(),
                    U_L,
                    U_H,
                    Theta,
                    det_ll_dict_dev,
                    det_hl_dict_dev,
                    omega=omega,
                )
                if torch.isnan(max_objective):
                    print(f"\nNaN in Max objective at iter {iteration + 1}. Keeping previous Theta.")
                    optimizer_max.zero_grad()
                    break
                max_objective.backward()
                torch.nn.utils.clip_grad_norm_([Theta], max_norm=clip_Theta)
                optimizer_max.step()

                with torch.no_grad():
                    max_norm_theta = epsilon * np.sqrt(N) if epsilon > 0 else 0.0
                    if max_norm_theta > 0:
                        Theta.data = project_onto_frobenius_ball(Theta, max_norm_theta)

            with torch.no_grad():
                final_adv_objective_in_iter = empirical_objective_cmnist_opt(
                    T_opt,
                    U_L,
                    U_H,
                    Theta,
                    det_ll_dict_dev,
                    det_hl_dict_dev,
                    omega=omega,
                )
        else:
            final_adv_objective_in_iter = final_T_objective_in_iter
            Theta.zero_()

        t_obj_item = (
            final_T_objective_in_iter.item()
            if torch.isfinite(final_T_objective_in_iter)
            else np.nan
        )
        adv_obj_item = (
            final_adv_objective_in_iter.item()
            if torch.isfinite(final_adv_objective_in_iter)
            else np.nan
        )

        monitor.track_iteration(
            iteration=iteration + 1,
            T_objective=t_obj_item,
            T_matrix=T_opt,
            T_matrix_prev=T_prev,
            adv_objective=adv_obj_item if robust_L else None,
            Theta=Theta if robust_L else None,
            Phi=None,
            epsilon=epsilon if robust_L else None,
            delta=None,
        )

        if not np.isnan(t_obj_item):
            if abs(prev_T_objective - t_obj_item) < tol:
                print(f"\nConverged at iteration {iteration + 1} (delta_obj < {tol})")
                break
            prev_T_objective = t_obj_item
        elif iteration > 0:
            print(f"\nStopping early due to NaN objective at iteration {iteration + 1}")
            break

    T_final = T_opt.detach().cpu().numpy()
    monitor.plot_summary(title_prefix=f"{method.upper()} Fold (Seed {seed})")

    Theta_final_np = Theta.detach().cpu().numpy()
    Phi_final_np = Phi.detach().cpu().numpy()

    paramsL = {
        "pert_U": Theta_final_np,
        "radius_worst": epsilon if robust_L else 0,
        "pert_hat": U_L.cpu().numpy(),
        "radius": epsilon if robust_L else 0,
    }
    paramsH = {
        "pert_U": Phi_final_np,
        "radius_worst": 0,
        "pert_hat": U_H.cpu().numpy(),
        "radius": 0,
    }
    opt_params = {"L": paramsL, "H": paramsH}

    return opt_params, T_final, Theta_final_np, Phi_final_np, monitor


def run_empirical_bary_optim_cmnist(
    U_ll_hat,
    U_hl_hat,
    det_ll_dict_opt,
    det_hl_dict_opt,
    omega,
    lr=1e-3,
    max_iter=5000,
    tol=1e-4,
    seed=42,
    monitor=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    U_ll_hat = torch.as_tensor(U_ll_hat, dtype=torch.float32, device=device)
    U_hl_hat = torch.as_tensor(U_hl_hat, dtype=torch.float32, device=device)
    if U_ll_hat.ndim > 2:
        U_ll_hat = U_ll_hat.view(U_ll_hat.shape[0], -1)
    if U_hl_hat.ndim > 2:
        U_hl_hat = U_hl_hat.view(U_hl_hat.shape[0], -1)

    N, l = U_ll_hat.shape
    _, h = U_hl_hat.shape

    det_ll_dict_dev = {k: v.to(device) for k, v in det_ll_dict_opt.items()}
    det_hl_dict_dev = {k: v.to(device) for k, v in det_hl_dict_opt.items()}

    T = torch.randn(h, l, device=device, requires_grad=True)
    optimizer = optim.Adam([T], lr=lr)

    prev_loss = float("inf")

    if monitor is None:
        monitor = EmpiricalMonitor()

    for iteration in tqdm(range(max_iter), desc="BaryCA Optimization (OPT view)"):
        T_prev = T.detach().clone()

        optimizer.zero_grad()
        loss = barycentric_objective_cmnist_opt(
            T,
            U_ll_hat,
            U_hl_hat,
            det_ll_dict_dev,
            det_hl_dict_dev,
            omega,
        )

        if torch.isnan(loss):
            print(f"NaN loss encountered at iteration {iteration}. Stopping.")
            break

        loss.backward()
        optimizer.step()

        current_loss_item = loss.item()
        monitor.track_iteration(
            iteration=iteration + 1,
            T_objective=current_loss_item,
            T_matrix=T,
            T_matrix_prev=T_prev,
        )

        if iteration % 50 == 0:
            tqdm.write(f"[BaryCA] Iter {iteration}, Loss={current_loss_item:.6f}")

        if abs(prev_loss - current_loss_item) < tol:
            print(f"Converged at iteration {iteration} (Δloss < {tol})")
            break

        prev_loss = current_loss_item

    print(f"Final BaryCA loss: {current_loss_item:.6f}")
    monitor.plot_summary(title_prefix=f"BaryCA Fold (Seed {seed})")

    T_final = T.detach().cpu().numpy()
    return T_final, monitor


# ============================================================
# Training wrappers
# ============================================================
def train_diroca_single_run(
    cv_folds,
    U_ll_hat,
    U_hl_hat,
    det_ll_dict,
    det_hl_dict,
    omega,
    run_hyperparams,
    fixed_params,
    epsilon,
    delta,
):
    print(f"Starting DiRoCA Training Run (OPT view): eps={epsilon}, del={delta} (delta ignored)")
    print(f"Hyperparameters: {run_hyperparams}")

    diroca_training_results = {}
    monitors = {}

    for i, fold_info in enumerate(cv_folds):
        fold_key = f"fold_{i}"
        print(f"\n--- Processing Fold {i+1}/{len(cv_folds)} ---")
        train_indices, test_indices = fold_info["train"], fold_info["test"]

        U_ll_train = torch.as_tensor(U_ll_hat[train_indices], dtype=torch.float32)
        U_hl_train = torch.as_tensor(U_hl_hat[train_indices], dtype=torch.float32)
        if U_ll_train.ndim > 2:
            U_ll_train = U_ll_train.view(U_ll_train.shape[0], -1)
        if U_hl_train.ndim > 2:
            U_hl_train = U_hl_train.view(U_hl_train.shape[0], -1)

        det_ll_train = {k: v[train_indices] for k, v in det_ll_dict.items()}
        det_hl_train = {k: v[train_indices] for k, v in det_hl_dict.items()}

        current_run_params = {**fixed_params, **run_hyperparams}
        fold_monitor = EmpiricalMonitor()

        run_args = {
            "U_L": U_ll_train,
            "U_H": U_hl_train,
            "det_ll_dict": det_ll_train,
            "det_hl_dict": det_hl_train,
            "omega": omega,
            "epsilon": epsilon,
            "delta": delta,
            **current_run_params,
            "monitor": fold_monitor,
        }

        try:
            opt_params, trained_T, final_Theta, final_Phi, trained_monitor = run_empirical_erica_optimization_fixed(
                **run_args
            )

            method_key = f"eps_{epsilon}_delta_{delta}"
            diroca_training_results[fold_key] = {
                method_key: {
                    "T_matrix": torch.tensor(trained_T, dtype=torch.float32),
                    "optimization_params": opt_params,
                    "test_indices": test_indices,
                    "epsilon": epsilon,
                    "delta": delta,
                    "final_Theta_ll": final_Theta,
                    "final_Phi_hl": final_Phi,
                }
            }
            monitors[fold_key] = trained_monitor
            print(f"  ✓ Training completed for Fold {i+1}")
        except Exception as e:
            print(f"  ✗ ERROR during training for Fold {i+1}: {e}")
            diroca_training_results[fold_key] = {"error": str(e)}
            monitors[fold_key] = None

        # NOTE: keep this break if you only want fold_0 for quick experiments
        break

    print("\nDiRoCA Training complete.")
    return {"diroca": diroca_training_results}, monitors


def train_gradca_single_run(
    cv_folds,
    U_ll_hat,
    U_hl_hat,
    det_ll_dict,
    det_hl_dict,
    omega,
    run_hyperparams,
    fixed_params,
):
    print("Starting GradCA Training (OPT view)...")
    print(f"Hyperparameters: {run_hyperparams}")

    gradca_training_results = {}
    monitors = {}
    gradca_hyperparams = run_hyperparams.copy()

    for i, fold_info in enumerate(cv_folds):
        fold_key = f"fold_{i}"
        print(f"\n--- Processing Fold {i+1}/{len(cv_folds)} ---")
        train_indices, test_indices = fold_info["train"], fold_info["test"]

        U_ll_train = torch.as_tensor(U_ll_hat[train_indices], dtype=torch.float32)
        U_hl_train = torch.as_tensor(U_hl_hat[train_indices], dtype=torch.float32)
        if U_ll_train.ndim > 2:
            U_ll_train = U_ll_train.view(U_ll_train.shape[0], -1)
        if U_hl_train.ndim > 2:
            U_hl_train = U_hl_train.view(U_hl_train.shape[0], -1)

        det_ll_train = {k: v[train_indices] for k, v in det_ll_dict.items()}
        det_hl_train = {k: v[train_indices] for k, v in det_hl_dict.items()}

        current_run_params = {
            **fixed_params,
            **gradca_hyperparams,
            "eta_max": 0.0,
            "num_steps_max": 0,
        }
        fold_monitor = EmpiricalMonitor()

        run_args = {
            "U_L": U_ll_train,
            "U_H": U_hl_train,
            "det_ll_dict": det_ll_train,
            "det_hl_dict": det_hl_train,
            "omega": omega,
            "epsilon": 0.0,
            "delta": 0.0,
            "robust_L": False,
            "robust_H": False,
            **current_run_params,
            "monitor": fold_monitor,
        }

        try:
            opt_params, trained_T, final_Theta, final_Phi, trained_monitor = run_empirical_erica_optimization_fixed(
                **run_args
            )

            method_key = "gradca_run"
            gradca_training_results[fold_key] = {
                method_key: {
                    "T_matrix": torch.tensor(trained_T, dtype=torch.float32),
                    "optimization_params": opt_params,
                    "test_indices": test_indices,
                    "final_Theta_ll": final_Theta,
                    "final_Phi_hl": final_Phi,
                }
            }
            monitors[fold_key] = trained_monitor
            print(f"  ✓ Training completed for Fold {i+1}")
        except Exception as e:
            print(f"  ✗ ERROR during training for Fold {i+1}: {e}")
            gradca_training_results[fold_key] = {"error": str(e)}
            monitors[fold_key] = None

        break

    print("\nGradCA Training complete.")
    return {"gradca": gradca_training_results}, monitors


def train_baryca_single_run(
    cv_folds,
    U_ll_hat,
    U_hl_hat,
    det_ll_dict_opt,
    det_hl_dict_opt,
    omega,
    run_hyperparams,
    fixed_params,
):
    print("Starting BaryCA Training (OPT view)...")
    print(f"Hyperparameters: {run_hyperparams}")

    baryca_training_results = {}
    monitors = {}

    lr = run_hyperparams.get("lr", 1e-3)
    max_iter = run_hyperparams.get("max_iter", 5000)
    tol = run_hyperparams.get("tol", 1e-4)
    seed = fixed_params.get("seed", 42)

    for i, fold_info in enumerate(cv_folds):
        fold_key = f"fold_{i}"
        print(f"\n--- Processing Fold {i+1}/{len(cv_folds)} ---")
        train_indices, test_indices = fold_info["train"], fold_info["test"]

        U_ll_train = torch.as_tensor(U_ll_hat[train_indices], dtype=torch.float32)
        U_hl_train = torch.as_tensor(U_hl_hat[train_indices], dtype=torch.float32)
        if U_ll_train.ndim > 2:
            U_ll_train = U_ll_train.view(U_ll_train.shape[0], -1)
        if U_hl_train.ndim > 2:
            U_hl_train = U_hl_train.view(U_hl_train.shape[0], -1)

        det_ll_train = {k: v[train_indices] for k, v in det_ll_dict_opt.items()}
        det_hl_train = {k: v[train_indices] for k, v in det_hl_dict_opt.items()}

        fold_monitor = EmpiricalMonitor()

        try:
            T_opt, trained_monitor = run_empirical_bary_optim_cmnist(
                U_ll_train,
                U_hl_train,
                det_ll_train,
                det_hl_train,
                omega,
                lr=lr,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                monitor=fold_monitor,
            )

            method_key = "baryca_run"
            baryca_training_results[fold_key] = {
                method_key: {
                    "T_matrix": torch.tensor(T_opt, dtype=torch.float32),
                    "test_indices": test_indices,
                }
            }
            monitors[fold_key] = trained_monitor
            print(f"  ✓ BaryCA training completed for Fold {i+1}")
        except Exception as e:
            print(f"  ✗ ERROR during BaryCA training for Fold {i+1}: {e}")
            baryca_training_results[fold_key] = {"error": str(e)}
            monitors[fold_key] = None

        break

    print("\nBaryCA Training complete.")
    return {"baryca": baryca_training_results}, monitors


# ============================================================
# Abs-LiNGAM wrapper (pixels → z only)
# ============================================================
def train_abslingam_single_run(
    cv_folds,
    det_ll_dict,
    det_hl_dict,
    run_hyperparams,
    obs_key: str = "obs",
):
    """
    Wrapper to compute Abs-LiNGAM for all folds, using the *observational*
    deterministic LL/HL vectors from det_ll_dict/det_hl_dict.

    IMPORTANT:
      - We learn T : pixels (3072) → z (64) ONLY.
      - Digit/color one-hot vectors are *ignored* in the fitting.
      - This matches the OPT setting of DiRoCA / GradCA / BaryCA.
    """
    print("\n" + "=" * 20 + " Training Abs-LiNGAM (pixels → z only) " + "=" * 20)

    abslingam_hyperparams = run_hyperparams or {}
    tau_p = abslingam_hyperparams.get("tau_perfect", 1e-2)
    tau_n = abslingam_hyperparams.get("tau_noisy", 1e-1)

    if obs_key not in det_ll_dict or obs_key not in det_hl_dict:
        print(f"[Error] obs_key='{obs_key}' not found in det_ll_dict or det_hl_dict.")
        return {"abslingam": {"error": f"obs_key '{obs_key}' not found"}}

    Dll_obs_all = det_ll_dict[obs_key]  # (N, 3092)
    Dhl_obs_all = det_hl_dict[obs_key]  # (N, 84) or (N, 64)

    print(f"det_ll_obs_all shape: {Dll_obs_all.shape}")
    print(f"det_hl_obs_all shape: {Dhl_obs_all.shape}")

    abslingam_training_results = {}

    for i, fold_info in enumerate(cv_folds):
        fold_key = f"fold_{i}"
        print(f"\n--- Processing Fold {i+1}/{len(cv_folds)} ---")
        train_indices, test_indices = fold_info["train"], fold_info["test"]

        try:
            Dll_obs_full = Dll_obs_all[train_indices]
            Dhl_obs_full = Dhl_obs_all[train_indices]

            abslingam_run_result = run_abs_lingam_cmnist(
                Dll_obs_full,
                Dhl_obs_full,
                tau_perfect=tau_p,
                tau_noisy=tau_n,
            )

            abslingam_training_results[fold_key] = {}

            if abslingam_run_result["Perfect"]["T"] is not None:
                abslingam_training_results[fold_key]["Abs-LiNGAM (Perfect)"] = {
                    "T_matrix": torch.tensor(
                        abslingam_run_result["Perfect"]["T"],
                        dtype=torch.float32,
                    ),
                    "test_indices": test_indices,
                }
            else:
                abslingam_training_results[fold_key]["Abs-LiNGAM (Perfect)"] = {
                    "error": "Calculation Failed"
                }

            if abslingam_run_result["Noisy"]["T"] is not None:
                abslingam_training_results[fold_key]["Abs-LiNGAM (Noisy)"] = {
                    "T_matrix": torch.tensor(
                        abslingam_run_result["Noisy"]["T"],
                        dtype=torch.float32,
                    ),
                    "test_indices": test_indices,
                }
            else:
                abslingam_training_results[fold_key]["Abs-LiNGAM (Noisy)"] = {
                    "error": "Calculation Failed"
                }

            print(f"  ✓ Abs-LiNGAM completed for Fold {i+1}")
        except Exception as e:
            print(f"  ✗ ERROR during Abs-LiNGAM for Fold {i+1}: {e}")
            abslingam_training_results[fold_key] = {"error": str(e)}

        break

    print("\nAbs-LiNGAM Training finished.")
    return {"abslingam": abslingam_training_results}


# ============================================================
# Merging results
# ============================================================
def merge_all_results(all_results_diroca, results_gradca, results_baryca, results_abslingam):
    print("\nMerging results from all training runs...")

    all_results = {}

    def _is_valid_method_dict(x):
        return isinstance(x, dict) and len(x) > 0

    # DiRoCA radii
    if "all_results_diroca" in locals() or all_results_diroca is not None:
        if isinstance(all_results_diroca, dict):
            for radius_key, results_outer in all_results_diroca.items():
                if not _is_valid_method_dict(results_outer):
                    print(f"  ✗ Skipped DiRoCA results for {radius_key} (None/invalid).")
                    continue

                if "diroca" not in results_outer or not isinstance(results_outer["diroca"], dict):
                    print(f"  ✗ Skipped DiRoCA results for {radius_key} (missing 'diroca' key).")
                    continue

                method_name = f"diroca_{radius_key}"
                all_results[method_name] = results_outer["diroca"]
                print(f"  ✓ Added DiRoCA results for {radius_key}.")
        else:
            print("  ✗ Warning: DiRoCA results ('all_results_diroca') not found or invalid.")
    else:
        print("  ✗ Warning: DiRoCA results ('all_results_diroca') not found or invalid.")

    # GradCA
    if results_gradca is not None and isinstance(results_gradca, dict) and "gradca" in results_gradca:
        if isinstance(results_gradca["gradca"], dict):
            all_results["gradca"] = results_gradca["gradca"]
            print("  ✓ Added GradCA results (single run).")
        else:
            print("  ✗ Warning: results_gradca['gradca'] not a dict.")
    else:
        print("  ✗ Warning: GradCA results not found or invalid.")

    # BaryCA
    if results_baryca is not None and isinstance(results_baryca, dict) and "baryca" in results_baryca:
        if isinstance(results_baryca["baryca"], dict):
            all_results["baryca"] = results_baryca["baryca"]
            print("  ✓ Added BaryCA results (single run).")
        else:
            print("  ✗ Warning: results_baryca['baryca'] not a dict.")
    else:
        print("  ✗ Warning: BaryCA results not found or invalid.")

    # Abs-LiNGAM
    if results_abslingam is not None and isinstance(results_abslingam, dict) and "abslingam" in results_abslingam:
        if isinstance(results_abslingam["abslingam"], dict):
            all_results["abslingam"] = results_abslingam["abslingam"]
            print("  ✓ Added Abs-LiNGAM results (single run).")
        else:
            print("  ✗ Warning: results_abslingam['abslingam'] not a dict.")
    else:
        print("  ✗ Warning: Abs-LiNGAM results not found or invalid.")

    print("\nStructure of merged 'all_results':")
    if not all_results:
        print("  'all_results' is empty. Cannot proceed with evaluation.")
    else:
        for method_name, fold_dict in all_results.items():
            if not isinstance(fold_dict, dict):
                print(f"  - {method_name}: invalid (not a dict).")
                continue

            total_folds = len(fold_dict)
            valid_folds = 0
            for fold_key, fold_data in fold_dict.items():
                if isinstance(fold_data, dict) and "error" not in fold_data:
                    valid_folds += 1

            print(f"  - {method_name}: Contains results for {valid_folds}/{total_folds} folds.")

            if method_name == "abslingam" and valid_folds > 0:
                first_valid_fold = None
                for fk, fd in fold_dict.items():
                    if isinstance(fd, dict) and "error" not in fd:
                        first_valid_fold = fk
                        break
                if first_valid_fold is not None:
                    print(
                        f"    (Abs-LiNGAM variants in {first_valid_fold}: "
                        f"{list(fold_dict[first_valid_fold].keys())})"
                    )

    print("\nMerging complete.")
    return all_results


# ============================================================
# Main pipeline
# ============================================================
def main(args):
    data_dir = args.data_dir
    N_SUB = args.n_sub
    max_iter_cli = args.max_iter

    (
        omega,
        Ill_relevant,
        Ihl_relevant,
        Dll_samples,
        Dhl_samples,
        ll_model,
        hl_model,
        U_ll_hat,
        U_hl_hat,
        d_z,
    ) = load_cmnist_data(data_dir)

    ll_interventions, hl_interventions = build_intervention_objects(Ill_relevant, Ihl_relevant)

    iota_obs, eta_obs, parent_info_ll_tuple, parent_info_hl = rebuild_parent_infos(Dll_samples, Dhl_samples)

    det_ll_dict, det_hl_dict = rebuild_deterministic_dicts(
        omega,
        Ill_relevant,
        Ihl_relevant,
        ll_interventions,
        hl_interventions,
        ll_model,
        hl_model,
        parent_info_ll_tuple,
        parent_info_hl,
    )

    (
        Dll_samples,
        Dhl_samples,
        U_ll_hat,
        U_hl_hat,
        det_ll_dict,
        det_hl_dict,
        det_ll_dict_opt,
        det_hl_dict_opt,
        iota_obs,
    ) = subsample_data(
        Dll_samples,
        Dhl_samples,
        U_ll_hat,
        U_hl_hat,
        det_ll_dict,
        det_hl_dict,
        N_SUB=N_SUB,
    )

    cv_folds = setup_cv_folds(Dll_samples[iota_obs], n_splits=5, seed=23, iota_obs=iota_obs)

    U_ll_hat_fixed, U_hl_hat_fixed, U_ll_hat_opt, U_hl_hat_opt = fix_noise_data(U_ll_hat, U_hl_hat)

    print("Core optimization functions defined (OPT view: pixels -> z, delta=0).")

    # ---------------------------------------------------------
    # Training DiRoCA multiple radii (including eps_emp)
    # ---------------------------------------------------------
    print("\n" + "=" * 20 + " Training DiRoCA Multiple Radii (OPT view: δ=0) " + "=" * 20)

    diroca_hyperparams = {
        "eta_min": 1e-3,
        "eta_max": 1e-3,
        "max_iter": 10000,
        "num_steps_min": 5,
        "num_steps_max": 2,
        "initialization": "random",
        "optimizers": "adam",
    }

    fixed_params_diroca = {
        "tol": 1e-4,
        "seed": 23,
        "robust_L": True,
        "robust_H": False,
        "experiment": "cmnist",
        "gain": 0.0,
    }

    # test mode: override max_iter for quick run
    if max_iter_cli is not None:
        diroca_hyperparams["max_iter"] = max_iter_cli
        print("\n[TEST MODE] Overriding max_iter to {} for DiRoCA/GradCA/BaryCA".format(max_iter_cli))

    U_ll_hat_run = U_ll_hat_fixed
    U_hl_hat_run = U_hl_hat_fixed
    print(f"Using U_ll_hat_run with shape: {U_ll_hat_run.shape}")
    print(f"Using U_hl_hat_run with shape: {U_hl_hat_run.shape}  (ignored in optimization if δ=0)")

    N_emp = U_ll_hat_run.shape[0]
    eps_emp = compute_empirical_radius(N_emp)
    print(f"Computed empirical radius eps_emp = {eps_emp:.6f} using N={N_emp}")

    eps_list = [0.5, 1.0, 2.0, 4.0, 8.0, eps_emp, 20, 40, 100]
    radius_combinations = [(eps, 0.0) for eps in eps_list]

    print(f"OPT view active: sweeping ε over {eps_list}, always using δ=0.0")

    all_results_diroca = {}
    all_monitors_diroca = {}

    print(f"\nRunning DiRoCA for {len(radius_combinations)} radius combinations...")

    for i, (epsilon, delta) in enumerate(radius_combinations):
        print(f"\n--- Running DiRoCA {i+1}/{len(radius_combinations)}: ε={epsilon}, δ={delta} ---")

        try:
            results_diroca, monitors_diroca = train_diroca_single_run(
                cv_folds,
                U_ll_hat_run,
                U_hl_hat_run,
                det_ll_dict,
                det_hl_dict,
                omega,
                diroca_hyperparams,
                fixed_params_diroca,
                epsilon,
                delta,
            )

            radius_key = f"eps_{epsilon}_delta_{delta}"
            all_results_diroca[radius_key] = results_diroca
            all_monitors_diroca[radius_key] = monitors_diroca

            print(f"✓ Completed {radius_key}")
        except Exception as e:
            radius_key = f"eps_{epsilon}_delta_{delta}"
            print(f"✗ Failed {radius_key}: {str(e)}")
            all_results_diroca[radius_key] = None
            all_monitors_diroca[radius_key] = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("DiRoCA Multi-Radius Training Complete (OPT view)!")
    print("=" * 50)

    # ---------------------------------------------------------
    # Training GradCA
    # ---------------------------------------------------------
    print("\n" + "=" * 20 + " Training GradCA (δ = 0 OPT view) " + "=" * 20)

    gradca_hyperparams = {
        "eta_min": 1e-3,
        "max_iter": 10000,
        "num_steps_min": 1,
        "initialization": "zeros",
        "optimizers": "adam",
    }

    if max_iter_cli is not None:
        gradca_hyperparams["max_iter"] = max_iter_cli

    fixed_params_gradca = {
        "tol": 1e-4,
        "seed": 23,
        "experiment": "cmnist",
        "gain": 0.0,
    }

    U_ll_hat_run = U_ll_hat_fixed
    U_hl_hat_run = U_hl_hat_fixed

    print(f"Using U_ll_hat_run with shape: {U_ll_hat_run.shape}")
    print(f"Using U_hl_hat_run with shape: {U_hl_hat_run.shape}  (ignored since δ = 0)")

    results_gradca, monitors_gradca = train_gradca_single_run(
        cv_folds,
        U_ll_hat_run,
        U_hl_hat_run,
        det_ll_dict,
        det_hl_dict,
        omega,
        gradca_hyperparams,
        fixed_params_gradca,
    )

    print("\nGradCA Training finished.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Training BaryCA
    # ---------------------------------------------------------
    print("\n" + "=" * 20 + " Training BaryCA (pixels → z only) " + "=" * 20)

    baryca_hyperparams = {
        "lr": 1e-3,
        "max_iter": 10000,
        "tol": 1e-4,
    }

    if max_iter_cli is not None:
        baryca_hyperparams["max_iter"] = max_iter_cli

    fixed_params_baryca = {
        "seed": 23,
    }

    U_ll_hat_run = U_ll_hat_fixed
    U_hl_hat_run = U_hl_hat_fixed

    print(f"Using U_ll_hat_run with shape: {U_ll_hat_run.shape}  (pixels noise)")
    print(f"Using U_hl_hat_run with shape: {U_hl_hat_run.shape}  (raw z noise)")

    obs_key = "obs"
    assert obs_key in det_hl_dict, f"{obs_key=} not in det_hl_dict keys."

    hl_full_dim = det_hl_dict[obs_key].shape[1]
    z_dim = hl_full_dim - 20
    assert z_dim > 0, f"Inferred z_dim={z_dim} is not positive. Check det_hl_dict shapes."
    print(f"Inferred z_dim from det_hl_dict['obs']: {z_dim}")

    det_ll_dict_opt_local = {k: v[:, :3072] for k, v in det_ll_dict.items()}
    det_hl_dict_opt_local = {k: v[:, 20 : 20 + z_dim] for k, v in det_hl_dict.items()}

    if U_hl_hat_run.shape[1] != z_dim:
        print(f"[Aligning] U_hl_hat_run dim {U_hl_hat_run.shape[1]} → {z_dim}")
        U_hl_hat_run = U_hl_hat_run[:, :z_dim]

    some_iota = next(iter(det_ll_dict_opt_local.keys()))
    some_eta = next(iter(det_hl_dict_opt_local.keys()))
    print(f"det_ll_dict_opt['{some_iota}'] shape: {det_ll_dict_opt_local[some_iota].shape}")
    print(f"det_hl_dict_opt['{some_eta}'] shape: {det_hl_dict_opt_local[some_eta].shape}")
    print(f"U_hl_hat_run aligned shape: {U_hl_hat_run.shape}")

    results_baryca, monitors_baryca = train_baryca_single_run(
        cv_folds,
        U_ll_hat_run,
        U_hl_hat_run,
        det_ll_dict_opt_local,
        det_hl_dict_opt_local,
        omega,
        baryca_hyperparams,
        fixed_params_baryca,
    )

    print("\nBaryCA Training finished.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Training Abs-LiNGAM
    # ---------------------------------------------------------
    print("\n" + "=" * 20 + " Training Abs-LiNGAM (pixels → z only, SELF-CONTAINED) " + "=" * 20)

    abslingam_hyperparams = {
        "tau_perfect": 1e-2,
        "tau_noisy": 1e-1,
    }

    results_abslingam = train_abslingam_single_run(
        cv_folds=cv_folds,
        det_ll_dict=det_ll_dict,
        det_hl_dict=det_hl_dict,
        run_hyperparams=abslingam_hyperparams,
        obs_key="obs",
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Merge results
    # ---------------------------------------------------------
    all_results = merge_all_results(all_results_diroca, results_gradca, results_baryca, results_abslingam)

    # ==========================================================
    # SAVE EVAL STATE FOR SEPARATE EVALUATION NOTEBOOK
    # ==========================================================
    print("\nSaving evaluation state for CMNIST...")

    eval_dir = os.path.join("data", "cmnist", "results_empirical")
    os.makedirs(eval_dir, exist_ok=True)

    # 1) merged all_results
    save_all_results_path = os.path.join(eval_dir, "cmnist_all_results.pkl")
    with open(save_all_results_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"  ✓ Saved all_results to {save_all_results_path}")

    # 2) cv_folds
    cv_folds_path = os.path.join(eval_dir, "cmnist_cv_folds.pkl")
    with open(cv_folds_path, "wb") as f:
        pickle.dump(cv_folds, f)
    print(f"  ✓ Saved cv_folds to {cv_folds_path}")

    # 3) omega (env pair mapping)
    omega_path = os.path.join(eval_dir, "cmnist_omega.pkl")
    with open(omega_path, "wb") as f:
        pickle.dump(omega, f)
    print(f"  ✓ Saved omega to {omega_path}")

    # 4) deterministic opt-view dicts
    det_ll_opt_path = os.path.join(eval_dir, "cmnist_det_ll_dict_opt.pt")
    torch.save(det_ll_dict_opt, det_ll_opt_path)
    print(f"  ✓ Saved det_ll_dict_opt to {det_ll_opt_path}")

    det_hl_opt_path = os.path.join(eval_dir, "cmnist_det_hl_dict_opt.pt")
    torch.save(det_hl_dict_opt, det_hl_opt_path)
    print(f"  ✓ Saved det_hl_dict_opt to {det_hl_opt_path}")

    # 5) opt-view noise
    U_ll_opt_path = os.path.join(eval_dir, "cmnist_U_ll_hat_opt.pt")
    torch.save(U_ll_hat_opt, U_ll_opt_path)
    print(f"  ✓ Saved U_ll_hat_opt to {U_ll_opt_path}")

    U_hl_opt_path = os.path.join(eval_dir, "cmnist_U_hl_hat_opt.pt")
    torch.save(U_hl_hat_opt, U_hl_opt_path)
    print(f"  ✓ Saved U_hl_hat_opt to {U_hl_opt_path}")

    # 6) Dll_samples / Dhl_samples
    dll_samples_path = os.path.join(eval_dir, "cmnist_Dll_samples.pkl")
    with open(dll_samples_path, "wb") as f:
        pickle.dump(Dll_samples, f)
    print(f"  ✓ Saved Dll_samples to {dll_samples_path}")

    dhl_samples_path = os.path.join(eval_dir, "cmnist_Dhl_samples.pkl")
    with open(dhl_samples_path, "wb") as f:
        pickle.dump(Dhl_samples, f)
    print(f"  ✓ Saved Dhl_samples to {dhl_samples_path}")

    print("CMNIST eval state saved.\n")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMNIST optimization: DiRoCA/GradCA/BaryCA/AbsLiNGAM")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/cmnist_new",
        help="Path to CMNIST data directory.",
    )
    parser.add_argument(
        "--n-sub",
        type=int,
        default=2000,
        help="Number of subsampled points per environment.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Max iterations for optimization (if set, overrides defaults for a quick test).",
    )

    args = parser.parse_args()
    main(args)

