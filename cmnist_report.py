#!/usr/bin/env python3
"""
CMNIST report (Xia encoder version).

- LL: same Film-U-Net-style image model at 32x32 + residuals U_ll_hat.
- HL: z-space model f_hl: (D,C) -> z with residuals U_hl_hat_z.

This script mirrors the old cmnist_make_reports.py diagnostics, but:
  * treats HL as a vector-valued latent z instead of a 16x16 image;
  * adds z-level diagnostics: residual independence, deterministic fit,
    and distances between observational and interventional z.
"""

import os
import argparse
import glob
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import rgb_to_hsv

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib


# ============================================================
# Basic helpers
# ============================================================

def to01_any(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    mn, mx = x.min().item(), x.max().item()

    # already [0,1]
    if -1e-3 <= mn and mx <= 1.0 + 1e-3:
        return x.clamp(0, 1).numpy()

    # looks like [-1,1]
    if -1.05 <= mn and mx <= 1.05:
        return ((x + 1.0) / 2.0).clamp(0, 1).numpy()

    # fallback: min-max
    m, M = x.min(), x.max()
    return ((x - m) / (M - m + 1e-8)).clamp(0, 1).numpy()


def get_obs_entry(D):
    """Robustly fetch observational entry."""
    if "obs" in D:
        return D["obs"]
    if None in D:
        return D[None]
    return next(iter(D.values()))


def hue_hist(img_chw, bins=36, v_thresh=0.1):
    """Hue histogram for one CHW image (for color diversity)."""
    x = img_chw.detach().cpu().float()
    if x.min() < -1.05 or x.max() > 1.05:
        # weird range → min-max
        m, M = x.min(), x.max()
        x = (x - m) / (M - m + 1e-8)
    else:
        # assume [-1,1] → [0,1]
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    rgb = x.permute(1, 2, 0).numpy()
    hsv = rgb_to_hsv(rgb)
    H, _, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = V > v_thresh
    if mask.sum() == 0:
        return np.ones(bins) / bins
    h = H[mask].ravel()
    hist, _ = np.histogram(h, bins=bins, range=(0, 1), density=False)
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    return hist


def js_divergence(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        a = np.clip(a, 1e-12, 1.0)
        b = np.clip(b, 1e-12, 1.0)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ============================================================
# LL model architecture (must match cmnist_train_new.py)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="bn"):
        super().__init__()
        Norm = {
            "bn": nn.BatchNorm2d,
            "in": nn.InstanceNorm2d,
            "none": lambda c: nn.Identity()
        }.get(norm_type, nn.BatchNorm2d)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="bn"):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, norm_type)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm_type="bn"):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        nn.init.kaiming_normal_(self.reduce.weight, nonlinearity="relu")
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, norm_type)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FiLM(nn.Module):
    def __init__(self, num_digits=10, num_colors=10, feat_channels=128, hidden=64):
        super().__init__()
        self.d_embed = nn.Embedding(num_digits, hidden)
        self.c_embed = nn.Embedding(num_colors, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_channels * 2),
        )
        nn.init.normal_(self.d_embed.weight, std=0.02)
        nn.init.normal_(self.c_embed.weight, std=0.02)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x, digit, color):
        d = self.d_embed(digit)
        c = self.c_embed(color)
        h = torch.cat([d, c], dim=-1)
        params = self.mlp(h)
        C = x.size(1)
        gamma, beta = params.split(C, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ImageColorizerUNet(nn.Module):
    """32×32 U-Net with FiLM conditioning."""
    def __init__(self, norm_type="bn"):
        super().__init__()
        self.inc = ConvBlock(1, 32, norm_type)
        self.down1 = Down(32, 64, norm_type)
        self.down2 = Down(64, 128, norm_type)
        self.film_bottleneck = FiLM(feat_channels=128)
        self.film_up1 = FiLM(feat_channels=64)
        self.up1 = Up(128, 64, 64, norm_type)
        self.up2 = Up(64, 32, 32, norm_type)
        self.outc = nn.Conv2d(32, 3, 1)
        nn.init.kaiming_normal_(self.outc.weight, nonlinearity="linear")

    def forward(self, x, digit, color):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.film_bottleneck(x3, digit, color)
        y = self.up1(x3, x2)
        y = self.film_up1(y, digit, color)
        y = self.up2(y, x1)
        return torch.tanh(self.outc(y))


# ============================================================
# HL z-model class (must match cmnist_train_new.py)
# ============================================================

class HLCellMeansShrunkVec:
    """
    Vector-valued version of the cell-means + shrinkage model.

    - Input: one-hot digit Xd (N,10), one-hot color Xc (N,10)
    - Output: z in R^{d_z}
    """
    def __init__(self, lam=10.0):
        self.lam = lam

    def fit(self, Xd, Xc, Z):
        """
        Xd, Xc: numpy arrays one-hot, shape (N,10)
        Z: numpy array shape (N,d_z)
        """
        Xd = np.asarray(Xd)
        Xc = np.asarray(Xc)
        Z = np.asarray(Z)
        d_idx = Xd.argmax(1)
        c_idx = Xc.argmax(1)
        N, d_z = Z.shape

        # cell-wise sums & counts
        sums = np.zeros((10, 10, d_z), dtype=np.float64)
        cnts = np.zeros((10, 10), dtype=np.float64)
        for di, ci, zi in zip(d_idx, c_idx, Z):
            sums[di, ci] += zi
            cnts[di, ci] += 1.0
        means = np.where(cnts[..., None] > 0,
                         sums / np.maximum(cnts[..., None], 1.0),
                         0.0)

        # additive baseline: mu0, mu_d, mu_c (vector-valued)
        mu0 = Z.mean(axis=0)  # (d_z,)
        mu_d = np.zeros((10, d_z), dtype=np.float64)
        mu_c = np.zeros((10, d_z), dtype=np.float64)
        for i in range(10):
            mask_d = (d_idx == i)
            if mask_d.any():
                mu_d[i] = Z[mask_d].mean(axis=0)
            else:
                mu_d[i] = mu0
        for j in range(10):
            mask_c = (c_idx == j)
            if mask_c.any():
                mu_c[j] = Z[mask_c].mean(axis=0)
            else:
                mu_c[j] = mu0

        base = mu_d[:, None, :] + mu_c[None, :, :] - mu0[None, None, :]

        denom = cnts + self.lam
        w = np.where(cnts > 0, cnts / denom, 0.0)  # (10,10)
        w = w[..., None]  # (10,10,1) for broadcasting

        self.table = w * means + (1.0 - w) * base
        return self

    def predict(self, Xd, Xc):
        Xd = np.asarray(Xd)
        Xc = np.asarray(Xc)
        d_idx = Xd.argmax(1)
        c_idx = Xc.argmax(1)
        return self.table[d_idx, c_idx]


# ============================================================
# Metric helpers (images + z)
# ============================================================

def map_to01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().float()
    if x.min() >= -1.05 and x.max() <= 1.05:
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)


def bg_level(x01: torch.Tensor, pct: float = 5.0) -> torch.Tensor:
    x = x01.view(x01.shape[0], -1)
    return torch.quantile(x, pct / 100.0, dim=1, keepdim=True)


def apply_bg(x01: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    x = x01.view(x01.shape[0], -1) - bg
    return x.clamp(0, 1).view_as(x01)


def metrics_all_images(images, model, U_hat, shapes, digits, colors, device):
    """
    Average MSE / corr for:
      - true vs deterministic D
      - true vs reconstruction D+U
    """
    N = images.size(0)
    mses_D, mses_DU, cors_D, cors_DU = [], [], [], []

    model.eval()
    with torch.no_grad():
        for i in range(N):
            det = model(
                shapes[i:i+1].to(device),
                digits[i:i+1].to(device),
                colors[i:i+1].to(device),
            )[0].cpu()

            true = map_to01(images[i])
            pred = map_to01(det)
            recon = map_to01(det + U_hat[i].cpu())

            bg = bg_level(true.unsqueeze(0), pct=5.0)  # (1,1)
            true_d = apply_bg(true.unsqueeze(0), bg)[0].view(-1).numpy()
            pred_d = apply_bg(pred.unsqueeze(0), bg)[0].view(-1).numpy()
            recon_d = apply_bg(recon.unsqueeze(0), bg)[0].view(-1).numpy()

            mses_D.append(float(np.mean((true_d - pred_d) ** 2)))
            mses_DU.append(float(np.mean((true_d - recon_d) ** 2)))
            cors_D.append(float(np.corrcoef(true_d, pred_d)[0, 1]))
            cors_DU.append(float(np.corrcoef(true_d, recon_d)[0, 1]))

    return (
        float(np.mean(mses_D)),
        float(np.mean(cors_D)),
        float(np.mean(mses_DU)),
        float(np.mean(cors_DU)),
    )


def metrics_all_z(z_true, model, U_z, digits, colors):
    """
    z_true: (N,d_z) tensor
    model:  HLCellMeansShrunkVec
    U_z:    (N,d_z) tensor
    digits/colors: Long tensors (0-9)
    """
    Z = z_true.detach().cpu().numpy()
    U = U_z.detach().cpu().numpy()
    d = digits.detach().cpu().numpy()
    c = colors.detach().cpu().numpy()

    Xd = np.eye(10)[d]
    Xc = np.eye(10)[c]

    Z_det = model.predict(Xd, Xc)          # (N,d_z)
    Z_recon = Z_det + U                    # (N,d_z)

    mse_D = float(np.mean((Z - Z_det) ** 2))
    mse_DU = float(np.mean((Z - Z_recon) ** 2))

    # average per-dimension correlation
    cors_D = []
    cors_DU = []
    for j in range(Z.shape[1]):
        z = Z[:, j]
        zd = Z_det[:, j]
        zr = Z_recon[:, j]
        if np.std(z) < 1e-8 or np.std(zd) < 1e-8 or np.std(zr) < 1e-8:
            continue
        cors_D.append(float(np.corrcoef(z, zd)[0, 1]))
        cors_DU.append(float(np.corrcoef(z, zr)[0, 1]))
    corr_D = float(np.mean(cors_D)) if cors_D else 0.0
    corr_DU = float(np.mean(cors_DU)) if cors_DU else 0.0
    return mse_D, corr_D, mse_DU, corr_DU


# ============================================================
# Core: build report for one dataset (z version)
# ============================================================

def _load_hl_model_z(ds_dir):
    """
    Load the vector cell-means HL model from joblib.
    Class name must match training: HLCellMeansShrunkVec.
    """
    path = os.path.join(ds_dir, "hl_model_cellmeans_z.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HL model not found at: {path}")
    print(f"  using HL model file: {os.path.basename(path)}")
    # Because we defined HLCellMeansShrunkVec above in __main__,
    # joblib can unpickle it.
    return joblib.load(path)


def make_report_for_dataset_z(ds_dir, bins=36, v_thresh=0.1,
                              num_resid_pixels=5, seed=0, device="cpu"):

    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    # ---------- Load artifacts ----------
    ds_dir = os.path.abspath(ds_dir)
    Dll = torch.load(os.path.join(ds_dir, "dll_samples.pkl"), map_location="cpu")
    Dhl = torch.load(os.path.join(ds_dir, "dhl_samples.pkl"), map_location="cpu")
    U_ll = torch.load(os.path.join(ds_dir, "U_ll_hat.pkl"), map_location="cpu")
    U_z  = torch.load(os.path.join(ds_dir, "U_hl_hat_z.pkl"), map_location="cpu")

    # LL model
    ll_model = ImageColorizerUNet()
    ll_model.load_state_dict(torch.load(os.path.join(ds_dir, "ll_model_unet.pth"), map_location="cpu"))
    ll_model.to(dev).eval()

    # HL z-model
    hl_model = _load_hl_model_z(ds_dir)

    # LL observations
    ll_obs = get_obs_entry(Dll)
    ll_imgs_32, ll_shapes_32, ll_digits, ll_colors = ll_obs
    assert ll_imgs_32.shape[2:] == (32, 32)

    # HL observations: Dhl['obs'] = [digit one-hot(10) | color one-hot(10) | z(d_z)]
    hl_obs = get_obs_entry(Dhl)
    if not isinstance(hl_obs, torch.Tensor) or hl_obs.dim() != 2:
        raise TypeError("Expected Dhl['obs'] to be a 2D tensor (N, 20 + d_z).")

    d_z = U_z.shape[1]
    if hl_obs.shape[1] < 20 + d_z:
        raise ValueError(
            f"hl_obs has shape {hl_obs.shape}, but we expected at least 20 + d_z={20 + d_z} columns."
        )

    Xd_obs = hl_obs[:, :10]   # digit one-hot
    Xc_obs = hl_obs[:, 10:20] # color one-hot
    z_obs  = hl_obs[:, -d_z:] # (N, d_z)

    assert U_z.shape == z_obs.shape, (
        f"U_hl_hat_z must match z_obs shape. Got U_z {U_z.shape}, z_obs {z_obs.shape}."
    )

    # Recover digits/colors as ints
    hl_digits = Xd_obs.argmax(dim=1).long()
    hl_colors = Xc_obs.argmax(dim=1).long()

    # =======================================================
    # 1) LL residual independence (same as old)
    # =======================================================
    U_flat = U_ll.view(U_ll.size(0), -1).cpu().numpy()
    pix_idx = rng.choice(
        U_flat.shape[1],
        size=min(num_resid_pixels, U_flat.shape[1]),
        replace=False,
    )
    resid_samples = U_flat[:, pix_idx]

    X_lab = np.stack([ll_digits.numpy(), ll_colors.numpy()], axis=1)
    y = resid_samples[:, 0]
    lin = LinearRegression().fit(X_lab, y)
    r2_resid = r2_score(y, lin.predict(X_lab))

    fig1, axs1 = plt.subplots(1, len(pix_idx), figsize=(3.0 * len(pix_idx), 3))
    if len(pix_idx) == 1:
        axs1 = [axs1]
    for i, p in enumerate(pix_idx):
        axs1[i].hist(resid_samples[:, i], bins=40, alpha=0.8)
        axs1[i].set_title(f"LL residual pixel {p}")
    fig1.suptitle("LL residual distributions (sampled pixels)")
    fig1.tight_layout()

    # =======================================================
    # 1b) HL residual independence in z-space
    # =======================================================
    U_z_np = U_z.view(U_z.size(0), -1).cpu().numpy()
    # pick a few z dimensions
    z_idx = rng.choice(
        U_z_np.shape[1],
        size=min(num_resid_pixels, U_z_np.shape[1]),
        replace=False,
    )
    resid_z_samples = U_z_np[:, z_idx]

    X_lab_hl = np.stack([hl_digits.numpy(), hl_colors.numpy()], axis=1)
    y_hl = resid_z_samples[:, 0]
    lin_hl = LinearRegression().fit(X_lab_hl, y_hl)
    r2_resid_hl = r2_score(y_hl, lin_hl.predict(X_lab_hl))

    fig1b, axs1b = plt.subplots(1, len(z_idx), figsize=(3.0 * len(z_idx), 3))
    if len(z_idx) == 1:
        axs1b = [axs1b]
    for i, p in enumerate(z_idx):
        axs1b[i].hist(resid_z_samples[:, i], bins=40, alpha=0.8)
        axs1b[i].set_title(f"HL residual z[{p}]")
    fig1b.suptitle("HL residual distributions in z (sampled dims)")
    fig1b.tight_layout()

    # =======================================================
    # 2) LL qualitative example (32x32) + metrics
    # =======================================================
    idx_vis = int(rng.integers(0, len(ll_imgs_32)))

    with torch.no_grad():
        det32 = ll_model(
            ll_shapes_32[idx_vis:idx_vis+1].to(dev),
            ll_digits[idx_vis:idx_vis+1].to(dev),
            ll_colors[idx_vis:idx_vis+1].to(dev),
        )[0].cpu()

    true32 = map_to01(ll_imgs_32[idx_vis])
    pred32 = map_to01(det32)
    resid32 = U_ll[idx_vis].detach().cpu()
    recon32 = map_to01(det32 + resid32)

    bg32 = bg_level(true32.unsqueeze(0), pct=5.0)
    true32_d = apply_bg(true32.unsqueeze(0), bg32)[0]
    pred32_d = apply_bg(pred32.unsqueeze(0), bg32)[0]
    recon32_d = apply_bg(recon32.unsqueeze(0), bg32)[0]

    resid_np = resid32.permute(1, 2, 0).numpy()
    mag = np.percentile(np.abs(resid_np).ravel(), 99)
    mag = max(mag, 1e-6)

    fig2, axs2 = plt.subplots(1, 4, figsize=(11, 3))
    axs2[0].imshow(true32_d.permute(1, 2, 0).numpy()); axs2[0].set_title("True 32x32"); axs2[0].axis("off")
    axs2[1].imshow(pred32_d.permute(1, 2, 0).numpy()); axs2[1].set_title("Deterministic D"); axs2[1].axis("off")
    axs2[2].imshow(resid_np, cmap="RdBu_r", vmin=-mag, vmax=mag); axs2[2].set_title("Residual U"); axs2[2].axis("off")
    axs2[3].imshow(recon32_d.permute(1, 2, 0).numpy()); axs2[3].set_title("D + U"); axs2[3].axis("off")
    fig2.suptitle("Low-level decomposition (32x32)")
    fig2.tight_layout()

    mse_D_32, corr_D_32, mse_DU_32, corr_DU_32 = metrics_all_images(
        ll_imgs_32, ll_model, U_ll, ll_shapes_32, ll_digits, ll_colors, dev
    )

    # =======================================================
    # 3) HL qualitative example in z-space + metrics
    # =======================================================
    z_true = z_obs          # (N,d_z)
    z_true_np = z_true.detach().cpu().numpy()
    U_z_np = U_z.detach().cpu().numpy()

    # deterministic & recon for full observational set
    Xd_np = Xd_obs.detach().cpu().numpy()
    Xc_np = Xc_obs.detach().cpu().numpy()
    z_det_np = hl_model.predict(Xd_np, Xc_np)
    z_recon_np = z_det_np + U_z_np

    # metrics on entire obs set
    mse_D_z, corr_D_z, mse_DU_z, corr_DU_z = metrics_all_z(
        z_true, hl_model, U_z, hl_digits, hl_colors
    )

    # single example (same idx_vis)
    z_true_ex = z_true_np[idx_vis]
    z_det_ex  = z_det_np[idx_vis]
    z_resid_ex = U_z_np[idx_vis]
    z_recon_ex = z_recon_np[idx_vis]
    dims = np.arange(len(z_true_ex))

    fig3, axs3 = plt.subplots(2, 2, figsize=(10, 6))
    axs3 = axs3.reshape(2, 2)
    axs3[0, 0].bar(dims, z_true_ex);  axs3[0, 0].set_title("z_true")
    axs3[0, 1].bar(dims, z_det_ex);   axs3[0, 1].set_title("z_det = f_hl(D,C)")
    axs3[1, 0].bar(dims, z_resid_ex); axs3[1, 0].set_title("U_z")
    axs3[1, 1].bar(dims, z_recon_ex); axs3[1, 1].set_title("z_det + U_z")
    for ax in axs3.ravel():
        ax.set_xticks([])
    fig3.suptitle("High-level decomposition in z-space")
    fig3.tight_layout()

    # =======================================================
    # 4) LL intervention histograms: mean intensity
    # =======================================================
    def get_means(entry):
        """Extract mean intensity from LL entry (images)."""
        if isinstance(entry, (tuple, list)):
            img = entry[0]
        else:
            img = entry
        arr = to01_any(img)
        return arr.mean(axis=(1, 2, 3))

    ll_keys = list(Dll.keys())
    ll_k_obs = "obs" if "obs" in ll_keys else ll_keys[0]
    ll_k_d = "D=8" if "D=8" in ll_keys else next(
        (k for k in ll_keys if k.startswith("D=") and "," not in k and k != ll_k_obs),
        ll_keys[1] if len(ll_keys) > 1 else ll_k_obs
    )
    ll_k_c = "C=0" if "C=0" in ll_keys else next(
        (k for k in ll_keys if k.startswith("C=") and "," not in k and k != ll_k_obs),
        ll_keys[2] if len(ll_keys) > 2 else ll_k_obs
    )
    ll_k_both = "D=8,C=0" if "D=8,C=0" in ll_keys else next(
        (k for k in ll_keys if "," in k),
        ll_keys[-1] if len(ll_keys) > 0 else ll_k_obs
    )

    ll_obs_entry = get_obs_entry(Dll) if ll_k_obs == "obs" else (Dll[ll_k_obs] if ll_k_obs in Dll else get_obs_entry(Dll))
    ll_obs_avgs = get_means(ll_obs_entry)
    ll_d_entry = Dll[ll_k_d] if ll_k_d in Dll else get_obs_entry(Dll)
    ll_d_avgs = get_means(ll_d_entry) if ll_k_d in Dll else ll_obs_avgs
    ll_c_entry = Dll[ll_k_c] if ll_k_c in Dll else get_obs_entry(Dll)
    ll_c_avgs = get_means(ll_c_entry) if ll_k_c in Dll else ll_obs_avgs
    ll_both_entry = Dll[ll_k_both] if ll_k_both in Dll else get_obs_entry(Dll)
    ll_both_avgs = get_means(ll_both_entry) if ll_k_both in Dll else ll_obs_avgs

    fig4a = plt.figure(figsize=(12, 7))
    plt.hist(ll_obs_avgs, bins=50, alpha=0.5, density=True, label="obs")
    if ll_k_d in Dll:
        plt.hist(ll_d_avgs, bins=50, alpha=0.6, density=True, label=f"do({ll_k_d})")
    if ll_k_c in Dll:
        plt.hist(ll_c_avgs, bins=50, alpha=0.6, density=True, label=f"do({ll_k_c})")
    if ll_k_both in Dll:
        plt.hist(ll_both_avgs, bins=50, alpha=0.6, density=True, label=f"do({ll_k_both})")
    plt.title("LL 32x32 abstraction: mean intensity under interventions")
    plt.xlabel("mean pixel intensity")
    plt.ylabel("density")
    plt.grid(alpha=0.4, ls="--")
    plt.legend()
    plt.tight_layout()

    # =======================================================
    # 5) HL intervention histograms in z-space
    #     (we use ||z||_2 per sample as a scalar summary)
    # =======================================================
    def get_z_norms(entry):
        """Entry is high-level tensor (N,20+d_z); return ||z||_2 per sample."""
        if not isinstance(entry, torch.Tensor):
            raise TypeError("Expected high-level entry to be a tensor.")
        # Use last d_z dims
        Z = entry[:, -d_z:].detach().cpu().numpy()
        return np.linalg.norm(Z, axis=1)

    hl_keys = list(Dhl.keys())
    hl_k_obs = "obs" if "obs" in hl_keys else hl_keys[0]
    hl_k_d = "D=8" if "D=8" in hl_keys else next(
        (k for k in hl_keys if k.startswith("D=") and "," not in k and k != hl_k_obs),
        hl_keys[1] if len(hl_keys) > 1 else hl_k_obs
    )
    hl_k_c = "C=0" if "C=0" in hl_keys else next(
        (k for k in hl_keys if k.startswith("C=") and "," not in k and k != hl_k_obs),
        hl_keys[2] if len(hl_keys) > 2 else hl_k_obs
    )
    hl_k_both = "D=8,C=0" if "D=8,C=0" in hl_keys else next(
        (k for k in hl_keys if "," in k),
        hl_keys[-1] if len(hl_keys) > 0 else hl_k_obs
    )

    hl_obs_entry = get_obs_entry(Dhl) if hl_k_obs == "obs" else (Dhl[hl_k_obs] if hl_k_obs in Dhl else get_obs_entry(Dhl))
    hl_obs_norms = get_z_norms(hl_obs_entry)
    hl_d_entry = Dhl[hl_k_d] if hl_k_d in Dhl else get_obs_entry(Dhl)
    hl_d_norms = get_z_norms(hl_d_entry) if hl_k_d in Dhl else hl_obs_norms
    hl_c_entry = Dhl[hl_k_c] if hl_k_c in Dhl else get_obs_entry(Dhl)
    hl_c_norms = get_z_norms(hl_c_entry) if hl_k_c in Dhl else hl_obs_norms
    hl_both_entry = Dhl[hl_k_both] if hl_k_both in Dhl else get_obs_entry(Dhl)
    hl_both_norms = get_z_norms(hl_both_entry) if hl_k_both in Dhl else hl_obs_norms

    fig4b = plt.figure(figsize=(12, 7))
    plt.hist(hl_obs_norms, bins=50, alpha=0.5, density=True, label="obs")
    if hl_k_d in Dhl:
        plt.hist(hl_d_norms, bins=50, alpha=0.6, density=True, label=f"do({hl_k_d})")
    if hl_k_c in Dhl:
        plt.hist(hl_c_norms, bins=50, alpha=0.6, density=True, label=f"do({hl_k_c})")
    if hl_k_both in Dhl:
        plt.hist(hl_both_norms, bins=50, alpha=0.6, density=True, label=f"do({hl_k_both})")
    plt.title("HL z abstraction: ||z||_2 under interventions")
    plt.xlabel("||z||_2")
    plt.ylabel("density")
    plt.grid(alpha=0.4, ls="--")
    plt.legend()
    plt.tight_layout()

    # =======================================================
    # 6) LL color sweep + hue JS (deterministic + with U)
    # =======================================================
    with torch.no_grad():
        base_shape = ll_shapes_32[idx_vis:idx_vis+1].to(dev)
        base_digit = ll_digits[idx_vis:idx_vis+1].to(dev)
        U_fixed = U_ll[idx_vis:idx_vis+1].to(dev)

        dets, outs = [], []
        for c in range(10):
            c_t = torch.tensor([c], device=dev)
            det = ll_model(base_shape, base_digit, c_t)
            out = det + U_fixed
            dets.append(det.cpu())
            outs.append(out.cpu())

    D_cf = torch.stack(dets, dim=0)
    O_cf = torch.stack(outs, dim=0)

    Hs = [hue_hist(D_cf[c][0], bins=bins, v_thresh=v_thresh) for c in range(10)]
    js_vals = [js_divergence(Hs[i], Hs[j]) for i in range(10) for j in range(i + 1, 10)]
    js_hue = float(np.mean(js_vals)) if js_vals else 0.0

    fig5, axs5 = plt.subplots(2, 10, figsize=(16, 3.5))
    for c in range(10):
        axs5[0, c].imshow(to01_any(D_cf[c][0]).transpose(1, 2, 0)); axs5[0, c].axis("off")
        axs5[0, c].set_title(f"c={c}", fontsize=7)
        axs5[1, c].imshow(to01_any(O_cf[c][0]).transpose(1, 2, 0)); axs5[1, c].axis("off")
    axs5[0, 0].set_ylabel("Det", fontsize=8)
    axs5[1, 0].set_ylabel("Det+U", fontsize=8)
    fig5.suptitle("Counterfactual color sweep (fixed digit & noise) - LL")
    fig5.tight_layout()

    # =======================================================
    # 7) z-space color sweep (HL deterministic paths)
    # =======================================================
    with torch.no_grad():
        d0 = int(ll_digits[idx_vis].item())
        # we ignore U_z here and look at deterministic f_hl(D,C)
        z_sweep = []
        for c in range(10):
            Xd_one = np.zeros((1, 10), dtype=np.float32)
            Xc_one = np.zeros((1, 10), dtype=np.float32)
            Xd_one[0, d0] = 1.0
            Xc_one[0, c] = 1.0
            z_det = hl_model.predict(Xd_one, Xc_one)[0]  # shape (d_z,)
            z_sweep.append(z_det)
        z_sweep = np.stack(z_sweep, axis=0)  # (10,d_z)

    fig7, axs7 = plt.subplots(2, 5, figsize=(14, 5))
    axs7 = axs7.reshape(2, 5)
    for i in range(10):
        r = i // 5
        c = i % 5
        axs7[r, c].bar(np.arange(d_z), z_sweep[i])
        axs7[r, c].set_title(f"color = {i}", fontsize=9)
        axs7[r, c].set_xticks([])
    fig7.suptitle("Counterfactual color sweep in z (fixed digit, deterministic f_hl)")
    fig7.tight_layout()

    # =======================================================
    # 8) PCA & z-distance between obs and interventions
    # =======================================================
    # We'll approximate PCA with numpy eig on cov of z_obs.
    Z_centered = z_obs - z_obs.mean(dim=0, keepdim=True)
    Z_np = Z_centered.detach().cpu().numpy()
    C_cov = np.cov(Z_np, rowvar=False)  # (d_z,d_z)
    eigvals, eigvecs = np.linalg.eigh(C_cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    V2 = eigvecs[:, :2]  # (d_z,2)
    Z2 = Z_np @ V2       # (N,2)

    Z2_d = Z2.copy()
    Z2_c = Z2.copy()
    d_int = hl_digits.detach().cpu().numpy()
    c_int = hl_colors.detach().cpu().numpy()

    fig8, axs8 = plt.subplots(1, 2, figsize=(10, 4))
    sc0 = axs8[0].scatter(Z2_d[:, 0], Z2_d[:, 1], c=d_int, cmap="tab10", s=5)
    axs8[0].set_title("z (PCA2) colored by digit")
    axs8[0].set_xlabel("PC1"); axs8[0].set_ylabel("PC2")
    fig8.colorbar(sc0, ax=axs8[0], ticks=range(10))

    sc1 = axs8[1].scatter(Z2_c[:, 0], Z2_c[:, 1], c=c_int, cmap="tab10", s=5)
    axs8[1].set_title("z (PCA2) colored by color")
    axs8[1].set_xlabel("PC1"); axs8[1].set_ylabel("PC2")
    fig8.colorbar(sc1, ax=axs8[1], ticks=range(10))
    fig8.tight_layout()

    # Distances ||z_obs - z_int||^2 for a few interventions
    # (compare distribution shifts in latent space)
    z_obs_np = z_obs.detach().cpu().numpy()

    def z_for_key(key):
        entry = Dhl[key]
        return entry[:, -d_z:].detach().cpu().numpy()

    z_obs_dist = None
    z_dist_digit = z_dist_color = z_dist_both = None
    if "obs" in Dhl:
        z_obs_int = z_for_key("obs")
        z_obs_dist = np.sum((z_obs_np - z_obs_int) ** 2, axis=1)

    if "D=8" in Dhl:
        z_d = z_for_key("D=8")
        z_dist_digit = np.sum((z_obs_np - z_d) ** 2, axis=1)
    if "C=0" in Dhl:
        z_c = z_for_key("C=0")
        z_dist_color = np.sum((z_obs_np - z_c) ** 2, axis=1)
    if "D=8,C=0" in Dhl:
        z_b = z_for_key("D=8,C=0")
        z_dist_both = np.sum((z_obs_np - z_b) ** 2, axis=1)

    fig9 = plt.figure(figsize=(10, 5))
    if z_obs_dist is not None:
        plt.hist(z_obs_dist, bins=60, alpha=0.4, label="obs vs obs")
    if z_dist_digit is not None:
        plt.hist(z_dist_digit, bins=60, alpha=0.4, label="obs vs do(D=8)")
    if z_dist_color is not None:
        plt.hist(z_dist_color, bins=60, alpha=0.4, label="obs vs do(C=0)")
    if z_dist_both is not None:
        plt.hist(z_dist_both, bins=60, alpha=0.4, label="obs vs do(D=8,C=0)")
    plt.xlabel("||z_obs - z_int||^2")
    plt.ylabel("count")
    plt.title("Latent z distance between observational and interventional regimes")
    plt.legend()
    plt.grid(alpha=0.4, ls="--")
    plt.tight_layout()

    # =======================================================
    # 9) Summary page
    # =======================================================
    fig6 = plt.figure(figsize=(8.5, 11))
    plt.axis("off")

    def stats(arr):
        return f"mean={np.mean(arr):.4f}, std={np.std(arr):.4f}"

    lines = []
    lines.append(f"Dataset (Xia-encoder): {os.path.basename(ds_dir)}")
    lines.append("")
    lines.append("1. LL residual independence:")
    lines.append(f"   R²(resid_pixel ~ [digit,color]) = {r2_resid:.4f}")
    lines.append("")
    lines.append("1b. HL residual independence in z:")
    lines.append(f"   R²(resid_z_dim ~ [digit,color]) = {r2_resid_hl:.4f}")
    lines.append("")
    lines.append("2. LL reconstruction quality (32x32):")
    lines.append(f"   avg MSE(true,D)   = {mse_D_32:.6f},   avg corr(true,D)   = {corr_D_32:.4f}")
    lines.append(f"   avg MSE(true,D+U) = {mse_DU_32:.6f}, avg corr(true,D+U) = {corr_DU_32:.4f}")
    lines.append("")
    lines.append("3. HL reconstruction quality in z:")
    lines.append(f"   avg MSE(z_true,z_det)   = {mse_D_z:.6f},   avg corr(z_true,z_det)   = {corr_D_z:.4f}")
    lines.append(f"   avg MSE(z_true,z_det+U) = {mse_DU_z:.6f}, avg corr(z_true,z_det+U) = {corr_DU_z:.4f}")
    lines.append("")
    lines.append("4. LL mean-intensity stats (32x32):")
    lines.append(f"   {ll_k_obs:10s}: {stats(ll_obs_avgs)}")
    if ll_k_d in Dll:
        lines.append(f"   {ll_k_d:10s}: {stats(ll_d_avgs)}")
    if ll_k_c in Dll:
        lines.append(f"   {ll_k_c:10s}: {stats(ll_c_avgs)}")
    if ll_k_both in Dll:
        lines.append(f"   {ll_k_both:10s}: {stats(ll_both_avgs)}")
    lines.append("")
    lines.append("5. HL z-norm stats (||z||_2):")
    lines.append(f"   {hl_k_obs:10s}: {stats(hl_obs_norms)}")
    if hl_k_d in Dhl:
        lines.append(f"   {hl_k_d:10s}: {stats(hl_d_norms)}")
    if hl_k_c in Dhl:
        lines.append(f"   {hl_k_c:10s}: {stats(hl_c_norms)}")
    if hl_k_both in Dhl:
        lines.append(f"   {hl_k_both:10s}: {stats(hl_both_norms)}")
    lines.append("")
    lines.append("6. Color controllability (LL deterministic hues):")
    lines.append(f"   JS-divergence across colors = {js_hue:.4f}")
    lines.append("")
    lines.append("7. z-space separation:")
    lines.append("   - PCA2 plots show clustering by digit/color in z.")
    if z_dist_digit is not None or z_dist_color is not None:
        lines.append("   - ||z_obs - z_int||^2 histograms show how far interventions move in latent space.")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(" - LL U-Net + U_ll_hat is the nonlinear ANM at image level.")
    lines.append(" - HL model f_hl + U_hl_hat_z is a vector-valued ANM on z.")
    lines.append(" - This checks whether the z-latent SCM is a good abstraction of the LL image SCM.")

    plt.text(0.05, 0.95, "\n".join(lines),
             va="top", ha="left", family="monospace", fontsize=10)

    # =======================================================
    # Save all pages
    # =======================================================
    out_dir = os.path.join(ds_dir, "reports_xia")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"{os.path.basename(ds_dir)}_report_xia.pdf")

    with PdfPages(out_pdf) as pdf:
        for fig in [fig1, fig1b, fig2, fig3, fig4a, fig4b, fig5, fig7, fig8, fig9, fig6]:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return out_pdf


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate CMNIST reports for Xia-encoder setting ((D,C)->image, (D,C)->z)."
    )
    ap.add_argument("--data-dir", required=True,
                    help="Dataset folder containing dll_samples.pkl, dhl_samples.pkl, "
                         "ll_model_unet.pth, hl_model_cellmeans_z.joblib, "
                         "U_ll_hat.pkl, U_hl_hat_z.pkl")
    ap.add_argument("--bins", type=int, default=36, help="Bins for hue hist")
    ap.add_argument("--v-thresh", type=float, default=0.1, help="Value threshold for hue hist")
    ap.add_argument("--num-resid-pixels", type=int, default=5,
                    help="# residual dims/pixels for histograms")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device for U-Net forward passes")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        args.device = "cpu"

    ds_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

    print(f">>> Building Xia-encoder report for: {ds_dir}")
    pdf_path = make_report_for_dataset_z(
        ds_dir,
        bins=args.bins,
        v_thresh=args.v_thresh,
        num_resid_pixels=args.num_resid_pixels,
        device=args.device,
    )
    print(f"✓ Saved: {pdf_path}")


if __name__ == "__main__":
    main()
