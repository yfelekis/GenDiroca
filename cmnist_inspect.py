#!/usr/bin/env python3
# cmnist_inspect.py
# Produce a multi-page PDF analysis for a selected CMNIST dataset.
# Usage:
#   python cmnist_inspect.py --data-root "data/cmnist*_clean" --select 0 --out analysis.pdf
#   python cmnist_inspect.py --data-dir data/cmnist_..._clean --out analysis.pdf

import os
import glob
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import rgb_to_hsv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import normaltest, shapiro, anderson, skew, kurtosis
import scipy.stats as stats

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def ensure_tanh_scaled(images: torch.Tensor) -> torch.Tensor:
    """Convert [0,1] → [-1,1]; leave [-1,1] as-is."""
    with torch.no_grad():
        imin, imax = float(images.min()), float(images.max())
    if -1.05 <= imin and imax <= 1.05:
        return images
    if -1e-6 <= imin and imax <= 1.0 + 1e-6:
        return images * 2.0 - 1.0
    raise ValueError(f"Unexpected image range [{imin:.3f}, {imax:.3f}].")

def to01_any(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    if x.numel() == 0:
        return np.array([])
    if x.min() >= -1.05 and x.max() <= 1.05:
        x = (x + 1.0) / 2.0
    else:
        m, M = x.min(), x.max()
        x = (x - m) / (M - m + 1e-8)
    return x.clamp(0, 1).numpy()

def add_text_page(pdf: PdfPages, title: str, lines: list[str]):
    fig = plt.figure(figsize=(11, 8.5))
    plt.axis("off")
    txt = title + "\n" + "\n".join(lines)
    plt.text(0.02, 0.98, txt, va="top", ha="left", family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def pick_default_keys(hl_keys):
    def _pick_obs(keys):
        if "obs" in keys: return "obs"
        # fallback: first key
        return list(keys)[0]
    def _pick_first_with_prefix(keys, prefix):
        cands = [k for k in keys if isinstance(k, str) and k.startswith(prefix)]
        return cands[0] if cands else None
    obs_key = _pick_obs(hl_keys)
    dig_key = "D=8" if "D=8" in hl_keys else _pick_first_with_prefix(hl_keys, "D=")
    col_key = "C=0" if "C=0" in hl_keys else _pick_first_with_prefix(hl_keys, "C=")
    combo_literal = "D=8,C=0"
    if combo_literal in hl_keys:
        combo_key = combo_literal
    else:
        combo_candidates = [k for k in hl_keys if isinstance(k, str) and "," in k and "D=" in k and "C=" in k]
        combo_key = combo_candidates[0] if combo_candidates else None
    return obs_key, dig_key, col_key, combo_key

# ---------------------------------------------------------
# (Optional) Model – we import from your training module if possible
# ---------------------------------------------------------

def get_unet_class():
    try:
        # Preferred: import your training module's class
        from cmnist_train import ImageColorizerUNet
        return ImageColorizerUNet
    except Exception:
        # Minimal fallback (matches architecture used)
        class ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch, norm_type="bn"):
                super().__init__()
                Norm = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d, "none": lambda c: nn.Identity()}.get(
                    str(norm_type).lower(), nn.BatchNorm2d
                )
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
            def forward(self, x): return self.block(x)

        class Down(nn.Module):
            def __init__(self, in_ch, out_ch, norm_type="bn"):
                super().__init__()
                self.pool = nn.MaxPool2d(2)
                self.conv = ConvBlock(in_ch, out_ch, norm_type)
            def forward(self, x): return self.conv(self.pool(x))

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
                d = self.d_embed(digit); c = self.c_embed(color)
                h = torch.cat([d, c], dim=-1)
                params = self.mlp(h)
                C = x.size(1)
                gamma, beta = params.split(C, dim=-1)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta  = beta.unsqueeze(-1).unsqueeze(-1)
                return gamma * x + beta

        class ImageColorizerUNet(nn.Module):
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
                y  = self.up1(x3, x2)
                y  = self.film_up1(y, digit, color)
                y  = self.up2(y, x1)
                return torch.tanh(self.outc(y))
        return ImageColorizerUNet

# ---------------------------------------------------------
# Diagnostics (non-interactive, all saved to PDF)
# ---------------------------------------------------------

def comprehensive_counterfactual_diagnostics_pdf(pdf, ll_obs_tuple, ll_model, U_ll_hat, num_samples=5, bins=36, v_thresh=0.1):
    final_images, img_shapes, digits, colors = ll_obs_tuple

    # ===== 1. Residual diagnostics =====
    U = U_ll_hat.view(U_ll_hat.size(0), -1).detach().cpu().numpy()
    rng = np.random.default_rng(0)
    sample_pixels = rng.choice(U.shape[1], size=min(num_samples, U.shape[1]), replace=False)
    resid_samples = U[:, sample_pixels]

    # Histograms (sampled pixels)
    fig, axs = plt.subplots(1, resid_samples.shape[1], figsize=(3*resid_samples.shape[1], 3))
    if resid_samples.shape[1] == 1: axs = np.array([axs])
    for i in range(resid_samples.shape[1]):
        axs[i].hist(resid_samples[:, i], bins=40, alpha=0.7)
        axs[i].set_title(f"Residuals (pixel {sample_pixels[i]})")
    plt.suptitle("Residual distributions — sampled pixels")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Dependence check
    X = np.stack([digits.cpu().numpy(), colors.cpu().numpy()], axis=1)
    y = resid_samples[:, 0]
    linreg = LinearRegression().fit(X, y)
    y_pred = linreg.predict(X)
    r2_resid = r2_score(y, y_pred)

    # Visual sanity check
    idx = int(np.random.randint(0, len(U_ll_hat)))
    resid_img = to01_any(U_ll_hat[idx]).transpose(1,2,0)
    true_img  = to01_any(final_images[idx]).transpose(1,2,0)
    pred_img  = np.clip(true_img - resid_img, 0.0, 1.0)

    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    axs[0].imshow(true_img); axs[0].set_title("True image"); axs[0].axis("off")
    axs[1].imshow(pred_img); axs[1].set_title("Pred deterministic"); axs[1].axis("off")
    axs[2].imshow(resid_img); axs[2].set_title("Residual (noise)"); axs[2].axis("off")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ===== 2. Counterfactual consistency =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ll_model = ll_model.to(device).eval()
    with torch.no_grad():
        base_shape = img_shapes[idx:idx+1].to(device)
        base_digit = digits[idx:idx+1].to(device)
        U_fixed    = U_ll_hat[idx:idx+1].to(device)
        dets, outs = [], []
        for c in range(10):
            c_tensor = torch.tensor([c], device=device)
            det = ll_model(base_shape, base_digit, c_tensor)
            ycf = det + U_fixed
            dets.append(det.cpu()); outs.append(ycf.cpu())
    D = torch.stack(dets, dim=0)  # (10,1,3,H,W)
    O = torch.stack(outs, dim=0)  # (10,1,3,H,W)
    D_gray = D.mean(2)  # (10,1,H,W)
    var_digit = float(D_gray.var(0).mean().item())
    mean_colors = O.view(O.size(0), O.size(1), -1).mean(-1)
    var_color = float(mean_colors.var(0).mean().item())
    U_cmp = O - D
    U_flat = U_cmp.view(U_cmp.size(0), -1).cpu().numpy()
    corr_mat = np.corrcoef(U_flat)
    iu = np.triu_indices_from(corr_mat, k=1)
    mean_corr = float(corr_mat[iu].mean())

    # Save sweep figure
    def to01(img_t):
        x = img_t.detach().cpu().float()
        if x.min() >= -1.05 and x.max() <= 1.05:
            x = (x + 1.0) / 2.0
        else:
            m, M = x.min(), x.max()
            x = (x - m) / (M - m + 1e-8)
        return x.clamp(0, 1).numpy()

    fig, axs = plt.subplots(2, 10, figsize=(15, 3))
    for c in range(10):
        axs[0, c].imshow(to01(dets[c][0].permute(1,2,0))); axs[0, c].axis("off")
        axs[0, c].set_title(f"c={c}", fontsize=8)
        axs[1, c].imshow(to01(outs[c][0].permute(1,2,0))); axs[1, c].axis("off")
    axs[0,0].set_ylabel("deterministic", fontsize=9)
    axs[1,0].set_ylabel("with noise", fontsize=9)
    plt.suptitle("Counterfactual color sweep (fixed U)", y=1.02)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ===== 3. Color hue analysis =====
    def hue_hist(img_chw, bins=36, v_thresh=0.25):
        x = img_chw.detach().cpu().float()
        if x.min() >= -1.05 and x.max() <= 1.05:
            x = (x + 1) / 2
        else:
            m, M = x.min(), x.max()
            x = (x - m) / (M - m + 1e-8)
        rgb = x.permute(1,2,0).numpy()
        hsv = rgb_to_hsv(rgb)
        H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
        mask = V > v_thresh
        if mask.sum() == 0:
            return np.ones(bins) / bins
        h = H[mask].ravel()
        hist, _ = np.histogram(h, bins=bins, range=(0,1), density=False)
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-12
        return hist

    def js_divergence(p, q):
        m = 0.5*(p+q)
        def kl(a,b):
            a = np.clip(a, 1e-12, 1.0)
            b = np.clip(b, 1e-12, 1.0)
            return np.sum(a * np.log(a/b))
        return 0.5*kl(p,m) + 0.5*kl(q,m)

    Hs = [hue_hist(det[0], bins=bins, v_thresh=v_thresh) for det in dets]
    js_vals = [js_divergence(Hs[i], Hs[j]) for i in range(len(Hs)) for j in range(i+1, len(Hs))]
    js_hue = float(np.mean(js_vals)) if js_vals else 0.0

    # Summary page
    lines = [
        "RESIDUAL DIAGNOSTICS",
        f"  R^2(residual ~ [digit,color]) = {r2_resid:.6f}",
        "",
        "COUNTERFACTUAL CONSISTENCY (U-Net)",
        f"  Digit variance (shape invariance proxy): {var_digit:.6f}",
        f"  Color variance (brightness sensitivity): {var_color:.6f}",
        f"  Noise stability (mean corr across colors): {mean_corr:.6f}",
        "",
        "COLOR HUE DIVERSITY",
        f"  Mean pairwise JS divergence (hue histograms): {js_hue:.6f}",
    ]
    add_text_page(pdf, "Comprehensive Counterfactual Diagnostics — Summary", lines)

    return {
        'residual_r2': r2_resid,
        'consistency_scores': {
            'digit_variance': var_digit,
            'color_variance': var_color,
            'noise_corr': mean_corr
        },
        'js_hue': js_hue
    }

def background_edge_analysis_pdf(pdf, U_ll_hat):
    assert U_ll_hat.dim() == 4 and U_ll_hat.size(1) in (1,3)
    N, C, H, W = U_ll_hat.shape
    background_threshold = float(0.5 * U_ll_hat.std().item())
    edge_frac = 0.125
    edge_w = max(1, int(round(min(H, W) * edge_frac)))

    bg_mask = (U_ll_hat.abs() < background_threshold)
    bg_pixels = U_ll_hat[bg_mask]
    bg_ratio = 100.0 * (bg_pixels.numel() / max(1, U_ll_hat.numel()))

    edge_mask = torch.zeros((H, W), dtype=torch.bool, device=U_ll_hat.device)
    edge_mask[:edge_w, :] = True; edge_mask[-edge_w:, :] = True
    edge_mask[:, :edge_w] = True; edge_mask[:, -edge_w:] = True
    edge_mask_b = edge_mask.view(1,1,H,W).expand(N,C,H,W)
    center_mask_b = ~edge_mask_b
    edge_pixels = U_ll_hat[edge_mask_b]
    center_pixels = U_ll_hat[center_mask_b]

    def _s(x): 
        return (float(x.mean().item()), float(x.std().item()), float(x.min().item()), float(x.max().item()), int(x.numel()))

    center_stats = _s(center_pixels) if center_pixels.numel() else None
    edge_stats   = _s(edge_pixels) if edge_pixels.numel() else None
    exact_zero = int((U_ll_hat == 0).sum().item())
    lt_1e3 = int((U_ll_hat.abs() < 1e-3).sum().item())
    lt_1e2 = int((U_ll_hat.abs() < 1e-2).sum().item())
    lt_1e1 = int((U_ll_hat.abs() < 1e-1).sum().item())

    vals = U_ll_hat.detach().cpu().numpy().reshape(-1)
    vlim = float(np.percentile(np.abs(vals), 99)) if vals.size else 0.1
    vlim = max(vlim, 1e-3)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0,0].hist(vals, bins=80, alpha=0.7, label='All pixels')
    if bg_pixels.numel() > 0:
        axes[0,0].hist(bg_pixels.detach().cpu().numpy(), bins=60, alpha=0.6, label='Background-like')
    axes[0,0].axvline(+background_threshold, color='k', ls='--', lw=1)
    axes[0,0].axvline(-background_threshold, color='k', ls='--', lw=1)
    axes[0,0].set_title('All vs Background-like pixels'); axes[0,0].legend()

    if edge_pixels.numel() > 0:
        axes[0,1].hist(edge_pixels.detach().cpu().numpy().reshape(-1), bins=60, alpha=0.8)
    axes[0,1].set_title('Edge pixels distribution')

    if center_pixels.numel() > 0 and edge_pixels.numel() > 0:
        axes[0,2].hist(center_pixels.detach().cpu().numpy().reshape(-1), bins=60, alpha=0.7, label='Center')
        axes[0,2].hist(edge_pixels.detach().cpu().numpy().reshape(-1), bins=60, alpha=0.7, label='Edge')
        axes[0,2].legend()
    axes[0,2].set_title('Center vs Edge')

    idx = 0
    img = U_ll_hat[idx].detach().cpu()
    if C == 1:
        axes[1,0].imshow(img[0], cmap='RdBu_r', vmin=-vlim, vmax=+vlim)
    else:
        axes[1,0].imshow(img.mean(0), cmap='RdBu_r', vmin=-vlim, vmax=+vlim)
    axes[1,0].set_title(f"Sample {idx} residual (mean over channels)"); axes[1,0].axis('off')

    bg_mask_sample = (img.abs() < background_threshold)
    axes[1,1].imshow(bg_mask_sample.any(0), cmap='gray')
    axes[1,1].set_title(f"Background-like mask (τ={background_threshold:.3f})"); axes[1,1].axis('off')

    axes[1,2].imshow(edge_mask.cpu(), cmap='gray')
    axes[1,2].set_title(f"Edge mask (width={edge_w})"); axes[1,2].axis('off')

    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    lines = [
        f"Image size: {H}x{W} | channels: {C}",
        f"Background threshold τ: {background_threshold:.6f}",
        f"Edge band width: {edge_w} px",
        f"Background-like pixels: {bg_pixels.numel()}/{U_ll_hat.numel()} ({bg_ratio:.2f}%)",
        f"Center stats: mean={center_stats[0]:.6f} std={center_stats[1]:.6f} min={center_stats[2]:.6f} max={center_stats[3]:.6f} count={center_stats[4]}" if center_stats else "Center stats: (empty)",
        f"Edge stats:   mean={edge_stats[0]:.6f} std={edge_stats[1]:.6f} min={edge_stats[2]:.6f} max={edge_stats[3]:.6f} count={edge_stats[4]}" if edge_stats else "Edge stats: (empty)",
        "Near-zero counts:",
        f"  exactly 0: {exact_zero}",
        f"  |x| < 1e-3: {lt_1e3}  |x| < 1e-2: {lt_1e2}  |x| < 1e-1: {lt_1e1}",
    ]
    add_text_page(pdf, "Background & Edge Analysis (U_ll_hat)", lines)

def hl_distribution_analysis_pdf(pdf, Dhl_samples):
    hl_keys = list(Dhl_samples.keys())
    obs_key, dig_key, col_key, combo_key = pick_default_keys(hl_keys)

    def _to_numpy(x):
        if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
        return np.asarray(x)

    def get_lastcol_for_key(k, synth_from=None):
        if k is not None and k in Dhl_samples:
            arr = _to_numpy(Dhl_samples[k])
            return arr[:, -1]
        if synth_from is None: return None
        dkey, ckey = synth_from
        if dkey not in Dhl_samples or ckey not in Dhl_samples: return None
        Hd = _to_numpy(Dhl_samples[dkey]); Hc = _to_numpy(Dhl_samples[ckey])
        mask = np.all(np.isclose(Hd[:, :20], Hc[:, :20]), axis=1)
        if not np.any(mask): return None
        return 0.5 * (Hd[mask, -1] + Hc[mask, -1])

    vals = {}
    obs_vals = get_lastcol_for_key(obs_key)
    if obs_vals is None:
        add_text_page(pdf, "HL Distribution Analysis", ["ERROR: Could not locate observational HL rows."])
        return
    vals["Observational"] = obs_vals
    if dig_key:
        v = get_lastcol_for_key(dig_key)
        if v is not None: vals[f"do({dig_key})"] = v
    if col_key:
        v = get_lastcol_for_key(col_key)
        if v is not None: vals[f"do({col_key})"] = v
    if combo_key:
        v = get_lastcol_for_key(combo_key)
        if v is not None:
            vals[f"do({combo_key})"] = v
    else:
        if dig_key and col_key:
            v = get_lastcol_for_key(None, synth_from=(dig_key, col_key))
            if v is not None: vals[f"do({dig_key},{col_key})"] = v

    # Print stats page
    lines = ["HL Image_ Distribution across Interventions"]
    for name, arr in vals.items():
        lines.append(f"{name}:")
        lines.append(f"  Mean: {arr.mean():.6f} | Std: {arr.std():.6f} | Min: {arr.min():.6f} | Max: {arr.max():.6f}")
    add_text_page(pdf, "HL Image_ Distribution — Summary Stats", lines)

    # Hist plot
    fig = plt.figure(figsize=(12, 8))
    for i, (name, arr) in enumerate(vals.items()):
        plt.hist(arr, bins=50, alpha=0.6, density=True, label=name)
    plt.title('HL Image_ Distribution across Interventions')
    plt.xlabel('Mean pixel intensity (HL Image_)'); plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.5); plt.legend()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Mean deltas page
    obs_mean = vals["Observational"].mean()
    lines = ["INTERVENTION RESPONSE (mean shift vs obs)"]
    for name, arr in vals.items():
        if name == "Observational": continue
        lines.append(f"{name}: Δ mean = {arr.mean() - obs_mean:+.6f}")
    add_text_page(pdf, "HL Intervention Response — Mean Shifts", lines)

def gaussianity_tests_pdf(pdf, U_ll_hat, U_hl_hat, num_samples=1000):
    lines_head = ["GAUSSIANITY TESTS FOR NOISE"]
    add_text_page(pdf, "Gaussianity Tests — Header", lines_head)

    # LL flatten
    U_ll_flat = U_ll_hat.view(U_ll_hat.size(0), -1).detach().cpu().numpy()
    n_pixels = min(num_samples, U_ll_flat.shape[1])
    rng = np.random.default_rng(0)
    sample_pixels = rng.choice(U_ll_flat.shape[1], size=n_pixels, replace=False)
    U_ll_samples = U_ll_flat[:, sample_pixels]

    # Test first 10 pixels
    ll_results = []
    lines = ["LOW-LEVEL NOISE (first 10 sampled pixels)"]
    for i in range(min(10, len(sample_pixels))):
        pixel = int(sample_pixels[i])
        pixel_data = U_ll_samples[:, i]
        stat_dp, p_dp = normaltest(pixel_data)
        if len(pixel_data) <= 5000:
            stat_sw, p_sw = shapiro(pixel_data)
        else:
            stat_sw, p_sw = np.nan, np.nan
        stat_ad, critical_values, _ = anderson(pixel_data, dist='norm')
        ll_results.append({
            'pixel': pixel,
            'dp_stat': float(stat_dp), 'dp_p': float(p_dp),
            'sw_stat': float(stat_sw), 'sw_p': float(p_sw),
            'ad_stat': float(stat_ad), 'ad_critical_5pct': float(critical_values[2])
        })
        lines.append(f"Pixel {pixel}: DP p={p_dp:.4f}, SW p={p_sw if not np.isnan(p_sw) else float('nan'):.4f}, AD stat={stat_ad:.4f} (5% crit {critical_values[2]:.4f})")
    add_text_page(pdf, "Gaussianity — LL Stats (first 10)", lines)

    # LL hist overlays (6 pixels)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i in range(6):
        if i >= len(sample_pixels): break
        pixel = int(sample_pixels[i])
        pixel_data = U_ll_samples[:, i]
        ax = axes[i//3, i%3]
        ax.hist(pixel_data, bins=50, density=True, alpha=0.7, label='Data')
        mu, sigma = pixel_data.mean(), pixel_data.std()
        xs = np.linspace(pixel_data.min(), pixel_data.max(), 100)
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), linewidth=2, label='Normal')
        ax.set_title(f'LL pixel {pixel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # HL tests
    U_hl_flat = U_hl_hat.detach().cpu().numpy().flatten()
    stat_dp_hl, p_dp_hl = normaltest(U_hl_flat)
    if len(U_hl_flat) <= 5000:
        stat_sw_hl, p_sw_hl = shapiro(U_hl_flat)
    else:
        stat_sw_hl, p_sw_hl = np.nan, np.nan
    stat_ad_hl, critical_values_hl, _ = anderson(U_hl_flat, dist='norm')

    lines = [
        "HIGH-LEVEL NOISE (scalar Image_ residuals)",
        f"D'Agostino–Pearson: p={p_dp_hl:.4f}",
        f"Shapiro–Wilk: p={p_sw_hl if not np.isnan(p_sw_hl) else float('nan'):.4f}",
        f"Anderson–Darling: stat={stat_ad_hl:.4f} (5% crit {critical_values_hl[2]:.4f})",
    ]
    add_text_page(pdf, "Gaussianity — HL Stats", lines)

    # HL visuals
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(U_hl_flat, bins=50, density=True, alpha=0.7, label='Data')
    mu_hl, sigma_hl = U_hl_flat.mean(), U_hl_flat.std()
    x_hl = np.linspace(U_hl_flat.min(), U_hl_flat.max(), 100)
    axes[0].plot(x_hl, stats.norm.pdf(x_hl, mu_hl, sigma_hl), linewidth=2, label='Normal')
    axes[0].set_title('HL: Histogram vs Normal'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    stats.probplot(U_hl_flat, dist="norm", plot=axes[1]); axes[1].set_title('HL: Q-Q Plot'); axes[1].grid(True, alpha=0.3)
    axes[2].boxplot(U_hl_flat); axes[2].set_title('HL: Box Plot'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # Summaries & interpretations
    ll_means = U_ll_samples.mean(axis=0)
    ll_stds = U_ll_samples.std(axis=0)
    ll_skews = skew(U_ll_samples, axis=0)
    ll_kurt = kurtosis(U_ll_samples, axis=0)

    lines = [
        "SUMMARY STATISTICS",
        f"LL: mean(mean)={ll_means.mean():.6f}, mean(std)={ll_stds.mean():.6f}, mean(skew)={ll_skews.mean():.6f}, mean(kurt)={ll_kurt.mean():.6f}",
        f"HL: mean={U_hl_flat.mean():.6f}, std={U_hl_flat.std():.6f}, skew={skew(U_hl_flat):.6f}, kurt={kurtosis(U_hl_flat):.6f}",
        "",
        "INTERPRETATION",
        ("LL: ✓ Appears Gaussian on average" if np.mean([r['dp_p'] for r in ll_results]) > 0.05
         else "LL: ✗ Non-Gaussian indications (mean DP p ≤ 0.05)"),
        ("HL: ✓ Appears Gaussian" if p_dp_hl > 0.05 else "HL: ✗ Non-Gaussian indications (DP p ≤ 0.05)"),
    ]
    add_text_page(pdf, "Gaussianity — Summary & Interpretation", lines)

def quick_u_stats_page(pdf, U_ll_hat, U_hl_hat):
    lines = [
        "U_ll_hat quick stats:",
        f"  shape={tuple(U_ll_hat.shape)} dtype={U_ll_hat.dtype}",
        f"  min={U_ll_hat.min().item():.6f} max={U_ll_hat.max().item():.6f}",
        f"  mean={U_ll_hat.mean().item():.6f} std={U_ll_hat.std().item():.6f}",
        f"  nonzero={(U_ll_hat != 0).sum().item()}/{U_ll_hat.numel()} "
        f"({(U_ll_hat != 0).sum().item()/U_ll_hat.numel()*100:.2f}%)",
        "",
        "U_hl_hat quick stats:",
        f"  shape={tuple(U_hl_hat.shape)} dtype={U_hl_hat.dtype}",
        f"  min={U_hl_hat.min().item():.6f} max={U_hl_hat.max().item():.6f}",
        f"  mean={U_hl_hat.mean().item():.6f} std={U_hl_hat.std().item():.6f}",
        f"  nonzero={(U_hl_hat != 0).sum().item()}/{U_hl_hat.numel()} "
        f"({(U_hl_hat != 0).sum().item()/U_hl_hat.numel()*100:.2f}%)",
    ]
    add_text_page(pdf, "Residuals — Quick Statistics", lines)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate a PDF analysis for a CMNIST dataset.")
    ap.add_argument("--data-root", type=str, default="data/cmnist*_clean",
                    help="Glob to discover dataset folders (ignored if --data-dir is set).")
    ap.add_argument("--select", type=int, default=0, help="Index of discovered dataset to use.")
    ap.add_argument("--data-dir", type=str, default=None, help="Direct path to a single dataset folder.")
    ap.add_argument("--out", type=str, default="cmnist_analysis.pdf", help="Output PDF filename.")
    args = ap.parse_args()

    # Resolve dataset folder
    if args.data_dir:
        data_dir = args.data_dir
    else:
        datasets = sorted(glob.glob(args.data_root))
        if not datasets:
            raise RuntimeError(f"No datasets found for glob: {args.data_root}")
        if args.select < 0 or args.select >= len(datasets):
            raise RuntimeError(f"--select out of range. Found {len(datasets)} datasets.")
        data_dir = datasets[args.select]

    # Load artifacts
    dll_path = os.path.join(data_dir, "dll_samples.pkl")
    dhl_path = os.path.join(data_dir, "dhl_samples.pkl")
    omega_path = os.path.join(data_dir, "intervention_mapping.pkl")
    ll_state_path = os.path.join(data_dir, "ll_model_unet.pth")
    U_ll_path = os.path.join(data_dir, "U_ll_hat.pkl")
    U_hl_path = os.path.join(data_dir, "U_hl_hat.pkl")

    assert os.path.isfile(dll_path), f"Missing {dll_path}"
    assert os.path.isfile(dhl_path), f"Missing {dhl_path}"
    assert os.path.isfile(omega_path), f"Missing {omega_path}"
    assert os.path.isfile(ll_state_path), f"Missing {ll_state_path} (train first)"
    assert os.path.isfile(U_ll_path), f"Missing {U_ll_path} (train first)"
    assert os.path.isfile(U_hl_path), f"Missing {U_hl_path} (train first)"

    Dll_samples = torch.load(dll_path)
    Dhl_samples = torch.load(dhl_path)
    omega_labels = torch.load(omega_path)

    # Grab observational tuples
    def _get_obs_tuple(D):
        return D.get("obs") or D.get(None) or next(iter(D.values()))
    ll_obs = _get_obs_tuple(Dll_samples)
    hl_obs = Dhl_samples.get("obs", next(iter(Dhl_samples.values())))

    # Load model
    ImageColorizerUNet = get_unet_class()
    ll_model = ImageColorizerUNet(norm_type="bn")
    ll_model.load_state_dict(torch.load(ll_state_path, map_location="cpu"))
    ll_model.eval()

    # Load residuals
    U_ll_hat = torch.load(U_ll_path)
    U_hl_hat = torch.load(U_hl_path)

    # PDF analysis
    with PdfPages(args.out) as pdf:
        # Cover
        lines = [
            f"Dataset: {os.path.basename(data_dir)}",
            f"Source folder: {data_dir}",
            "",
            "This report summarizes:",
            "  • Residual diagnostics & counterfactual consistency (U-Net)",
            "  • HL Image_ distribution across interventions",
            "  • Background & edge analysis for U_ll_hat",
            "  • Gaussianity tests (LL & HL noise)",
        ]
        add_text_page(pdf, "CMNIST — Data & Abduction Diagnostics", lines)

        # Shapes page
        imgs, shapes, digs, cols = ll_obs
        lines = [
            "Observational tensors:",
            f"  LL images:  {tuple(imgs.shape)}  range≈[{float(imgs.min()):.3f},{float(imgs.max()):.3f}]",
            f"  LL shapes:  {tuple(shapes.shape)}  range≈[{float(shapes.min()):.3f},{float(shapes.max()):.3f}]",
            f"  digits:     {tuple(digs.shape)}",
            f"  colors:     {tuple(cols.shape)}",
            f"  HL obs:     {tuple(hl_obs.shape)}",
            "",
            "Residual artifacts:",
            f"  U_ll_hat:   {tuple(U_ll_hat.shape)}",
            f"  U_hl_hat:   {tuple(U_hl_hat.shape)}",
        ]
        add_text_page(pdf, "Dataset Summary", lines)

        # Comprehensive diagnostics
        comp = comprehensive_counterfactual_diagnostics_pdf(pdf, ll_obs, ll_model, U_ll_hat, num_samples=5, bins=36, v_thresh=0.1)

        # Quick residual stats
        quick_u_stats_page(pdf, U_ll_hat, U_hl_hat)

        # Background & edge analysis
        background_edge_analysis_pdf(pdf, U_ll_hat)

        # HL distributions
        hl_distribution_analysis_pdf(pdf, Dhl_samples)

        # Gaussianity tests
        gaussianity_tests_pdf(pdf, U_ll_hat, U_hl_hat, num_samples=1000)

        # Final summary page
        lines = [
            "SUMMARY",
            f"Residual independence (R^2): {comp['residual_r2']:.6f}",
            f"Digit variance (shape invariance): {comp['consistency_scores']['digit_variance']:.6f}",
            f"Color variance (sensitivity):      {comp['consistency_scores']['color_variance']:.6f}",
            f"Noise stability (mean corr):       {comp['consistency_scores']['noise_corr']:.6f}",
            f"Hue diversity (JS):                {comp['js_hue']:.6f}",
            "",
            "Heuristics:",
            ("  • Residuals ≈ independent (good)" if abs(comp['residual_r2']) < 0.05 else "  • Residuals show label structure (needs work)"),
            ("  • Shape preserved if digit variance < 1e-2" if comp['consistency_scores']['digit_variance'] < 1e-2 else "  • Noticeable shape drift"),
            ("  • Color sensitivity good if variance > 1e-2" if comp['consistency_scores']['color_variance'] > 1e-2 else "  • Low color sensitivity"),
            ("  • Noise stable if corr > 0.8" if comp['consistency_scores']['noise_corr'] > 0.8 else "  • Noise varies across colors"),
        ]
        add_text_page(pdf, "Overall Summary & Heuristics", lines)

    print(f"✓ Analysis PDF written to: {args.out}")

if __name__ == "__main__":
    main()
