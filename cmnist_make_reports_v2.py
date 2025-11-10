#!/usr/bin/env python3
import os, glob, re, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import rgb_to_hsv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
# Discover v2 datasets
# ============================================================

def discover_datasets_v2(root, pattern=None):
    """
    v2-style dataset = folder containing:
      dll_samples.pkl, dhl_samples.pkl,
      ll_unet_32.pth, hl_unet_16.pth,
      U_ll_hat_32.pkl, U_hl_hat_16.pkl
    """
    root = os.path.abspath(root)
    # candidate directories named cmnist* under root (recursive search)
    candidates = sorted(glob.glob(os.path.join(root, "**/cmnist*"), recursive=True))
    # Filter to only directories
    candidates = [d for d in candidates if os.path.isdir(d)]
    if pattern:
        rx = re.compile(pattern)
        candidates = [p for p in candidates if rx.search(os.path.basename(p))]

    out = []
    required = [
        "dll_samples.pkl",
        "dhl_samples.pkl",
        "ll_unet_32.pth",
        "hl_unet_16.pth",
        "U_ll_hat_32.pkl",
        "U_hl_hat_16.pkl",
    ]

    for d in candidates:
        if all(os.path.isfile(os.path.join(d, f)) for f in required):
            out.append(d)

    # also: if root itself looks like a dataset, include it
    if all(os.path.isfile(os.path.join(root, f)) for f in required):
        if root not in out:
            out.append(root)

    return out


# ============================================================
# HL obs normalization
# ============================================================

def normalize_hl_obs(hl_entry, ll_shapes_32, ll_digits, ll_colors):
    """
    Normalize Dhl['obs'] into:
      (imgs_16, shapes_16, digits, colors).

    Cases:
      - Tensor (N,3,16,16): use LL digits/colors and pooled shapes.
      - Tuple/list: assume (imgs_16, shapes_16, digits, colors).
      - Dict: try common keys.
    """
    if isinstance(hl_entry, torch.Tensor):
        imgs_16 = hl_entry
        with torch.no_grad():
            shapes_16 = F.avg_pool2d(ll_shapes_32, kernel_size=2, stride=2)
        return imgs_16, shapes_16, ll_digits, ll_colors

    if isinstance(hl_entry, (tuple, list)):
        if len(hl_entry) < 4:
            raise ValueError(f"HL obs tuple too short: len={len(hl_entry)}")
        return hl_entry[0], hl_entry[1], hl_entry[2], hl_entry[3]

    if isinstance(hl_entry, dict):
        def _find(d, keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        imgs_16 = _find(hl_entry, ["images", "final_images", "imgs"])
        shapes_16 = _find(hl_entry, ["shapes", "img_shapes"])
        digits = _find(hl_entry, ["digits", "D", "labels"])
        colors = _find(hl_entry, ["colors", "C"])
        if any(v is None for v in [imgs_16, shapes_16, digits, colors]):
            raise KeyError("HL obs dict missing images/shapes/digits/colors.")
        return imgs_16, shapes_16, digits, colors

    raise TypeError(f"Unsupported HL obs type: {type(hl_entry)}")


# ============================================================
# Metric helpers
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


def metrics_all(images, model, U_hat, shapes, digits, colors, device):
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

            bg = bg_level(true, pct=5.0)
            true_d = apply_bg(true, bg).view(-1).numpy()
            pred_d = apply_bg(pred, bg).view(-1).numpy()
            recon_d = apply_bg(recon, bg).view(-1).numpy()

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


# ============================================================
# Core: build report for one dataset
# ============================================================

def make_report_for_dataset_v2(ds_dir, bins=36, v_thresh=0.1,
                               num_resid_pixels=5, seed=0, device="cpu"):

    from cmnist_train_v2 import FilmUNet  # same arch used for training

    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    # ---------- Load artifacts ----------
    Dll = torch.load(os.path.join(ds_dir, "dll_samples.pkl"), map_location="cpu")
    Dhl = torch.load(os.path.join(ds_dir, "dhl_samples.pkl"), map_location="cpu")
    U_ll = torch.load(os.path.join(ds_dir, "U_ll_hat_32.pkl"), map_location="cpu")
    U_hl = torch.load(os.path.join(ds_dir, "U_hl_hat_16.pkl"), map_location="cpu")

    ll_model = FilmUNet()
    ll_model.load_state_dict(torch.load(os.path.join(ds_dir, "ll_unet_32.pth"), map_location="cpu"))
    ll_model.to(dev).eval()

    hl_model = FilmUNet()
    hl_model.load_state_dict(torch.load(os.path.join(ds_dir, "hl_unet_16.pth"), map_location="cpu"))
    hl_model.to(dev).eval()

    # LL observations
    ll_obs = get_obs_entry(Dll)
    ll_imgs_32, ll_shapes_32, ll_digits, ll_colors = ll_obs
    assert ll_imgs_32.shape[2:] == (32, 32)

    # HL observations
    hl_obs_raw = get_obs_entry(Dhl)
    hl_imgs_16, hl_shapes_16, hl_digits, hl_colors = normalize_hl_obs(
        hl_obs_raw, ll_shapes_32, ll_digits, ll_colors
    )
    assert hl_imgs_16.shape[2:] == (16, 16)

    # =======================================================
    # 1) LL residual independence
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
        axs1[i].set_title(f"Residual pixel {p}")
    fig1.suptitle("LL residual distributions (sampled pixels)")
    fig1.tight_layout()

    # =======================================================
    # 1b) HL residual independence
    # =======================================================
    U_hl_flat = U_hl.view(U_hl.size(0), -1).cpu().numpy()
    pix_idx_hl = rng.choice(
        U_hl_flat.shape[1],
        size=min(num_resid_pixels, U_hl_flat.shape[1]),
        replace=False,
    )
    resid_samples_hl = U_hl_flat[:, pix_idx_hl]

    X_lab_hl = np.stack([hl_digits.numpy(), hl_colors.numpy()], axis=1)
    y_hl = resid_samples_hl[:, 0]
    lin_hl = LinearRegression().fit(X_lab_hl, y_hl)
    r2_resid_hl = r2_score(y_hl, lin_hl.predict(X_lab_hl))

    fig1b, axs1b = plt.subplots(1, len(pix_idx_hl), figsize=(3.0 * len(pix_idx_hl), 3))
    if len(pix_idx_hl) == 1:
        axs1b = [axs1b]
    for i, p in enumerate(pix_idx_hl):
        axs1b[i].hist(resid_samples_hl[:, i], bins=40, alpha=0.8)
        axs1b[i].set_title(f"Residual pixel {p}")
    fig1b.suptitle("HL residual distributions (sampled pixels)")
    fig1b.tight_layout()

    # =======================================================
    # 2) LL qualitative example (32x32)
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

    bg32 = bg_level(true32, pct=5.0)
    true32_d = apply_bg(true32, bg32)
    pred32_d = apply_bg(pred32, bg32)
    recon32_d = apply_bg(recon32, bg32)

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

    mse_D_32, corr_D_32, mse_DU_32, corr_DU_32 = metrics_all(
        ll_imgs_32, ll_model, U_ll, ll_shapes_32, ll_digits, ll_colors, dev
    )

    # =======================================================
    # 3) HL qualitative example (16x16)
    # =======================================================
    with torch.no_grad():
        det16 = hl_model(
            hl_shapes_16[idx_vis:idx_vis+1].to(dev),
            hl_digits[idx_vis:idx_vis+1].to(dev),
            hl_colors[idx_vis:idx_vis+1].to(dev),
        )[0].cpu()

    true16 = map_to01(hl_imgs_16[idx_vis])
    pred16 = map_to01(det16)
    resid16 = U_hl[idx_vis].detach().cpu()
    recon16 = map_to01(det16 + resid16)

    bg16 = bg_level(true16, pct=5.0)
    true16_d = apply_bg(true16, bg16)
    pred16_d = apply_bg(pred16, bg16)
    recon16_d = apply_bg(recon16, bg16)

    resid16_np = resid16.permute(1, 2, 0).numpy()
    mag16 = np.percentile(np.abs(resid16_np).ravel(), 99)
    mag16 = max(mag16, 1e-6)

    fig3, axs3 = plt.subplots(1, 4, figsize=(11, 3))
    axs3[0].imshow(true16_d.permute(1, 2, 0).numpy()); axs3[0].set_title("True 16x16"); axs3[0].axis("off")
    axs3[1].imshow(pred16_d.permute(1, 2, 0).numpy()); axs3[1].set_title("Deterministic D"); axs3[1].axis("off")
    axs3[2].imshow(resid16_np, cmap="RdBu_r", vmin=-mag16, vmax=mag16); axs3[2].set_title("Residual U"); axs3[2].axis("off")
    axs3[3].imshow(recon16_d.permute(1, 2, 0).numpy()); axs3[3].set_title("D + U"); axs3[3].axis("off")
    fig3.suptitle("High-level decomposition (16x16)")
    fig3.tight_layout()

    mse_D_16, corr_D_16, mse_DU_16, corr_DU_16 = metrics_all(
        hl_imgs_16, hl_model, U_hl, hl_shapes_16, hl_digits, hl_colors, dev
    )

    # =======================================================
    # 4) LL intervention histograms: mean intensity
    # =======================================================
    def get_means(entry):
        """Extract mean intensity from LL or HL entry."""
        if isinstance(entry, torch.Tensor):
            arr = to01_any(entry)
        elif isinstance(entry, (tuple, list)):
            arr = to01_any(entry[0])
        elif isinstance(entry, dict):
            # For dict entries, try to get the first image tensor
            arr = to01_any(next(iter(entry.values())) if entry else entry)
        else:
            arr = to01_any(entry)
        return arr.mean(axis=(1, 2, 3))

    ll_keys = list(Dll.keys())
    ll_k_obs = "obs" if "obs" in ll_keys else ll_keys[0]
    ll_k_d = "D=8" if "D=8" in ll_keys else next(
        (k for k in ll_keys if k.startswith("D=") and "," not in k and k != ll_k_obs), ll_keys[1] if len(ll_keys) > 1 else ll_k_obs
    )
    ll_k_c = "C=0" if "C=0" in ll_keys else next(
        (k for k in ll_keys if k.startswith("C=") and "," not in k and k != ll_k_obs), ll_keys[2] if len(ll_keys) > 2 else ll_k_obs
    )
    ll_k_both = "D=8,C=0" if "D=8,C=0" in ll_keys else next(
        (k for k in ll_keys if "," in k), ll_keys[-1] if len(ll_keys) > 0 else ll_k_obs
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
    # 5) HL intervention histograms: mean intensity
    # =======================================================
    hl_keys = list(Dhl.keys())
    hl_k_obs = "obs" if "obs" in hl_keys else hl_keys[0]
    hl_k_d = "D=8" if "D=8" in hl_keys else next(
        (k for k in hl_keys if k.startswith("D=") and "," not in k and k != hl_k_obs), hl_keys[1] if len(hl_keys) > 1 else hl_k_obs
    )
    hl_k_c = "C=0" if "C=0" in hl_keys else next(
        (k for k in hl_keys if k.startswith("C=") and "," not in k and k != hl_k_obs), hl_keys[2] if len(hl_keys) > 2 else hl_k_obs
    )
    hl_k_both = "D=8,C=0" if "D=8,C=0" in hl_keys else next(
        (k for k in hl_keys if "," in k), hl_keys[-1] if len(hl_keys) > 0 else hl_k_obs
    )

    hl_obs_entry = get_obs_entry(Dhl) if hl_k_obs == "obs" else (Dhl[hl_k_obs] if hl_k_obs in Dhl else get_obs_entry(Dhl))
    hl_obs_avgs = get_means(hl_obs_entry)
    hl_d_entry = Dhl[hl_k_d] if hl_k_d in Dhl else get_obs_entry(Dhl)
    hl_d_avgs = get_means(hl_d_entry) if hl_k_d in Dhl else hl_obs_avgs
    hl_c_entry = Dhl[hl_k_c] if hl_k_c in Dhl else get_obs_entry(Dhl)
    hl_c_avgs = get_means(hl_c_entry) if hl_k_c in Dhl else hl_obs_avgs
    hl_both_entry = Dhl[hl_k_both] if hl_k_both in Dhl else get_obs_entry(Dhl)
    hl_both_avgs = get_means(hl_both_entry) if hl_k_both in Dhl else hl_obs_avgs

    fig4b = plt.figure(figsize=(12, 7))
    plt.hist(hl_obs_avgs, bins=50, alpha=0.5, density=True, label="obs")
    if hl_k_d in Dhl:
        plt.hist(hl_d_avgs, bins=50, alpha=0.6, density=True, label=f"do({hl_k_d})")
    if hl_k_c in Dhl:
        plt.hist(hl_c_avgs, bins=50, alpha=0.6, density=True, label=f"do({hl_k_c})")
    if hl_k_both in Dhl:
        plt.hist(hl_both_avgs, bins=50, alpha=0.6, density=True, label=f"do({hl_k_both})")
    plt.title("HL 16x16 abstraction: mean intensity under interventions")
    plt.xlabel("mean pixel intensity")
    plt.ylabel("density")
    plt.grid(alpha=0.4, ls="--")
    plt.legend()
    plt.tight_layout()

    # =======================================================
    # 6) Color sweep + hue JS (LL deterministic)
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
    # 7) Color sweep + hue JS (HL deterministic)
    # =======================================================
    with torch.no_grad():
        base_shape_hl = hl_shapes_16[idx_vis:idx_vis+1].to(dev)
        base_digit_hl = hl_digits[idx_vis:idx_vis+1].to(dev)
        U_fixed_hl = U_hl[idx_vis:idx_vis+1].to(dev)

        dets_hl, outs_hl = [], []
        for c in range(10):
            c_t = torch.tensor([c], device=dev)
            det_hl = hl_model(base_shape_hl, base_digit_hl, c_t)
            out_hl = det_hl + U_fixed_hl
            dets_hl.append(det_hl.cpu())
            outs_hl.append(out_hl.cpu())

    D_cf_hl = torch.stack(dets_hl, dim=0)
    O_cf_hl = torch.stack(outs_hl, dim=0)

    Hs_hl = [hue_hist(D_cf_hl[c][0], bins=bins, v_thresh=v_thresh) for c in range(10)]
    js_vals_hl = [js_divergence(Hs_hl[i], Hs_hl[j]) for i in range(10) for j in range(i + 1, 10)]
    js_hue_hl = float(np.mean(js_vals_hl)) if js_vals_hl else 0.0

    fig7, axs7 = plt.subplots(2, 10, figsize=(16, 3.5))
    for c in range(10):
        axs7[0, c].imshow(to01_any(D_cf_hl[c][0]).transpose(1, 2, 0)); axs7[0, c].axis("off")
        axs7[0, c].set_title(f"c={c}", fontsize=7)
        axs7[1, c].imshow(to01_any(O_cf_hl[c][0]).transpose(1, 2, 0)); axs7[1, c].axis("off")
    axs7[0, 0].set_ylabel("Det", fontsize=8)
    axs7[1, 0].set_ylabel("Det+U", fontsize=8)
    fig7.suptitle("Counterfactual color sweep (fixed digit & noise) - HL")
    fig7.tight_layout()

    # =======================================================
    # 8) Summary page
    # =======================================================
    fig6 = plt.figure(figsize=(8.5, 11))
    plt.axis("off")

    def stats(arr):
        return f"mean={np.mean(arr):.4f}, std={np.std(arr):.4f}"

    lines = []
    lines.append(f"Dataset: {os.path.basename(ds_dir)} (v2: LL 32x32, HL 16x16)")
    lines.append("")
    lines.append("1. LL residual independence:")
    lines.append(f"   R²(resid_pixel ~ [digit,color]) = {r2_resid:.4f}")
    lines.append("")
    lines.append("1b. HL residual independence:")
    lines.append(f"   R²(resid_pixel ~ [digit,color]) = {r2_resid_hl:.4f}")
    lines.append("")
    lines.append("2. LL reconstruction quality (32x32):")
    lines.append(f"   avg MSE(true,D)   = {mse_D_32:.6f},   avg corr(true,D)   = {corr_D_32:.4f}")
    lines.append(f"   avg MSE(true,D+U) = {mse_DU_32:.6f}, avg corr(true,D+U) = {corr_DU_32:.4f}")
    lines.append("")
    lines.append("3. HL reconstruction quality (16x16):")
    lines.append(f"   avg MSE(true,D)   = {mse_D_16:.6f},   avg corr(true,D)   = {corr_D_16:.4f}")
    lines.append(f"   avg MSE(true,D+U) = {mse_DU_16:.6f}, avg corr(true,D+U) = {corr_DU_16:.4f}")
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
    lines.append("5. HL mean-intensity stats (16x16):")
    lines.append(f"   {hl_k_obs:10s}: {stats(hl_obs_avgs)}")
    if hl_k_d in Dhl:
        lines.append(f"   {hl_k_d:10s}: {stats(hl_d_avgs)}")
    if hl_k_c in Dhl:
        lines.append(f"   {hl_k_c:10s}: {stats(hl_c_avgs)}")
    if hl_k_both in Dhl:
        lines.append(f"   {hl_k_both:10s}: {stats(hl_both_avgs)}")
    lines.append("")
    lines.append("6. Color controllability (LL deterministic hues):")
    lines.append(f"   JS-divergence across colors = {js_hue:.4f}")
    lines.append("")
    lines.append("7. Color controllability (HL deterministic hues):")
    lines.append(f"   JS-divergence across colors = {js_hue_hl:.4f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(" - LL FilmU-Net + U_ll_hat_32 is the nonlinear ANM at 32x32.")
    lines.append(" - HL FilmU-Net + U_hl_hat_16 is the same mechanism on 2x2 pooled 16x16.")
    lines.append(" - This checks whether the 16x16 model is a causal abstraction of the 32x32 model.")

    plt.text(0.05, 0.95, "\n".join(lines),
             va="top", ha="left", family="monospace", fontsize=10)

    # =======================================================
    # Save all pages
    # =======================================================
    out_dir = os.path.join(ds_dir, "reports_v2")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"{os.path.basename(ds_dir)}_report_v2.pdf")

    with PdfPages(out_pdf) as pdf:
        for fig in [fig1, fig1b, fig2, fig3, fig4a, fig4b, fig5, fig7, fig6]:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return out_pdf


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate CMNIST v2 reports (LL 32x32, HL 16x16 FiLM-U-Nets)."
    )
    ap.add_argument("--root", default=".", help="Root directory (default: current dir)")
    ap.add_argument("--pattern", default=None, help="Regex to filter dataset folders")
    ap.add_argument("--bins", type=int, default=36, help="Bins for hue hist")
    ap.add_argument("--v-thresh", type=float, default=0.1, help="Value threshold for hue hist")
    ap.add_argument("--num-resid-pixels", type=int, default=5,
                    help="# residual pixels for histograms")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device for U-Net forward passes")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        args.device = "cpu"

    ds_list = discover_datasets_v2(args.root, args.pattern)

    if not ds_list:
        print("No v2-style trained datasets found.")
        print("Expecting in each dataset folder:")
        print("  dll_samples.pkl, dhl_samples.pkl, ll_unet_32.pth, hl_unet_16.pth,")
        print("  U_ll_hat_32.pkl, U_hl_hat_16.pkl")
        return

    print(f"Found {len(ds_list)} dataset(s):")
    for d in ds_list:
        print(" -", d)

    for d in ds_list:
        print(f"\n>>> Building v2 report for: {d}")
        pdf_path = make_report_for_dataset_v2(
            d,
            bins=args.bins,
            v_thresh=args.v_thresh,
            num_resid_pixels=args.num_resid_pixels,
            device=args.device,
        )
        print(f"✓ Saved: {pdf_path}")


if __name__ == "__main__":
    main()
