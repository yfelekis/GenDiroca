#!/usr/bin/env python3
import os, glob, re, argparse, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import rgb_to_hsv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- Utils ----------------

def to01_any(x):
    x = x.detach().cpu().float()
    mn, mx = x.min().item(), x.max().item()

    # already [0,1] (allow tiny numeric slack) → return as-is
    if -1e-3 <= mn and mx <= 1.0 + 1e-3:
        return x.clamp(0, 1).numpy()

    # clearly [-1,1] → map to [0,1]
    if -1.0 - 1e-3 <= mn and mx <= 1.0 + 1e-3:
        return ((x + 1.0) / 2.0).clamp(0, 1).numpy()

    # fallback: min–max
    m, M = x.min(), x.max()
    return ((x - m) / (M - m + 1e-8)).clamp(0, 1).numpy()


def get_obs_tuple(D):
    return D.get("obs") or D.get(None) or next(iter(D.values()))

def hue_hist(img_chw, bins=36, v_thresh=0.1):
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
        return float(np.sum(a * np.log(a/b)))
    return 0.5*kl(p,m) + 0.5*kl(q,m)

def discover_datasets(root, pattern=None):
    # Accept both *_clean and *_noise* sets
    paths = sorted(glob.glob(os.path.join(root, "cmnist*")))
    if pattern:
        rx = re.compile(pattern)
        paths = [p for p in paths if rx.search(os.path.basename(p))]
    # keep only those that look trained (all required files exist)
    out = []
    for d in paths:
        need = ["dll_samples.pkl", "dhl_samples.pkl", "ll_model_unet.pth", "U_ll_hat.pkl"]
        if all(os.path.isfile(os.path.join(d, f)) for f in need):
            out.append(d)
    return out

# ------------- Core report per dataset -------------

def make_report_for_dataset(ds_dir, bins=36, v_thresh=0.1, num_resid_pixels=5, seed=0, device="cpu"):
    import importlib
    try:
        cmnist_train = importlib.import_module("cmnist_train")
        ImageColorizerUNet = getattr(cmnist_train, "ImageColorizerUNet")
    except Exception as e:
        raise RuntimeError("Could not import ImageColorizerUNet from cmnist_train.py") from e

    # Load artifacts
    Dll = torch.load(os.path.join(ds_dir, "dll_samples.pkl"), map_location="cpu")
    Dhl = torch.load(os.path.join(ds_dir, "dhl_samples.pkl"), map_location="cpu")
    U_ll_hat = torch.load(os.path.join(ds_dir, "U_ll_hat.pkl"), map_location="cpu")

    ll_model = ImageColorizerUNet(norm_type="bn")
    ll_model.load_state_dict(torch.load(os.path.join(ds_dir, "ll_model_unet.pth"), map_location="cpu"))
    ll_model.eval().to(device)

    ll_obs = get_obs_tuple(Dll)  # (images, shapes, digits, colors)
    images, shapes, digits, colors = ll_obs
    images = images.to("cpu")
    shapes = shapes.to(device)
    digits = digits.to(device)
    colors = colors.to(device)

    # 1) Residual diagnostics
    rng = np.random.default_rng(seed)
    U = U_ll_hat.view(U_ll_hat.size(0), -1).cpu().numpy()
    pix_idx = rng.choice(U.shape[1], size=min(num_resid_pixels, U.shape[1]), replace=False)
    resid_samples = U[:, pix_idx]

    # Fit simple linear model of residual vs labels (one pixel)
    X = np.stack([digits.cpu().numpy(), colors.cpu().numpy()], axis=1)
    y = resid_samples[:, 0]
    linreg = LinearRegression().fit(X, y)
    y_pred = linreg.predict(X)
    r2_resid = r2_score(y, y_pred)

    # Figure: histograms of sampled pixel residuals
    fig1, axs1 = plt.subplots(1, len(pix_idx), figsize=(3.2*len(pix_idx), 3))
    if len(pix_idx) == 1: axs1 = [axs1]
    for i, p in enumerate(pix_idx):
        axs1[i].hist(resid_samples[:, i], bins=40, alpha=0.8)
        axs1[i].set_title(f"Residual pixel {p}")
    fig1.suptitle("Residual distributions (sampled pixels)")
    fig1.tight_layout()

    # One qualitative triplet (true, pred, residual)
    idx_vis = rng.integers(0, len(U_ll_hat))
    resid_img = to01_any(U_ll_hat[idx_vis]).transpose(1,2,0)
    true_img  = to01_any(images[idx_vis]).transpose(1,2,0)
    pred_img  = np.clip(true_img - resid_img, 0.0, 1.0)
    fig2, axs2 = plt.subplots(1, 3, figsize=(9,3))
    axs2[0].imshow(true_img); axs2[0].set_title("True image"); axs2[0].axis("off")
    axs2[1].imshow(pred_img); axs2[1].set_title("Pred deterministic"); axs2[1].axis("off")
    axs2[2].imshow(resid_img); axs2[2].set_title("Residual (noise)"); axs2[2].axis("off")
    fig2.suptitle("Qualitative check")
    fig2.tight_layout()

    # 2) Counterfactual sweep (change color 0..9 for a fixed digit and fixed noise)
    with torch.no_grad():
        base_shape = shapes[idx_vis:idx_vis+1]
        base_digit = digits[idx_vis:idx_vis+1]
        U_fixed    = U_ll_hat[idx_vis:idx_vis+1].to(device)

        dets, outs = [], []
        for c in range(10):
            c_tensor = torch.tensor([c], device=device)
            det = ll_model(base_shape, base_digit, c_tensor)
            ycf = det + U_fixed
            dets.append(det.cpu())
            outs.append(ycf.cpu())
    D = torch.stack(dets, dim=0)  # (10,1,3,H,W)
    O = torch.stack(outs, dim=0)

    # Consistency scores
    D_gray = D.mean(2)  # (10,1,H,W)
    var_digit = D_gray.var(0).mean().item()
    mean_colors = O.view(O.size(0), O.size(1), -1).mean(-1)
    var_color = mean_colors.var(0).mean().item()
    U_cmp = O - D
    U_flat = U_cmp.view(U_cmp.size(0), -1).numpy()
    corr_mat = np.corrcoef(U_flat)
    iu = np.triu_indices_from(corr_mat, k=1)
    mean_corr = float(np.mean(corr_mat[iu]))

    # Figure: counterfactual grid (deterministic vs with noise)
    fig3, axs3 = plt.subplots(2, 10, figsize=(15, 3))
    for c in range(10):
        axs3[0, c].imshow(to01_any(D[c][0]).transpose(1,2,0)); axs3[0, c].axis("off")
        axs3[0, c].set_title(f"c={c}", fontsize=8)
        axs3[1, c].imshow(to01_any(O[c][0]).transpose(1,2,0)); axs3[1, c].axis("off")
    axs3[0,0].set_ylabel("deterministic", fontsize=9)
    axs3[1,0].set_ylabel("with noise", fontsize=9)
    fig3.suptitle("Counterfactual color sweep (fixed U)")
    fig3.tight_layout()

    # 3) HL four-way histogram page
    def hl_last_col(entry): return entry[:, -1].numpy()
    # choose keys robustly
    keys = list(Dhl.keys())
    k_obs = "obs" if "obs" in keys else keys[0]
    k_d   = "D=8" if "D=8" in keys else next((k for k in keys if k.startswith("D=") and "," not in k and k!=k_obs), keys[1])
    k_c   = "C=0" if "C=0" in keys else next((k for k in keys if k.startswith("C=") and "," not in k and k!=k_obs), keys[2])
    k_both = "D=8,C=0" if "D=8,C=0" in keys else next((k for k in keys if "," in k), keys[-1])

    obs_averages = hl_last_col(Dhl[k_obs])
    int_d_avgs   = hl_last_col(Dhl[k_d])
    int_c_avgs   = hl_last_col(Dhl[k_c])
    int_both     = hl_last_col(Dhl[k_both])

    fig4 = plt.figure(figsize=(12,7))
    plt.hist(obs_averages, bins=50, alpha=0.50, density=True, label='Observational', color='blue')
    plt.hist(int_d_avgs,  bins=50, alpha=0.70, density=True, label=f'do({k_d})', color='orange')
    plt.hist(int_c_avgs,  bins=50, alpha=0.70, density=True, label=f'do({k_c})', color='green')
    plt.hist(int_both,    bins=50, alpha=0.85, density=True, label=f'do({k_both})', color='purple')
    plt.title('HL "Image_" Distribution')
    plt.xlabel('Mean pixel intensity (Image_)'); plt.ylabel('Density')
    plt.grid(ls='--', alpha=0.5); plt.legend()
    plt.tight_layout()

    # 4) Hue diversity score (JS over hues from deterministic dets)
    Hs = [hue_hist(D[c][0], bins=bins, v_thresh=max(v_thresh, 0.1)) for c in range(10)]
    js_vals = [js_divergence(Hs[i], Hs[j]) for i in range(10) for j in range(i+1, 10)]
    js_hue = float(np.mean(js_vals)) if js_vals else 0.0

    # 5) Summary page (text)
    fig5 = plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    txt = []
    txt.append(f"Dataset: {os.path.basename(ds_dir)}")
    txt.append("")
    txt.append("Residual independence (R² of residual ~ [digit,color]):")
    txt.append(f"  R² = {r2_resid:.4f}")
    txt.append("")
    txt.append("Counterfactual consistency scores:")
    txt.append(f"  digit_variance (shape invariance) : {var_digit:.6f}")
    txt.append(f"  color_variance (color sensitivity) : {var_color:.6f}")
    txt.append(f"  noise_corr (stability across colors): {mean_corr:.6f}")
    txt.append("")
    txt.append(f"Color hue diversity (JS over hues, deterministic dets): {js_hue:.4f}")
    txt.append("")
    txt.append("HL 'Image_' stats:")
    def stats(x): return f"mean={np.mean(x):.4f}, std={np.std(x):.4f}"
    txt.append(f"  {k_obs:10s} : {stats(obs_averages)}")
    txt.append(f"  {k_d:10s} : {stats(int_d_avgs)}")
    txt.append(f"  {k_c:10s} : {stats(int_c_avgs)}")
    txt.append(f"  {k_both:10s} : {stats(int_both)}")
    plt.text(0.05, 0.95, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=11)

    # 6) Save all figures to a single PDF
    out_dir = os.path.join(ds_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"{os.path.basename(ds_dir)}_report.pdf")
    with PdfPages(out_pdf) as pdf:
        for fig in [fig1, fig2, fig3, fig4, fig5]:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return out_pdf

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Generate CMNIST PDF reports for trained datasets.")
    ap.add_argument("--root", default="data", help="Root data directory (default: data)")
    ap.add_argument("--pattern", default=None, help="Regex to filter datasets by folder name")
    ap.add_argument("--bins", type=int, default=36, help="Bins for hue histograms")
    ap.add_argument("--v-thresh", type=float, default=0.1, help="Value threshold for hue histograms")
    ap.add_argument("--num-resid-pixels", type=int, default=5, help="How many residual pixels to histogram")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="Device for U-Net forward")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"

    ds_list = discover_datasets(args.root, args.pattern)
    if not ds_list:
        print("No trained datasets found. (Need dll_samples.pkl, dhl_samples.pkl, ll_model_unet.pth, U_ll_hat.pkl)")
        return

    print(f"Found {len(ds_list)} dataset(s):")
    for d in ds_list:
        print(" -", d)

    for d in ds_list:
        try:
            print(f"\n>>> Building report for: {d}")
            pdf_path = make_report_for_dataset(
                d, bins=args.bins, v_thresh=args.v_thresh,
                num_resid_pixels=args.num_resid_pixels, device=args.device
            )
            print(f"✓ Saved: {pdf_path}")
        except Exception as e:
            print(f"⚠️  Failed on {d}: {e}")

if __name__ == "__main__":
    main()
