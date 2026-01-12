#!/usr/bin/env python3
"""
Train ColorMNIST Models (Xia-encoder version)

- Low level (LL): same as before
    U-Net with FiLM, learning Pixels | (shape, Digit, Color)
    and abducing U_ll_hat as residuals.

- High level (HL): now fits a structural function
        f_hl : (Digit_, Color_) -> z  in R^{d_z}
  where z is the Xia representation.

  We model   z = m_{D,C} + U_hl
  with m_{D,C} estimated via cell-means + shrinkage
  towards an additive baseline (digit effect + color effect - global mean),
  extended to vector-valued outputs.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score
import joblib
import argparse

# ============================================================
# Reproducibility & scaling
# ============================================================

def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_tanh_scaled(images: torch.Tensor) -> torch.Tensor:
    """
    Convert [0,1] → [-1,1]; leave [-1,1] as-is.
    New generator uses [0,1], so this keeps LL code backward-compatible.
    """
    with torch.no_grad():
        imin, imax = images.min().item(), images.max().item()
    if -1.05 <= imin and imax <= 1.05:
        return images
    if -1e-6 <= imin and imax <= 1.0 + 1e-6:
        return images * 2.0 - 1.0
    raise ValueError(f"Unexpected image range [{imin:.3f}, {imax:.3f}].")


# ============================================================
# Model blocks (U-Net for LL) — unchanged
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="bn"):
        super().__init__()
        Norm = {
            "bn": nn.BatchNorm2d,
            "in": nn.InstanceNorm2d,
            None: lambda c: nn.Identity()
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
# Training utilities
# ============================================================

class EarlyStopper:
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def r2_imagewise(y, yhat):
    """
    Image-wise R^2 over flattened pixels.
    """
    N = y.size(0)
    yv, yh = y.view(N, -1), yhat.view(N, -1)
    ss_res = torch.sum((yv - yh) ** 2)
    ss_tot = torch.sum((yv - yv.mean()) ** 2)
    return (1 - ss_res / (ss_tot + 1e-12)).item(), None


# ============================================================
# LL training (U-Net) — unchanged
# ============================================================

def train_cnn_model_unet(
    model, ll_samples_tuple, epochs=5, batch_size=256, lr=1e-3,
    huber_beta=0.02, val_frac=0.1, weight_decay=1e-5, seed=42,
    use_amp=True, grad_clip=1.0
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Training on device: {device}")
    model.to(device)

    final_images, img_shapes, digits, colors = ll_samples_tuple
    targets = ensure_tanh_scaled(final_images)
    ds = TensorDataset(img_shapes, digits, colors, targets)

    n_val = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.SmoothL1Loss(beta=huber_beta)
    stopper = EarlyStopper(patience=30)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state, best_val, best_val_r2 = None, math.inf, -float("inf")

    for ep in tqdm(range(1, epochs + 1), desc="  - Training Epochs"):
        model.train()
        train_loss = 0.0
        for s, d, c, t in train_loader:
            s, d, c, t = s.to(device), d.to(device), c.to(device), t.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(s, d, c)
                loss = crit(pred, t)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
            train_loss += loss.item() * s.size(0)
        train_loss /= n_train

        # Validation
        model.eval()
        val_loss, tgts, preds = 0.0, [], []
        with torch.no_grad():
            for s, d, c, t in val_loader:
                s, d, c, t = s.to(device), d.to(device), c.to(device), t.to(device)
                pred = model(s, d, c)
                val_loss += crit(pred, t).item() * s.size(0)
                tgts.append(t.cpu()); preds.append(pred.cpu())
        val_loss /= n_val
        tgts, preds = torch.cat(tgts), torch.cat(preds)
        r2_val, _ = r2_imagewise(tgts, preds)
        best_val_r2 = max(best_val_r2, r2_val)
        sched.step()

        tqdm.write(
            f"Epoch {ep:04d} | train {train_loss:.6f} | val {val_loss:.6f} "
            f"| val_R2 {r2_val:.4f} | lr {sched.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {ep}. Best val_R2={best_val_r2:.4f}")
            break

    # Restore best and compute final train/val R²
    if best_state is not None:
        model.load_state_dict(best_state)

    def _r2(loader):
        model.eval(); tgts, preds = [], []
        with torch.no_grad():
            for s, d, c, t in loader:
                pred = model(s.to(device), d.to(device), c.to(device))
                tgts.append(t); preds.append(pred.cpu())
        tgts, preds = torch.cat(tgts), torch.cat(preds)
        return r2_imagewise(tgts, preds)[0]

    r2_train_final = _r2(train_loader)
    r2_val_final   = _r2(val_loader)
    model.to("cpu")
    return model, r2_train_final, r2_val_final


def abduce_ll_noise_unet(ll_obs_data, seed=42, norm_type="bn"):
    print("Abducing low-level noise (U_ll_hat) using: U-Net with FiLM...")
    model = ImageColorizerUNet(norm_type=norm_type)
    model, r2_train_final, r2_val_final = train_cnn_model_unet(model, ll_obs_data, seed=seed)
    print(f"[U-Net] R^2 — train: {r2_train_final:.4f} | val: {r2_val_final:.4f}")

    # Residuals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    imgs, shapes, digits, colors = ll_obs_data
    imgs_scaled = ensure_tanh_scaled(imgs)
    preds = []
    with torch.no_grad():
        for i in range(0, len(shapes), 1024):
            s = shapes[i:i+1024].to(device)
            d = digits[i:i+1024].to(device)
            c = colors[i:i+1024].to(device)
            preds.append(model(s, d, c).cpu())
    preds = torch.cat(preds)
    U_ll_hat = imgs_scaled - preds
    model.to("cpu")
    print("✓ Low-level abduction complete.")
    return model, U_ll_hat, r2_train_final, r2_val_final


# ============================================================
# HL abduction — vector-valued cell-means + shrinkage
# ============================================================

def _extract_hl_arrays(hl_data: torch.Tensor):
    """
    hl_data: (N, 20 + d_z)
      columns 0..9   : digit one-hot
      columns 10..19 : color one-hot
      columns 20..   : z (latent, dim d_z)

    Returns:
      Xd, Xc    : (N,10) numpy
      Z         : (N,d_z) numpy
      d_idx, c_idx : integer class indices (N,)
    """
    hl_data_np = hl_data.cpu().numpy()
    Xd = hl_data_np[:, :10]
    Xc = hl_data_np[:, 10:20]
    Z  = hl_data_np[:, 20:]
    d_idx = Xd.argmax(1)
    c_idx = Xc.argmax(1)
    return Xd, Xc, Z, d_idx, c_idx


def _additive_baseline_vec(Z, d_idx, c_idx, n_classes=10):
    """
    Vector-valued additive baseline:
      mu0      : global mean in R^{d_z}
      mu_d[i]  : mean of Z | D=i
      mu_c[j]  : mean of Z | C=j
    """
    mu0 = Z.mean(axis=0)  # (d_z,)

    d_z = Z.shape[1]
    mu_d = np.zeros((n_classes, d_z))
    mu_c = np.zeros((n_classes, d_z))

    for i in range(n_classes):
        mask = (d_idx == i)
        if np.any(mask):
            mu_d[i] = Z[mask].mean(axis=0)
        else:
            mu_d[i] = mu0

    for j in range(n_classes):
        mask = (c_idx == j)
        if np.any(mask):
            mu_c[j] = Z[mask].mean(axis=0)
        else:
            mu_c[j] = mu0

    return mu0, mu_d, mu_c


class HLCellMeansShrunkVec:
    """
    Vector-valued extension of the scalar HLCellMeansShrunk.

    For each (digit, color) cell we estimate:
      m_{d,c} in R^{d_z}

    with shrinkage toward an additive baseline:
      base_{d,c} = mu_d[d] + mu_c[c] - mu0

    so m_{d,c} = w * empirical_mean + (1-w) * base_{d,c}.

    This corresponds to the SCM:
      Z = m_{D,C} + U_hl
    with additive noise U_hl in the latent space.
    """
    def __init__(self, lam=10.0, n_classes=10):
        self.lam = lam
        self.n_classes = n_classes
        self.table = None  # (10,10,d_z)

    def fit(self, Xd, Xc, Z):
        # Xd/Xc are one-hot (N,10), Z is (N,d_z)
        d_idx = Xd.argmax(1)
        c_idx = Xc.argmax(1)
        n_classes = self.n_classes
        n_z = Z.shape[1]

        sums = np.zeros((n_classes, n_classes, n_z))
        cnts = np.zeros((n_classes, n_classes, 1))

        for di, ci, zi in zip(d_idx, c_idx, Z):
            sums[di, ci] += zi
            cnts[di, ci, 0] += 1.0

        # empirical means (with safe divide)
        means = np.divide(sums, np.maximum(cnts, 1.0))

        # additive baseline
        mu0, mu_d, mu_c = _additive_baseline_vec(Z, d_idx, c_idx, n_classes=n_classes)
        base = (mu_d[:, None, :] + mu_c[None, :, :] - mu0[None, None, :])  # (10,10,d_z)

        denom = cnts + self.lam
        w = np.where(cnts > 0, cnts / denom, 0.0)  # (10,10,1)

        self.table = w * means + (1.0 - w) * base  # (10,10,d_z)
        return self

    def predict(self, Xd, Xc):
        assert self.table is not None, "Model not fitted."
        d_idx = Xd.argmax(1)
        c_idx = Xc.argmax(1)
        return self.table[d_idx, c_idx, :]  # (N,d_z)


def abduce_hl_noise_cellmeans_vec(
    hl_data: torch.Tensor,
    val_frac: float = 0.2,
    seed: int = 42,
    lam_grid=(2, 5, 10, 20, 50)
):
    """
    Abduce HL noise when HL variable is z in R^{d_z}.

    Splits by (digit,color) cells so validation respects intervention structure.
    Uses multi-output R^2 (variance-weighted) as objective.
    """
    rng = np.random.default_rng(seed)
    Xd, Xc, Z, d_idx, c_idx = _extract_hl_arrays(hl_data)
    keys = d_idx * 10 + c_idx

    train_idx, val_idx = [], []
    for k in range(100):
        idx = np.where(keys == k)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_val = int(len(idx) * val_frac)
        val_idx.extend(list(idx[:n_val]))
        train_idx.extend(list(idx[n_val:]))

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)

    Xd_tr, Xc_tr, Z_tr = Xd[train_idx], Xc[train_idx], Z[train_idx]
    Xd_val, Xc_val, Z_val = Xd[val_idx], Xc[val_idx], Z[val_idx]

    best_lam, best_val_r2 = None, -np.inf
    for lam in lam_grid:
        m = HLCellMeansShrunkVec(lam=lam).fit(Xd_tr, Xc_tr, Z_tr)
        Z_val_pred = m.predict(Xd_val, Xc_val)
        r2 = r2_score(Z_val, Z_val_pred, multioutput="variance_weighted")
        if r2 > best_val_r2:
            best_val_r2 = r2
            best_lam = lam

    model = HLCellMeansShrunkVec(lam=best_lam).fit(Xd_tr, Xc_tr, Z_tr)
    Z_tr_pred = model.predict(Xd_tr, Xc_tr)
    r2_tr = r2_score(Z_tr, Z_tr_pred, multioutput="variance_weighted")

    print(
        f"✓ High-level abduction (vector cell-means + shrinkage). "
        f"λ={best_lam} | train R²={r2_tr:.4f} | val R²={best_val_r2:.4f}"
    )

    # Refit on full data for the final SCM parameters
    full_model = HLCellMeansShrunkVec(lam=best_lam).fit(Xd, Xc, Z)
    Z_full_pred = full_model.predict(Xd, Xc)
    U = torch.tensor((Z - Z_full_pred), dtype=torch.float32)  # (N,d_z)
    return full_model, U, r2_tr, best_val_r2, best_lam


# ============================================================
# Training driver
# ============================================================

def _get_obs_tuple(D):
    """Return observational tuple from dict of LL samples."""
    return D.get("obs") or D.get(None) or next(iter(D.values()))


def train_models(
    Dll,
    Dhl_z,
    seed: int = 42,
    norm_type: str = "bn",
    save_models: bool = True,
    output_dir: str = "data/cmnist_new",
):
    set_global_seed(seed)
    ll_obs = _get_obs_tuple(Dll)
    hl_obs = Dhl_z.get("obs", next(iter(Dhl_z.values())))

    print("\n" + "=" * 60)
    print("TRAINING COLOR MNIST MODELS (Xia-encoder version)")
    print("=" * 60)

    # ---------------- HL MODEL ----------------
    print("\n--- Training High-Level Model f_hl: (D,C) -> z ---")
    hl_model, U_hl_hat, r2_hl_tr, r2_hl_val, lam = abduce_hl_noise_cellmeans_vec(hl_obs)
    print(f"[HL] R² — train: {r2_hl_tr:.4f} | val: {r2_hl_val:.4f} | λ={lam:g}")

    # ---------------- LL MODEL ----------------
    print("\n--- Training Low-Level Model (U-Net) ---")
    ll_model_unet, U_ll_hat, r2_ll_tr, r2_ll_val = abduce_ll_noise_unet(
        ll_obs, seed=seed, norm_type=norm_type
    )
    print(f"[LL] R² — train: {r2_ll_tr:.4f} | val: {r2_ll_val:.4f}")

    # ---------------- Save models ----------------
    if save_models:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving models and residuals to {output_dir}...")
        torch.save(ll_model_unet.state_dict(), os.path.join(output_dir, "ll_model_unet.pth"))
        joblib.dump(hl_model, os.path.join(output_dir, "hl_model_cellmeans_z.joblib"))
        torch.save(U_ll_hat, os.path.join(output_dir, "U_ll_hat.pkl"))
        torch.save(U_hl_hat, os.path.join(output_dir, "U_hl_hat_z.pkl"))
        print("✓ Models and residuals saved successfully!")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"[HL] R² — train: {r2_hl_tr:.4f} | val: {r2_hl_val:.4f} | λ={lam:g}")
    print(f"[LL] R² — train: {r2_ll_tr:.4f} | val: {r2_ll_val:.4f}")
    print("✓ Both models trained and saved. Ready for abstraction experiments.\n")

    return ll_model_unet, hl_model, U_ll_hat, U_hl_hat, r2_hl_tr, r2_hl_val, r2_ll_tr, r2_ll_val


def main():
    p = argparse.ArgumentParser(
        description=(
            "Train ColorMNIST models on a Xia-aligned dataset folder.\n"
            "Folder must contain: dll_samples.pkl, dhl_samples_z.pkl, intervention_mapping.pkl"
        )
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default="data/cmnist_new",
        help="Folder containing dll_samples.pkl, dhl_samples_z.pkl, intervention_mapping.pkl",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Where to save models/residuals (default: input-dir)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--norm-type",
        type=str,
        default="bn",
        choices=["bn", "in", "none"],
    )
    args = p.parse_args()

    in_dir = args.input_dir
    out_dir = args.save_dir or in_dir

    print("ColorMNIST Model Training Pipeline (Xia-encoder version)")
    print("=" * 50)
    set_global_seed(args.seed)

    Dll_samples = torch.load(os.path.join(in_dir, "dll_samples.pkl"))
    Dhl_samples_z = torch.load(os.path.join(in_dir, "dhl_samples.pkl"))
    omega = torch.load(os.path.join(in_dir, "intervention_mapping.pkl"))
    _ = omega  # kept for compatibility; not used directly here

    (
        ll_model,
        hl_model,
        U_ll_hat,
        U_hl_hat,
        r2_hl_tr,
        r2_hl_val,
        r2_ll_tr,
        r2_ll_val,
    ) = train_models(
        Dll_samples,
        Dhl_samples_z,
        seed=args.seed,
        norm_type=args.norm_type,
        save_models=True,
        output_dir=out_dir,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"[HL] R² — train: {r2_hl_tr:.4f} | val: {r2_hl_val:.4f}")
    print(f"[LL] R² — train: {r2_ll_tr:.4f} | val: {r2_ll_val:.4f}")
    print("You can now use these models for causal abstraction experiments on z.")
    return ll_model, hl_model, U_ll_hat, U_hl_hat


if __name__ == "__main__":
    main()
