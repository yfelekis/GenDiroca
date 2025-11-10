#!/usr/bin/env python3
"""
Train ColorMNIST Models (v2): identical FiLM-U-Net for LL (32x32) and HL (16x16)

- Loads dll_samples.pkl and dhl_samples.pkl saved by cmnist_gen_v2.py
- Trains two FiLM-conditioned U-Nets with the same architecture:
    * LL: (shape_32 -> color_32)
    * HL: (shape_16 -> color_16)
- Computes residuals U_ll_hat and U_hl_hat
- Reports train/val R² for both levels and saves everything.
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import joblib


# ----------------------------
# Repro + small utilities
# ----------------------------

def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_tanh_scaled(images: torch.Tensor) -> torch.Tensor:
    """Convert [0,1] → [-1,1]; leave [-1,1] as-is."""
    with torch.no_grad():
        imin, imax = images.min().item(), images.max().item()
    if -1.05 <= imin and imax <= 1.05 and (imin < -0.5 or imax > 0.5):
        return images
    if -1e-6 <= imin and imax <= 1.0 + 1e-6:
        return images * 2.0 - 1.0
    raise ValueError(f"Unexpected image range [{imin:.3f}, {imax:.3f}].")


def r2_imagewise(y: torch.Tensor, yhat: torch.Tensor):
    N = y.size(0)
    yv, yh = y.view(N, -1), yhat.view(N, -1)
    ss_res = torch.sum((yv - yh) ** 2)
    ss_tot = torch.sum((yv - yv.mean()) ** 2)
    return (1 - ss_res / (ss_tot + 1e-12)).item()


# ----------------------------
# FiLM U-Net (shared LL/HL)
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="bn"):
        super().__init__()
        Norm = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d, "none": lambda _: nn.Identity()}.get(norm_type, nn.BatchNorm2d)
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


class FilmUNet(nn.Module):
    """Single architecture for both 32×32 and 16×16."""
    def __init__(self, norm_type="bn"):
        super().__init__()
        self.inc = ConvBlock(1, 32, norm_type)
        self.down1 = Down(32, 64, norm_type)
        self.down2 = Down(64, 128, norm_type)
        self.film_bottleneck = FiLM(feat_channels=128)
        self.up1 = Up(128, 64, 64, norm_type)
        self.film_up1 = FiLM(feat_channels=64)
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


# ----------------------------
# Training utils
# ----------------------------

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


def train_unet(
    model, inputs_1hw, digits, colors, targets_3hw,
    *, epochs=40, batch_size=256, lr=1e-3, huber_beta=0.02,
    val_frac=0.1, weight_decay=1e-5, seed=42, use_amp=True, grad_clip=1.0,
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    targets_scaled = ensure_tanh_scaled(targets_3hw)

    ds = TensorDataset(inputs_1hw, digits, colors, targets_scaled)
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

    for ep in tqdm(range(1, epochs + 1), desc="  - Training epochs"):
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

        # validation
        model.eval(); val_loss, tgts, preds = 0.0, [], []
        with torch.no_grad():
            for s, d, c, t in val_loader:
                s, d, c, t = s.to(device), d.to(device), c.to(device), t.to(device)
                pred = model(s, d, c)
                val_loss += crit(pred, t).item() * s.size(0)
                tgts.append(t.cpu()); preds.append(pred.cpu())
        val_loss /= n_val
        tgts, preds = torch.cat(tgts), torch.cat(preds)
        r2_val = r2_imagewise(tgts, preds)
        best_val_r2 = max(best_val_r2, r2_val)
        sched.step()

        tqdm.write(f"Epoch {ep:03d} | train {train_loss:.6f} | val {val_loss:.6f} | val_R2 {r2_val:.4f}")
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {ep}. Best val_R2={best_val_r2:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute residuals
    model.eval()
    preds_full = []
    with torch.no_grad():
        for i in range(0, len(inputs_1hw), 512):
            s = inputs_1hw[i:i+512].to(device)
            d = digits[i:i+512].to(device)
            c = colors[i:i+512].to(device)
            preds_full.append(model(s, d, c).cpu())
    preds_full = torch.cat(preds_full)
    targets_full = ensure_tanh_scaled(targets_3hw)
    U_hat = targets_full - preds_full

    return model.cpu(), U_hat, r2_val, best_val_r2


# ----------------------------
# Data unpacking helpers
# ----------------------------

def _get_obs_tuple_ll(Dll):
    """Returns (imgs_32, shapes_32, digits, colors) for observational split."""
    return Dll.get("obs", next(iter(Dll.values())))


def _normalize_hl_obs(hl_entry):
    """
    Standardize HL obs into (images, shapes, digits, colors) when already provided
    as a tuple or dict. If it's a tensor, the caller must supply shapes/digits/colors.
    """
    if isinstance(hl_entry, (tuple, list)):
        if len(hl_entry) < 4:
            raise ValueError(f"HL obs tuple has length {len(hl_entry)}; need ≥4.")
        return hl_entry[:4]
    if isinstance(hl_entry, dict):
        def _find(d, keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None
        images = _find(hl_entry, ["images", "final_images", "imgs"])
        shapes = _find(hl_entry, ["shapes", "img_shapes"])
        digits = _find(hl_entry, ["digits", "D", "labels"])
        colors = _find(hl_entry, ["colors", "C"])
        if any(x is None for x in [images, shapes, digits, colors]):
            raise KeyError("Missing one of images/shapes/digits/colors in HL obs.")
        return images, shapes, digits, colors
    # If it's a tensor, we'll handle it in the caller using LL info.
    if isinstance(hl_entry, torch.Tensor):
        return None
    raise TypeError(f"Unsupported HL obs type: {type(hl_entry)}")


# ----------------------------
# Training driver
# ----------------------------

def train_models_v2(Dll, Dhl, *, seed=42, norm_type="bn", save_models=True, output_dir="data/cmnist"):
    set_global_seed(seed)

    print("\n" + "=" * 60)
    print("TRAINING COLOR MNIST MODELS (v2: identical U-Net @ 32 and 16)")
    print("=" * 60)

    # LL observational tuple (N,3,32,32), (N,1,32,32), (N,), (N,)
    ll_obs = _get_obs_tuple_ll(Dll)
    ll_imgs_32, ll_shapes_32, ll_digits, ll_colors = ll_obs

    # HL observational entry may be:
    # - a tuple/list (imgs_16, shapes_16, digits, colors)
    # - a dict with those keys
    # - or (in our generator v2) simply a Tensor imgs_16 of shape (N,3,16,16)
    hl_obs_raw = Dhl.get("obs", next(iter(Dhl.values())))

    hl_tuple = _normalize_hl_obs(hl_obs_raw)
    if hl_tuple is None:
        # It's a Tensor of images only: build shapes_16/digits/colors from LL obs.
        hl_imgs_16 = hl_obs_raw  # (N,3,16,16)
        with torch.no_grad():
            # downsample LL shapes_32 -> shapes_16 by avg pooling
            hl_shapes_16 = F.avg_pool2d(ll_shapes_32, kernel_size=2, stride=2)
        hl_digits = ll_digits
        hl_colors = ll_colors
    else:
        hl_imgs_16, hl_shapes_16, hl_digits, hl_colors = hl_tuple

    # --- HL 16x16 ---
    print("\n--- Training High-Level U-Net (16x16) ---")
    hl_model = FilmUNet(norm_type=norm_type)
    hl_model, U_hl_hat, r2_hl_tr, r2_hl_val = train_unet(
        hl_model, hl_shapes_16, hl_digits.long(), hl_colors.long(), hl_imgs_16,
        epochs=40, batch_size=256, lr=1e-3,
    )
    print(f"[HL-16] R² — train: {r2_hl_tr:.4f} | val: {r2_hl_val:.4f}")

    # --- LL 32x32 ---
    print("\n--- Training Low-Level U-Net (32x32) ---")
    ll_model = FilmUNet(norm_type=norm_type)
    ll_model, U_ll_hat, r2_ll_tr, r2_ll_val = train_unet(
        ll_model, ll_shapes_32, ll_digits.long(), ll_colors.long(), ll_imgs_32,
        epochs=40, batch_size=256, lr=1e-3,
    )
    print(f"[LL-32] R² — train: {r2_ll_tr:.4f} | val: {r2_ll_val:.4f}")

    if save_models:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving models and residuals to {output_dir}...")
        torch.save(ll_model.state_dict(), os.path.join(output_dir, "ll_unet_32.pth"))
        torch.save(hl_model.state_dict(), os.path.join(output_dir, "hl_unet_16.pth"))
        torch.save(U_ll_hat, os.path.join(output_dir, "U_ll_hat_32.pkl"))
        torch.save(U_hl_hat, os.path.join(output_dir, "U_hl_hat_16.pkl"))
        meta = dict(
            r2_hl_train=r2_hl_tr, r2_hl_val=r2_hl_val,
            r2_ll_train=r2_ll_tr, r2_ll_val=r2_ll_val,
        )
        joblib.dump(meta, os.path.join(output_dir, "train_meta_v2.joblib"))
        print("✓ Saved.")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY (v2)")
    print("=" * 60)
    print(f"[HL-16] R² — train: {r2_hl_tr:.4f} | val: {r2_hl_val:.4f}")
    print(f"[LL-32] R² — train: {r2_ll_tr:.4f} | val: {r2_ll_val:.4f}")
    print("✓ Models ready for abstraction alignment.\n")

    return ll_model, hl_model, U_ll_hat, U_hl_hat


# ----------------------------
# CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Train ColorMNIST (v2) with identical FiLM-U-Net @ 32x32 & 16x16.")
    p.add_argument("--input-dir", type=str, default="data/cmnist")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--norm-type", type=str, default="bn", choices=["bn", "in", "none"])
    args = p.parse_args()

    in_dir = args.input_dir
    out_dir = args.save_dir or in_dir

    print("ColorMNIST Model Training Pipeline (v2)")
    print("=" * 50)
    print(f"Input dir: {in_dir}")
    print(f"Save  dir: {out_dir}")

    Dll_samples = torch.load(os.path.join(in_dir, "dll_samples.pkl"))
    Dhl_samples = torch.load(os.path.join(in_dir, "dhl_samples.pkl"))

    train_models_v2(Dll_samples, Dhl_samples, seed=args.seed, norm_type=args.norm_type,
                    save_models=True, output_dir=out_dir)


if __name__ == "__main__":
    main()
