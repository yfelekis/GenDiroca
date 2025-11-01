def abduce_hl_noise(hl_observational_data):
    """
    RidgeCV on one-hots (digits/colors) -> image_feature; returns (model, U_hl_hat, r2_train)
    """
    X_digits = hl_observational_data[:, :10].numpy()
    X_colors = hl_observational_data[:, 10:20].numpy()
    y = hl_observational_data[:, -1].numpy()  # [0,1] mean intensity
    X = np.concatenate([X_digits, X_colors], axis=1)

    model = RidgeCV(alphas=np.logspace(-4, 4, 25))
    model.fit(X, y)
    y_hat = model.predict(X)
    r2_tr = r2_score(y, y_hat)
    U_hl_hat = torch.tensor((y - y_hat), dtype=torch.float32).unsqueeze(1)

    print(f"✓ High-level abduction complete. HL R^2 (train): {r2_tr:.4f}")
    return model, U_hl_hat, r2_tr

#!/usr/bin/env python3
"""
Train ColorMNIST Models (UNet + FiLM-all-scales + Huber+SSIM)

What's new vs your last run:
- FiLM conditioning at ALL scales: inc, down1, down2, bottleneck, up1, up2.
- Mixed loss: total = huber_weight * SmoothL1 + ssim_weight * (1 - SSIM).
  SSIM is computed on [0,1] range from model tanh outputs; 3×3 Gaussian window.
- EMA evaluation, warmup+cosine LR, stratified split, HL Ridge baseline.

Expected effect:
- FiLM-all-scales improves conditioning signal flow.
- SSIM term tends to sharpen structures/colors and nudges R² upward.

Note:
- SSIM here is implementation-light and self-contained (no external deps).
- If you later want VGG 'perceptual' features, we can add that behind a flag.
"""

import os
import math
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import joblib


# ----------------------------
# Reproducibility helpers
# ----------------------------

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Data scaling helper
# ----------------------------

def ensure_tanh_scaled(images: torch.Tensor) -> torch.Tensor:
    """
    Ensure targets live in [-1, 1]. If they appear to be in [0, 1], rescale.
    Assumes float tensor (N, C, H, W) or similar.
    """
    with torch.no_grad():
        imin = images.min().item()
        imax = images.max().item()
    if imin >= -1.05 and imax <= 1.05:
        if imin >= -0.01 and imax <= 1.01:
            return images * 2.0 - 1.0
        return images
    eps = 1e-8
    images = (images - images.min()) / (images.max() - images.min() + eps)
    return images * 2.0 - 1.0


# ----------------------------
# SSIM (self-contained)
# ----------------------------

def _gaussian_kernel_2d(kernel_size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def _ssim_per_channel(x, y, kernel, C1, C2):
    # x, y: (N,1,H,W); kernel: (1,1,ks,ks)
    mu_x = F.conv2d(x, kernel, padding=kernel.shape[-1]//2, groups=1)
    mu_y = F.conv2d(y, kernel, padding=kernel.shape[-1]//2, groups=1)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=kernel.shape[-1]//2, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=kernel.shape[-1]//2, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=kernel.shape[-1]//2, groups=1) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim = num / (den + 1e-12)
    return ssim

def ssim_torch(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    SSIM for RGB images, averaged over channels and spatial dims.
    x, y expected in [0,1], shape (N, 3, H, W).
    Returns mean SSIM over batch.
    """
    assert x.shape == y.shape
    device, dtype = x.device, x.dtype
    C = x.size(1)
    kernel = _gaussian_kernel_2d(kernel_size, sigma, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

    # Constants per SSIM paper (L=1 for [0,1])
    L = 1.0
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    ssim_vals = []
    for c in range(C):
        ssim_map = _ssim_per_channel(x[:, c:c+1], y[:, c:c+1], kernel, C1, C2)
        ssim_vals.append(ssim_map.mean(dim=(-1, -2)))  # (N,)
    ssim_stack = torch.stack(ssim_vals, dim=1)  # (N, C)
    return ssim_stack.mean()  # scalar


# ----------------------------
# Core building blocks
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="bn"):
        super().__init__()
        norm = {
            "bn": nn.BatchNorm2d,
            "in": nn.InstanceNorm2d,
            None: lambda c: nn.Identity(),
        }.get(norm_type, nn.BatchNorm2d)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_ch),
            nn.ReLU(inplace=True),
        )
        self._init()

    def _init(self):
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
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.reduce.weight, nonlinearity="relu")
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, norm_type)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        # pad if needed for odd sizes
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""
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
                if m.bias is not None:
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


# ----------------------------
# U-Net with FiLM at ALL scales
# ----------------------------

class ImageColorizerUNet(nn.Module):
    """
    Light U-Net for 32×32 inputs with FiLM conditioning at every stage.
    Input: grayscale (N,1,32,32), digit (N,), color (N,) -> output (N,3,32,32) in [-1,1] via tanh
    """
    def __init__(self, norm_type="bn"):
        super().__init__()
        # encoder
        self.inc = ConvBlock(1, 32, norm_type)
        self.film_inc = FiLM(feat_channels=32)

        self.down1 = Down(32, 64, norm_type)
        self.film_down1 = FiLM(feat_channels=64)

        self.down2 = Down(64, 128, norm_type)
        self.film_down2 = FiLM(feat_channels=128)

        # bottleneck film
        self.film_bottleneck = FiLM(feat_channels=128)

        # decoder
        self.up1 = Up(128, 64, 64, norm_type)
        self.film_up1 = FiLM(feat_channels=64)

        self.up2 = Up(64, 32, 32, norm_type)
        self.film_up2 = FiLM(feat_channels=32)

        self.outc = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.kaiming_normal_(self.outc.weight, nonlinearity="linear")
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)

    def forward(self, x, digit, color):
        x1 = self.inc(x)
        x1 = self.film_inc(x1, digit, color)

        x2 = self.down1(x1)
        x2 = self.film_down1(x2, digit, color)

        x3 = self.down2(x2)
        x3 = self.film_down2(x3, digit, color)

        x3 = self.film_bottleneck(x3, digit, color)

        y = self.up1(x3, x2)
        y = self.film_up1(y, digit, color)

        y = self.up2(y, x1)
        y = self.film_up2(y, digit, color)

        out = torch.tanh(self.outc(y))
        return out


# ----------------------------
# EMA helper
# ----------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995, device=None):
        self.decay = decay
        self.shadow = {}
        self.device = device
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().float()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            new = p.detach().float()
            self.shadow[name].mul_(self.decay).add_(new, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        # swap in shadow params (store originals)
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.dtype).to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in getattr(self, "backup", {}):
                p.data.copy_(self.backup[name])
        self.backup = {}


# ----------------------------
# Scheduler: warmup + cosine
# ----------------------------

def make_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs=10, min_lr_mult=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # cosine from 1.0 -> min_lr_mult
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_mult + (1.0 - min_lr_mult) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------
# R^2 helpers
# ----------------------------

def r2_imagewise(final_images, predictions):
    """Compute global and per-image R^2 on the same scale for both tensors."""
    N = final_images.size(0)
    y = final_images.view(N, -1)
    yhat = predictions.view(N, -1)
    ss_res = torch.sum((y - yhat) ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2)
    r2_global = 1 - (ss_res / (ss_tot + 1e-12))
    y_mean = y.mean(dim=1, keepdim=True)
    ss_res_i = torch.sum((y - yhat) ** 2, dim=1)
    ss_tot_i = torch.sum((y - y_mean) ** 2, dim=1) + 1e-12
    r2_per_image = 1 - ss_res_i / ss_tot_i
    return r2_global.item(), r2_per_image


# ----------------------------
# Training (Huber + SSIM + EMA)
# ----------------------------

class EarlyStopper:
    def __init__(self, patience=50, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_cnn_model_unet(
    model: nn.Module,
    ll_samples_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    epochs=300,
    batch_size=256,
    lr=3e-3,
    weight_decay=1e-5,
    huber_beta=0.02,
    huber_weight=1.0,
    ssim_weight=0.15,
    warmup_epochs=10,
    min_lr_mult=0.01,
    seed=42,
    use_amp=False,
    ema_decay=0.9995,
    stratify_by_color=True,
):
    """
    Train UNet with FiLM-all-scales using mixed Huber + SSIM loss and EMA eval.
    """
    set_global_seed(seed)

    # device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"  - Training on device: {device}")

    # Unpack tuple
    final_images, img_shapes, digits, colors = ll_samples_tuple
    final_images_scaled = ensure_tanh_scaled(final_images)

    # Dataset & split (stratified by color optionally)
    N = img_shapes.size(0)
    indices = torch.arange(N)
    if stratify_by_color:
        # 90/10 split within each color id
        val_mask = torch.zeros(N, dtype=torch.bool)
        for c in torch.unique(colors):
            c_idx = indices[(colors == c)]
            # deterministic shuffle
            g = torch.Generator().manual_seed(seed + int(c.item()))
            perm = c_idx[torch.randperm(len(c_idx), generator=g)]
            k_val = max(1, int(0.1 * len(perm)))
            val_mask[perm[:k_val]] = True
        train_idx = indices[~val_mask]
        val_idx = indices[val_mask]
    else:
        g = torch.Generator().manual_seed(seed)
        perm = indices[torch.randperm(N, generator=g)]
        k_val = max(1, int(0.1 * N))
        val_idx = perm[:k_val]
        train_idx = perm[k_val:]

    train_ds = TensorDataset(img_shapes[train_idx],
                             digits[train_idx],
                             colors[train_idx],
                             final_images_scaled[train_idx])
    val_ds = TensorDataset(img_shapes[val_idx],
                           digits[val_idx],
                           colors[val_idx],
                           final_images_scaled[val_idx])

    pin_mem = (device.type == "cuda")
    num_workers = 2 if pin_mem else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem)

    model.to(device)
    criterion_huber = nn.SmoothL1Loss(beta=huber_beta)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs=warmup_epochs, min_lr_mult=min_lr_mult)
    stopper = EarlyStopper(patience=50, min_delta=1e-5)

    # EMA
    ema = EMA(model, decay=ema_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    def mixed_loss(pred, target):
        # Huber term on [-1,1]
        l_h = criterion_huber(pred, target)
        # SSIM term expects [0,1]
        p01 = (pred + 1.0) * 0.5
        t01 = (target + 1.0) * 0.5
        p01 = p01.clamp(0, 1)
        t01 = t01.clamp(0, 1)
        l_ssim = 1.0 - ssim_torch(p01, t01, kernel_size=7, sigma=1.0)
        return huber_weight * l_h + ssim_weight * l_ssim, l_h.item(), l_ssim.item()

    best_state = None
    best_val_ema = math.inf

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        train_loss = 0.0
        train_huber = 0.0
        train_ssim = 0.0

        for shape_b, digit_b, color_b, target_b in train_loader:
            shape_b = shape_b.to(device)
            digit_b = digit_b.to(device)
            color_b = color_b.to(device)
            target_b = target_b.to(device)

            optimizer.zero_grad()
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    pred = model(shape_b, digit_b, color_b)
                    loss, l_h, l_s = mixed_loss(pred, target_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(shape_b, digit_b, color_b)
                loss, l_h, l_s = mixed_loss(pred, target_b)
                loss.backward()
                optimizer.step()

            ema.update(model)

            train_loss += loss.item() * shape_b.size(0)
            train_huber += l_h * shape_b.size(0)
            train_ssim += l_s * shape_b.size(0)

        train_loss /= len(train_ds)
        train_huber /= len(train_ds)
        train_ssim /= len(train_ds)

        # Validation – online
        model.eval()
        val_loss_online = 0.0
        with torch.no_grad():
            for shape_b, digit_b, color_b, target_b in val_loader:
                shape_b = shape_b.to(device)
                digit_b = digit_b.to(device)
                color_b = color_b.to(device)
                target_b = target_b.to(device)
                pred = model(shape_b, digit_b, color_b)
                loss, _, _ = mixed_loss(pred, target_b)
                val_loss_online += loss.item() * shape_b.size(0)
        val_loss_online /= len(val_ds)

        # Validation – EMA
        ema.apply_to(model)
        val_loss_ema = 0.0
        with torch.no_grad():
            for shape_b, digit_b, color_b, target_b in val_loader:
                shape_b = shape_b.to(device)
                digit_b = digit_b.to(device)
                color_b = color_b.to(device)
                target_b = target_b.to(device)
                pred = model(shape_b, digit_b, color_b)
                loss, _, _ = mixed_loss(pred, target_b)
                val_loss_ema += loss.item() * shape_b.size(0)
        val_loss_ema /= len(val_ds)

        # R² (EMA) quick peek
        # (Compute on a single small batch to keep it light during training)
        try:
            (shape_b, digit_b, color_b, target_b) = next(iter(val_loader))
            shape_b = shape_b.to(device); digit_b = digit_b.to(device); color_b = color_b.to(device); target_b = target_b.to(device)
            with torch.no_grad():
                pred_ema = model(shape_b, digit_b, color_b)
            ema_r2, _ = r2_imagewise(target_b.detach().cpu(), pred_ema.detach().cpu())
        except StopIteration:
            ema_r2 = float('nan')
        ema.restore(model)

        scheduler.step()

        tqdm.write(
            f"Epoch {epoch:04d} | "
            f"train {train_loss:.6f} (huber {train_huber:.4f}, ssim {train_ssim:.4f}) | "
            f"val_online {val_loss_online:.6f} | val_ema {val_loss_ema:.6f} | "
            f"R2(val_ema) {ema_r2:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping on EMA val
        if val_loss_ema < best_val_ema:
            best_val_ema = val_loss_ema
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(val_loss_ema):
            print(f"Early stopping at epoch {epoch} (best EMA val {best_val_ema:.6f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final: move to CPU
    model.to("cpu")
    return model, val_idx  # return indices so we can recompute proper val R² later


# ----------------------------
# High-level abduction (Ridge on one-hots)
# ----------------------------

def abduce_hl_noise_better(hl_observational_data):
    """
    HL abduction on one-hot digit+color -> scalar feature, RidgeCV.
    Returns model, residuals tensor, and R² (train).
    """
    X_digits = hl_observational_data[:, :10].numpy()
    X_colors = hl_observational_data[:, 10:20].numpy()
    y = hl_observational_data[:, -1].numpy()
    X = np.concatenate([X_digits, X_colors], axis=1)

    model = RidgeCV(alphas=np.logspace(-4, 4, 25))
    model.fit(X, y)
    y_hat = model.predict(X)
    r2_tr = r2_score(y, y_hat)
    U_hl_hat = torch.tensor((y - y_hat), dtype=torch.float32).unsqueeze(1)

    print(f"✓ High-level abduction complete. HL R^2 (train): {r2_tr:.4f}")
    return model, U_hl_hat, r2_tr


# ----------------------------
# IO
# ----------------------------

def load_cmnist_data(data_dir='data/cmnist'):
    print(f"Loading ColorMNIST data from {data_dir}...")
    Dll_samples = torch.load(os.path.join(data_dir, 'dll_samples.pkl'))
    Dhl_samples = torch.load(os.path.join(data_dir, 'dhl_samples.pkl'))
    omega = torch.load(os.path.join(data_dir, 'intervention_mapping.pkl'))
    print("Data loaded successfully!")
    print(f"  - Low-level interventions: {len(Dll_samples)}")
    print(f"  - High-level interventions: {len(Dhl_samples)}")
    print(f"  - Intervention mappings: {len(omega)}")
    return Dll_samples, Dhl_samples, omega


# ----------------------------
# Driver
# ----------------------------

def train_models(Dll_samples, Dhl_samples, save_models=True, output_dir='data/cmnist', seed=42):
    set_global_seed(seed)

    # Observational data
    ll_obs_data_tuple = Dll_samples[None]
    hl_obs_data = Dhl_samples[None]

    print("\n" + "="*60)
    print("TRAINING COLORMNIST MODELS")
    print("="*60)

    # HL
    print("\n--- Training High-Level Model ---")
    hl_model, U_hl_hat, r2_hl = abduce_hl_noise_better(hl_obs_data)

    # LL
    print("\n--- Training Low-Level Model ---")
    print("Abducing low-level noise (U_ll_hat) using: ImageColorizerUNet (EMA + Huber+SSIM)...")
    model = ImageColorizerUNet()
    model, val_idx = train_cnn_model_unet(
        model,
        ll_obs_data_tuple,
        epochs=300,
        batch_size=256,
        lr=3e-3,
        weight_decay=1e-5,
        huber_beta=0.02,
        huber_weight=1.0,
        ssim_weight=0.15,       # <-- tweak 0.1–0.3 if needed
        warmup_epochs=10,
        min_lr_mult=0.01,       # lower floor helps late training
        seed=seed,
        use_amp=False,          # MPS doesn't support CUDA AMP; safe False
        ema_decay=0.9995,
        stratify_by_color=True
    )

    # Predict deterministic outputs on all obs with the (best) model
    print("\n- Predicting deterministic outputs to calculate residuals (noise)...")
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    final_images, img_shapes, digits, colors = ll_obs_data_tuple
    final_images_scaled = ensure_tanh_scaled(final_images)

    with torch.no_grad():
        preds = []
        B = 1024
        for i in range(0, len(img_shapes), B):
            p = model(img_shapes[i:i+B].to(device),
                      digits[i:i+B].to(device),
                      colors[i:i+B].to(device)).cpu()
            preds.append(p)
        predictions = torch.cat(preds, dim=0)

    # Residuals
    U_ll_hat = final_images_scaled - predictions
    print("\n✓ Low-level abduction complete with FiLM-all-scales + Huber+SSIM.")

    # R² (full set)
    print("\n--- Final LL R² (full observational set) ---")
    r2_full, r2_per_img_full = r2_imagewise(final_images_scaled, predictions)
    print(f"  - LL R² (full set): {r2_full:.4f} | Per-image mean {r2_per_img_full.mean().item():.4f}, median {r2_per_img_full.median().item():.4f}")

    # R² (validation-only)
    print("\n--- Validation-only LL R² ---")
    val_idx = torch.as_tensor(val_idx, dtype=torch.long)
    r2_val, r2_per_img_val = r2_imagewise(final_images_scaled[val_idx], predictions[val_idx])
    print(f"  - LL R² (val-only): {r2_val:.6f} | Per-image mean {r2_per_img_val.mean().item():.6f}, median {r2_per_img_val.median().item():.6f}")

    # Save
    if save_models:
        os.makedirs(output_dir, exist_ok=True)
        print("\nSaving models to", output_dir, "...")
        torch.save(model.state_dict(), os.path.join(output_dir, 'll_model_unet_filmall_ssim.pth'))
        joblib.dump(hl_model, os.path.join(output_dir, 'hl_model_ridge.joblib'))
        torch.save(U_ll_hat, os.path.join(output_dir, 'U_ll_hat.pkl'))
        torch.save(U_hl_hat, os.path.join(output_dir, 'U_hl_hat.pkl'))
        # also persist the last split for reproducibility
        torch.save({"val_indices": val_idx.cpu()}, os.path.join(output_dir, 'll_val_split.pt'))
        print("Models & split saved successfully!")

    return model, hl_model, U_ll_hat, U_hl_hat, r2_hl, r2_val, r2_full


def main():
    print("ColorMNIST Model Training Pipeline (FiLM-all-scales + Huber+SSIM + EMA)")
    print("==================================================")
    set_global_seed(42)

    Dll_samples, Dhl_samples, omega = load_cmnist_data()
    ll_model, hl_model, U_ll_hat, U_hl_hat, r2_hl, r2_val, r2_full = train_models(Dll_samples, Dhl_samples)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"HL train R^2: {r2_hl:.4f}")
    print(f"LL R^2 (val-only): {r2_val:.4f}")
    print(f"LL R^2 (full set): {r2_full:.4f}")
    print("Models trained and saved successfully.")
    # === Visualize abduction results ===
    try:
        save_du_triplets(
            final_images=final_images,        # the observational dataset tensor
            residuals=U_ll_hat_unet,          # residuals computed by your U-Net
            out_dir="outputs/cmnist_du",
            num_samples=36,
            seed=42,
        )
    except Exception as e:
        print("[DU VIS] Visualization skipped:", e)

    return ll_model, hl_model, U_ll_hat, U_hl_hat


if __name__ == "__main__":
    ll_model, hl_model, U_ll_hat, U_hl_hat = main()
