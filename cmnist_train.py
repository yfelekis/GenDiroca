#!/usr/bin/env python3
"""
Train ColorMNIST Models (paired with robust generator)

Highlights:
- HL abduction: cell-means + count-aware shrinkage toward additive baseline.
- LL abduction: U-Net with FiLM conditioning, now reporting train/val R².
- Deterministic splits, consistent return signatures.
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

# ----------------------------
# Reproducibility & scaling
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
    if -1.05 <= imin and imax <= 1.05:
        return images
    if -1e-6 <= imin and imax <= 1.0 + 1e-6:
        return images * 2.0 - 1.0
    raise ValueError(f"Unexpected image range [{imin:.3f}, {imax:.3f}].")


# ----------------------------
# Model blocks (U-Net for LL)
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="bn"):
        super().__init__()
        Norm = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d, None: lambda c: nn.Identity()}.get(norm_type, nn.BatchNorm2d)
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


# ----------------------------
# Training utilities
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


def r2_imagewise(y, yhat):
    N = y.size(0)
    yv, yh = y.view(N, -1), yhat.view(N, -1)
    ss_res = torch.sum((yv - yh) ** 2)
    ss_tot = torch.sum((yv - yv.mean()) ** 2)
    return (1 - ss_res / (ss_tot + 1e-12)).item(), None


# ----------------------------
# LL training (U-Net)
# ----------------------------

def train_cnn_model_unet(
    model, ll_samples_tuple, epochs=40, batch_size=256, lr=1e-3,
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
        model.eval(); val_loss, tgts, preds = 0.0, [], []
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

        tqdm.write(f"Epoch {ep:04d} | train {train_loss:.6f} | val {val_loss:.6f} "
                   f"| val_R2 {r2_val:.4f} | lr {sched.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {ep}. Best val_R2={best_val_r2:.4f}")
            break

    # Restore best and compute final train/val R²
    if best_state is not None: model.load_state_dict(best_state)
    def _r2(loader):
        model.eval(); tgts, preds = [], []
        with torch.no_grad():
            for s, d, c, t in loader:
                pred = model(s.to(device), d.to(device), c.to(device))
                tgts.append(t); preds.append(pred.cpu())
        tgts, preds = torch.cat(tgts), torch.cat(preds)
        return r2_imagewise(tgts, preds)[0]
    r2_train_final = _r2(train_loader)
    r2_val_final = _r2(val_loader)
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
            s, d, c = shapes[i:i+1024].to(device), digits[i:i+1024].to(device), colors[i:i+1024].to(device)
            preds.append(model(s, d, c).cpu())
    preds = torch.cat(preds)
    U_ll_hat = imgs_scaled - preds
    model.to("cpu")
    print("✓ Low-level abduction complete.")
    return model, U_ll_hat, r2_train_final, r2_val_final


# ----------------------------
# HL abduction — cell-means + shrinkage
# ----------------------------

def _extract_hl_arrays(hl_data):
    Xd, Xc = hl_data[:, :10].numpy(), hl_data[:, 10:20].numpy()
    y = hl_data[:, -1].numpy()
    return Xd, Xc, y, Xd.argmax(1), Xc.argmax(1)


def _additive_baseline(y, d, c):
    mu0 = y.mean()
    mu_d = np.array([y[d == i].mean() if np.any(d == i) else mu0 for i in range(10)])
    mu_c = np.array([y[c == j].mean() if np.any(c == j) else mu0 for j in range(10)])
    return mu0, mu_d, mu_c


class HLCellMeansShrunk:
    def __init__(self, lam=10.0): self.lam = lam
    def fit(self, Xd, Xc, y):
        d, c = Xd.argmax(1), Xc.argmax(1)
        sums, cnts = np.zeros((10,10)), np.zeros((10,10))
        for di, ci, yi in zip(d, c, y): sums[di,ci]+=yi; cnts[di,ci]+=1
        means = np.divide(sums, np.maximum(cnts,1))
        mu0, mu_d, mu_c = _additive_baseline(y, d, c)
        base = mu_d[:,None]+mu_c[None,:]-mu0
        denom = cnts + self.lam
        w = np.where(cnts>0, cnts/denom, 0.0)
        self.table = w*means+(1-w)*base
        return self
    def predict(self, Xd, Xc):
        d, c = Xd.argmax(1), Xc.argmax(1)
        return self.table[d,c]


def abduce_hl_noise_cellmeans(hl_data, val_frac=0.2, seed=42, lam_grid=(2,5,10,20,50)):
    rng = np.random.default_rng(seed)
    Xd,Xc,y,d,c=_extract_hl_arrays(hl_data)
    keys=d*10+c
    train,val=[],[]
    for k in range(100):
        idx=np.where(keys==k)[0]; 
        if len(idx)==0: continue
        rng.shuffle(idx)
        n_val=int(len(idx)*val_frac)
        val+=list(idx[:n_val]); train+=list(idx[n_val:])
    Xd_tr,Xc_tr,y_tr=Xd[train],Xc[train],y[train]
    Xd_val,Xc_val,y_val=Xd[val],Xc[val],y[val]

    best_lam,best_val=-1,-np.inf
    for lam in lam_grid:
        m=HLCellMeansShrunk(lam).fit(Xd_tr,Xc_tr,y_tr)
        r2=r2_score(y_val,m.predict(Xd_val,Xc_val))
        if r2>best_val: best_val,r2_val= r2,r2; best_lam=lam
    model=HLCellMeansShrunk(best_lam).fit(Xd_tr,Xc_tr,y_tr)
    r2_tr=r2_score(y_tr,model.predict(Xd_tr,Xc_tr))
    print(f"✓ High-level abduction (cell-means + shrinkage). λ={best_lam} | "
          f"train R²={r2_tr:.4f} | val R²={r2_val:.4f}")
    full=HLCellMeansShrunk(best_lam).fit(Xd,Xc,y)
    U=torch.tensor((y-full.predict(Xd,Xc)),dtype=torch.float32).unsqueeze(1)
    return full,U,r2_tr,r2_val,best_lam


# ----------------------------
# Training driver
# ----------------------------

def _get_obs_tuple(D):
    """Return observational tuple from dict of LL samples."""
    return D.get("obs") or D.get(None) or next(iter(D.values()))


def train_models(Dll, Dhl, seed=42, norm_type="bn", save_models=True, output_dir="data/cmnist"):
    set_global_seed(seed)
    ll_obs = _get_obs_tuple(Dll)
    hl_obs = Dhl.get("obs", next(iter(Dhl.values())))

    print("\n" + "="*60)
    print("TRAINING COLOR MNIST MODELS")
    print("="*60)

    # ---------------- HL MODEL ----------------
    print("\n--- Training High-Level Model (cell-means) ---")
    hl_model, U_hl_hat, r2_hl_tr, r2_hl_val, lam = abduce_hl_noise_cellmeans(hl_obs)
    print(f"[HL] R² — train: {r2_hl_tr:.4f} | val: {r2_hl_val:.4f} | λ={lam:g}")

    # ---------------- LL MODEL ----------------
    print("\n--- Training Low-Level Model (U-Net) ---")
    ll_model_unet, U_ll_hat, r2_ll_tr, r2_ll_val = abduce_ll_noise_unet(ll_obs, seed=seed, norm_type=norm_type)
    print(f"[LL] R² — train: {r2_ll_tr:.4f} | val: {r2_ll_val:.4f}")

    # ---------------- Save models ----------------
    if save_models:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving models and residuals to {output_dir}...")
        torch.save(ll_model_unet.state_dict(), os.path.join(output_dir, 'll_model_unet.pth'))
        joblib.dump(hl_model, os.path.join(output_dir, 'hl_model_cellmeans.joblib'))
        torch.save(U_ll_hat, os.path.join(output_dir, 'U_ll_hat.pkl'))
        torch.save(U_hl_hat, os.path.join(output_dir, 'U_hl_hat.pkl'))
        print("✓ Models and residuals saved successfully!")

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"[HL] R² — train: {r2_hl_tr:.4f} | val: {r2_hl_val:.4f} | λ={lam:g}")
    print(f"[LL] R² — train: {r2_ll_tr:.4f} | val: {r2_ll_val:.4f}")
    print("✓ Both models trained and saved. Ready for abstraction experiments.\n")

    return ll_model_unet, hl_model, U_ll_hat, U_hl_hat, r2_hl_tr, r2_hl_val, r2_ll_tr, r2_ll_val



def main():
    p = argparse.ArgumentParser(description="Train ColorMNIST models on a generated dataset folder.")
    p.add_argument("--input-dir", type=str, default="data/cmnist",
                   help="Folder containing dll_samples.pkl, dhl_samples.pkl, intervention_mapping.pkl")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Where to save models/residuals (default: input-dir)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--norm-type", type=str, default="bn", choices=["bn","in","none"])
    args = p.parse_args()

    in_dir = args.input_dir
    out_dir = args.save_dir or in_dir

    print("ColorMNIST Model Training Pipeline")
    print("="*50)
    set_global_seed(args.seed)

    Dll_samples = torch.load(os.path.join(in_dir, "dll_samples.pkl"))
    Dhl_samples = torch.load(os.path.join(in_dir, "dhl_samples.pkl"))
    omega = torch.load(os.path.join(in_dir, "intervention_mapping.pkl"))

    ll_model, hl_model, U_ll_hat, U_hl_hat, r2_hl_tr, r2_hl_val, r2_ll_tr, r2_ll_val = train_models(
        Dll_samples, Dhl_samples,
        seed=args.seed, norm_type=args.norm_type,
        save_models=True, output_dir=out_dir
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"[HL] R² — train: {r2_hl_tr:.4f} | val: {r2_hl_val:.4f}")
    print(f"[LL] R² — train: {r2_ll_tr:.4f} | val: {r2_ll_val:.4f}")
    print("You can now use these models for causal abstraction experiments.")
    return ll_model, hl_model, U_ll_hat, U_hl_hat



if __name__ == "__main__":
    main()
