# #!/usr/bin/env python3
# """
# Train ColorMNIST Models

# This script loads the generated ColorMNIST data and trains neural network models
# for both low-level and high-level causal abstraction tasks.
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader, random_split
# from tqdm import tqdm
# import math
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import os


# def ensure_tanh_scaled(images: torch.Tensor) -> torch.Tensor:
#     """
#     Ensure targets live in [-1, 1]. If they appear to be in [0, 1], rescale.
#     Assumes float tensor (N, 3, H, W) or similar.
#     """
#     with torch.no_grad():
#         imin = images.min().item()
#         imax = images.max().item()
#     if imin >= -1.05 and imax <= 1.05:
#         if imin >= -0.01 and imax <= 1.01:
#             return images * 2.0 - 1.0
#         return images
#     eps = 1e-8
#     images = (images - images.min()) / (images.max() - images.min() + eps)
#     return images * 2.0 - 1.0


# # ----------------------------
# # Core building blocks
# # ----------------------------

# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_type="bn"):
#         super().__init__()
#         norm = {
#             "bn": nn.BatchNorm2d,
#             "in": nn.InstanceNorm2d,
#             None: lambda c: nn.Identity(),
#         }.get(norm_type, nn.BatchNorm2d)
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             norm(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             norm(out_ch),
#             nn.ReLU(inplace=True),
#         )
#         self._init()

#     def _init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

#     def forward(self, x):
#         return self.block(x)


# class Down(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_type="bn"):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2)
#         self.conv = ConvBlock(in_ch, out_ch, norm_type)

#     def forward(self, x):
#         return self.conv(self.pool(x))


# class Up(nn.Module):
#     def __init__(self, in_ch, skip_ch, out_ch, norm_type="bn"):
#         super().__init__()
#         self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
#         nn.init.kaiming_normal_(self.reduce.weight, nonlinearity="relu")
#         self.conv = ConvBlock(out_ch + skip_ch, out_ch, norm_type)

#     def forward(self, x, skip):
#         x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
#         x = self.reduce(x)
#         dh = skip.size(-2) - x.size(-2)
#         dw = skip.size(-1) - x.size(-1)
#         if dh != 0 or dw != 0:
#             x = F.pad(x, (0, dw, 0, dh))
#         x = torch.cat([skip, x], dim=1)
#         return self.conv(x)


# class FiLM(nn.Module):
#     """Feature-wise Linear Modulation for conditioning.
#     Maps discrete labels (digit, color) into per-channel gamma/beta.
#     """
#     def __init__(self, num_digits=10, num_colors=10, feat_channels=128, hidden=64):
#         super().__init__()
#         self.d_embed = nn.Embedding(num_digits, hidden)
#         self.c_embed = nn.Embedding(num_colors, hidden)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden * 2, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, feat_channels * 2),
#         )
#         nn.init.normal_(self.d_embed.weight, std=0.02)
#         nn.init.normal_(self.c_embed.weight, std=0.02)
#         for m in self.mlp:
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x, digit, color):
#         d = self.d_embed(digit)
#         c = self.c_embed(color)
#         h = torch.cat([d, c], dim=-1)
#         params = self.mlp(h)
#         C = x.size(1)
#         gamma, beta = params.split(C, dim=-1)
#         gamma = gamma.unsqueeze(-1).unsqueeze(-1)
#         beta = beta.unsqueeze(-1).unsqueeze(-1)
#         return gamma * x + beta


# # ----------------------------
# # U-Net with FiLM conditioning
# # ----------------------------

# class ImageColorizerUNet(nn.Module):
#     """
#     A light U-Net for 32×32 inputs with FiLM conditioning on digit and color.
#     Input: grayscale image (N,1,32,32), digit label (N,), color label (N,)
#     Output: colored image in [-1,1] via tanh (N,3,32,32)
#     """
#     def __init__(self, norm_type="bn"):
#         super().__init__()
#         self.inc = ConvBlock(1, 32, norm_type)
#         self.down1 = Down(32, 64, norm_type)
#         self.down2 = Down(64, 128, norm_type)
#         self.film_bottleneck = FiLM(feat_channels=128)
#         self.film_up1 = FiLM(feat_channels=64)
#         self.up1 = Up(128, 64, 64, norm_type)
#         self.up2 = Up(64, 32, 32, norm_type)
#         self.outc = nn.Conv2d(32, 3, kernel_size=1)
#         nn.init.kaiming_normal_(self.outc.weight, nonlinearity="linear")
#         if self.outc.bias is not None:
#             nn.init.zeros_(self.outc.bias)

#     def forward(self, x, digit, color):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x3 = self.film_bottleneck(x3, digit, color)
#         y = self.up1(x3, x2)
#         y = self.film_up1(y, digit, color)
#         y = self.up2(y, x1)
#         out = torch.tanh(self.outc(y))
#         return out


# # ----------------------------
# # Training with Huber loss, cosine LR, early stopping, validation split
# # ----------------------------

# class EarlyStopper:
#     def __init__(self, patience=30, min_delta=0.0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.best = math.inf
#         self.counter = 0

#     def step(self, val_loss):
#         if val_loss < self.best - self.min_delta:
#             self.best = val_loss
#             self.counter = 0
#             return False
#         else:
#             self.counter += 1
#             return self.counter >= self.patience


# def train_cnn_model_unet(model, ll_samples_tuple, epochs=300, batch_size=256, lr=1e-3, huber_beta=0.02, val_frac=0.1, weight_decay=1e-5):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"  - Training on device: {device}")
#     model.to(device)

#     # Unpack
#     final_images, img_shapes, digits, colors = ll_samples_tuple

#     # Ensure targets match tanh scaling
#     final_images_scaled = ensure_tanh_scaled(final_images)

#     # Dataset & split
#     full_ds = TensorDataset(img_shapes, digits, colors, final_images_scaled)
#     n_val = max(1, int(len(full_ds) * val_frac))
#     n_train = len(full_ds) - n_val
#     train_ds, val_ds = random_split(full_ds, [n_train, n_val])

#     pin_mem = device.type == "cuda"
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2 if pin_mem else 0, pin_memory=pin_mem)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if pin_mem else 0, pin_memory=pin_mem)

#     criterion = nn.SmoothL1Loss(beta=huber_beta)  # Huber / Charbonnier-like
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#     stopper = EarlyStopper(patience=30, min_delta=1e-5)

#     best_state = None
#     best_val = math.inf

#     for epoch in tqdm(range(1, epochs + 1), desc="  - Training Epochs"):
#         model.train()
#         train_loss = 0.0
#         for shape_batch, digit_batch, color_batch, target_batch in train_loader:
#             shape_batch = shape_batch.to(device)
#             digit_batch = digit_batch.to(device)
#             color_batch = color_batch.to(device)
#             target_batch = target_batch.to(device)

#             optimizer.zero_grad()
#             pred = model(shape_batch, digit_batch, color_batch)
#             loss = criterion(pred, target_batch)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * shape_batch.size(0)

#         train_loss /= n_train

#         # Validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for shape_batch, digit_batch, color_batch, target_batch in val_loader:
#                 shape_batch = shape_batch.to(device)
#                 digit_batch = digit_batch.to(device)
#                 color_batch = color_batch.to(device)
#                 target_batch = target_batch.to(device)
#                 pred = model(shape_batch, digit_batch, color_batch)
#                 loss = criterion(pred, target_batch)
#                 val_loss += loss.item() * shape_batch.size(0)
#         val_loss /= n_val

#         scheduler.step()

#         tqdm.write(f"Epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e}")

#         # Early stopping
#         if val_loss < best_val:
#             best_val = val_loss
#             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
#         if stopper.step(val_loss):
#             print(f"Early stopping at epoch {epoch} (best val {best_val:.6f}).")
#             break

#     if best_state is not None:
#         model.load_state_dict(best_state)
#     model.to("cpu")
#     return model


# # --------------------
# # High-level abduction
# # --------------------

# def abduce_hl_noise(hl_observational_data):
#     print("Fitting high-level model to abduce noise (U_hl_hat)...")
#     features_hl = hl_observational_data[:, :-1].numpy()
#     target_hl = hl_observational_data[:, -1].numpy()
#     hl_deterministic_model = LinearRegression()
#     hl_deterministic_model.fit(features_hl, target_hl)
#     predictions = hl_deterministic_model.predict(features_hl)
#     residuals = target_hl - predictions
#     print("✓ High-level abduction complete.")

#     return hl_deterministic_model, torch.tensor(residuals, dtype=torch.float32).unsqueeze(1)


# # -------------------
# # Low-level abduction
# # -------------------

# def abduce_ll_noise_unet(ll_observational_data_tuple):
#     print("Abducing low-level noise (U_ll_hat) using: ImageColorizerUNet...")
#     ll_deterministic_model = ImageColorizerUNet()
#     ll_deterministic_model = train_cnn_model_unet(ll_deterministic_model, ll_observational_data_tuple)

#     # Predict deterministic outputs on all obs
#     print("\n- Predicting deterministic outputs to calculate residuals (noise)...")
#     ll_deterministic_model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ll_deterministic_model.to(device)

#     final_images, img_shapes, digits, colors = ll_observational_data_tuple
#     final_images_scaled = ensure_tanh_scaled(final_images)  # same scaling as training

#     with torch.no_grad():
#         preds = []
#         B = 1024
#         for i in range(0, len(img_shapes), B):
#             shape_batch = img_shapes[i:i+B].to(device)
#             digit_batch = digits[i:i+B].to(device)
#             color_batch = colors[i:i+B].to(device)
#             p = ll_deterministic_model(shape_batch, digit_batch, color_batch).cpu()
#             preds.append(p)
#         predicted_deterministic_images = torch.cat(preds, dim=0)

#     U_ll_hat = final_images_scaled - predicted_deterministic_images
#     print("\n✓ Low-level abduction complete using U-Net with FiLM.")
#     ll_deterministic_model.to("cpu")
#     return ll_deterministic_model, U_ll_hat


# # ----------------------------
# # R^2 evaluation consistent with scaling
# # ----------------------------

# def r2_imagewise(final_images, predictions):
#     """Compute global and per-image R^2 on the same scale for both tensors."""
#     # Flatten per image
#     N = final_images.size(0)
#     y = final_images.view(N, -1)
#     yhat = predictions.view(N, -1)
#     # Global R^2
#     ss_res = torch.sum((y - yhat) ** 2)
#     ss_tot = torch.sum((y - y.mean()) ** 2)
#     r2_global = 1 - (ss_res / (ss_tot + 1e-12))
#     # Per-image R^2
#     y_mean = y.mean(dim=1, keepdim=True)
#     ss_res_i = torch.sum((y - yhat) ** 2, dim=1)
#     ss_tot_i = torch.sum((y - y_mean) ** 2, dim=1) + 1e-12
#     r2_per_image = 1 - ss_res_i / ss_tot_i
#     return r2_global.item(), r2_per_image


# def load_cmnist_data(data_dir='data/cmnist'):
#     """Load the generated ColorMNIST data."""
#     print(f"Loading ColorMNIST data from {data_dir}...")
    
#     Dll_samples = torch.load(f'{data_dir}/dll_samples.pkl')
#     Dhl_samples = torch.load(f'{data_dir}/dhl_samples.pkl')
#     omega = torch.load(f'{data_dir}/intervention_mapping.pkl')
    
#     print("Data loaded successfully!")
#     print(f"  - Low-level interventions: {len(Dll_samples)}")
#     print(f"  - High-level interventions: {len(Dhl_samples)}")
#     print(f"  - Intervention mappings: {len(omega)}")
    
#     return Dll_samples, Dhl_samples, omega


# def train_models(Dll_samples, Dhl_samples, save_models=True, output_dir='data/cmnist'):
#     """Train both low-level and high-level models."""
    
#     # Get observational data
#     ll_obs_data_tuple = Dll_samples[None]
#     hl_obs_data = Dhl_samples[None]
    
#     print("\n" + "="*60)
#     print("TRAINING COLORMNIST MODELS")
#     print("="*60)
    
#     # Train high-level model
#     print("\n--- Training High-Level Model ---")
#     hl_model, U_hl_hat = abduce_hl_noise(hl_obs_data)
    
#     # Train low-level model
#     print("\n--- Training Low-Level Model ---")
#     ll_model_unet, U_ll_hat_unet = abduce_ll_noise_unet(ll_obs_data_tuple)
    
#     # R^2 evaluation
#     print("\n--- Performing R-squared Test for ImageColorizerUNet ---")
#     final_images, img_shapes, digits, colors = ll_obs_data_tuple
#     final_images_scaled = ensure_tanh_scaled(final_images)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ll_model_unet.to(device).eval()

#     with torch.no_grad():
#         preds = []
#         B = 1024
#         for i in range(0, len(img_shapes), B):
#             shape_batch = img_shapes[i:i+B].to(device)
#             digit_batch = digits[i:i+B].to(device)
#             color_batch = colors[i:i+B].to(device)
#             p = ll_model_unet(shape_batch, digit_batch, color_batch).cpu()
#             preds.append(p)
#         predictions = torch.cat(preds, dim=0)

#     r2_global, r2_per_image = r2_imagewise(final_images_scaled, predictions)
#     print(f"  - Overall R-squared (R^2): {r2_global:.4f}")
#     print(f"  - Per-image R^2: mean={r2_per_image.mean().item():.4f}, median={r2_per_image.median().item():.4f}")
#     print("Abduction and validation for upgraded CNN complete.")
    
#     # Save models if requested
#     if save_models:
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"\nSaving models to {output_dir}...")
#         torch.save(ll_model_unet.state_dict(), f'{output_dir}/ll_model_unet.pth')
#         torch.save(hl_model, f'{output_dir}/hl_model.pkl')
#         torch.save(U_ll_hat_unet, f'{output_dir}/U_ll_hat.pkl')
#         torch.save(U_hl_hat, f'{output_dir}/U_hl_hat.pkl')
#         print("Models saved successfully!")
    
#     return ll_model_unet, hl_model, U_ll_hat_unet, U_hl_hat


# def main():
#     """Main function to run the complete training pipeline."""
#     print("ColorMNIST Model Training Pipeline")
#     print("="*50)
    
#     # Load data
#     Dll_samples, Dhl_samples, omega = load_cmnist_data()
    
#     # Train models
#     ll_model, hl_model, U_ll_hat, U_hl_hat = train_models(Dll_samples, Dhl_samples)
    
#     print("\n" + "="*60)
#     print("TRAINING COMPLETE!")
#     print("="*60)
#     print("Models trained and saved successfully.")
#     print("You can now use these models for causal abstraction experiments.")
    
#     return ll_model, hl_model, U_ll_hat, U_hl_hat


# if __name__ == "__main__":
#     # Run the training pipeline
#     ll_model, hl_model, U_ll_hat, U_hl_hat = main()

#!/usr/bin/env python3
"""
Train ColorMNIST Models — updated

Changes:
- Deterministic RNG & splits
- HL abduction: Standardized RidgeCV (+ optional D×C interactions)
- Save sklearn with joblib; torch with state_dict
- Report U-Net val R^2 (held-out) in addition to train R^2
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

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import joblib

# ----------------------------
# Reproducibility helpers
# ----------------------------

def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Scaling utility
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
# U-Net with FiLM conditioning
# ----------------------------

class ImageColorizerUNet(nn.Module):
    """
    A light U-Net for 32×32 inputs with FiLM conditioning on digit and color.
    Input: grayscale (N,1,32,32), digit (N,), color (N,) -> output (N,3,32,32) in [-1,1]
    """
    def __init__(self, norm_type="bn"):
        super().__init__()
        self.inc = ConvBlock(1, 32, norm_type)
        self.down1 = Down(32, 64, norm_type)
        self.down2 = Down(64, 128, norm_type)
        self.film_bottleneck = FiLM(feat_channels=128)
        self.film_up1 = FiLM(feat_channels=64)
        self.up1 = Up(128, 64, 64, norm_type)
        self.up2 = Up(64, 32, 32, norm_type)
        self.outc = nn.Conv2d(32, 3, kernel_size=1)
        nn.init.kaiming_normal_(self.outc.weight, nonlinearity="linear")
        if self.outc.bias is not None:
            nn.init.zeros_(self.outc.bias)

    def forward(self, x, digit, color):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.film_bottleneck(x3, digit, color)
        y = self.up1(x3, x2)
        y = self.film_up1(y, digit, color)
        y = self.up2(y, x1)
        out = torch.tanh(self.outc(y))
        return out

# ----------------------------
# Training with Huber loss, cosine LR, early stopping, val R^2
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
        else:
            self.counter += 1
            return self.counter >= self.patience

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

def train_cnn_model_unet(
    model,
    ll_samples_tuple,
    epochs=300,
    batch_size=256,
    lr=1e-3,
    huber_beta=0.02,
    val_frac=0.1,
    weight_decay=1e-5,
    seed=42,
    norm_type="bn",
    use_amp=False,
    grad_clip=None,
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Training on device: {device}")
    model.to(device)

    # Unpack
    final_images, img_shapes, digits, colors = ll_samples_tuple

    # Ensure targets match tanh scaling
    final_images_scaled = ensure_tanh_scaled(final_images)

    # Dataset & deterministic split
    full_ds = TensorDataset(img_shapes, digits, colors, final_images_scaled)
    n_val = max(1, int(len(full_ds) * val_frac))
    n_train = len(full_ds) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    pin_mem = device.type == "cuda"
    num_workers = 2 if pin_mem else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem)

    criterion = nn.SmoothL1Loss(beta=huber_beta)  # Huber-like
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper = EarlyStopper(patience=30, min_delta=1e-5)

    best_state = None
    best_val = math.inf

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in tqdm(range(1, epochs + 1), desc="  - Training Epochs"):
        model.train()
        train_loss = 0.0
        for shape_batch, digit_batch, color_batch, target_batch in train_loader:
            shape_batch = shape_batch.to(device)
            digit_batch = digit_batch.to(device)
            color_batch = color_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(shape_batch, digit_batch, color_batch)
                    loss = criterion(pred, target_batch)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(shape_batch, digit_batch, color_batch)
                loss = criterion(pred, target_batch)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            train_loss += loss.item() * shape_batch.size(0)

        train_loss /= n_train

        # Validation loss + R^2
        model.eval()
        val_loss = 0.0
        all_tgts, all_preds = [], []
        with torch.no_grad():
            for shape_batch, digit_batch, color_batch, target_batch in val_loader:
                shape_batch = shape_batch.to(device)
                digit_batch = digit_batch.to(device)
                color_batch = color_batch.to(device)
                target_batch = target_batch.to(device)
                pred = model(shape_batch, digit_batch, color_batch)
                loss = criterion(pred, target_batch)
                val_loss += loss.item() * shape_batch.size(0)
                all_tgts.append(target_batch.detach().cpu())
                all_preds.append(pred.detach().cpu())
        val_loss /= n_val
        if len(all_tgts):
            tgts = torch.cat(all_tgts, dim=0)
            preds = torch.cat(all_preds, dim=0)
            r2_val_global, _ = r2_imagewise(tgts, preds)
        else:
            r2_val_global = float('nan')

        scheduler.step()

        tqdm.write(
            f"Epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f} "
            f"| val_R2 {r2_val_global:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch} (best val {best_val:.6f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return model

# --------------------
# High-level abduction (improved)
# --------------------

# def abduce_hl_noise_better(hl_observational_data, add_interactions=True):
#     """
#     Improved HL abduction:
#     - Standardize features
#     - Optionally add digit×color interactions
#     - RidgeCV for stability
#     Returns: (fitted_model_pipeline, U_hl_hat_tensor, r2_train)
#     """
#     X = hl_observational_data[:, :-1].numpy()
#     y = hl_observational_data[:, -1].numpy()

#     if add_interactions:
#         poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#         X_aug = poly.fit_transform(X)
#     else:
#         X_aug = X

#     model = make_pipeline(
#         StandardScaler(with_mean=True, with_std=True),
#         RidgeCV(alphas=np.logspace(-4, 4, 25), fit_intercept=True)
#     )
#     model.fit(X_aug, y)
#     y_hat = model.predict(X_aug)
#     r2_tr = r2_score(y, y_hat)
#     U_hl_hat = torch.tensor((y - y_hat), dtype=torch.float32).unsqueeze(1)
#     print(f"✓ High-level abduction complete. HL R^2 (train): {r2_tr:.4f}")
#     return model, U_hl_hat, r2_tr
def abduce_hl_noise_better(hl_observational_data, add_interactions=False):
    """
    Fixed HL abduction that properly handles one-hot encoded data:
    - Use one-hot features directly (no polynomial interactions)
    - RidgeCV for stability
    Returns: (fitted_model, U_hl_hat_tensor, r2_train)
    """
    # Extract the correct features (already one-hot encoded)
    X_digits = hl_observational_data[:, :10].numpy()  # first 10 columns (one-hot digits)
    X_colors = hl_observational_data[:, 10:20].numpy()  # next 10 columns (one-hot colors)
    y = hl_observational_data[:, -1].numpy()  # last column (image_feature)
    
    # Combine one-hot features
    X = np.concatenate([X_digits, X_colors], axis=1)
    
    print(f"Training hl_model on {X.shape[0]} samples...")
    print(f"Input features: one-hot encoded digits and colors ({X.shape[1]} features)")
    print(f"Target: image_feature (mean={y.mean():.4f}, std={y.std():.4f})")
    
    # Create and fit the model (no polynomial features needed)
    model = RidgeCV(alphas=np.logspace(-4, 4, 25))
    model.fit(X, y)
    y_hat = model.predict(X)
    r2_tr = r2_score(y, y_hat)
    
    # Calculate residuals
    U_hl_hat = torch.tensor((y - y_hat), dtype=torch.float32).unsqueeze(1)
    
    print(f"✓ High-level abduction complete. HL R^2 (train): {r2_tr:.4f}")
    
    # Test intervention response
    test_cases = [(0, 0), (8, 0), (5, 5)]
    print("Intervention response test:")
    for digit, color in test_cases:
        test_digit_onehot = np.zeros(10)
        test_color_onehot = np.zeros(10)
        test_digit_onehot[digit] = 1
        test_color_onehot[color] = 1
        test_input = np.concatenate([test_digit_onehot, test_color_onehot]).reshape(1, -1)
        pred = model.predict(test_input)[0]
        print(f"  - do(Digit={digit}, Color={color}): {pred:.6f}")
    
    return model, U_hl_hat, r2_tr
    
# -------------------
# Low-level abduction
# -------------------

def abduce_ll_noise_unet(ll_observational_data_tuple, seed=42):
    print("Abducing low-level noise (U_ll_hat) using: ImageColorizerUNet...")
    ll_deterministic_model = ImageColorizerUNet()
    ll_deterministic_model = train_cnn_model_unet(
        ll_deterministic_model,
        ll_observational_data_tuple,
        seed=seed
    )

    # Predict deterministic outputs on all obs
    print("\n- Predicting deterministic outputs to calculate residuals (noise)...")
    ll_deterministic_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ll_deterministic_model.to(device)

    final_images, img_shapes, digits, colors = ll_observational_data_tuple
    final_images_scaled = ensure_tanh_scaled(final_images)  # same scaling as training

    with torch.no_grad():
        preds = []
        B = 1024
        for i in range(0, len(img_shapes), B):
            shape_batch = img_shapes[i:i+B].to(device)
            digit_batch = digits[i:i+B].to(device)
            color_batch = colors[i:i+B].to(device)
            p = ll_deterministic_model(shape_batch, digit_batch, color_batch).cpu()
            preds.append(p)
        predicted_deterministic_images = torch.cat(preds, dim=0)

    U_ll_hat = final_images_scaled - predicted_deterministic_images
    print("\n✓ Low-level abduction complete using U-Net with FiLM.")
    ll_deterministic_model.to("cpu")
    return ll_deterministic_model, U_ll_hat

# ----------------------------
# Data IO
# ----------------------------

def load_cmnist_data(data_dir='data/cmnist'):
    """Load the generated ColorMNIST data."""
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
# Training driver
# ----------------------------

def train_models(Dll_samples, Dhl_samples, save_models=True, output_dir='data/cmnist', seed=42):
    """Train both low-level and high-level models."""
    set_global_seed(seed)

    # Get observational data
    ll_obs_data_tuple = Dll_samples[None]
    hl_obs_data = Dhl_samples[None]

    print("\n" + "="*60)
    print("TRAINING COLORMNIST MODELS")
    print("="*60)

    # Train high-level model (improved)
    print("\n--- Training High-Level Model ---")
    hl_model, U_hl_hat, r2_hl = abduce_hl_noise_better(hl_obs_data, add_interactions=False)

    # Train low-level model
    print("\n--- Training Low-Level Model ---")
    ll_model_unet, U_ll_hat_unet = abduce_ll_noise_unet(ll_obs_data_tuple, seed=seed)

    # R^2 evaluation (train set)
    print("\n--- Performing R-squared Test for ImageColorizerUNet ---")
    final_images, img_shapes, digits, colors = ll_obs_data_tuple
    final_images_scaled = ensure_tanh_scaled(final_images)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ll_model_unet.to(device).eval()

    with torch.no_grad():
        preds = []
        B = 1024
        for i in range(0, len(img_shapes), B):
            shape_batch = img_shapes[i:i+B].to(device)
            digit_batch = digits[i:i+B].to(device)
            color_batch = colors[i:i+B].to(device)
            p = ll_model_unet(shape_batch, digit_batch, color_batch).cpu()
            preds.append(p)
        predictions = torch.cat(preds, dim=0)

    r2_global_train, r2_per_image_train = r2_imagewise(final_images_scaled, predictions)
    print(f"  - Overall R^2 (train): {r2_global_train:.4f}")
    print(f"  - Per-image R^2 (train): mean={r2_per_image_train.mean().item():.4f}, median={r2_per_image_train.median().item():.4f}")

    print("\nAbduction and validation for upgraded CNN complete.")

    # Save models if requested
    if save_models:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving models to {output_dir}...")
        # Torch model as state_dict (safer across versions)
        torch.save(ll_model_unet.state_dict(), os.path.join(output_dir, 'll_model_unet.pth'))
        # Sklearn model via joblib
        joblib.dump(hl_model, os.path.join(output_dir, 'hl_model.joblib'))
        # Residuals
        torch.save(U_ll_hat_unet, os.path.join(output_dir, 'U_ll_hat.pkl'))
        torch.save(U_hl_hat, os.path.join(output_dir, 'U_hl_hat.pkl'))
        print("Models saved successfully!")

    return ll_model_unet, hl_model, U_ll_hat_unet, U_hl_hat, r2_hl

# ----------------------------
# Main
# ----------------------------

def main():
    """Main function to run the complete training pipeline."""
    print("ColorMNIST Model Training Pipeline")
    print("="*50)
    set_global_seed(42)

    # Load data
    Dll_samples, Dhl_samples, omega = load_cmnist_data()

    # Train models
    ll_model, hl_model, U_ll_hat, U_hl_hat, r2_hl = train_models(Dll_samples, Dhl_samples)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"HL train R^2: {r2_hl:.4f}")
    print("Models trained and saved successfully.")
    print("You can now use these models for causal abstraction experiments.")

    return ll_model, hl_model, U_ll_hat, U_hl_hat

if __name__ == "__main__":
    # Run the training pipeline
    ll_model, hl_model, U_ll_hat, U_hl_hat = main()
