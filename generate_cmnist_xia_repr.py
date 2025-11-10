#!/usr/bin/env python3
"""
Generate Colored MNIST data + Xia-style representation z as HL variable.

Design:
- LL data: EXACTLY your ColorMNISTDataGenerator (torchvision MNIST, your confounding + interventions).
- HL data: concat[ one-hot(D), one-hot(C), z ],
    where z is produced by a RepresentationalNN (Xia et al.) trained here as an autoencoder on obs images.

We:
  1) Generate obs data with your LL generator.
  2) Train RepresentationalNN encoder/decoder on obs colored images to reconstruct them (repr="auto_enc").
  3) For each intervention, generate LL samples and pass images through the trained encoder to get z.
  4) Save dll_samples / dhl_samples / intervention_mapping in your original format.
"""

import os
import json
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from operations import Intervention  # your project

# Xia code
from src_xia.scm.repr_nn.representation_nn import RepresentationalNN
from src_xia.datagen.scm_datagen import SCMDataTypes as sdt


# =========================
# Globals & interventions
# =========================

D_LL, C_LL, P_LL = 'Digit', 'Color', 'Pixels'
D_HL, C_HL, I_HL = 'Digit_', 'Color_', 'Image_'

INTERVENTION_LABELS = [
    "obs",
    "D=6",
    "D=8",
    "D=4",
    "C=7",
    "C=0",
    "C=4",
    "D=6,C=7",
    "D=8,C=0",
    "D=4,C=4",
]


def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_ll_interventions() -> Dict[str, Optional[Intervention]]:
    return {
        "obs": None,
        "D=6": Intervention({D_LL: 6}),
        "D=8": Intervention({D_LL: 8}),
        "D=4": Intervention({D_LL: 4}),
        "C=7": Intervention({C_LL: 7}),
        "C=0": Intervention({C_LL: 0}),
        "C=4": Intervention({C_LL: 4}),
        "D=6,C=7": Intervention({D_LL: 6, C_LL: 7}),
        "D=8,C=0": Intervention({D_LL: 8, C_LL: 0}),
        "D=4,C=4": Intervention({D_LL: 4, C_LL: 4}),
    }


def make_hl_interventions() -> Dict[str, Optional[Intervention]]:
    return {
        "obs": None,
        "D=6": Intervention({D_HL: 6}),
        "D=8": Intervention({D_HL: 8}),
        "D=4": Intervention({D_HL: 4}),
        "C=7": Intervention({C_HL: 7}),
        "C=0": Intervention({C_HL: 0}),
        "C=4": Intervention({C_HL: 4}),
        "D=6,C=7": Intervention({D_HL: 6, C_HL: 7}),
        "D=8,C=0": Intervention({D_HL: 8, C_HL: 0}),
        "D=4,C=4": Intervention({D_HL: 4, C_HL: 4}),
    }


def define_interventions():
    ll_int = make_ll_interventions()
    hl_int = make_hl_interventions()
    omega_labels = {lab: lab for lab in INTERVENTION_LABELS}
    return omega_labels, ll_int, hl_int


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


# =========================
# Your ColorMNIST generator
# =========================

class ColorMNISTDataGenerator:
    """
    Generates samples for Colored MNIST.
    - Shapes: grayscale tensors in [-1,1]
    - Colored images: tensors in [0,1]
    """

    def __init__(
        self,
        image_size: int = 32,
        correlation: float = 0.0,
        noise_enable: bool = False,
        sensor_noise_std: float = 0.03,
        brightness_jitter: float = 0.05,
        contrast_jitter: float = 0.05,
        color_jitter_std: float = 0.015,
    ):
        self.image_size = int(image_size)
        self.correlation = float(correlation)

        self.noise_enable = bool(noise_enable)
        self.sensor_noise_std = float(sensor_noise_std)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.color_jitter_std = float(color_jitter_std)

        # RGB per color
        self.colors = {
            0: (1.0, 0.0, 0.0), 1: (1.0, 0.6, 0.0), 2: (0.8, 1.0, 0.0),
            3: (0.2, 1.0, 0.0), 4: (0.0, 1.0, 0.4), 5: (0.0, 1.0, 1.0),
            6: (0.0, 0.4, 1.0), 7: (0.2, 0.0, 1.0), 8: (0.8, 0.0, 1.0),
            9: (1.0, 0.0, 0.6)
        }

        self.shape_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),                 # [0,1]
            transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
        ])

        print("Loading and preparing MNIST dataset via torchvision...")
        mnist = torchvision.datasets.MNIST('data/', train=True, download=True)
        self.mnist_data: Dict[int, List[torch.Tensor]] = {i: [] for i in range(10)}
        for img, target in zip(mnist.data, mnist.targets):
            self.mnist_data[int(target)].append(img)
        print("MNIST dataset ready.")

    @torch.no_grad()
    def _apply_photometric_jitter(self, x3chw: torch.Tensor) -> torch.Tensor:
        if self.brightness_jitter > 0:
            b = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * self.brightness_jitter
            x3chw = x3chw * b
        if self.contrast_jitter > 0:
            mean = x3chw.mean()
            c = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * self.contrast_jitter
            x3chw = (x3chw - mean) * c + mean
        if self.color_jitter_std > 0:
            jitter = torch.randn(3, 1, 1) * self.color_jitter_std
            x3chw = x3chw + jitter
        return clamp01(x3chw)

    @torch.no_grad()
    def _apply_sensor_noise(self, x3chw: torch.Tensor) -> torch.Tensor:
        if self.sensor_noise_std <= 0:
            return x3chw
        noise = torch.randn_like(x3chw) * self.sensor_noise_std
        return clamp01(x3chw + noise)

    def _sample_digit_color(self, intervention: Optional[Intervention]):
        iv_dict = intervention.vv() if intervention is not None else {}
        u_conf = np.random.randint(10)
        digit = iv_dict.get(D_LL, u_conf if np.random.random() < self.correlation else np.random.randint(10))
        color = iv_dict.get(C_LL, u_conf if np.random.random() < self.correlation else np.random.randint(10))
        return int(digit), int(color)

    def generate_samples(self, n: int, intervention: Optional[Intervention] = None):
        final_images, img_shapes, digits, colors = [], [], [], []

        for _ in range(n):
            digit, color = self._sample_digit_color(intervention)
            idx = np.random.randint(len(self.mnist_data[digit]))
            original_shape_uint8: torch.Tensor = self.mnist_data[digit][idx]

            # shape [-1,1]
            shape_tensor = self.shape_transform(original_shape_uint8)
            img_shapes.append(shape_tensor)

            # colored [0,1]
            img_gray01 = (original_shape_uint8.to(torch.float32) / 255.0).unsqueeze(0)
            img_color = img_gray01.repeat(3, 1, 1)
            r, g, b = self.colors[color]
            img_color[0].mul_(r)
            img_color[1].mul_(g)
            img_color[2].mul_(b)

            final_img_tensor = TF.resize(img_color, [self.image_size, self.image_size], antialias=True)

            if self.noise_enable:
                final_img_tensor = self._apply_photometric_jitter(final_img_tensor)
                final_img_tensor = self._apply_sensor_noise(final_img_tensor)

            final_images.append(final_img_tensor)
            digits.append(digit)
            colors.append(color)

        return (
            torch.stack(final_images, dim=0).to(torch.float32),   # (N,3,H,W) [0,1]
            torch.stack(img_shapes, dim=0).to(torch.float32),     # (N,1,H,W) [-1,1]
            torch.tensor(digits, dtype=torch.long),
            torch.tensor(colors, dtype=torch.long),
        )


# =========================
# Train Xia-style encoder HERE
# =========================

def train_xia_repr_autoencoder_on_obs(
    obs_images: torch.Tensor,
    img_size: int,
    rep_size: int,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """
    Train RepresentationalNN (from Xia et al.) as an autoencoder on obs images.

    Setup:
      - Single IMAGE variable "img"
      - repr="auto_enc", rep-image-only
      - loss: sum_v MSE(recon[v], input[v]) -> here just v="img"

    Returns:
      repr_model (trained), to be used for computing z = encode(img)["img"].
    """

    # Dummy causal graph with one image node
    class DummyCG:
        def __init__(self):
            self.v = ["img"]
            self.pa = {"img": []}

    cg = DummyCG()

    v_type = {"img": sdt.IMAGE}
    v_size = {"img": 3 * img_size * img_size}

    hyper = {
        "rep-size": rep_size,
        "img-size": img_size,
        "rep-h-size": 128,
        "rep-feature-maps": 64,
        "rep-h-layers": 2,
        "batch-norm": True,
        "gan-arch": "dcgan",
        "repr": "auto_enc",
        "rep-image-only": True,
    }

    model = RepresentationalNN(cg, v_size, v_type, hyperparams=hyper)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = list(model.encoders.parameters()) + list(model.decoders.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(obs_images)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    print("\n[Repr] Training Xia-style RepresentationalNN (autoencoder) on obs images...")
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (x,) in dl:
            x = x.to(device)
            batch = {"img": x}
            out = model(batch)              # returns dict: {"img": recon}
            x_hat = out["img"]
            loss = loss_fn(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        total /= len(ds)
        print(f"  [Repr AE] Epoch {ep:02d}/{epochs} - recon loss: {total:.6f}")
    print("[Repr] Encoder training complete.\n")

    model.eval()
    model.to(device)
    return model


def encode_batch_with_repr_model(model: RepresentationalNN, images: torch.Tensor, rep_size: int) -> torch.Tensor:
    """
    Apply trained RepresentationalNN to a batch of images.
    images: (N,3,H,W) in [0,1]
    Returns: (N, rep_size) tensor z
    """
    device = next(model.parameters()).device
    zs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, images.size(0), 512):
            xb = images[i:i+512].to(device)
            batch = {"img": xb}
            rep_dict = model.encode(batch)
            z = rep_dict["img"]
            if z.dim() > 2:
                z = z.view(z.size(0), -1)
            zs.append(z.cpu())
    return torch.cat(zs, dim=0)


# =========================
# Main generation with z
# =========================

def generate_cmnist_data_with_z(
    n_samples: int = 1000,
    save_data: bool = True,
    output_dir: str = 'data/cmnist_xia_z',
    seed: int = 42,
    image_size: int = 32,
    correlation: float = 0.0,
    noise_enable: bool = False,
    sensor_noise_std: float = 0.03,
    brightness_jitter: float = 0.05,
    contrast_jitter: float = 0.05,
    color_jitter_std: float = 0.015,
    rep_size: int = 64,
):
    set_global_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    omega_labels, ll_ints, hl_ints = define_interventions()
    gen = ColorMNISTDataGenerator(
        image_size=image_size,
        correlation=correlation,
        noise_enable=noise_enable,
        sensor_noise_std=sensor_noise_std,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        color_jitter_std=color_jitter_std,
    )

    # 1) Generate obs data once
    obs_ll = gen.generate_samples(n_samples, intervention=ll_ints["obs"])
    obs_images, obs_shapes, obs_digits, obs_colors = obs_ll

    # 2) Train Xia-style RepresentationalNN as autoencoder on obs_images
    repr_model = train_xia_repr_autoencoder_on_obs(
        obs_images, img_size=image_size, rep_size=rep_size,
        epochs=10, batch_size=256, lr=1e-3,
    )

    # 3) Generate all interventions + encode z
    print(f"Generating {n_samples} samples for each of {len(INTERVENTION_LABELS)} interventions...")
    Dll_samples: Dict[str, Tuple[torch.Tensor, ...]] = {}
    Dhl_samples: Dict[str, torch.Tensor] = {}

    for lab in tqdm(INTERVENTION_LABELS, desc="Generating Data w/ z"):
        if lab == "obs":
            ll = obs_ll
        else:
            ll = gen.generate_samples(n_samples, intervention=ll_ints[lab])

        final_images, img_shapes, digits, colors = ll
        Dll_samples[lab] = (final_images, img_shapes, digits, colors)

        # Encode to z via trained RepresentationalNN
        z = encode_batch_with_repr_model(repr_model, final_images, rep_size=rep_size)

        d_oh = F.one_hot(digits, num_classes=10).float()
        c_oh = F.one_hot(colors, num_classes=10).float()
        hl = torch.cat([d_oh, c_oh, z], dim=1)   # (N, 20 + rep_size)
        Dhl_samples[lab] = hl

    # 4) Range check
    obs_ll_images = Dll_samples["obs"][0]
    imin, imax = obs_ll_images.min().item(), obs_ll_images.max().item()
    print(f"[Range check] LL colored images (obs) in [{imin:.3f}, {imax:.3f}] (expected [0,1])")

    # 5) Save
    if save_data:
        print(f"\nSaving data to {output_dir}...")
        torch.save(Dll_samples, os.path.join(output_dir, 'dll_samples.pkl'))
        torch.save(Dhl_samples, os.path.join(output_dir, 'dhl_samples.pkl'))
        torch.save(omega_labels, os.path.join(output_dir, 'intervention_mapping.pkl'))
        with open(os.path.join(output_dir, 'intervention_mapping.json'), 'w') as f:
            json.dump(omega_labels, f, indent=2)
        print("Data saved successfully!")

    return Dll_samples, Dhl_samples, omega_labels


# =========================
# CLI
# =========================

def build_argparser():
    p = argparse.ArgumentParser(description="Generate CMNIST + Xia-style z (train encoder inside).")
    p.add_argument("--n-samples", type=int, default=1000, help="Samples per intervention.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--correlation", type=float, default=0.0)
    p.add_argument("--noise-enable", action="store_true")
    p.add_argument("--sensor-noise-std", type=float, default=0.03)
    p.add_argument("--brightness-jitter", type=float, default=0.05)
    p.add_argument("--contrast-jitter", type=float, default=0.05)
    p.add_argument("--color-jitter-std", type=float, default=0.015)
    p.add_argument("--rep-size", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="data/cmnist_xia_z")
    p.add_argument("--no-save", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()

    print("ColorMNIST + z Generator (Xia-style encoder trained inline)")
    print("=" * 60)
    print(f"  - Samples per intervention: {args.n_samples}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Image size: {args.image_size}")
    print(f"  - Correlation: {args.correlation}")
    print(f"  - Rep size (z dim): {args.rep_size}")
    print(f"  - Output dir: {args.output_dir}")

    Dll_samples, Dhl_samples, omega_labels = generate_cmnist_data_with_z(
        n_samples=args.n_samples,
        save_data=not args.no_save,
        output_dir=args.output_dir,
        seed=args.seed,
        image_size=args.image_size,
        correlation=args.correlation,
        noise_enable=args.noise_enable,
        sensor_noise_std=args.sensor_noise_std,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
        color_jitter_std=args.color_jitter_std,
        rep_size=args.rep_size,
    )

    print("\nDone.")
    print(f"Generated data for {len(omega_labels)} intervention labels.")
    return Dll_samples, Dhl_samples, omega_labels


if __name__ == "__main__":
    main()
