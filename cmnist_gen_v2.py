#!/usr/bin/env python3
"""
cmnist_gen_v2.py

ColorMNIST Data Generator (parameterized) with a structured high-level abstraction:
- Low-level (LL): colored images at resolution R x R (default 32), shapes (grayscale) in [-1,1], digits, colors.
- High-level (HL): the SAME colored images but downsampled to H x H (default 16) via 2x2 average pooling.

Key points:
- Pass experiment + noise params via args or CLI.
- LL colored images in [0,1]; LL shapes in [-1,1].
- HL images are in [0,1], shape (3, H, H) with H = hl_downsample_size.
- Noise and jitters applied in [0,1] and gated by NOISE_ENABLE (LL stage only).
"""

import os
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
from operations import Intervention  # provided in your project
import argparse

# ----------------------------
# Variable names (LL / HL)
# ----------------------------
D_LL, C_LL, P_LL = 'Digit', 'Color', 'Pixels'
D_HL, C_HL, I_HL = 'Digit_', 'Color_', 'Image_'  # kept for completeness (mapping labels stay identity)

# ----------------------------
# Stable intervention labels (order matters)
# ----------------------------
INTERVENTION_LABELS = [
    "obs",          # iota0 / eta0 (None)
    "D=6",          # iota1 / eta1
    "D=8",          # iota2 / eta2
    "D=4",          # iota3 / eta3
    "C=7",          # iota4 / eta4
    "C=0",          # iota5 / eta5
    "C=4",          # iota6 / eta6
    "D=6,C=7",      # iota7 / eta7
    "D=8,C=0",      # iota8 / eta8
    "D=4,C=4",      # iota9 / eta9
]

# ----------------------------
# Utils
# ----------------------------
def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def make_ll_interventions() -> Dict[str, Optional[Intervention]]:
    """Return mapping label -> LL Intervention object."""
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
    """Return mapping label -> HL Intervention object (for completeness)."""
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
    """
    Return (omega_labels, ll_interventions, hl_interventions) where
    - omega_labels: dict[str,str] mapping LL labels -> HL labels (identity here)
    - *_interventions: dict[str, Intervention]
    """
    ll_int = make_ll_interventions()
    hl_int = make_hl_interventions()
    omega_labels = {lab: lab for lab in INTERVENTION_LABELS}
    return omega_labels, ll_int, hl_int


# ----------------------------
# Generator
# ----------------------------
class ColorMNISTDataGenerator:
    """
    Generates samples for the Colored MNIST experiment.
    - Shapes (grayscale) in [-1,1] at (image_size x image_size).
    - Colored images in [0,1] at (image_size x image_size).
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

        # Noise settings
        self.noise_enable = bool(noise_enable)
        self.sensor_noise_std = float(sensor_noise_std)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.color_jitter_std = float(color_jitter_std)

        # RGB per color class (0..9)
        self.colors = {
            0: (1.0, 0.0, 0.0), 1: (1.0, 0.6, 0.0), 2: (0.8, 1.0, 0.0),
            3: (0.2, 1.0, 0.0), 4: (0.0, 1.0, 0.4), 5: (0.0, 1.0, 1.0),
            6: (0.0, 0.4, 1.0), 7: (0.2, 0.0, 1.0), 8: (0.8, 0.0, 1.0),
            9: (1.0, 0.0, 0.6)
        }

        # Transform for the original grayscale shape -> [-1,1]
        self.shape_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),                 # [0,1]
            transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
        ])

        # Load and bucket MNIST by digit
        print("Loading and preparing MNIST dataset...")
        mnist = torchvision.datasets.MNIST('data/', train=True, download=True)
        self.mnist_data: Dict[int, List[torch.Tensor]] = {i: [] for i in range(10)}
        for img, target in zip(mnist.data, mnist.targets):
            self.mnist_data[int(target)].append(img)  # uint8 (H,W)
        print("MNIST dataset ready.")

    # ---- noise helpers (instance methods; operate in [0,1]) ----
    @torch.no_grad()
    def _apply_photometric_jitter(self, x3chw: torch.Tensor) -> torch.Tensor:
        """
        x in [0,1], shape (3,H,W). Applies brightness/contrast and small color jitter.
        """
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
        """
        x in [0,1], shape (3,H,W). Adds small Gaussian readout noise and clamps.
        """
        if self.sensor_noise_std <= 0:
            return x3chw
        noise = torch.randn_like(x3chw) * self.sensor_noise_std
        return clamp01(x3chw + noise)

    # ------------------------------------------------------------

    def _sample_digit_color(self, intervention: Optional[Intervention]):
        iv_dict = intervention.vv() if intervention is not None else {}
        # sample confounder
        u_conf = np.random.randint(10)
        # decide digit/color
        digit = iv_dict.get(D_LL, u_conf if np.random.random() < self.correlation else np.random.randint(10))
        color = iv_dict.get(C_LL, u_conf if np.random.random() < self.correlation else np.random.randint(10))
        return int(digit), int(color)

    def generate_samples(self, n: int, intervention: Optional[Intervention] = None):
        """
        Returns (final_images, img_shapes, digits, colors)
        - final_images: (N,3,H,W) in [0,1]  with H=W=self.image_size
        - img_shapes:   (N,1,H,W) in [-1,1] (same self.image_size)
        - digits/colors: Long tensors
        """
        final_images, img_shapes, digits, colors = [], [], [], []

        for _ in range(n):
            digit, color = self._sample_digit_color(intervention)
            # pick MNIST sample for that digit
            idx = np.random.randint(len(self.mnist_data[digit]))
            original_shape_uint8: torch.Tensor = self.mnist_data[digit][idx]  # (H,W), uint8

            # Shape path ([-1,1])
            shape_tensor = self.shape_transform(original_shape_uint8)  # (1,H,W)
            img_shapes.append(shape_tensor)

            # Colored path ([0,1]) — colorize BEFORE resize, then resize tensor
            img_gray01 = (original_shape_uint8.to(torch.float32) / 255.0).unsqueeze(0)  # (1,H0,W0)
            img_color = img_gray01.repeat(3, 1, 1)           # (3,H0,W0)
            r, g, b = self.colors[color]
            img_color[0].mul_(r)
            img_color[1].mul_(g)
            img_color[2].mul_(b)

            # Resize tensor to (H,W) with antialias
            final_img_tensor = TF.resize(img_color, [self.image_size, self.image_size], antialias=True)  # [0,1]

            # Optional photometric jitter + sensor noise in [0,1]
            if self.noise_enable:
                final_img_tensor = self._apply_photometric_jitter(final_img_tensor)
                final_img_tensor = self._apply_sensor_noise(final_img_tensor)

            final_images.append(final_img_tensor)
            digits.append(digit)
            colors.append(color)

        return (
            torch.stack(final_images, dim=0).to(torch.float32),   # (N,3,H,W) in [0,1]
            torch.stack(img_shapes, dim=0).to(torch.float32),     # (N,1,H,W) in [-1,1]
            torch.tensor(digits, dtype=torch.long),
            torch.tensor(colors, dtype=torch.long),
        )


# ----------------------------
# Abstraction (HL) = downsampled images 16x16x3
# ----------------------------
def low_to_high_downsample(
    ll_samples_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    hl_downsample_size: int = 16,
) -> torch.Tensor:
    """
    Map low-level colored images (N,3,H,W) to high-level colored images (N,3,hH,hW),
    by 2x2 average pooling (implemented via antialiased resize).

    Returns:
        HL_images: (N,3,hl_downsample_size,hl_downsample_size) in [0,1]
    """
    final_images, _, _, _ = ll_samples_tuple  # final_images in [0,1]
    if final_images.ndim != 4 or final_images.size(1) != 3:
        raise ValueError(f"Expected final_images of shape (N,3,H,W); got {tuple(final_images.shape)}")
    # Use TF.resize with antialias=True to emulate average pooling downsample
    hl_imgs = TF.resize(final_images, [hl_downsample_size, hl_downsample_size], antialias=True)  # keeps [0,1]
    return hl_imgs.to(torch.float32)


# ----------------------------
# Driver
# ----------------------------
def generate_cmnist_data(
    n_samples: int = 1000,
    save_data: bool = True,
    output_dir: str = 'data/cmnist',
    seed: int = 42,
    resolution: int = 32,
    correlation: float = 0.0,
    noise_enable: bool = False,
    sensor_noise_std: float = 0.03,
    brightness_jitter: float = 0.05,
    contrast_jitter: float = 0.05,
    color_jitter_std: float = 0.015,
    hl_downsample_size: int = 16,
):
    """
    Generate CMNIST data across all interventions with fully parameterized settings.

    Saves:
      - dll_samples.pkl : dict[label] -> (final_images[0,1], img_shapes[-1,1], digits, colors)
      - dhl_samples.pkl : dict[label] -> HL_images (N,3,hl_downsample_size,hl_downsample_size) in [0,1]
      - intervention_mapping.pkl/json : dict[str,str] omega_labels (LL label -> HL label)
    """
    set_global_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    omega_labels, ll_ints, hl_ints = define_interventions()
    gen = ColorMNISTDataGenerator(
        image_size=resolution,
        correlation=correlation,
        noise_enable=noise_enable,
        sensor_noise_std=sensor_noise_std,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        color_jitter_std=color_jitter_std,
    )

    print(f"Generating {n_samples} samples for each of {len(INTERVENTION_LABELS)} low-level interventions...")
    Dll_samples: Dict[str, Tuple[torch.Tensor, ...]] = {}
    for lab in tqdm(INTERVENTION_LABELS, desc="Generating Low-Level Data"):
        Dll_samples[lab] = gen.generate_samples(n_samples, intervention=ll_ints[lab])

    print("\nGenerating high-level data by applying 2x2 pooling (downsample to "
          f"{hl_downsample_size}x{hl_downsample_size})...")
    Dhl_samples: Dict[str, torch.Tensor] = {}
    for lab in tqdm(INTERVENTION_LABELS, desc="Generating High-Level Data"):
        Dhl_samples[lab] = low_to_high_downsample(Dll_samples[lab], hl_downsample_size=hl_downsample_size)

    # Quick diagnostics on ranges (LL + HL)
    obs_ll_images = Dll_samples["obs"][0]
    imin, imax = obs_ll_images.min().item(), obs_ll_images.max().item()
    print(f"[Range check] LL colored images (obs) in [{imin:.3f}, {imax:.3f}] (expected [0,1])")
    if not (-1e-6 <= imin <= 0.0 + 1e-6 and 1.0 - 1e-6 <= imax <= 1.0 + 1e-6):
        print("WARNING: LL colored images are not strictly in [0,1]. Check the pipeline.")

    obs_hl_images = Dhl_samples["obs"]
    hmin, hmax = obs_hl_images.min().item(), obs_hl_images.max().item()
    print(f"[Range check] HL downsampled images (obs) in [{hmin:.3f}, {hmax:.3f}] (expected [0,1])")

    if save_data:
        print(f"\nSaving data to {output_dir}...")
        torch.save(Dll_samples, os.path.join(output_dir, 'dll_samples.pkl'))
        torch.save(Dhl_samples, os.path.join(output_dir, 'dhl_samples.pkl'))
        torch.save(omega_labels, os.path.join(output_dir, 'intervention_mapping.pkl'))
        with open(os.path.join(output_dir, 'intervention_mapping.json'), 'w') as f:
            json.dump(omega_labels, f, indent=2)
        print("Data saved successfully!")

    return Dll_samples, Dhl_samples, omega_labels


def build_argparser():
    p = argparse.ArgumentParser(description="Generate Colored MNIST with structured HL downsampling.")
    p.add_argument("--resolution", type=int, default=32, help="LL output image resolution (HxW).")
    p.add_argument("--hl-size", type=int, default=16, help="HL downsampled resolution (HxW).")
    p.add_argument("--correlation", type=float, default=0.0, help="Digit–color correlation in [0,1].")
    p.add_argument("--n-samples", type=int, default=1000, help="Samples per intervention.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Noise & jitter
    p.add_argument("--noise-enable", action="store_true", help="Enable photometric jitter and sensor noise.")
    p.add_argument("--sensor-noise-std", type=float, default=0.03, help="Std of Gaussian readout noise in [0,1].")
    p.add_argument("--brightness-jitter", type=float, default=0.05, help="Multiplicative brightness jitter amplitude.")
    p.add_argument("--contrast-jitter", type=float, default=0.05, help="Multiplicative contrast jitter amplitude.")
    p.add_argument("--color-jitter-std", type=float, default=0.015, help="Per-channel additive jitter std in [0,1].")

    # IO
    p.add_argument("--output-dir", type=str, default="data/cmnist", help="Output directory.")
    p.add_argument("--no-save", action="store_true", help="Do not save artifacts to disk.")
    return p


def main():
    args = build_argparser().parse_args()

    print("ColorMNIST Data Generator (HL = 2x2 pooled LL)")
    print("=" * 50)
    print(f"  - LL Resolution: {args.resolution}x{args.resolution}")
    print(f"  - HL Resolution: {args.hl_size}x{args.hl_size}")
    print(f"  - Correlation: {args.correlation}")
    print(f"  - Samples per intervention: {args.n_samples}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Noise enabled: {args.noise_enable} (σ={args.sensor_noise_std}, "
          f"brightness±{args.brightness_jitter}, contrast±{args.contrast_jitter}, "
          f"color_std={args.color_jitter_std})")

    Dll_samples, Dhl_samples, omega_labels = generate_cmnist_data(
        n_samples=args.n_samples,
        save_data=not args.no_save,
        output_dir=args.output_dir,
        seed=args.seed,
        resolution=args.resolution,
        correlation=args.correlation,
        noise_enable=args.noise_enable,
        sensor_noise_std=args.sensor_noise_std,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
        color_jitter_std=args.color_jitter_std,
        hl_downsample_size=args.hl_size,
    )

    print("\nDone.")
    print(f"Generated data for {len(omega_labels)} intervention labels.")
    return Dll_samples, Dhl_samples, omega_labels


if __name__ == "__main__":
    main()
