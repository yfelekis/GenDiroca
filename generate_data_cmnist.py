#!/usr/bin/env python3
"""
Unified CMNIST Generator (LL images + z via TorchScript encoder)

- Low level (LL): generate Colored MNIST images:
    * load MNIST via python-mnist from dat/mnist/
    * colorize with the same 10-color palette
    * Resize -> ToTensor -> scale to [-1, 1]
  We then convert to [0,1] ONLY for saving in Dll_samples (compatibility).

- High level (HL): z = encoder_ts(I_P_norm) where I_P_norm is the [-1,1] image.
  We pack [onehot(D) | onehot(C) | z].

Outputs (same as before):
  - dll_samples.pkl : dict[str] -> (final_images[0,1], img_shapes[-1,1], digits, colors)
  - dhl_samples.pkl : dict[str] -> (N, 84) with [10 digit one-hots | 10 color one-hots | 64-dim z]
  - intervention_mapping.pkl / .json
"""

import os
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

from operations import Intervention

# Variable names (LL / HL)
D_LL, C_LL, P_LL = 'Digit', 'Color', 'Pixels'
D_HL, C_HL, I_HL = 'Digit_', 'Color_', 'Image_'

# Stable intervention labels 
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

def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    """Return mapping label -> HL Intervention object."""
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

# -----------------------------
# color MNIST maker
# -----------------------------

class XiaAlignedColorMNIST:
    """
    Minimal re-implementation of Xia's ColorMNISTDataGenerator image path:
      * Load MNIST via python-mnist (files under dat/mnist/)
      * Colorize with same palette
      * Resize to (image_size, image_size)
      * ToTensor() -> [0,1] then map to [-1,1] if normalize=True

    We also produce a grayscale "shape" channel in [-1,1] (matching your LL tuple spec).
    """
    PALETTE = {
        0: (1.0, 0.0, 0.0),
        1: (1.0, 0.6, 0.0),
        2: (0.8, 1.0, 0.0),
        3: (0.2, 1.0, 0.0),
        4: (0.0, 1.0, 0.4),
        5: (0.0, 1.0, 1.0),
        6: (0.0, 0.4, 1.0),
        7: (0.2, 0.0, 1.0),
        8: (0.8, 0.0, 1.0),
        9: (1.0, 0.0, 0.6)
    }

    def __init__(self, image_size: int = 32, mnist_dir: str = "third_party/xia_nca/NeuralCausalAbstractions/dat/mnist"):
        from torchvision import transforms
        from mnist import MNIST

        self.image_size = int(image_size)
        self.transform_rgb = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size, antialias=True),
            transforms.ToTensor()  # -> [0,1]
        ])
        self.transform_shape = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size, antialias=True),
            transforms.ToTensor(),                 # [0,1]
            transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
        ])

        self.mnist_path = mnist_dir
        must_exist = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        ]
        for f in must_exist:
            if not os.path.exists(os.path.join(self.mnist_path, f)):
                raise FileNotFoundError(
                    f"MNIST file missing: {os.path.join(self.mnist_path, f)}\n"
                    f"Make sure the idx files are under this folder."
                )

        mn = MNIST(self.mnist_path)
        images, labels = mn.load_training()
        images = np.array(images).reshape((-1, 28, 28))
        labels = np.array(labels)

        # Bucket by digit for fast sampling
        self.images_by_digit: Dict[int, List[np.ndarray]] = {i: [] for i in range(10)}
        for x, y in zip(images, labels):
            self.images_by_digit[int(y)].append(x.astype(np.uint8))

    @staticmethod
    def _colorize_uint8(gray_uint8_hw: np.ndarray, rgb: Tuple[float, float, float]) -> np.ndarray:
        """gray_uint8_hw -> (H,W,3) uint8 multiply + rounding."""
        h, w = gray_uint8_hw.shape
        g = gray_uint8_hw.astype(np.float32) / 255.0  # 0..1
        r = np.clip(np.round(g * rgb[0] * 255.0), 0, 255)
        g2 = np.clip(np.round(g * rgb[1] * 255.0), 0, 255)
        b = np.clip(np.round(g * rgb[2] * 255.0), 0, 255)
        out = np.stack([r, g2, b], axis=-1).astype(np.uint8)
        return out  # (H,W,3) uint8

    def _sample_one(self, digit: int, color: int):
        """Return (img_rgb_norm, shape_norm, digit_int, color_int)."""
        import numpy.random as npr
        from torchvision.transforms.functional import to_pil_image

        # pick MNIST sample for that digit
        pool = self.images_by_digit[int(digit)]
        idx = npr.randint(len(pool))
        gray28 = pool[idx]  # (28,28) uint8

        # RGB colorization in uint8
        rgb = self._colorize_uint8(gray28, self.PALETTE[int(color)])

        # Resize/color path: ToTensor -> [0,1] -> scale to [-1,1]
        img_rgb_01 = self.transform_rgb(rgb)              # (3,H,W) in [0,1]
        img_rgb_norm = img_rgb_01 * 2.0 - 1.0            # [-1,1]

        # Shape path (grayscale) -> [-1,1]
        shape_01 = self.transform_shape.transforms[0:2]  # we’ll just re-run cleanly below
        
        gray28_3d = gray28.reshape(28, 28)               # uint8
        shape_norm = self.transform_shape(gray28_3d)     # (1,H,W) in [-1,1]

        return img_rgb_norm, shape_norm, int(digit), int(color)

    def sample_batch(self, n: int, digit: np.ndarray, color: np.ndarray):
        imgs_norm, shapes_norm, digits, colors = [], [], [], []
        for i in range(n):
            x, s, d, c = self._sample_one(int(digit[i]), int(color[i]))
            imgs_norm.append(x)
            shapes_norm.append(s)
            digits.append(d)
            colors.append(c)
        imgs_norm = torch.stack(imgs_norm, dim=0)        # (N,3,H,W) [-1,1]
        shapes_norm = torch.stack(shapes_norm, dim=0)    # (N,1,H,W) [-1,1]
        return imgs_norm, shapes_norm, torch.tensor(digits), torch.tensor(colors)

# -----------------------------
# Intervention & sampling utils
# -----------------------------

def _one_hot(idx: torch.Tensor, n: int = 10) -> torch.Tensor:
    return F.one_hot(idx.long(), num_classes=n).float()

def _expand_do(val, n):
    return np.ones(n, dtype=int) * int(val)

def _draw_u(n, p_align=0.85):
    u_conf = np.random.randint(10, size=n)
    u_digit = np.random.randint(10, size=n)
    u_color = np.random.randint(10, size=n)
    u_dig_align = np.random.binomial(1, p_align, size=n)
    u_color_align = np.random.binomial(1, p_align, size=n)
    return u_conf, u_digit, u_color, u_dig_align, u_color_align

def _apply_do_or_align(n, do: Optional[Intervention], p_align=0.85):
    U = _draw_u(n, p_align=p_align)
    u_conf, u_digit, u_color, u_dig_align, u_color_align = U

    if do is not None:
        do_map = do.vv()
    else:
        do_map = {}

    if D_LL in do_map:
        digit = _expand_do(do_map[D_LL], n)
    else:
        digit = np.where(u_dig_align == 1, u_conf, u_digit)

    if C_LL in do_map:
        color = _expand_do(do_map[C_LL], n)
    else:
        color = np.where(u_color_align == 1, u_conf, u_color)

    return digit.astype(int), color.astype(int)

# -----------------------------
# Main
# -----------------------------

def generate_cmnist_data_new(
    n_samples: int,
    encoder_ts_path: str,
    output_dir: str = "data/cmnist",
    seed: int = 42,
    resolution: int = 32,
    p_align: float = 0.85,
    mnist_dir: str = "third_party/xia_nca/NeuralCausalAbstractions/dat/mnist",
):
    """
    Generate CMNIST data LL images and HL = z from TorchScript encoder.
    Saves:
      - dll_samples.pkl : dict[str] -> (final_images[0,1], img_shapes[-1,1], digits, colors)
      - dhl_samples.pkl : dict[str] -> (N, 84)  [10 one-hot D | 10 one-hot C | 64-dim z]
      - intervention_mapping.pkl / .json
    """
    set_global_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Build intervention sets & omega mapping
    omega_labels, ll_ints, hl_ints = define_interventions()

    # image generator
    gen = XiaAlignedColorMNIST(image_size=resolution, mnist_dir=mnist_dir)

    # Load TorchScript encoder (your traced adapter: [B,3,H,W] -> [B,64])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.jit.load(encoder_ts_path, map_location=device).eval()

    # Per-intervention sampling, encode z, and pack outputs
    Dll_samples: Dict[str, Tuple[torch.Tensor, ...]] = {}
    Dhl_samples: Dict[str, torch.Tensor] = {}

    for lab in INTERVENTION_LABELS:
        print(f"[{lab}] generating {n_samples} samples...")
        digit_np, color_np = _apply_do_or_align(n_samples, ll_ints[lab], p_align=p_align)

        # Generate images/shapes in [-1,1]
        imgs_norm, shapes_norm, digits_t, colors_t = gen.sample_batch(n_samples, digit_np, color_np)  # [-1,1]

        # LL save format: images in [0,1], shapes in [-1,1]
        imgs_01 = (imgs_norm + 1.0) / 2.0
        Dll_samples[lab] = (
            imgs_01.to(torch.float32),         # (N,3,H,W) [0,1]
            shapes_norm.to(torch.float32),     # (N,1,H,W) [-1,1]
            digits_t.long(),                   # (N,)
            colors_t.long(),                   # (N,)
        )

        # HL: z = encoder([-1,1] images])
        with torch.no_grad():
            z = []
            bs = 512
            for i in range(0, n_samples, bs):
                x = imgs_norm[i:i+bs].to(device)
                z.append(encoder(x).cpu())
            z = torch.cat(z, dim=0)  # (N,64)

        # Pack HL tensor: [onehot(D) | onehot(C) | z]
        Xd = _one_hot(digits_t, 10)
        Xc = _one_hot(colors_t, 10)
        Dhl_samples[lab] = torch.cat([Xd, Xc, z], dim=1).to(torch.float32)  # (N, 20+64=84)

    # Range check (obs)
    obs_ll_images = Dll_samples["obs"][0]
    imin, imax = obs_ll_images.min().item(), obs_ll_images.max().item()
    print(f"[Range check] LL colored images (obs) in [{imin:.3f}, {imax:.3f}] (expected [0,1])")

    # Save
    torch.save(Dll_samples, os.path.join(output_dir, 'dll_samples.pkl'))
    torch.save(Dhl_samples, os.path.join(output_dir, 'dhl_samples.pkl'))
    torch.save(omega_labels, os.path.join(output_dir, 'intervention_mapping.pkl'))
    with open(os.path.join(output_dir, 'intervention_mapping.json'), 'w') as f:
        json.dump(omega_labels, f, indent=2)

    print(f"Saved to {output_dir}")
    return Dll_samples, Dhl_samples, omega_labels

# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Generate Xia-aligned CMNIST with z from TorchScript encoder.")
    p.add_argument("--encoder-ts", type=str, required=True,
                   help="Path to TorchScript encoder (rep_encoder_only_traced.pt).")
    p.add_argument("--output-dir", type=str, default="data/cmnist",
                   help="Where to save dll_samples.pkl / dhl_samples.pkl / intervention_mapping.*")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="Samples per intervention.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--p-align", type=float, default=0.85,
                   help="Probability that D and C align with u_conf.")
    p.add_argument("--mnist-dir", type=str,
                   default="third_party/xia_nca/NeuralCausalAbstractions/dat/mnist",
                   help="Folder with raw MNIST idx files (train-images-idx3-ubyte, etc.).")
    return p

def main():
    args = build_argparser().parse_args()
    print("CMNIST Generator — start")
    print("=" * 60)
    print(f"  encoder_ts: {args.encoder_ts}")
    print(f"  out dir   : {args.output_dir}")
    print(f"  per-int N : {args.n_samples}")
    print(f"  resolution: {args.resolution}")
    print(f"  p_align   : {args.p_align}")
    print(f"  mnist_dir : {args.mnist_dir}")
    generate_cmnist_data_new(
        n_samples=args.n_samples,
        encoder_ts_path=args.encoder_ts,
        output_dir=args.output_dir,
        seed=args.seed,
        resolution=args.resolution,
        p_align=args.p_align,
        mnist_dir=args.mnist_dir,
    )
    print("Done.")

if __name__ == "__main__":
    main()
