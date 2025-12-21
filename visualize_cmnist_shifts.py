#!/usr/bin/env python3
import argparse
import os
import pickle
import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. Setup & Style
# ============================================================
def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "text.usetex": False,                 
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",             
        "mathtext.rm": "serif",
        "axes.titlesize": 16,
        "axes.labelsize": 14
    })

# ============================================================
# 2. Data Loading
# ============================================================
def load_data(eval_dir):
    path = os.path.join(eval_dir, "cmnist_Dll_samples.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    print(f"Loading samples from: {path}")
    with open(path, "rb") as f:
        Dll_samples = pickle.load(f)
    
    # Extract images from the first available key
    first_key = list(Dll_samples.keys())[0]
    ll_imgs, _, _, _ = Dll_samples[first_key]
    return ll_imgs

# ============================================================
# 3. Transform Logic
# ============================================================
def build_camera_transform(rotation_deg, brightness_factor):
    transforms = []
    
    # 1. Rotation (Deterministic)
    if rotation_deg != 0:
        transforms.append(T.RandomAffine(
            degrees=(rotation_deg, rotation_deg), 
            translate=(0.1, 0.1), 
            scale=None, 
            fill=0 
        ))
    
    # 2. Brightness (Deterministic)
    if brightness_factor != 1.0:
        transforms.append(T.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))
        
    if len(transforms) == 0:
        return T.Lambda(lambda x: x)
    return T.Compose(transforms)

def apply_transform_batch(X_flat, transform, seed):
    """Applies rotation/lighting in [0,1] space."""
    N = X_flat.shape[0]
    X_img = X_flat.view(N, 3, 32, 32)
    
    # Denormalize [-1,1] -> [0,1] for correct lighting/fill
    X_01 = (X_img + 1) * 0.5
    X_shifted = torch.empty_like(X_01)
    
    base_seed = int(seed) % (2**32)
    for i in range(N):
        s = (base_seed + i) % (2**32)
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        X_shifted[i] = transform(X_01[i])
    
    # Renormalize -> [-1,1]
    X_out = X_shifted * 2 - 1
    return torch.clamp(X_out, -1, 1).view(N, -1)

def apply_huber_noise(X_flat, alpha, noise_scale, seed):
    """Applies Huber contamination (alpha=0 is clean, alpha=1 is noisy)."""
    if alpha <= 0: return X_flat
    
    N = X_flat.shape[0]
    base_seed = int(seed) % (2**32)
    torch.manual_seed(base_seed); np.random.seed(base_seed); random.seed(base_seed)
    
    mask = torch.rand(N, 1) < alpha
    noise = torch.randn_like(X_flat) * noise_scale
    X_corrupt = torch.where(mask, X_flat + noise, X_flat)
    return torch.clamp(X_corrupt, -1, 1)

# ============================================================
# 4. Main Execution
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize CMNIST Shifts (Rotation, Lighting, Huber)")
    
    # Data params
    parser.add_argument("--eval-dir", default="data/cmnist/results_empirical", help="Path to data directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of digits to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection and noise")
    
    # Shift params
    parser.add_argument("--rotation", type=float, default=90.0, help="Rotation degrees (e.g. 30.0, 90.0)")
    parser.add_argument("--lighting", type=float, default=3.0, help="Brightness factor (e.g. 0.6, 0.2, 3.0)")
    parser.add_argument("--huber-alpha", type=float, default=1.0, help="Huber probability (0.0 to 1.0)")
    parser.add_argument("--noise-scale", type=float, default=0.5, help="Huber noise intensity")
    
    args = parser.parse_args()
    
    setup_style()
    
    # 1. Load Data
    try:
        raw_images_pool = load_data(args.eval_dir)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Select Random Samples
    np.random.seed(args.seed)
    indices = np.random.choice(len(raw_images_pool), args.num_samples, replace=False)
    batch_raw = raw_images_pool[indices]
    
    # Flatten for processing
    X_flat = (batch_raw * 2 - 1).view(args.num_samples, -1)
    
    # 3. Apply Camera Shift (Rotation + Lighting)
    print(f"Applying Rotation={args.rotation}° and Lighting={args.lighting}x...")
    tf = build_camera_transform(args.rotation, args.lighting)
    X_shifted = apply_transform_batch(X_flat, tf, seed=args.seed)
    
    # 4. Apply Huber Noise
    print(f"Applying Huber Noise (α={args.huber_alpha})...")
    X_final = apply_huber_noise(X_shifted, args.huber_alpha, noise_scale=args.noise_scale, seed=args.seed)

    # 5. Prepare for Plotting (Reshape & Scale to 0-1)
    def to_plot(tensor):
        img = tensor.view(-1, 3, 32, 32).permute(0, 2, 3, 1).numpy()
        return np.clip((img + 1) / 2.0, 0, 1)

    imgs_orig = to_plot(X_flat)
    imgs_shift = to_plot(X_shifted)
    imgs_huber = to_plot(X_final)

    # 6. Plot
    fig, axes = plt.subplots(3, args.num_samples, figsize=(3 * args.num_samples, 10))
    
    rows = [imgs_orig, imgs_shift, imgs_huber]
    row_titles = [
        "Original", 
        f"Shifted\n(Rot={args.rotation}°, Light={args.lighting})", 
        f"Huber Contaminated\n(α={args.huber_alpha})"
    ]

    for r in range(3):
        # Add Row Label
        axes[r, 0].annotate(row_titles[r], xy=(0, 0.5), xytext=(-20, 0),
                            xycoords='axes fraction', textcoords='offset points',
                            size='large', ha='right', va='center', fontweight='bold',
                            fontfamily='serif')
        
        for c in range(args.num_samples):
            ax = axes[r, c]
            ax.imshow(rows[r][c])
            ax.axis('off')
            
    plt.tight_layout()
    output_file = "visualize_shifts.png"
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()