#!/usr/bin/env python3
"""
ColorMNIST Data Generator

This script generates colored MNIST data for causal abstraction experiments.
It creates both low-level (pixel-level) and high-level (abstracted) datasets
with various interventions to test causal abstraction methods.

Based on the Colored MNIST experiment from the DiRoCA_TBS project.
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
from operations import Intervention


# --- Experiment Configuration ---
RESOLUTION = 32
CORRELATION = 0.0
N_SAMPLES = 1000
K_FOLDS = 5
SEED = 42

# Variable names
# Low-Level (LL)
D_LL, C_LL, P_LL = 'Digit', 'Color', 'Pixels'
# High-Level (HL)
D_HL, C_HL, I_HL = 'Digit_', 'Color_', 'Image_'


class ColorMNISTDataGenerator:
    """
    Generates samples for the Colored MNIST experiment based on the specified
    causal structure and spurious correlation.
    """
    
    def __init__(self, image_size=RESOLUTION, correlation=CORRELATION):
        self.correlation = correlation
        self.image_size = image_size
        
        # Defines the color for each digit label 0-9
        self.colors = {
            0: (1.0, 0.0, 0.0), 1: (1.0, 0.6, 0.0), 2: (0.8, 1.0, 0.0),
            3: (0.2, 1.0, 0.0), 4: (0.0, 1.0, 0.4), 5: (0.0, 1.0, 1.0),
            6: (0.0, 0.4, 1.0), 7: (0.2, 0.0, 1.0), 8: (0.8, 0.0, 1.0),
            9: (1.0, 0.0, 0.6)
        }
        
        # Transform for the final colored image
        self.color_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        
        # Transform for the original grayscale shape
        self.shape_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize shape to [-1, 1]
        ])
        
        # Load and pre-sort MNIST data for efficient sampling
        print("Loading and preparing MNIST dataset...")
        mnist = torchvision.datasets.MNIST('data/', train=True, download=True)
        self.mnist_data = {i: [] for i in range(10)}
        for img, target in zip(mnist.data, mnist.targets):
            self.mnist_data[target.item()].append(img)
        print("MNIST dataset ready.")

    def generate_samples(self, n, intervention=None):
        """
        The core causal data generation process happens here.
        Returns: (final_images, img_shapes, digits, colors)
        """
        final_images, img_shapes, digits, colors = [], [], [], []

        for _ in range(n):
            # 1. Sample the unobserved confounder U_CD
            u_conf = np.random.randint(10)

            # 2. Determine digit and color based on confounder, correlation, and intervention
            iv_dict = intervention.vv() if intervention is not None else {}
            digit = iv_dict.get(D_LL, u_conf if np.random.random() < self.correlation else np.random.randint(10))
            color = iv_dict.get(C_LL, u_conf if np.random.random() < self.correlation else np.random.randint(10))

            # 3. Generate the image I based on parents D and C
            idx = np.random.randint(len(self.mnist_data[digit]))
            original_shape = self.mnist_data[digit][idx]  # D -> I (shape)

            # Process the original grayscale shape
            shape_tensor = self.shape_transform(original_shape)
            img_shapes.append(shape_tensor)

            # C -> I (color)
            color_rgb = self.colors[color]
            # Create a 3-channel image and apply color
            img_color = torch.tensor(original_shape).float().unsqueeze(0).repeat(3, 1, 1)
            for c in range(3):
                img_color[c] *= color_rgb[c]

            # Process the final colored image
            final_img_tensor = self.color_transform(img_color)
            final_images.append(final_img_tensor)
            
            digits.append(digit)
            colors.append(color)

        # Stack the tensors and return them separately
        return (
            torch.stack(final_images),
            torch.stack(img_shapes),
            torch.tensor(digits, dtype=torch.long),
            torch.tensor(colors, dtype=torch.long)
        )


def low_to_high_level(ll_samples_tuple):
    """
    Abstraction function that maps low-level data to high-level representation.
    Converts pixel-level data to abstract features.
    """
    final_images, _, digits, colors = ll_samples_tuple
    
    # Extract image feature (mean intensity across all pixels)
    image_feature = final_images.mean(dim=[1, 2, 3]).unsqueeze(1)
    
    # One-hot encode digits and colors
    digits_onehot = torch.nn.functional.one_hot(digits, num_classes=10).float()
    colors_onehot = torch.nn.functional.one_hot(colors, num_classes=10).float()
    
    # Concatenate all features: [digits_onehot, colors_onehot, image_feature]
    hl_samples = torch.cat([digits_onehot, colors_onehot, image_feature], axis=1)
    
    return hl_samples


def define_interventions():
    """Define all low-level and high-level interventions."""
    
    # Low-Level Interventions (iota)
    iota0 = None
    iota1 = Intervention({D_LL: 6})
    iota2 = Intervention({D_LL: 8})
    iota3 = Intervention({D_LL: 4})
    iota4 = Intervention({C_LL: 7})
    iota5 = Intervention({C_LL: 0})
    iota6 = Intervention({C_LL: 4})
    iota7 = Intervention({D_LL: 6, C_LL: 7})
    iota8 = Intervention({D_LL: 8, C_LL: 0})
    iota9 = Intervention({D_LL: 4, C_LL: 4})

    # High-Level Interventions (eta)
    eta0 = None
    eta1 = Intervention({D_HL: 6})
    eta2 = Intervention({D_HL: 8})
    eta3 = Intervention({D_HL: 4})
    eta4 = Intervention({C_HL: 7})
    eta5 = Intervention({C_HL: 0})
    eta6 = Intervention({C_HL: 4})
    eta7 = Intervention({D_HL: 6, C_HL: 7})
    eta8 = Intervention({D_HL: 8, C_HL: 0})
    eta9 = Intervention({D_HL: 4, C_HL: 4})

    # The ground-truth mapping between interventions
    omega = {
        iota0: eta0, iota1: eta1, iota2: eta2, iota3: eta3, iota4: eta4,
        iota5: eta5, iota6: eta6, iota7: eta7, iota8: eta8, iota9: eta9
    }

    # The set of relevant interventions for the experiment
    Ill_relevant = list(omega.keys())
    Ihl_relevant = list(omega.values())
    
    return omega, Ill_relevant, Ihl_relevant


def generate_cmnist_data(n_samples=N_SAMPLES, save_data=True, output_dir='data/cmnist'):
    """
    Generate ColorMNIST data for all interventions.
    
    Args:
        n_samples: Number of samples to generate per intervention
        save_data: Whether to save the generated data to disk
        output_dir: Directory to save the data
    
    Returns:
        tuple: (Dll_samples, Dhl_samples, omega) - low-level data, high-level data, and intervention mapping
    """
    
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Create output directory if it doesn't exist
    if save_data:
        os.makedirs(output_dir, exist_ok=True)
    
    # Define interventions
    omega, Ill_relevant, Ihl_relevant = define_interventions()
    
    # Instantiate the data generator
    data_generator = ColorMNISTDataGenerator()
    
    # Generate all low-level interventional datasets
    print(f"Generating {n_samples} samples for each of the {len(Ill_relevant)} low-level interventions...")
    Dll_samples = {}
    for iota in tqdm(Ill_relevant, desc="Generating Low-Level Data"):
        Dll_samples[iota] = data_generator.generate_samples(n_samples, intervention=iota)
    
    # Generate all high-level interventional datasets by applying the abstraction
    print(f"\nGenerating high-level data by applying the abstraction map...")
    Dhl_samples = {}
    for iota, eta in tqdm(omega.items(), desc="Generating High-Level Data"):
        Dhl_samples[eta] = low_to_high_level(Dll_samples[iota])
    
    # Verification
    print("Data generation and abstraction complete.")
    obs_ll_images_shape = Dll_samples[None][0].shape
    obs_ll_shapes_shape = Dll_samples[None][1].shape
    obs_hl_shape = Dhl_samples[None].shape

    print(f"   - Shape of low-level COLORED IMAGES: {obs_ll_images_shape} (N, C, H, W)")
    print(f"   - Shape of low-level ORIGINAL SHAPES: {obs_ll_shapes_shape} (N, C, H, W)")
    print(f"   - Shape of high-level data (D_, C_, I_): {obs_hl_shape}")
    
    # Save data if requested
    if save_data:
        print(f"\nSaving data to {output_dir}...")
        torch.save(Dll_samples, os.path.join(output_dir, 'dll_samples.pkl'))
        torch.save(Dhl_samples, os.path.join(output_dir, 'dhl_samples.pkl'))
        torch.save(omega, os.path.join(output_dir, 'intervention_mapping.pkl'))
        print("Data saved successfully!")
    
    return Dll_samples, Dhl_samples, omega


def main():
    """Main function to run the ColorMNIST data generation."""
    print("ColorMNIST Data Generator")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  - Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"  - Correlation: {CORRELATION}")
    print(f"  - Samples per intervention: {N_SAMPLES}")
    print(f"  - Random seed: {SEED}")
    print()
    
    # Generate the data
    Dll_samples, Dhl_samples, omega = generate_cmnist_data()
    
    print("\nData generation completed successfully!")
    print(f"Generated data for {len(omega)} intervention pairs.")
    
    return Dll_samples, Dhl_samples, omega


if __name__ == "__main__":
    # Run the data generation
    Dll_samples, Dhl_samples, omega = main()
