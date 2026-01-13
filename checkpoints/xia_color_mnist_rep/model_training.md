# Model Training & Reproducibility Report

## Overview

This repository utilizes a pre-trained **Representational Encoder ($E_\phi$)** derived from the *Neural Causal Abstractions* framework (Xia et al., 2024). This document details the exact training procedure, hyperparameters, and environment used to produce the checkpoints provided in `checkpoints/`.

The included model achieves a test set auxiliary accuracy of **95.7%**, confirming that the learned latent space $z$ successfully captures the causal factors (Digit and Color).

---

## 1. Source Framework

- **Repository:** https://github.com/CausalAILab/NeuralCausalAbstractions  
- **Paper:** *Neural Causal Abstractions* — Xia et al., 2024  
- **Methodology:**  
  The encoder was trained using the `auto_enc_conditional` representation setting, which jointly optimizes:
  - Image reconstruction via an autoencoder objective
  - Causal disentanglement via GAN-based causal regularization

---

## 2. Training

```bash
!python -m src.main representational sampling mnist gan \
  --h-layers 3 \
  --h-size 2 \
  --scale-h-size \
  --scale-u-size \
  --batch-norm \
  --gan-mode wgan \
  --gan-arch biggan \
  --disc-type biggan \
  --img-size 32 \
  --repr auto_enc_conditional \
  --rep-size 64 \
  --rep-image-only \
  --rep-h-layers 3 \
  --rep-h-size 128 \
  --gpu 0
```
---

## 4. Hyperparameters

Based on the training command above, the encoder architecture is defined as follows. These settings are baked into the provided `rep_encoder_only_traced.pt` artifact.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input Shape | $32 \times 32 \times 3$ | RGB images scaled to $[-1, 1]$ |
| Latent Dim ($z$) | 64 | Output representation size |
| Hidden Layers | 3 | Depth of encoder network |
| Hidden Size | 128 | Width of hidden layers |
| Representation | `auto_enc_conditional` | AE + causal GAN objective |
| GAN Mode | `wgan` | Wasserstein GAN |
| GAN Architecture | `biggan` | Generator and discriminator |
| Batch Norm | Enabled | Stabilizes training |

---

## 5. Artifact Extraction

The full training process produces a PyTorch Lightning checkpoint (`.ckpt`).

To enable lightweight downstream usage (e.g., in `generate_data_cmnist.py`), the encoder module was extracted and traced using TorchScript.

### Source Checkpoint

```
epoch=961-step=225108.ckpt
```

### Extraction Procedure

1. Instantiate `RepresentationalNN` using the hyperparameters above
2. Load checkpoint weights
3. Isolate the image encoder:
   ```python
   model.encoders.image
   ```
4. Freeze and trace with TorchScript using dummy input:
   ```python
   dummy = torch.randn(1, 3, 32, 32)
   traced = torch.jit.trace(encoder, dummy)
   ```
5. Save artifact

### Final Artifact

```
rep_encoder_only_traced.pt
```

This artifact contains a fully self-contained TorchScript encoder that maps:

$$x \in \mathbb{R}^{3 \times 32 \times 32} \rightarrow z \in \mathbb{R}^{64}$$

---

