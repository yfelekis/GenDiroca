#!/usr/bin/env bash
set -euo pipefail

# Optional: activate environment
# source ~/venvs/your_env/bin/activate

echo "=== Generating noisy ColorMNIST variants (32x32 & 16x16) ==="

# Define grids
RESOLUTIONS=(32 16)
CORRELATIONS=(0.00 0.25 0.50 0.85)

# Noise parameter sets (20 total configs)
declare -a NOISE_SETTINGS=(
  # sensor std, brightness, contrast, color jitter
  "0.01 0.02 0.02 0.005"
  "0.02 0.02 0.02 0.010"
  "0.03 0.03 0.03 0.010"
  "0.04 0.03 0.03 0.015"
  "0.05 0.05 0.05 0.015"
  "0.06 0.05 0.05 0.020"
  "0.08 0.05 0.05 0.025"
  "0.10 0.05 0.05 0.030"
  "0.02 0.05 0.00 0.010"
  "0.02 0.00 0.05 0.010"
  "0.02 0.10 0.10 0.010"
  "0.03 0.08 0.08 0.015"
  "0.04 0.08 0.08 0.020"
  "0.05 0.10 0.10 0.025"
  "0.06 0.10 0.10 0.030"
  "0.08 0.15 0.15 0.030"
  "0.10 0.15 0.15 0.035"
  "0.12 0.20 0.20 0.040"
  "0.15 0.20 0.20 0.050"
  "0.20 0.25 0.25 0.060"
)

# Loop over combinations
for res in "${RESOLUTIONS[@]}"; do
  for corr in "${CORRELATIONS[@]}"; do
    for i in "${!NOISE_SETTINGS[@]}"; do
      read sensor std_bright std_contrast std_color <<< "${NOISE_SETTINGS[$i]}"
      OUTDIR="data/cmnist${res}_c${corr/./}_noise${i}"
      echo "→ Generating ${OUTDIR} (σ=${sensor}, b=${std_bright}, c=${std_contrast}, j=${std_color})"
      
      python cmnist_gen.py \
        --resolution ${res} \
        --correlation ${corr} \
        --n-samples 1000 \
        --noise-enable \
        --sensor-noise-std ${sensor} \
        --brightness-jitter ${std_bright} \
        --contrast-jitter ${std_contrast} \
        --color-jitter-std ${std_color} \
        --output-dir ${OUTDIR}
    done
  done
done

echo "✅ All noisy CMNIST (32x32 & 16x16) variants generated successfully."
