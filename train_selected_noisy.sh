#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT=cmnist_train.py   # your trainer

SELECTED=(
  data/cmnist32_c000_noise0
  data/cmnist32_c000_noise1
  data/cmnist32_c000_noise2
  data/cmnist32_c000_noise3
  data/cmnist32_c000_noise4
  data/cmnist32_c000_noise5
  data/cmnist32_c000_noise6
  data/cmnist32_c000_noise7
  data/cmnist32_c000_noise8
  data/cmnist32_c000_noise9
  data/cmnist32_c000_noise10
  data/cmnist32_c000_noise11
  data/cmnist32_c000_noise12
  data/cmnist32_c000_noise13
  data/cmnist32_c000_noise14
  data/cmnist32_c000_noise15
  data/cmnist32_c000_noise16
)

echo "Found ${#SELECTED[@]} dataset(s)."
for dir in "${SELECTED[@]}"; do
  echo ">>> Training on: $dir"

  if [[ ! -f "$dir/dll_samples.pkl" || ! -f "$dir/dhl_samples.pkl" ]]; then
    echo "  ⚠️  Missing dll_samples.pkl/dhl_samples.pkl — skipping."
    continue
  fi

  # Skip if already trained (remove this block to force retrain)
  if [[ -f "$dir/ll_model_unet.pth" && -f "$dir/hl_model_cellmeans.joblib" ]]; then
    echo "  ↪️  Models already present. Skipping."
    continue
  fi

  $PY "$SCRIPT" --input-dir "$dir" --save-dir "$dir" 2>&1 | tee "$dir/train.log"
done

echo "✅ All selected noisy trainings finished."
