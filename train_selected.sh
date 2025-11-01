#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT=cmnist_train.py   # <— change to your actual training filename if different

SELECTED=(
  # 32×32
  data/cmnist32_c00_clean
  data/cmnist32_c25_clean
  data/cmnist32_c50_clean
  data/cmnist32_c85_clean

  # 16×16
  data/cmnist16_c00_clean
  data/cmnist16_c25_clean
  data/cmnist16_c50_clean
  data/cmnist16_c85_clean

  # 8×8
  data/cmnist8_c00_clean
  data/cmnist8_c25_clean
  data/cmnist8_c50_clean
  data/cmnist8_c85_clean
)

echo "Found ${#SELECTED[@]} target dataset(s)."

for dir in "${SELECTED[@]}"; do
  echo ">>> Training on: $dir"

  if [[ ! -f "$dir/dll_samples.pkl" || ! -f "$dir/dhl_samples.pkl" ]]; then
    echo "  ⚠️  Missing dataset files in $dir (need dll_samples.pkl and dhl_samples.pkl). Skipping."
    continue
  fi

  # Skip if models already exist; delete these lines if you want to retrain
  if [[ -f "$dir/ll_model_unet.pth" && -f "$dir/hl_model_cellmeans.joblib" ]]; then
    echo "  ↪️  Models already present. Skipping."
    continue
  fi

  $PY "$SCRIPT" --input-dir "$dir" --save-dir "$dir" 2>&1 | tee "$dir/train.log"
done

echo "✅ All selected trainings finished."
