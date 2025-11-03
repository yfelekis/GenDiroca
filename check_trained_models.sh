#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking trained ColorMNIST models ==="
printf "%-30s %-10s %-10s\n" "Dataset" "LL" "HL"
echo "-----------------------------------------------"

for d in $(find data -maxdepth 1 -type d -name 'cmnist*' | sort); do
  ll=$(test -f "$d/ll_model_unet.pth" && echo "✅" || echo "❌")
  hl=$(test -f "$d/hl_model_cellmeans.joblib" && echo "✅" || echo "❌")
  printf "%-30s %-10s %-10s\n" "$(basename "$d")" "$ll" "$hl"
done

echo "-----------------------------------------------"
echo "✅ Check complete."
