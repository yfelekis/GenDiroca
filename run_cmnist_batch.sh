#!/usr/bin/env bash
set -euo pipefail

# Optional: activate your environment
# source ~/venvs/your_env/bin/activate

# ===== 32x32 (clean, different correlations) =====
python cmnist_gen.py --resolution 32 --correlation 0.00 --n-samples 1000 --output-dir data/cmnist32_c00_clean
python cmnist_gen.py --resolution 32 --correlation 0.25 --n-samples 1000 --output-dir data/cmnist32_c25_clean
python cmnist_gen.py --resolution 32 --correlation 0.50 --n-samples 1000 --output-dir data/cmnist32_c50_clean
python cmnist_gen.py --resolution 32 --correlation 0.85 --n-samples 1000 --output-dir data/cmnist32_c85_clean

# ===== 16x16 (clean) =====
python cmnist_gen.py --resolution 16 --correlation 0.00 --n-samples 1000 --output-dir data/cmnist16_c00_clean
python cmnist_gen.py --resolution 16 --correlation 0.25 --n-samples 1000 --output-dir data/cmnist16_c25_clean
python cmnist_gen.py --resolution 16 --correlation 0.50 --n-samples 1000 --output-dir data/cmnist16_c50_clean
python cmnist_gen.py --resolution 16 --correlation 0.85 --n-samples 1000 --output-dir data/cmnist16_c85_clean

# ===== 8x8 (clean) =====
python cmnist_gen.py --resolution 8  --correlation 0.00 --n-samples 1000 --output-dir data/cmnist8_c00_clean
python cmnist_gen.py --resolution 8  --correlation 0.25 --n-samples 1000 --output-dir data/cmnist8_c25_clean
python cmnist_gen.py --resolution 8  --correlation 0.50 --n-samples 1000 --output-dir data/cmnist8_c50_clean
python cmnist_gen.py --resolution 8  --correlation 0.85 --n-samples 1000 --output-dir data/cmnist8_c85_clean

echo "✅ All CMNIST variants generated successfully."
