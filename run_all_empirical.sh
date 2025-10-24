#!/bin/bash
set -euo pipefail

EXP="battery"
RESULTS_DIR="data/${EXP}/results_empirical_9010"
EVAL_SCRIPT="evaluate_battery_empirical.py"   # change if you named it differently

echo "--- STARTING COMPLETE EMPIRICAL EXPERIMENT BATCH ---"

# =============================================================
# BATTERY EMPIRICAL EXPERIMENTS (3 total)
# =============================================================
echo -e "\n[1/3] Running: BATTERY Empirical, Additive Gaussian Noise..."
python "$EVAL_SCRIPT" \
  --experiment "$EXP" \
  --results-dir "$RESULTS_DIR" \
  --shift_type additive \
  --distribution gaussian \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 5.0 --noise_steps 20 \
  --trials 20

echo -e "\n[2/3] Running: BATTERY Empirical, Additive Student-t Noise..."
python "$EVAL_SCRIPT" \
  --experiment "$EXP" \
  --results-dir "$RESULTS_DIR" \
  --shift_type additive \
  --distribution student-t \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 5.0 --noise_steps 20 \
  --trials 20

echo -e "\n[3/3] Running: BATTERY Empirical, Additive Exponential Noise..."
python "$EVAL_SCRIPT" \
  --experiment "$EXP" \
  --results-dir "$RESULTS_DIR" \
  --shift_type additive \
  --distribution exponential \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 5.0 --noise_steps 20 \
  --trials 20

echo -e "\n--- EMPIRICAL EXPERIMENT BATCH COMPLETE ---"
