#!/bin/bash

set -e

echo "--- STARTING EVALUATION BATCH (Student-t & Exponential) ---"

# Settings:
# Alpha: 0.0 to 1.0 (10 steps)
# Noise: 0.0 to 10.0 (20 steps)
# Trials: 20

# ==========================================
# 1. Battery Evaluation
# ==========================================

echo -e "\n[1/4] Battery: Student-t noise..."
python run_battery_evaluation.py \
  --distribution student-t \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n[2/4] Battery: Exponential noise..."
python run_battery_evaluation.py \
  --distribution exponential \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

# ==========================================
# 2. nLUCAS Evaluation
# ==========================================

echo -e "\n[3/4] nLUCAS: Student-t noise..."
python run_nlucas_evaluation.py \
  --distribution student-t \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n[4/4] nLUCAS: Exponential noise..."
python run_nlucas_evaluation.py \
  --distribution exponential \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n--- ALL EVALUATIONS COMPLETE ---"