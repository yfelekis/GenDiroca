#!/bin/bash

set -e

echo "--- STARTING BATTERY EMPIRICAL EXPERIMENT BATCH ---"

# All runs:
#   alpha in [0,1], 10 steps
#   noise in [0,10], 20 steps
#   20 trials

echo -e "\n[1/3] Battery: Additive Gaussian noise..."
python run_battery_evaluation.py \
  --distribution gaussian \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n[2/3] Battery: Additive Student-t noise..."
python run_battery_evaluation.py \
  --distribution student-t \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n[3/3] Battery: Additive Exponential noise..."
python run_battery_evaluation.py \
  --distribution exponential \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n--- BATTERY EMPIRICAL EXPERIMENT BATCH COMPLETE ---"
