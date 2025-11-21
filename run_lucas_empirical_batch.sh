#!/bin/bash

set -e

echo "--- STARTING LUCAS EMPIRICAL EXPERIMENT BATCH ---"

# All runs:
#   alpha in [0,1], 10 steps
#   noise in [0,5], 20 steps
#   20 trials
#   shift_type additive (default)

echo -e "\n[1/3] LUCAS: Additive Gaussian noise..."
python run_general_evaluation.py \
  --experiment lucas \
  --distribution gaussian \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n[2/3] LUCAS: Additive Student-t noise..."
python run_general_evaluation.py \
  --experiment lucas \
  --distribution student-t \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n[3/3] LUCAS: Additive Exponential noise..."
python run_general_evaluation.py \
  --experiment lucas \
  --distribution exponential \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n--- LUCAS EMPIRICAL EXPERIMENT BATCH COMPLETE ---"
