#!/bin/bash

set -e

echo "--- STARTING LUCAS GRADCA EMPIRICAL EXPERIMENT BATCH ---"

# All runs:
#   Target Method: GradCA
#   alpha in [0,1], 10 steps
#   noise in [0,10], 20 steps
#   20 trials
#   shift_type additive (default)

echo -e "\n[1/3] LUCAS (GradCA): Additive Gaussian noise..."
python grdca_focused_eval.py \
  --experiment lucas \
  --method gradca \
  --distribution gaussian \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n[2/3] LUCAS (GradCA): Additive Student-t noise..."
python grdca_focused_eval.py \
  --experiment lucas \
  --method gradca \
  --distribution student-t \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n[3/3] LUCAS (GradCA): Additive Exponential noise..."
python grdca_focused_eval.py \
  --experiment lucas \
  --method gradca \
  --distribution exponential \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20

echo -e "\n--- LUCAS GRADCA BATCH COMPLETE ---"