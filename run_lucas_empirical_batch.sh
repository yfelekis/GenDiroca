#!/bin/bash
set -e

echo "--- STARTING LUCAS FULL DUAL-MODE EVALUATION ---"

# Optional: Clear old results to ensure fresh files
# rm data/lucas/evaluation_results/*.csv 

# 1. Gaussian (The Main Plot)
echo -e "\n[1/3] LUCAS: Additive Gaussian noise (Dual Mode)..."
python run_general_evaluation.py \
  --experiment lucas \
  --distribution gaussian \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20 \
  --eval_mode both

# 2. Student-t (Appendix Plot 1)
echo -e "\n[2/3] LUCAS: Additive Student-t noise (Dual Mode)..."
python run_general_evaluation.py \
  --experiment lucas \
  --distribution student-t \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20 \
  --eval_mode both

# 3. Exponential (Appendix Plot 2)
echo -e "\n[3/3] LUCAS: Additive Exponential noise (Dual Mode)..."
python run_general_evaluation.py \
  --experiment lucas \
  --distribution exponential \
  --shift_type additive \
  --alpha_min 0.0 --alpha_max 1.0 --alpha_steps 10 \
  --noise_min 0.0 --noise_max 10.0 --noise_steps 20 \
  --trials 20 \
  --eval_mode both

echo -e "\n--- LUCAS FULL BATCH COMPLETE ---"