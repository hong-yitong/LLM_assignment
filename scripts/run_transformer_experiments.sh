#!/bin/bash
# scripts/run_transformer_experiments.sh

# Baseline
python train.py --exp_name trans_base --model transformer --d_model 256 --epochs 15

# Architecture: Norm [cite: 21]
python train.py --exp_name trans_rms --model transformer --norm_type rms --epochs 15

# Architecture: Positional Embedding [cite: 21]
python train.py --exp_name trans_pos_learn --model transformer --pos_type learnable --epochs 15