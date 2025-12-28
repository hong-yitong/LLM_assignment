#!/bin/bash
# scripts/run_ArchitecturalAblation.sh
# 1. 绝对位置 (Sinusoidal) - Baseline
python train.py --exp_name trans_abs_sin --model transformer --pos_type absolute --epochs 15
# 2. 可学习位置 (Learnable)
python train.py --exp_name trans_abs_learn --model transformer --pos_type learnable --epochs 15

# 1. LayerNorm (Baseline)
# (复用上面的 trans_abs_sin 结果即可，它默认是 LayerNorm)
# 2. RMSNorm
python train.py --exp_name trans_rms --model transformer --norm_type rms --pos_type absolute --epochs 15