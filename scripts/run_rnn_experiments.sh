#!/bin/bash
# scripts/run_rnn_experiments.sh

# Attention Ablation [cite: 15]
# python train.py --exp_name rnn_dot --model rnn --attn_type dot --epochs 10 --batch_size 256
# python train.py --exp_name rnn_add --model rnn --attn_type additive --epochs 10 --batch_size 256
python train.py --exp_name rnn_mul --model rnn --attn_type multiplicative --epochs 10 --batch_size 256

# Training Policy: Free Running vs Teacher Forcing [cite: 16]
# 纯 TF
python train.py --exp_name rnn_tf_only --model rnn --tf_ratio 1.0 --epochs 10
# Scheduled Sampling (Free Running gradually)
python train.py --exp_name rnn_schedule --model rnn --tf_ratio 1.0 --scheduled_sampling --epochs 10