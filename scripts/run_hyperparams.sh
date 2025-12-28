#!/bin/bash
# scripts/run_hyperparams.sh
# 改变 Batch Size
python train.py --exp_name trans_bs512 --model transformer --batch_size 512 --epochs 10
python train.py --exp_name trans_bs256 --model transformer --batch_size 256 --epochs 10