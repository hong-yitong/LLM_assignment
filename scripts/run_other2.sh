#!/bin/bash
set -e

# scripts/run_missing_experiments.sh
# 目标：用尽量少的实验补齐作业要求的缺口
# - RNN: Teacher Forcing vs Free Running（补 tf_ratio=0.0）
# - Transformer: 位置编码 absolute vs "relative"（用 relative_mock 作为 No-PE baseline）
# - Transformer: Hyperparam sensitivity（learning rate + model scale）
# - Decoding: 选一个代表模型跑 beam（你已有 rnn_dot/trans_base/t5，但补一个 transformer 的 beam 对照也可）

############################
# 0) 统一设置（可按需改）
############################
EPOCHS_RNN=10
EPOCHS_TRANS=10     # 补实验一般跑 10 足够写 report；如果你想更稳，可改 15
BS_TRANS=512
BS_RNN=256



############################
# 3) Transformer：Learning rate 敏感性（补 2 个点即可成段落）
############################
python train.py \
  --exp_name trans_lr1e4 \
  --model transformer \
  --lr 0.0001 \
  --epochs ${EPOCHS_TRANS} \
  --batch_size ${BS_TRANS}

python train.py \
  --exp_name trans_lr1e3 \
  --model transformer \
  --lr 0.001 \
  --epochs ${EPOCHS_TRANS} \
  --batch_size ${BS_TRANS}

############################
# 4) Transformer：Model scale 敏感性（d_model / layers）
# 尽量少：一个更小，一个更大
############################
# 小模型：d_model=128（注意需能被 nhead=4 整除）
python train.py \
  --exp_name trans_scale_small_d128 \
  --model transformer \
  --d_model 128 \
  --nhead 4 \
  --layers 2 \
  --epochs ${EPOCHS_TRANS} \
  --batch_size ${BS_TRANS}

# 大模型：layers=4（d_model 仍 256，保持对照清晰）
python train.py \
  --exp_name trans_scale_deep_L4 \
  --model transformer \
  --d_model 256 \
  --nhead 4 \
  --layers 4 \
  --epochs ${EPOCHS_TRANS} \
  --batch_size ${BS_TRANS}

echo "All missing experiments launched successfully."