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
# 1) RNN：补 Free Running（纯 0 teacher forcing）
############################
python train.py \
  --exp_name rnn_free_running_tf0 \
  --model rnn \
  --tf_ratio 0.0 \
  --epochs ${EPOCHS_RNN} \
  --batch_size ${BS_RNN}

# （可选）如果你想把 TF=1.0 也用同 batch_size 对齐再跑一遍，可打开
# python train.py \
#   --exp_name rnn_tf_only_bs256 \
#   --model rnn \
#   --tf_ratio 1.0 \
#   --epochs ${EPOCHS_RNN} \
#   --batch_size ${BS_RNN}

############################
# 2) Transformer：位置编码 "relative" 补齐
# 你的 relative_mock 实际是 No-PE baseline（不是标准相对位置 bias）
############################
python train.py \
  --exp_name trans_pos_relative_mock \
  --model transformer \
  --pos_type relative_mock \
  --epochs ${EPOCHS_TRANS} \
  --batch_size ${BS_TRANS}

