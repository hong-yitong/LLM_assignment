import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import pickle
from src.dataset import NMTDataset, collate_fn
from src.models_rnn import RNNSeq2Seq
from src.models_transformer import TransformerModel
from src.utils import ExperimentLogger
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    # 实验基础配置
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['rnn', 'transformer'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005)
    
    # RNN 专属: Attention & Teacher Forcing
    parser.add_argument('--attn_type', type=str, default='dot', choices=['dot', 'additive', 'multiplicative'])
    parser.add_argument('--tf_ratio', type=float, default=1.0, help="Initial Teacher Forcing Ratio")
    parser.add_argument('--scheduled_sampling', action='store_true', help="Decay TF ratio over epochs")
    
    # Transformer 专属: Ablation
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads for Transformer (d_model must be divisible by this)')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--norm_type', type=str, default='layer', choices=['layer', 'rms'])
    parser.add_argument('--pos_type', type=str, default='absolute', choices=['absolute', 'learnable', 'relative_mock'])
    args = parser.parse_args()
    
    # 1. 目录与日志
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = ExperimentLogger(exp_dir)
    logger.metrics['config'] = vars(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {args.exp_name} on {device}")
    
    # 2. 数据
    train_ds = NMTDataset('./dataset/train.jsonl', build_vocab=True)
    val_ds = NMTDataset('./dataset/val.jsonl', src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    
    # 保存词表 (Inference 必需)
    with open(os.path.join(exp_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump({'src': train_ds.src_vocab, 'tgt': train_ds.tgt_vocab}, f)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    
# 3. 模型初始化 (请确保这两行在 if args.model... 之前)
    # ==========================================
    src_v_sz = len(train_ds.src_vocab)  # <--- 这里定义 src_v_sz
    tgt_v_sz = len(train_ds.tgt_vocab)  # <--- 这里定义 tgt_v_sz
    
    print(f"Source Vocab: {src_v_sz}, Target Vocab: {tgt_v_sz}") # 打印一下确认

    if args.model == 'rnn':
        model = RNNSeq2Seq(
            src_v_sz,                   # 使用定义好的变量
            tgt_v_sz, 
            d_model=args.d_model, 
            n_layers=args.layers, 
            attn_type=args.attn_type,
            dropout=0.1
        ).to(device)
    else:
        model = TransformerModel(
            src_v_sz,                   # 使用定义好的变量
            tgt_v_sz, 
            d_model=args.d_model, 
            nhead=args.nhead,           # 确保 parser 里有 --nhead
            num_layers=args.layers, 
            norm_type=args.norm_type,   # 确保 parser 里有 --norm_type
            pos_type=args.pos_type,     # 确保 parser 里有 --pos_type
            dropout=0.1
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    

    # 4. 训练循环
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        # 动态调整 Teacher Forcing
        current_tf = args.tf_ratio
        if args.scheduled_sampling:
            current_tf = max(0.0, args.tf_ratio - (epoch / args.epochs))
        
        # === 修改开始：添加 tqdm 进度条 ===
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        
        for src, tgt, _, _, _ in pbar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            # Forward
            if args.model == 'rnn':
                output = model(src, tgt, teacher_forcing_ratio=current_tf)
            else:
                output = model(src, tgt[:, :-1]) # Transformer Input (No EOS)
            
            # Loss Calculation
            if args.model == 'transformer':
                output = output.reshape(-1, output.shape[-1])
                target = tgt[:, 1:].reshape(-1)
            else:
                output = output[:, 1:, :].reshape(-1, output.shape[-1])
                target = tgt[:, 1:].reshape(-1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 实时更新进度条上的 Loss 显示
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'tf': f"{current_tf:.2f}"})
        # === 修改结束 ===
            
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        
        # 这一行 print 其实可以保留，作为 log 记录
        print(f"Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        logger.log(epoch+1, avg_loss, 0.0, epoch_time)
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(exp_dir, f'ckpt.pt'))
if __name__ == "__main__":
    main()