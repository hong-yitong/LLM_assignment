import torch
import torch.nn as nn
import json
import time
import os
import math

# --- 1. 实验记录工具 ---
class ExperimentLogger:
    def __init__(self, exp_dir):
        self.log_path = os.path.join(exp_dir, 'metrics.json')
        self.metrics = {
            'train_loss': [], 'val_loss': [], 
            'epoch_times': [], 'bleu_scores': [],
            'config': {}
        }
    
    def log(self, epoch, train_loss, val_loss, time_taken):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['epoch_times'].append(time_taken)
        self.save()

    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

# --- 2. 架构消融组件: RMSNorm ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

# --- 3. 解码策略: Beam Search ---
def beam_search_decode(model, src, src_vocab, tgt_vocab, device, beam_width=3, max_len=50, model_type='transformer'):
    """
    通用 Beam Search 实现，支持 RNN 和 Transformer
    """
    model.eval()
    sos_idx = tgt_vocab['<sos>']
    eos_idx = tgt_vocab['<eos>']
    pad_idx = 0 # 假设 <pad> index 为 0
    
    # Encoder 阶段
    with torch.no_grad():
        src_tensor = src.unsqueeze(0).to(device) # [1, Seq]
        # 🟢 修复点 1: 生成 src_padding_mask
        src_padding_mask = (src_tensor == pad_idx).to(device)
        
        if model_type == 'rnn':
            enc_out, hidden = model.encoder(src_tensor)
            # RNN Decoder 初始状态
            # hidden: [Layers, Batch, Hid]
            dec_input = torch.tensor([[sos_idx]], device=device)
            # 状态元组: (log_prob, sequence, decoder_hidden)
            candidates = [(0.0, [sos_idx], hidden)]
        
        elif model_type == 'transformer':
            # Transformer Encoder
            memory = model.encode(src_tensor)
            # 状态元组: (log_prob, sequence) - Transformer 无需传递 hidden
            candidates = [(0.0, [sos_idx])]
            
    final_candidates = []
    
    for _ in range(max_len):
        new_candidates = []
        for score, seq, *args in candidates:
            if seq[-1] == eos_idx:
                final_candidates.append((score, seq, *args))
                continue
            
            # 准备 Decoder 输入
            if model_type == 'rnn':
                hidden_state = args[0]
                dec_input = torch.tensor([[seq[-1]]], device=device)
                
                # RNN 单步解码
                logits, new_hidden = model.decode_step(dec_input, hidden_state, enc_out)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0) # [Vocab]
            
            elif model_type == 'transformer':
                tgt_tensor = torch.tensor([seq], device=device)
                # 🟢 修复点 2: 传入 src_padding_mask
                logits = model.decode(tgt_tensor, memory, src_padding_mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            
            # 选取 Top-K
            topk_probs, topk_ids = torch.topk(log_probs, beam_width)
            
            for k in range(beam_width):
                new_score = score + topk_probs[k].item()
                new_seq = seq + [topk_ids[k].item()]
                
                if model_type == 'rnn':
                    new_candidates.append((new_score, new_seq, new_hidden))
                else:
                    new_candidates.append((new_score, new_seq))
        
        # 排序并截取 Beam Width
        ordered = sorted(new_candidates, key=lambda x: x[0], reverse=True)
        candidates = ordered[:beam_width]
        
        if len(candidates) == 0: break # 所有都结束了
        
    # 如果没有生成完，取当前的最佳
    if not final_candidates:
        final_candidates = candidates
        
    best_score, best_seq, *_ = sorted(final_candidates, key=lambda x: x[0], reverse=True)[0]
    
    # 转回文本
    tokens = [tgt_vocab.itos[i] for i in best_seq[1:] if i != eos_idx] # 去掉 SOS/EOS
    return " ".join(tokens)