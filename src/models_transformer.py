import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 基础组件: RMSNorm & Positional Encoding ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        return x + self.pe[:, :x.size(1), :]

# --- 2. 自定义 Transformer Layer (支持 RMSNorm 替换) ---

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_type='layer'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 核心修改点：支持切换 Norm 类型
        if norm_type == 'rms':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-Norm 结构 (更稳定，适合 Deep Transformer)
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_type='layer'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        if norm_type == 'rms':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self Attention
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross Attention
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        
        # FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

# --- 3. 完整的 Transformer 模型 ---

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, 
                 num_layers=3, dim_feedforward=1024, dropout=0.1, 
                 norm_type='layer', pos_type='absolute'):
        super().__init__()
        self.d_model = d_model
        self.pos_type = pos_type
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        
        # --- Positional Embedding Ablation ---
        # 1. Absolute (Sinusoidal): 固定公式
        if pos_type == 'absolute':
            self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        # 2. Learnable: 可学习参数
        elif pos_type == 'learnable':
            self.pos_encoder = nn.Embedding(1000, d_model) # 假设最大长度1000
        # 3. Relative: 不使用输入侧的位置编码，而是依赖 Attention (此处简化为无 PE，依赖 causal mask 隐式学习，或后续扩展 bias)
        # 作业中为了对比，通常 "No PE" 或 "Learnable" 是常见 Baseline。
        # 如果要实现 T5 风格的 relative bias，需要修改 attention 内部，代码量巨大。
        # 建议：对比 Sinusoidal (Fixed) vs Learnable (Trainable) 即可满足 "Schemes" 要求。
        elif pos_type == 'relative_mock': 
            self.pos_encoder = None 
            
        # --- Normalization Ablation ---
        # 使用自定义的 Encoder/Decoder Layer
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src):
        src_mask = (src == 0).to(src.device) # Padding mask
        x = self.src_emb(src) * math.sqrt(self.d_model)
        
        if self.pos_type == 'absolute':
            x = self.pos_encoder(x)
        elif self.pos_type == 'learnable':
            positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0)
            x = x + self.pos_encoder(positions)
            
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_mask)
        return x

    def decode(self, tgt, memory, src_padding_mask):
        tgt_pad_mask = (tgt == 0).to(tgt.device)
        tgt_len = tgt.size(1)
        # Causal Mask (上三角)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)
        
        x = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        
        if self.pos_type == 'absolute':
            x = self.pos_encoder(x)
        elif self.pos_type == 'learnable':
            positions = torch.arange(0, tgt_len, device=tgt.device).unsqueeze(0)
            x = x + self.pos_encoder(positions)
            
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, 
                      tgt_key_padding_mask=tgt_pad_mask, 
                      memory_key_padding_mask=src_padding_mask)
        
        return self.fc_out(x)

    def forward(self, src, tgt):
        # 训练专用
        memory = self.encode(src)
        src_padding_mask = (src == 0).to(src.device)
        return self.decode(tgt, memory, src_padding_mask)