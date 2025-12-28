import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, method='dot'):
        super().__init__()
        self.method = method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        if method == 'additive':
            self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        elif method == 'multiplicative':
            self.attn = nn.Linear(enc_hid_dim, dec_hid_dim)

    def forward(self, hidden, encoder_outputs):
        # hidden: [Layers, Batch, Dec_Hid] -> 我们需要取最后一层
        # encoder_outputs: [Batch, Seq_Len, Enc_Hid]
        
        # 修正：确保 hidden 维度正确。如果是多层 RNN，hidden 形状是 [Layers, Batch, Hid]
        # Attention 计算通常只需要最后一层的 Hidden State
        if hidden.dim() == 3:
            hidden = hidden[-1] # [Batch, Dec_Hid]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        if self.method == 'dot':
            # [B, 1, H] @ [B, H, L] -> [B, 1, L]
            attn_scores = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
            
        elif self.method == 'multiplicative':
            weighted_enc = self.attn(encoder_outputs)
            attn_scores = torch.bmm(weighted_enc, hidden.unsqueeze(2)).squeeze(2)
            
        else: # additive
            hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
            energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
            attn_scores = self.v(energy).squeeze(2)
            
        return F.softmax(attn_scores, dim=1)

class RNNSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, n_layers=2, dropout=0.1, attn_type='dot'):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.encoder_gru = nn.GRU(d_model, d_model, num_layers=n_layers, 
                                  batch_first=True, dropout=dropout, bidirectional=False)
        
        self.decoder_gru = nn.GRU(d_model + d_model, d_model, num_layers=n_layers, 
                                  batch_first=True, dropout=dropout)
        
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.attn = Attention(d_model, d_model, attn_type)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encoder(self, src):
        embedded = self.dropout(self.src_emb(src))
        outputs, hidden = self.encoder_gru(embedded)
        return outputs, hidden

    def decode_step(self, input_tok, hidden, enc_outputs):
        '''
        修复点：智能处理 input_tok 的维度
        '''
        # 1. 维度检查与修正
        # 如果是 [Batch]，unsqueeze 成 [Batch, 1]
        if input_tok.dim() == 1:
            input_tok = input_tok.unsqueeze(1)
        # 如果已经是 [Batch, 1] (Beam Search 时)，保持不变，不要再 unsqueeze 了！

        embedded = self.dropout(self.tgt_emb(input_tok)) # [Batch, 1, Emb]
        
        # 2. Attention 计算
        # hidden 是 [Layers, Batch, Hid]，Attention 内部会自动处理取最后一层
        a = self.attn(hidden, enc_outputs) # [Batch, Seq]
        
        a = a.unsqueeze(1) # [Batch, 1, Seq]
        weighted = torch.bmm(a, enc_outputs) # [Batch, 1, Hid]
        
        # 3. 拼接 (现在 embedded 和 weighted 都是 3维 的了)
        rnn_input = torch.cat((embedded, weighted), dim=2) 
        
        output, new_hidden = self.decoder_gru(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, new_hidden

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        
        enc_outputs, hidden = self.encoder(src)
        
        # 初始输入 <sos> 是 1D tensor: [Batch]
        input_tok = tgt[:, 0] 
        
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size).to(src.device)
        
        for t in range(1, tgt_len):
            prediction, hidden = self.decode_step(input_tok, hidden, enc_outputs)
            outputs[:, t] = prediction
            
            top1 = prediction.argmax(1)
            import random
            use_teacher = random.random() < teacher_forcing_ratio
            # 这里的 input_tok 仍然保持 1D [Batch]
            input_tok = tgt[:, t] if use_teacher else top1
            
        return outputs