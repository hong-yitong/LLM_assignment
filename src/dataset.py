import torch
from torch.utils.data import Dataset
import json
import jieba
import nltk
import os
from collections import Counter
from functools import partial
import multiprocessing

import os
import nltk
# 获取当前文件 (src/dataset.py) 的上级目录的上级目录 (即项目根目录)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nltk.data.path.append(project_root)

class Vocab:
    def __init__(self, tokens=None, min_freq=2, specials=['<pad>', '<unk>', '<sos>', '<eos>']):
        self.specials = specials
        self.stoi = {k: i for i, k in enumerate(specials)}
        self.itos = {i: k for i, k in enumerate(specials)}
        if tokens:
            self.build_vocab(tokens, min_freq)

    def build_vocab(self, tokens, min_freq):
        counter = Counter(tokens)
        idx = len(self.specials)
        for token, freq in counter.most_common():
            if freq >= min_freq:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1
        print(f"Vocab size: {len(self.stoi)}")

    def __len__(self):
        return len(self.stoi)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])

class NMTDataset(Dataset):
    def __init__(self, path, src_vocab=None, tgt_vocab=None, max_len=128, build_vocab=False):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        self.max_len = max_len
        
        # 定义 Tokenizers
        self.tok_zh = lambda x: jieba.lcut(x)
        # 使用本地 NLTK
        try:
            self.tok_en = lambda x: nltk.word_tokenize(x.lower())
        except LookupError:
            print("Warning: NLTK punkt not found in ./tokenizers, falling back to split()")
            self.tok_en = lambda x: x.lower().split()

        if build_vocab:
            print("Building Vocab...")
            all_zh = [w for d in self.data for w in self.tok_zh(d['zh'])]
            all_en = [w for d in self.data for w in self.tok_en(d['en'])]
            self.src_vocab = Vocab(all_zh, min_freq=2)
            self.tgt_vocab = Vocab(all_en, min_freq=2)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]['zh']
        tgt_text = self.data[idx]['en']
        
        src_toks = ['<sos>'] + self.tok_zh(src_text)[:self.max_len] + ['<eos>']
        tgt_toks = ['<sos>'] + self.tok_en(tgt_text)[:self.max_len] + ['<eos>']
        
        src_ids = [self.src_vocab[t] for t in src_toks]
        tgt_ids = [self.tgt_vocab[t] for t in tgt_toks]
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'src_len': len(src_ids),
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def collate_fn(batch):
    src_pad = 0 # <pad> index
    tgt_pad = 0
    
    src = torch.nn.utils.rnn.pad_sequence([x['src'] for x in batch], batch_first=True, padding_value=src_pad)
    tgt = torch.nn.utils.rnn.pad_sequence([x['tgt'] for x in batch], batch_first=True, padding_value=tgt_pad)
    src_lens = torch.tensor([x['src_len'] for x in batch])
    
    return src, tgt, src_lens, [x['src_text'] for x in batch], [x['tgt_text'] for x in batch]