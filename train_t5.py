import argparse
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
# 1. 修改这里：去掉 AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
# 2. 新增这里：使用 PyTorch 原生的 AdamW
from torch.optim import AdamW
from tqdm import tqdm

class T5Dataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:  # 加上 encoding='utf-8' 防止编码报错
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # T5 需要 task prefix，这里加上翻译前缀
        src = "translate Chinese to English: " + item['zh']
        tgt = item['en']
        
        # Tokenize
        enc = self.tokenizer(src, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        dec = self.tokenizer(tgt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'input_ids': enc.input_ids.squeeze(),
            'attention_mask': enc.attention_mask.squeeze(),
            'labels': dec.input_ids.squeeze()
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='t5_finetune')
    # 建议 batch_size 根据你的 H100 显存调整，512 可能稍大，如果 OOM 改成 128 或 256
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning T5 on {device}")
    
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 这里的模型名称 't5-small' 会自动下载。
    # [cite_start]如果你想用效果更好的 't5-base' (作业参考资料提到 [cite: 105])，可以改这里。
    model_name = 't5-small' 
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("请确保服务器能连接 HuggingFace，或者手动下载模型权重。")
        return
    
    train_ds = T5Dataset('./dataset/train.jsonl', tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4) # 加速数据加载
    
    # 使用 PyTorch 原生的 AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # 保存模型
        model.save_pretrained(exp_dir)
        tokenizer.save_pretrained(exp_dir)
        # 保存一个标记文件，防止 Inference 时找不到
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            f.write(json.dumps({'model_type': 't5', 'base_model': model_name}))

if __name__ == '__main__':
    main()