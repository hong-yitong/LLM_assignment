import os
import torch
import json
import pickle
import time
import argparse
import csv
from src.models_rnn import RNNSeq2Seq
from src.models_transformer import TransformerModel
from src.utils import beam_search_decode
from src.dataset import Vocab 
from tqdm import tqdm
import jieba

# 防止路径引用问题
import sys
sys.path.append(os.getcwd())

def save_predictions(exp_path, strategy, sources, refs, preds):
    """
    将详细的翻译结果保存到实验目录下，方便 Case Study
    """
    filename = f"predictions_{strategy}.txt"
    filepath = os.path.join(exp_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for s, r, p in zip(sources, refs, preds):
            f.write(f"Source: {s}\n")
            f.write(f"Ref   : {r}\n")
            f.write(f"Pred  : {p}\n")
            f.write("-" * 40 + "\n")
    print(f"  [Saved] Predictions saved to {filepath}")

def save_metrics(exp_path, strategy, metrics):
    """
    将评估指标保存为 JSON
    """
    filename = f"test_metrics_{strategy}.json"
    filepath = os.path.join(exp_path, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"  [Saved] Metrics saved to {filepath}")

def evaluate_model(exp_path, test_data, device, use_beam=False):
    # 策略名称
    strategy = 'Beam-3' if use_beam else 'Greedy'
    
    # 1. 识别模型类型
    is_t5 = os.path.exists(os.path.join(exp_path, 'config.json')) 
    metrics = {}
    
    preds, refs, sources = [], [], []
    latencies = []
    
    if is_t5:
        # --- T5 推理 ---
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        try:
            tokenizer = T5Tokenizer.from_pretrained(exp_path)
            model = T5ForConditionalGeneration.from_pretrained(exp_path).to(device)
        except Exception as e:
            print(f"Warning: T5 model error in {exp_path}: {e}")
            return None
            
        model.eval()
        for item in tqdm(test_data, desc=f"T5 ({strategy})"):
            start = time.time()
            input_text = "translate Chinese to English: " + item['zh']
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            
            outputs = model.generate(input_ids, max_length=128, num_beams=(3 if use_beam else 1))
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            latencies.append(time.time() - start)
            preds.append(pred)
            refs.append(item['en'])
            sources.append(item['zh'])
            
    else:
        # --- RNN / Transformer 推理 ---
        try:
            with open(os.path.join(exp_path, 'vocab.pkl'), 'rb') as f:
                vocabs = pickle.load(f)
            with open(os.path.join(exp_path, 'metrics.json'), 'r') as f:
                config = json.load(f)['config']
        except Exception as e:
            print(f"Skipping {exp_path}: {e}")
            return None 
            
        src_vocab, tgt_vocab = vocabs['src'], vocabs['tgt']
        
        # 初始化模型
        if config['model'] == 'rnn':
            model = RNNSeq2Seq(
                len(src_vocab), len(tgt_vocab), 
                d_model=config['d_model'], 
                n_layers=config['layers'], 
                attn_type=config['attn_type']
            ).to(device)
        else:
            model = TransformerModel(
                len(src_vocab), len(tgt_vocab), 
                d_model=config['d_model'], 
                nhead=config.get('nhead', 4),       
                num_layers=config['layers'],        
                norm_type=config['norm_type'], 
                pos_type=config['pos_type']
            ).to(device)
                                     
        try:
            model.load_state_dict(torch.load(os.path.join(exp_path, 'ckpt.pt'), map_location=device))
        except RuntimeError as e:
            print(f"FATAL: Architecture mismatch in {exp_path}. {e}")
            return None

        model.to(device)
        model.eval()
        
        for item in tqdm(test_data, desc=f"Eval {os.path.basename(exp_path)} ({strategy})"):
            start = time.time()
            
            tokens = ['<sos>'] + jieba.lcut(item['zh']) + ['<eos>']
            src_ids = [src_vocab[t] for t in tokens]
            src_tensor = torch.tensor(src_ids, dtype=torch.long).to(device)
            
            try:
                beam_w = 3 if use_beam else 1
                pred = beam_search_decode(model, src_tensor, src_vocab, tgt_vocab, device, beam_width=beam_w, model_type=config['model'])
            except Exception as e:
                print(f"Decode error: {e}")
                pred = ""

            latencies.append(time.time() - start)
            preds.append(pred)
            refs.append(item['en'])
            sources.append(item['zh'])

    # --- 计算指标 ---
    from sacrebleu.metrics import BLEU
    bleu = BLEU()
    score = bleu.corpus_score(preds, [refs])
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    # 长难句分析 (>20 words)
    long_data = [(p, r) for p, r in zip(preds, refs) if len(r.split()) > 20]
    if long_data:
        long_preds, long_refs = zip(*long_data)
        long_score = bleu.corpus_score(long_preds, [long_refs]).score
    else:
        long_score = 0.0
    
    # --- 结果打包 ---
    result_metrics = {
        'exp': os.path.basename(exp_path),
        'decoding': strategy,
        'bleu': score.score,
        'latency_ms': avg_latency * 1000,
        'long_sentence_bleu': long_score
    }
    
    # --- 保存文件 ---
    save_predictions(exp_path, strategy, sources, refs, preds)
    save_metrics(exp_path, strategy, result_metrics)
    
    return result_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference using device: {device}")
    
    # 读取测试集
    test_data = []
    # 确保此处路径正确
    data_path = './dataset/test.jsonl'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    results = []
    exp_root = 'experiments'
    
    if not os.path.exists(exp_root):
        print("No experiments found.")
        return

    # 遍历所有实验文件夹
    for exp_name in sorted(os.listdir(exp_root)):
        exp_path = os.path.join(exp_root, exp_name)
        if not os.path.isdir(exp_path): continue
        
        print(f"\n--- Evaluating Experiment: {exp_name} ---")
        
        # 1. Greedy 模式 (必跑)
        res_greedy = evaluate_model(exp_path, test_data, device, use_beam=False)
        if res_greedy:
            results.append(res_greedy)
            
        # 2. Beam Search 模式 (选跑)
        # 如果你想全跑，去掉 if 条件即可
        if 'rnn_dot' in exp_name or 'trans_base' in exp_name or 't5' in exp_name: 
             print(f"  > Running Beam Search for {exp_name}...")
             res_beam = evaluate_model(exp_path, test_data, device, use_beam=True)
             if res_beam:
                 results.append(res_beam)
    
    # --- 打印最终报表 ---
    print("\n" + "="*85)
    print(f"{'Experiment':<25} | {'Strategy':<8} | {'BLEU':<6} | {'Long-BLEU':<10} | {'Latency(ms)':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['exp']:<25} | {r['decoding']:<8} | {r['bleu']:<6.2f} | {r['long_sentence_bleu']:<10.2f} | {r['latency_ms']:<10.0f}")
    print("="*85 + "\n")

    # --- 保存汇总 CSV ---
    csv_file = "final_summary.csv"
    keys = ['exp', 'decoding', 'bleu', 'long_sentence_bleu', 'latency_ms']
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"All results saved to root directory: {csv_file}")
    except IOError:
        print("Error saving CSV file. Check permissions.")

if __name__ == '__main__':
    main()