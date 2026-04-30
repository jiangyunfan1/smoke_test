#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import List
from transformers import AutoTokenizer

# ------------------------- 优化点 3：预筛选安全 Token ID 池 -------------------------
def get_safe_token_pool(tokenizer):
    safe_ids = []
    special_ids = set(tokenizer.all_special_ids)
    # 扫描前 50000 个 token 即可，足够采样
    for tid in range(min(tokenizer.vocab_size, 50000)):
        text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if text.strip() and all(c.isalnum() or c == ' ' for c in text) and '\n' not in text:
            safe_ids.append(tid)
    return safe_ids if safe_ids else [1, 2, 3] # 极简状态下避空

# ------------------------- 优化逻辑：从本地 GSM8K 文件提取 -------------------------
def get_gsm8k_ids_local(tokenizer, gsm8k_path: str, target_len: int, rng: random.Random) -> List[int]:
    """从本地 JSONL 文件循环读取文本并编码，直到填满 target_len"""
    questions = []
    if not os.path.exists(gsm8k_path):
        raise FileNotFoundError(f"找不到 GSM8K 文件: {gsm8k_path}")
        
    with open(gsm8k_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data.get("question", ""))
    
    rng.shuffle(questions)
    
    all_ids = []
    idx = 0
    # 针对 100K 这种超长请求，循环拼接直到填满
    while len(all_ids) < target_len:
        q_text = questions[idx % len(questions)]
        ids = tokenizer.encode(q_text, add_special_tokens=False)
        all_ids.extend(ids)
        idx += 1
            
    return all_ids[:target_len]

# ------------------------- 优化点 1 & 2：并行 worker -------------------------
def worker_task(task_id, prefix_ids, total_len, safe_pool, tokenizer_dir, seed):
    # 这里的 total_len 是目标总 token 数
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    rng = random.Random(seed + task_id)
    
    suffix_len = total_len - len(prefix_ids)
    # 采样后缀
    suffix_ids = [rng.choice(safe_pool) for _ in range(max(0, suffix_len))]
    
    full_ids = prefix_ids + suffix_ids
    # 一次性 decode，避免循环产生的 N^2 开销
    full_text = tokenizer.decode(full_ids, clean_up_tokenization_spaces=False)
    return {"question": full_text, "answer": ""}

# ------------------------- 主流程 -------------------------
def main():
    ap = argparse.ArgumentParser(description="高性能语义化数据集生成器")
    ap.add_argument("--total", type=int, required=True, help="总条数")
    ap.add_argument("--num-prefixes", type=int, default=1, help="公共前缀数量")
    ap.add_argument("--length", type=int, required=True, help="总 Token 长度")
    ap.add_argument("--prefix-ratio", type=str, required=True, help="前缀比例，如 0.75 或 75%%")
    ap.add_argument("--tokenizer-dir", type=str, required=True, help="分词器路径")
    ap.add_argument("--gsm8k-path", type=str, required=True, help="本地 GSM8K train.jsonl 路径")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    # 参数转换
    ratio_str = args.prefix_ratio.strip()
    ratio = float(ratio_str[:-1])/100.0 if ratio_str.endswith('%') else float(ratio_str)
    prefix_target_len = int(args.length * ratio)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    safe_pool = get_safe_token_pool(tokenizer)
    rng_main = random.Random(args.seed)
    
    # 1. 构造公共前缀 (从本地 GSM8K)
    print(f"正在读取 {args.gsm8k_path} 生成 {args.num_prefixes} 组公共前缀...")
    all_prefixes_ids = []
    prefix_jsonl_content = []
    for i in range(args.num_prefixes):
        p_ids = get_gsm8k_ids_local(tokenizer, args.gsm8k_path, prefix_target_len, rng_main)
        all_prefixes_ids.append(p_ids)
        prefix_jsonl_content.append({"question": tokenizer.decode(p_ids), "answer": ""})

    # 2. 准备输出目录
    t_name = os.path.basename(os.path.normpath(args.tokenizer_dir))
    dir_name = f"dataset_{args.seed}_{args.length}_{ratio_str}_{t_name}"
    os.makedirs(f"{dir_name}_prefix", exist_ok=True)
    
    with open(f"{dir_name}_prefix/prefix.jsonl", "w", encoding="utf-8") as f:
        for item in prefix_jsonl_content:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 3. 多进程生成数据集
    print(f"开始并行生成 {args.total} 条 100K 级别的数据...")
    dataset = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(args.total):
            p_ids = all_prefixes_ids[i % args.num_prefixes]
            futures.append(executor.submit(
                worker_task, i, p_ids, args.length, safe_pool, args.tokenizer_dir, args.seed
            ))
        
        for future in futures:
            dataset.append(future.result())

    # 保存最终产物
    output_path = f"{dir_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"成功生成！\ngenerated_dataset: {output_path}\n前缀文件目录: {dir_name}_prefix/")

if __name__ == "__main__":
    main()

