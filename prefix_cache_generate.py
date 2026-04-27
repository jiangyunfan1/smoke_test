#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成仅由 英文 + 空格 + 数字 构成的数据集（无换行与特殊字符）。
- 入参：
  --tokenizer-dir  默认 /mnt/weight/deepseek_r1
  --seed           默认 202508167
  --tokens         默认 4096
- 共 4001 条：
  * 第 1 条为公共前缀，长度 tokens//2 个 token
  * 后 4000 条每条长度为 tokens，包含公共前缀，且彼此“额外公共前缀”<=16 token
- 输出到当前目录新建的文件夹：{tokens}_{seed}_dataset/
  * {tokens}_{seed}_prefix.jsonl   # 仅1行（公共前缀）
  * {tokens}_{seed}.jsonl          # 4000行（不含前缀行）
"""

import argparse
import json
import os
import random
import re
from typing import List, Tuple

from transformers import AutoTokenizer

ALLOWED_RE = re.compile(r'^[A-Za-z0-9 ]+$')

# ------------------------- 工具函数（不抛异常，内部自愈） -------------------------

def ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # 极端情况下（权限/文件名冲突），退化到当前目录
        pass

def decode_ids(tokenizer, ids: List[int]) -> str:
    try:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    except Exception:
        return ""

def encode_ids(tokenizer, text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return []

def encode_len(tokenizer, text: str) -> int:
    return len(encode_ids(tokenizer, text))

def is_allowed_text(s: str) -> bool:
    if not s:
        return False
    if any(c in s for c in ("\n", "\r", "\t")):
        return False
    return ALLOWED_RE.fullmatch(s) is not None

def filter_allowed(s: str) -> str:
    # 尽量保持原状，仅过滤非法字符（注意：过滤会影响 re-encode 长度，后续再调节）
    s2 = "".join(ch for ch in s if (ch.isalnum() or ch == " "))
    # 去除换行等
    s2 = s2.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s2

def build_safe_token_pools(tokenizer) -> Tuple[List[int], List[int], int]:
    """
    返回：nospace_ids, space_ids, filler_id
    - 优先从 vocab 里筛选“安全 token”（解码仅含 A-Za-z0-9 空格，且不含特殊/换行）
    - 若不足，退化用 encode(" a") / encode(" 0") 等得到的最后一个 token 作为“空格型”填充
    - filler_id：至少保证有一个稳定可追加的 token（一般是以空格开头的安全 token）
    """
    nospace_ids, space_ids = [], []
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_size = getattr(tokenizer, "vocab_size", 0) or 0

    # 1) 扫描词表
    for tid in range(vocab_size):
        if tid in special_ids:
            continue
        piece = decode_ids(tokenizer, [tid])
        if not piece:
            continue
        # 严格字符集
        if not is_allowed_text(piece):
            continue
        if piece.startswith(" "):
            space_ids.append(tid)
        else:
            nospace_ids.append(tid)

    # 2) 至少构造一个 filler_id
    filler_id = None
    if space_ids:
        filler_id = space_ids[0]
    else:
        # 尝试从常见文本取最后一个 token
        for cand in [" 0", " 1", " a", " A", " B", " 2"]:
            ids = encode_ids(tokenizer, cand)
            if ids:
                t = ids[-1]
                if t not in special_ids:
                    filler_id = t
                    break
        # 最差情况：从非空格候选里取一个
        if filler_id is None and nospace_ids:
            filler_id = nospace_ids[0]

    # 3) 若 pools 太少，尽量补齐（不抛异常，只尽力而为）
    def try_add_space_like(chs: List[str]):
        for ch in chs:
            ids = encode_ids(tokenizer, ch)
            if not ids:
                continue
            t = ids[-1]
            dec = decode_ids(tokenizer, [t])
            if is_allowed_text(dec):
                space_ids.append(t)
                if len(space_ids) >= 3:
                    return

    def try_add_nospace_like(chs: List[str]):
        for ch in chs:
            ids = encode_ids(tokenizer, ch)
            if not ids:
                continue
            t = ids[-1]
            dec = decode_ids(tokenizer, [t])
            if is_allowed_text(dec) and not dec.startswith(" "):
                nospace_ids.append(t)
                if len(nospace_ids) >= 2:
                    return

    if len(space_ids) < 3:
        try_add_space_like([" 0", " 1", " a", " b", " A", " B", " 2", " 3", " X", " Z"])
    if len(nospace_ids) < 2:
        try_add_nospace_like(["a", "b", "A", "B", "Z", "X", "0", "1", "2"])

    # 兜底：若仍为空，至少保证有1个 id 可用（极少发生）
    if not space_ids and filler_id is not None:
        space_ids.append(filler_id)
    if not nospace_ids and filler_id is not None:
        nospace_ids.append(filler_id)

    # 再兜底：如果 filler_id 仍为空，强制设为 vocab 的第一个非特殊 id
    if filler_id is None:
        for tid in range(vocab_size):
            if tid in special_ids:
                continue
            filler_id = tid
            break

    return nospace_ids, space_ids, (filler_id if filler_id is not None else 0)

def fix_to_target_token_len_by_ids(tokenizer, ids: List[int], target_len: int,
                                   add_token_id: int) -> List[int]:
    """
    通过“增删 token id”控制 re-encode 长度 == target_len。
    - 解码 -> 重新 encode 得到实际长度，与目标不符则：
        * 若不足：在尾部 append add_token_id
        * 若超出：在尾部 pop 一个 id
    - 循环至稳定。全程不抛异常。
    """
    ids = list(ids)
    for _ in range(4096):  # 上限迭代，足够保守
        text = decode_ids(tokenizer, ids)
        if not is_allowed_text(text):
            text = filter_allowed(text)
        cur_len = encode_len(tokenizer, text)
        if cur_len == target_len:
            return encode_ids(tokenizer, text)  # 返回 re-encode 后的 ids（确保一致）
        if cur_len < target_len:
            ids.append(add_token_id)
        else:
            if ids:
                ids.pop()
            else:
                # 空则先加再调
                ids.append(add_token_id)
    # 到此仍未达成，使用“直接构造”兜底：重复追加 add_token_id，再截断
    if not ids:
        ids = [add_token_id]
    while len(ids) < target_len:
        ids.append(add_token_id)
    ids = ids[:target_len]
    # 最终再走一遍 decode/encode，尽力贴近目标
    text = decode_ids(tokenizer, ids)
    text = filter_allowed(text)
    final_ids = encode_ids(tokenizer, text)
    # 若仍不等，简单按差额增删（再来一次有限步）
    for _ in range(256):
        if len(final_ids) == target_len:
            break
        if len(final_ids) < target_len:
            final_ids.append(add_token_id)
        else:
            final_ids.pop()
    return final_ids

def make_prefix_ids(tokenizer, nospace_ids: List[int], space_ids: List[int],
                    base_len: int, rng: random.Random, filler_id: int) -> List[int]:
    """
    生成长度为 base_len 的“公共前缀” ids：
    - 尝试：首个取 nospace，后续取 space，保证可读且 re-encode 稳定
    - 若不稳定，使用 fix_to_target_token_len_by_ids 调整
    """
    ids = []
    first = (nospace_ids[0] if nospace_ids else filler_id)
    ids.append(first)
    for _ in range(max(0, base_len - 1)):
        ids.append(space_ids[rng.randrange(len(space_ids))] if space_ids else filler_id)
    ids = fix_to_target_token_len_by_ids(tokenizer, ids, base_len,
                                         add_token_id=(space_ids[0] if space_ids else filler_id))
    return ids

def idx_to_bit_ids(idx: int, bits: int, bit0_id: int, bit1_id: int) -> List[int]:
    s = format(idx, f"0{bits}b")
    return [bit0_id if ch == "0" else bit1_id for ch in s]

# ------------------------- 主流程 -------------------------

def main():
    parser = argparse.ArgumentParser(description="生成仅含 英文+空格+数字 的 JSONL 数据集（无异常抛出，自愈长度）")
    parser.add_argument("--tokenizer-dir", type=str, default="/mnt/nfs/levis/DeepSeek-R1_w8a8_vllm")
    parser.add_argument("--seed", type=int, default=202508167)
    parser.add_argument("--tokens", type=int, default=3500)
    parser.add_argument("--data_num", type=int, default=200)
    parser.add_argument("--fraction", type=int, default=50)
    args = parser.parse_args()

    tokens = int(args.tokens)
    rng = random.Random(int(args.seed))

    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
    except Exception:
        # 兜底：如果加载失败，尝试不使用 fast
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)

    nospace_ids, space_ids, filler_id = build_safe_token_pools(tokenizer)

    # bit0 / bit1：尽量选择两个不同的“空格型”安全 token；不足则从 nospace 或 filler 补齐
    bit0_id = (space_ids[0] if len(space_ids) >= 1 else filler_id)
    bit1_id = (space_ids[1] if len(space_ids) >= 2 else (nospace_ids[0] if nospace_ids else filler_id))
    if bit1_id == bit0_id:
        # 强制找一个不同的
        pool = (space_ids + nospace_ids) or [filler_id]
        for t in pool:
            if t != bit0_id:
                bit1_id = t
                break

    # 公共前缀比例
    fraction = int(args.fraction)
    base_len = max(1, tokens*fraction // 100)
    # 允许的“额外公共前缀”上限（<=16），若 tokens 太小则自动缩小 bits
    max_extra = 16
    bits = max(1, min(max_extra, max(1, tokens - base_len)))

    # 构造公共前缀 ids 并稳定为 base_len
    prefix_ids = make_prefix_ids(tokenizer, nospace_ids, space_ids, base_len, rng, filler_id)

    # 输出目录
    out_dir = f"{tokens}_{args.seed}_dataset"
    ensure_dir(out_dir)

    prefix_path = os.path.join(out_dir, f"{tokens}_{args.seed}_bs1_{int(args.fraction)}_prefix.jsonl")
    data_path = os.path.join(out_dir, f"{tokens}_{args.seed}_bs{int(args.data_num)}_{int(args.fraction)}_prefix.jsonl")

    # 写入前缀文件（1 行）
    try:
        prefix_text = decode_ids(tokenizer, prefix_ids)
        if not is_allowed_text(prefix_text):
            prefix_text = filter_allowed(prefix_text)
        # 再次以长度为准修正（保证 re-encode 后长度 == base_len）
        prefix_ids_fixed = fix_to_target_token_len_by_ids(
            tokenizer, encode_ids(tokenizer, prefix_text), base_len,
            add_token_id=(space_ids[0] if space_ids else filler_id)
        )
        prefix_text = decode_ids(tokenizer, prefix_ids_fixed)
        with open(prefix_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question": prefix_text, "answer": ""}, ensure_ascii=True))
            f.write("\n")
    except Exception:
        # 极端容错：退化生成简单可控文本（A0 A1 ...），再修正到 base_len
        seed_text = " ".join(f"A{i%10}" for i in range(max(1, base_len)))
        prefix_ids_fixed = fix_to_target_token_len_by_ids(
            tokenizer, encode_ids(tokenizer, seed_text), base_len,
            add_token_id=(space_ids[0] if space_ids else filler_id)
        )
        prefix_text = decode_ids(tokenizer, prefix_ids_fixed)
        with open(prefix_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question": prefix_text, "answer": ""}, ensure_ascii=True))
            f.write("\n")

    # 生成后 4000 条
    total_follow = int(args.data_num)
    with open(data_path, "w", encoding="utf-8") as f:
        for i in range(total_follow):
            bit_ids = idx_to_bit_ids(i, bits=bits, bit0_id=bit0_id, bit1_id=bit1_id)
            # 先拼接 前缀 + bit 区，再用随机空格型 token 补齐
            cur_ids = list(prefix_ids_fixed) + bit_ids
            # 剩余长度
            remain = max(0, tokens - len(encode_ids(tokenizer, decode_ids(tokenizer, cur_ids))))
            # 先尽量用“空格型”安全 token 填充
            for _ in range(remain):
                cur_ids.append(space_ids[rng.randrange(len(space_ids))] if space_ids else filler_id)
            # 修正为精确 tokens 长度
            final_ids = fix_to_target_token_len_by_ids(
                tokenizer, cur_ids, tokens,
                add_token_id=(space_ids[0] if space_ids else filler_id)
            )
            q = decode_ids(tokenizer, final_ids)
            if not is_allowed_text(q):
                q = filter_allowed(q)
            f.write(json.dumps({"question": q, "answer": ""}, ensure_ascii=True))
            f.write("\n")

    print(f"[完成] 数据集已生成：\n  - {prefix_path}  (1 行)\n  - {data_path}    ({total_follow} 行)")
    print(f"[信息] 参数 tokens={tokens}, seed={args.seed}, bits(额外公共前缀控制)={bits}，公共前缀长度={base_len}, 公共前缀占比={fraction}")

if __name__ == "__main__":
    main()

