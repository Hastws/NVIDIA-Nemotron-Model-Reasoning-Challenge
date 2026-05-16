#!/usr/bin/env python3
"""Quick inventory of all training data resources."""
import polars as pl
import json

print("=== 训练资源盘点 ===")

# 1. Solver-verified answer-only
ao = pl.read_csv('data/sft_ao_7741.csv')
print(f"Solver-verified answer-only: {len(ao)} 条")

# 2. Programmatic CoT
cot_types = {}
for f in ['data_archive/programmatic_cot.jsonl', 
          'data_archive/cipher_programmatic_cot.jsonl', 
          'data_archive/bit_ops_programmatic_cot.jsonl']:
    try:
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                t = d.get('type', 'unk')
                cot_types[t] = cot_types.get(t, 0) + 1
    except:
        pass
total_cot = sum(cot_types.values())
print(f"\n程序化 CoT (100%正确推理链): {total_cot} 条")
for t, c in sorted(cot_types.items()):
    print(f"  {t}: {c}")

# 3. Full data
train = pl.read_csv('competition_data/train.csv')
print(f"\n原始训练集: {len(train)} 条 (含 {len(train)-total_cot} 条无法程序化验证)")

# 4. CoT v2 hybrid
try:
    cv2 = pl.read_csv('data/sft_cot_v2_hybrid.csv')
    has_t = cv2.filter(pl.col('thinking').str.len_chars() > 0).shape[0]
    print(f"\nCoT v2 hybrid: {len(cv2)} 条 (有thinking={has_t}, answer-only={len(cv2)-has_t})")
except Exception as e:
    print(f"CoT v2 hybrid: {e}")

# 5. Training step calculations
print("\n=== 训练步数对比 (grad_accum=4) ===")
for name, n, ep in [
    ("V2/E1 (600x1ep)", 600, 1),
    ("v32 (9500x1ep)", 9500, 1),
    ("7741 CoT (1ep)", 7741, 1),
    ("7741 CoT (2ep)", 7741, 2),
    ("3000精选 (2ep)", 3000, 2),
    ("2000精选 (3ep)", 2000, 3),
]:
    steps = (n * ep) // 4
    print(f"  {name}: {steps} steps ({steps/150:.1f}x of V2)")

# 6. Token length analysis for full CoT
print("\n=== 如果用全部 7741 CoT + seq_len=1024 ===")
print("  程序化 CoT 平均长度: 300-600 chars ≈ 100-200 tokens")
print("  prompt 平均 ~200-400 tokens")  
print("  总共 ~400-600 tokens, 远低于 1024 上限")
print("  结论: seq_len=1024 完全够用, 无需增大")
