#!/usr/bin/env python3
"""Deep analysis of sft_merged_v1.csv — data composition breakdown."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/sft_merged_v1.csv')
has_thinking = df['thinking'].notna() & (df['thinking'].str.strip() != '') & (df['thinking'].str.strip().str.lower() != 'nan')

print(f"=== sft_merged_v1.csv 完整分析 ===")
print(f"总行数: {len(df)}")
print(f"有 thinking: {has_thinking.sum()}")
print(f"纯 answer-only: {(~has_thinking).sum()}")

# Classify prompt types
def classify_type(prompt):
    p = str(prompt)[:100].lower()
    if 'bit manipulation' in p:
        return 'bit_ops'
    elif 'encryption' in p or 'cipher' in p:
        return 'cipher'
    elif 'gravitational' in p:
        return 'gravity'
    elif 'numeral' in p or 'converted into a different' in p:
        return 'numeral'
    elif 'unit conversion' in p:
        return 'unit_conv'
    elif 'transformation' in p:
        return 'symbol'
    else:
        return 'unknown'

df['type'] = df['prompt'].apply(classify_type)
df['has_thinking'] = has_thinking
df['thinking_len'] = df['thinking'].fillna('').str.len()

# --- Overall type distribution ---
print(f"\n=== 题型分布 (全部) ===")
print(df['type'].value_counts().to_string())

# --- By thinking vs answer-only ---
print(f"\n=== 有 thinking 的题型分布 ===")
thinking_df = df[df['has_thinking']]
print(thinking_df['type'].value_counts().to_string())

print(f"\n=== 纯 answer-only 的题型分布 ===")
ao_df = df[~df['has_thinking']]
print(ao_df['type'].value_counts().to_string())

# --- Thinking content categories ---
print(f"\n=== Thinking 内容分类 ===")

def classify_thinking(row):
    if not row['has_thinking']:
        return 'answer_only'
    t = str(row['thinking']).strip()
    tlen = len(t)
    if tlen < 30:
        return 'compact_rule'
    elif tlen < 100:
        return 'short_cot'
    elif tlen < 500:
        return 'medium_cot'
    else:
        return 'long_cot'

df['thinking_cat'] = df.apply(classify_thinking, axis=1)

# Cross-tabulation
ct = pd.crosstab(df['type'], df['thinking_cat'], margins=True)
print(ct.to_string())

# --- Stage 1 simulation (THINKING_ONLY + DROP_LONG) ---
print(f"\n=== Stage 1 模拟 (thinking_only + drop_long @ 512 tokens) ===")
stage1 = thinking_df.copy()
print(f"Thinking-only rows: {len(stage1)}")

# Approximate token count (chars / 3.5 for English, but these have mixed content)
# Better: estimate from thinking_len + prompt_len + answer_len + template overhead
stage1['approx_tokens'] = (stage1['prompt'].str.len() + stage1['thinking_len'] + stage1['answer'].astype(str).str.len() + 50) / 3.5
print(f"Approx tokens > 512: {(stage1['approx_tokens'] > 512).sum()}")
print(f"Approx tokens > 1024: {(stage1['approx_tokens'] > 1024).sum()}")
print(f"Approx tokens <= 512: {(stage1['approx_tokens'] <= 512).sum()}")

# --- Show sample thinking content by type ---
print(f"\n=== 每种题型的 Thinking 样本 ===")
for t in ['bit_ops', 'cipher', 'gravity', 'numeral', 'unit_conv', 'symbol']:
    sub = thinking_df[thinking_df['type'] == t]
    if len(sub) == 0:
        print(f"\n--- {t}: 无 thinking 数据 ---")
        continue
    print(f"\n--- {t} ({len(sub)} rows, avg thinking len={sub['thinking_len'].mean():.0f}) ---")
    # Show shortest, median, longest
    sorted_sub = sub.sort_values('thinking_len')
    for label, idx in [("SHORTEST", 0), ("MEDIAN", len(sorted_sub)//2), ("LONGEST", -1)]:
        row = sorted_sub.iloc[idx]
        print(f"  {label} ({row['thinking_len']} chars): {str(row['thinking'])[:120]}...")

# --- Duplicate check ---
print(f"\n=== 重复检查 ===")
dup_prompt = df['prompt'].duplicated().sum()
dup_full = df.duplicated(subset=['prompt', 'answer']).sum()
print(f"重复 prompt: {dup_prompt}")
print(f"重复 prompt+answer: {dup_full}")

# Check if same prompt appears in both thinking and answer-only
prompt_in_both = set(thinking_df['prompt']) & set(ao_df['prompt'])
print(f"同一 prompt 同时出现在 thinking 和 answer-only 中: {len(prompt_in_both)}")

# --- Data source estimation ---
print(f"\n=== 数据来源估算 ===")
# compact rules: thinking < 50 chars, typically from programmatic solvers
compact = thinking_df[thinking_df['thinking_len'] < 50]
print(f"Compact rules (<50 chars): {len(compact)}")
print(f"  Type breakdown: {compact['type'].value_counts().to_dict()}")

# full CoT: thinking >= 50 chars
full_cot = thinking_df[thinking_df['thinking_len'] >= 50]
print(f"Full CoT (>=50 chars): {len(full_cot)}")
print(f"  Type breakdown: {full_cot['type'].value_counts().to_dict()}")

# answer-only
print(f"Answer-only: {len(ao_df)}")
print(f"  Type breakdown: {ao_df['type'].value_counts().to_dict()}")
