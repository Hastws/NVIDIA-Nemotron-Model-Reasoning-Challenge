#!/usr/bin/env python3
"""Compare V2 (0.68) vs v33 (0.58) data sampling and config."""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from collections import Counter

train_df = pl.read_csv('competition_data/train.csv')
print(f"Total: {len(train_df)}")

def classify_type(p):
    p = p.lower()
    if 'bit manipulation' in p or '8-bit binary' in p: return 'bit_ops'
    elif 'encrypt' in p or 'decrypt' in p: return 'cipher'
    elif 'gravitational' in p or 'falling distance' in p: return 'gravity'
    elif 'numeral system' in p: return 'numeral'
    elif 'transformation rules' in p: return 'symbol'
    elif 'unit conversion' in p or 'convert the following measurement' in p: return 'unit_conv'
    return 'unknown'

# V2: simple random sample
v2_sample = train_df.sample(n=600, seed=42)
v2_types = [classify_type(p) for p in v2_sample['prompt'].to_list()]
v2_counts = Counter(v2_types)
print("\n=== V2: train_df.sample(n=600, seed=42) ===")
for t in sorted(v2_counts.keys()):
    print(f"  {t}: {v2_counts[t]}")

# v33: stratified 100/type
train_df2 = train_df.with_columns(
    pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('qtype')
)
v33_dfs = []
for qtype in sorted(train_df2['qtype'].unique().to_list()):
    subset = train_df2.filter(pl.col('qtype') == qtype)
    v33_dfs.append(subset.sample(n=min(100, len(subset)), seed=42))
v33_sample = pl.concat(v33_dfs)
print("\n=== v33: stratified 100/type ===")
for row in v33_sample['qtype'].value_counts().sort('qtype').iter_rows():
    print(f"  {row[0]}: {row[1]}")

# Overlap
v2_ids = set(v2_sample['id'].to_list())
v33_ids = set(v33_sample['id'].to_list())
overlap = v2_ids & v33_ids
print(f"\n=== Overlap ===")
print(f"V2: {len(v2_ids)} | v33: {len(v33_ids)} | Common: {len(overlap)} ({len(overlap)/600*100:.1f}%)")

print("\n=== KEY DIFFERENCES ===")
print("1. Sampling: V2=random, v33=stratified 100/type")
print("2. Suffix: V2='\\nPut your final answer inside \\\\boxed{}.'")
print("   v33='\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`'")
print("3. logging_steps: V2=5, v33=10")
print("4. V2 missing: cutlass mocks (V2 has no MagicMock for mamba_ssm.ops.cute)")
