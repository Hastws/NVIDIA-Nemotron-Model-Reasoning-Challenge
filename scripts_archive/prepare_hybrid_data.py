#!/usr/bin/env python3
"""
Prepare hybrid training data for SFT experiment:
- E1 base: 600 samples (seed=42) from train.csv
- For numeral/gravity/unit_conv: add thinking from programmatic_cot.jsonl
- For cipher/bit_ops/symbol: answer-only
- Output: data/sft_e1_hybrid_cot.csv with columns: id, prompt, answer, thinking
"""
import polars as pl
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
COMP_DIR = os.path.join(os.path.dirname(__file__), '..', 'competition_data')

# Load train.csv and replicate E1 sampling
train_df = pl.read_csv(os.path.join(COMP_DIR, 'train.csv'))
e1_sampled = train_df.sample(n=600, seed=42)
print(f"E1 sampled: {len(e1_sampled)} rows")

# Load programmatic CoT
cot_by_id = {}
with open(os.path.join(DATA_DIR, 'programmatic_cot.jsonl'), 'r') as f:
    for line in f:
        d = json.loads(line)
        cot_by_id[d['id']] = d['thinking']

# Add thinking column
ids = e1_sampled['id'].to_list()
thinking_col = []
with_cot = 0
without_cot = 0
for row_id in ids:
    if row_id in cot_by_id:
        thinking_col.append(cot_by_id[row_id])
        with_cot += 1
    else:
        thinking_col.append("")
        without_cot += 1

result_df = e1_sampled.with_columns(
    pl.Series("thinking", thinking_col)
)

# Save
out_path = os.path.join(DATA_DIR, 'sft_e1_hybrid_cot.csv')
result_df.write_csv(out_path)
print(f"Saved to {out_path}")
print(f"  With CoT: {with_cot}")
print(f"  Answer-only: {without_cot}")

# Verify
df = pl.read_csv(out_path)
print(f"  Columns: {df.columns}")
print(f"  Rows: {len(df)}")
print(f"  Non-empty thinking: {(df['thinking'].str.len_chars() > 0).sum()}")
print(f"  Sample with thinking: {df.filter(pl.col('thinking').str.len_chars() > 0)[0, 'thinking'][:100]}...")
