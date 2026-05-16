#!/usr/bin/env python3
"""
Prepare comprehensive training data with CoT for all solvable types.
Strategy:
- Use ALL 7071 solved samples with CoT (reasoning_content field)
- For symbol type: include as answer-only (from E1 base)
- Goal: maximize coverage while maintaining quality
"""
import polars as pl
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
COMP_DIR = os.path.join(os.path.dirname(__file__), '..', 'competition_data')

# Load all programmatic CoT data
cot_by_id = {}

# Easy types (numeral, gravity, unit_conv)
with open(os.path.join(DATA_DIR, 'programmatic_cot.jsonl'), 'r') as f:
    for line in f:
        d = json.loads(line)
        cot_by_id[d['id']] = {'thinking': d['thinking'], 'type': d['type']}

# Cipher
with open(os.path.join(DATA_DIR, 'cipher_programmatic_cot.jsonl'), 'r') as f:
    for line in f:
        d = json.loads(line)
        cot_by_id[d['id']] = {'thinking': d['thinking'], 'type': d['type']}

# Bit_ops
with open(os.path.join(DATA_DIR, 'bit_ops_programmatic_cot.jsonl'), 'r') as f:
    for line in f:
        d = json.loads(line)
        cot_by_id[d['id']] = {'thinking': d['thinking'], 'type': d['type']}

print(f"Total CoT entries: {len(cot_by_id)}")

# Count by type
from collections import Counter
type_counts = Counter(v['type'] for v in cot_by_id.values())
print(f"CoT by type: {dict(type_counts)}")

# Load train.csv
train_df = pl.read_csv(os.path.join(COMP_DIR, 'train.csv'))

# === Strategy 1: E1-size hybrid (600 samples) ===
# Same E1 sampling + add CoT where available
e1_sampled = train_df.sample(n=600, seed=42)
ids = e1_sampled['id'].to_list()
thinking_col = [cot_by_id[i]['thinking'] if i in cot_by_id else "" for i in ids]
e1_hybrid = e1_sampled.with_columns(pl.Series("thinking", thinking_col))
e1_with_cot = sum(1 for t in thinking_col if t)
e1_without_cot = sum(1 for t in thinking_col if not t)
print(f"\n=== E1 Hybrid (600) ===")
print(f"  With CoT: {e1_with_cot}, Without CoT: {e1_without_cot}")

# Save
e1_hybrid.write_csv(os.path.join(DATA_DIR, 'sft_e1_hybrid_cot.csv'))

# === Strategy 2: Full CoT (all solved, ~7071 samples) ===
# All samples that have CoT
all_ids_with_cot = set(cot_by_id.keys())
full_cot_df = train_df.filter(pl.col('id').is_in(list(all_ids_with_cot)))
ids2 = full_cot_df['id'].to_list()
thinking_col2 = [cot_by_id[i]['thinking'] for i in ids2]
full_cot = full_cot_df.with_columns(pl.Series("thinking", thinking_col2))
print(f"\n=== Full CoT ({len(full_cot)}) ===")
tc2 = Counter(cot_by_id[i]['type'] for i in ids2)
print(f"  By type: {dict(tc2)}")
full_cot.write_csv(os.path.join(DATA_DIR, 'sft_full_cot.csv'))

# === Strategy 3: Balanced 600 with max CoT coverage ===
# Sample ~100 per type, preferring samples with CoT
import random
random.seed(42)

balanced_ids = []
train_rows = train_df.to_dicts()

# Detect type for each row (quick)
def detect_type(prompt):
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p):
        return "bit_ops"
    elif "encrypt" in p or "cipher" in p or "secret code" in p:
        return "cipher"
    elif "gravit" in p:
        return "gravity"
    elif "numeral" in p:
        return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p) or ("secret unit" in p):
        return "unit_conv"
    else:
        return "symbol"

type_rows = {}
for row in train_rows:
    t = detect_type(row['prompt'])
    type_rows.setdefault(t, []).append(row)

print(f"\n=== Type detection ===")
for t, rows in sorted(type_rows.items()):
    print(f"  {t}: {len(rows)}")

# For balanced sampling: 100/type, prefer CoT
for t, rows in type_rows.items():
    with_cot = [r for r in rows if r['id'] in cot_by_id]
    without_cot = [r for r in rows if r['id'] not in cot_by_id]
    
    random.shuffle(with_cot)
    random.shuffle(without_cot)
    
    # Take up to 100 from with_cot, fill remainder from without_cot
    selected = with_cot[:100]
    if len(selected) < 100:
        remaining = 100 - len(selected)
        selected.extend(without_cot[:remaining])
    
    balanced_ids.extend([r['id'] for r in selected])

balanced_df = train_df.filter(pl.col('id').is_in(balanced_ids))
ids3 = balanced_df['id'].to_list()
thinking_col3 = [cot_by_id[i]['thinking'] if i in cot_by_id else "" for i in ids3]
balanced = balanced_df.with_columns(pl.Series("thinking", thinking_col3))
b_with_cot = sum(1 for t in thinking_col3 if t)
b_without_cot = sum(1 for t in thinking_col3 if not t)
print(f"\n=== Balanced 600 ===")
print(f"  Total: {len(balanced)}")
print(f"  With CoT: {b_with_cot}, Without CoT: {b_without_cot}")
balanced.write_csv(os.path.join(DATA_DIR, 'sft_balanced_cot_600.csv'))

# === Strategy 4: Medium size (1000-2000) with all CoT ===
# Take all solved + 100 from each unsolved type
medium_ids = list(all_ids_with_cot)

# Add symbol samples (no CoT)
symbol_rows = [r for r in type_rows.get('symbol', []) if r['id'] not in cot_by_id]
random.shuffle(symbol_rows)
medium_ids.extend([r['id'] for r in symbol_rows[:200]])

# Add more bit_ops without CoT
bitops_no_cot = [r for r in type_rows.get('bit_ops', []) if r['id'] not in cot_by_id]
random.shuffle(bitops_no_cot)
medium_ids.extend([r['id'] for r in bitops_no_cot[:100]])

medium_df = train_df.filter(pl.col('id').is_in(medium_ids))
ids4 = medium_df['id'].to_list()
thinking_col4 = [cot_by_id[i]['thinking'] if i in cot_by_id else "" for i in ids4]
medium = medium_df.with_columns(pl.Series("thinking", thinking_col4))
m_with_cot = sum(1 for t in thinking_col4 if t)
m_without_cot = sum(1 for t in thinking_col4 if not t)
print(f"\n=== Medium ({len(medium)}) ===")
print(f"  With CoT: {m_with_cot}, Without CoT: {m_without_cot}")
medium.write_csv(os.path.join(DATA_DIR, 'sft_medium_cot.csv'))

print(f"\n=== Summary ===")
print(f"  sft_e1_hybrid_cot.csv: 600 (E1 base + CoT where available)")
print(f"  sft_full_cot.csv: {len(full_cot)} (all solved with CoT)")
print(f"  sft_balanced_cot_600.csv: {len(balanced)} (balanced 100/type, max CoT)")
print(f"  sft_medium_cot.csv: {len(medium)} (all CoT + symbol/bit_ops padding)")
