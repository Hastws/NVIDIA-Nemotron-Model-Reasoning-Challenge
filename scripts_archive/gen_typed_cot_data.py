"""
生成分类型简洁 CoT 训练数据

策略:
- 从 7741 solver-verified 数据中随机采样 5 种类型 (各~105条)
- 加入 symbol 约75条 (answer-only, 空thinking)
- 总计约 600 条, 与 V2 规模一致
- 每条都有 thinking 列 (symbol为空字符串)
"""
import json, os, random
import polars as pl
from collections import defaultdict

os.chdir('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛')

random.seed(42)

# 1. Load all programmatic CoT data
cot_by_id = {}
for fname in ['data_archive/programmatic_cot.jsonl',
              'data_archive/cipher_programmatic_cot.jsonl',
              'data_archive/bit_ops_programmatic_cot.jsonl']:
    with open(fname) as f:
        for line in f:
            d = json.loads(line)
            cot_by_id[d['id']] = d

print(f"Total programmatic CoT entries: {len(cot_by_id)}")

# 2. Load train.csv
train_df = pl.read_csv('competition_data/train.csv')

# 3. Classify types
def classify_type(prompt):
    p = prompt.lower()
    if 'bit' in p and ('operation' in p or 'transformation' in p):
        return 'bit_ops'
    elif 'gravity' in p or 'falling' in p or 'free fall' in p:
        return 'gravity'
    elif 'numeral' in p or 'numbering' in p:
        return 'numeral'
    elif 'encrypt' in p or 'decrypt' in p or 'cipher' in p or 'ciphertext' in p:
        return 'cipher'
    elif 'unit' in p or 'convert' in p and ('measurement' in p or 'wonderland' in p):
        return 'unit_conv'
    elif 'symbol' in p or 'transform' in p:
        return 'symbol'
    else:
        return 'unknown'

train_df = train_df.with_columns(
    pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('type')
)

# 4. Split by type: those with CoT vs symbol
cot_rows = []  # rows with programmatic CoT
symbol_rows = []  # symbol rows (no CoT)
no_cot_rows = []  # non-symbol without CoT (bit_ops/cipher failures)

for row in train_df.iter_rows(named=True):
    if row['id'] in cot_by_id:
        cot_entry = cot_by_id[row['id']]
        cot_rows.append({
            'id': row['id'],
            'prompt': row['prompt'],
            'answer': str(row['answer']),
            'thinking': cot_entry['thinking'],
            'type': row['type'],
        })
    elif row['type'] == 'symbol':
        symbol_rows.append({
            'id': row['id'],
            'prompt': row['prompt'],
            'answer': str(row['answer']),
            'thinking': '',  # empty thinking for symbol
            'type': 'symbol',
        })
    else:
        no_cot_rows.append({
            'id': row['id'],
            'type': row['type'],
        })

print(f"\nRows with CoT: {len(cot_rows)}")
print(f"Symbol rows (no CoT): {len(symbol_rows)}")
print(f"Non-symbol without CoT: {len(no_cot_rows)}")

# Count by type
cot_type_counts = defaultdict(int)
for r in cot_rows:
    cot_type_counts[r['type']] += 1
print(f"\nCoT by type:")
for t in sorted(cot_type_counts.keys()):
    print(f"  {t}: {cot_type_counts[t]}")

# 5. Sample strategy: V2-like random distribution
# V2 had: bit_ops=94, cipher=115, gravity=100, numeral=92, symbol=74, unit_conv=125
# ~500 non-symbol + ~100 symbol → 600

# We'll sample proportionally from solver-verified pool
# Total solver-verified: 7741 across 5 types → each type ~1400+
# Sample ~525 from 5 types (proportional), ~75 from symbol
TARGET_TOTAL = 600
TARGET_SYMBOL = 75
TARGET_COT = TARGET_TOTAL - TARGET_SYMBOL

# Group CoT rows by type
cot_by_type = defaultdict(list)
for r in cot_rows:
    cot_by_type[r['type']].append(r)

# Sample proportionally (approximately V2's distribution, but capped)
# V2 ratios (excluding symbol): bit_ops=94, cipher=115, gravity=100, numeral=92, unit_conv=125 → total=526
# Scale to 525:
v2_ratios = {'bit_ops': 94, 'cipher': 115, 'gravity': 100, 'numeral': 92, 'unit_conv': 125}
v2_total = sum(v2_ratios.values())

type_targets = {}
remaining = TARGET_COT
for t in sorted(v2_ratios.keys()):
    n = round(v2_ratios[t] / v2_total * TARGET_COT)
    type_targets[t] = min(n, len(cot_by_type[t]))
    remaining -= type_targets[t]

# Distribute any remainder
if remaining > 0:
    for t in sorted(type_targets.keys()):
        if remaining <= 0:
            break
        extra = min(remaining, len(cot_by_type[t]) - type_targets[t])
        type_targets[t] += extra
        remaining -= extra

print(f"\nSampling targets:")
total_check = 0
for t in sorted(type_targets.keys()):
    print(f"  {t}: {type_targets[t]} (from pool of {len(cot_by_type[t])})")
    total_check += type_targets[t]
print(f"  symbol: {TARGET_SYMBOL} (from pool of {len(symbol_rows)})")
print(f"  Total: {total_check + TARGET_SYMBOL}")

# 6. Random sample
selected = []
for t, n in type_targets.items():
    pool = cot_by_type[t]
    random.shuffle(pool)
    selected.extend(pool[:n])

# Sample symbol
random.shuffle(symbol_rows)
selected.extend(symbol_rows[:TARGET_SYMBOL])

random.shuffle(selected)

print(f"\nFinal dataset: {len(selected)} examples")

# 7. Verify thinking length distribution
type_think_lens = defaultdict(list)
for row in selected:
    type_think_lens[row['type']].append(len(row['thinking']))

print(f"\nThinking length distribution:")
for t in sorted(type_think_lens.keys()):
    vals = type_think_lens[t]
    import statistics
    if vals and max(vals) > 0:
        print(f"  {t}: n={len(vals)}, avg={statistics.mean(vals):.0f} chars, range=[{min(vals)}, {max(vals)}]")
    else:
        print(f"  {t}: n={len(vals)}, EMPTY thinking (answer-only)")

# 8. Save to CSV
out_df = pl.DataFrame(selected).select(['id', 'prompt', 'answer', 'thinking'])
out_path = 'data/sft_typed_cot_600.csv'
out_df.write_csv(out_path)
print(f"\nSaved to {out_path}")
print(f"Columns: {out_df.columns}")
print(f"Shape: {out_df.shape}")

# 9. Verify a few examples
print(f"\n=== Sample examples ===")
for t in sorted(type_think_lens.keys()):
    for row in selected:
        if row['type'] == t:
            think_preview = row['thinking'][:100] + '...' if len(row['thinking']) > 100 else row['thinking']
            print(f"\n[{t}] id={row['id']}")
            print(f"  thinking({len(row['thinking'])} chars): {think_preview}")
            print(f"  answer: {str(row['answer'])[:60]}")
            break
