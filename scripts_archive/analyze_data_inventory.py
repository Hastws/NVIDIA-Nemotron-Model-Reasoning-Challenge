"""Analyze all available data files for the competition."""
import csv
from collections import Counter

def detect_type(p):
    p = p[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p): return "bit_ops"
    elif "encrypt" in p or "cipher" in p: return "cipher"
    elif "gravit" in p: return "gravity"
    elif "numeral" in p or "wonderland numbers" in p: return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p): return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p): return "symbol"
    return "unknown"

def analyze_csv(path, label):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"\n{label}: FILE NOT FOUND")
        return
    
    print(f"\n=== {label} ({len(rows)} rows) ===")
    print(f"  Columns: {list(rows[0].keys())}")
    types = Counter(detect_type(r['prompt']) for r in rows)
    for t in sorted(types):
        print(f"  {t}: {types[t]}")
    
    # Check answer format
    has_think = sum(1 for r in rows if '<think>' in r.get('answer', ''))
    avg_len = sum(len(r.get('answer', '')) for r in rows) / max(len(rows), 1)
    print(f"  Answers with <think>: {has_think}")
    print(f"  Avg answer length: {avg_len:.1f} chars")
    
    # Show a few sample answers
    for i in range(min(3, len(rows))):
        ans = rows[i].get('answer', '')[:100]
        t = detect_type(rows[i]['prompt'])
        print(f"  Sample {i} ({t}): {repr(ans)}")

# Full training set
analyze_csv('competition_data/train.csv', 'train.csv (full)')

# All data CSVs
for name in [
    'data/sft_balanced_100.csv',
    'data/sft_curated_700.csv',
    'data/sft_e1_plus_cipher100.csv',
    'data/sft_balanced_cot_600.csv',
    'data/sft_e1_hybrid_cot.csv',
    'data/sft_full_cot.csv',
]:
    analyze_csv(name, name.split('/')[-1])
