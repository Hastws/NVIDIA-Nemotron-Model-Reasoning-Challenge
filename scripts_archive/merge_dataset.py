#!/usr/bin/env python3
"""
Merge datasets into final SFT training CSV.

Strategy:
  1. Compact (DSL rules): ALL 8121 matched rows — thinking = compact DSL
  2. Full CoT: type-aware quality filter — thinking = natural reasoning (no boxed)
     - numeral:   0 (too simple)
     - unit_conv:  0 (too simple)
     - gravity:  score>=96 (~300)
     - symbol:   score>=85 (all ~230)
     - cipher:   score>=85 (all ~1500)
     - bit_ops:  score>=80 (all ~1500)
  3. Answer-only: random 4000 from 8121 matched rows — thinking = empty

Output: data/sft_merged_v1.csv  (columns: id, prompt, answer, thinking)
  - For compact: thinking = DSL string
  - For full CoT: thinking = natural reasoning text
  - For answer-only: thinking = "" (empty)
  - All rows: answer is the raw answer (no boxed — boxed is added by training script)
"""
import os, csv, json, random
from collections import Counter, defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# 1. Load sources
# ═══════════════════════════════════════════════════════════════════

# train_annotated.csv — all 9500 rows, 8121 matched
print("Loading train_annotated.csv...")
with open(os.path.join(DATA, 'train_annotated.csv')) as f:
    all_rows = {r['id']: r for r in csv.DictReader(f)}
matched = {k: v for k, v in all_rows.items() if v['match'] == 'True' and v.get('solution_process', '').strip()}
print(f"  Total: {len(all_rows)}, Matched: {len(matched)}")

# train_dsl_rules.jsonl — DSL compact rules
print("Loading train_dsl_rules.jsonl...")
with open(os.path.join(DATA, 'train_dsl_rules.jsonl')) as f:
    dsl_map = {}
    for line in f:
        if line.strip():
            r = json.loads(line)
            if r['score'] > 0:
                dsl_map[r['id']] = r['dsl']
print(f"  Valid DSL rules: {len(dsl_map)}")

# cot_best4_nobox.jsonl — full CoT (no boxed)
cot_file = os.path.join(DATA, 'cot_best4_nobox.jsonl')
print(f"Loading {os.path.basename(cot_file)}...")
cot_map = {}
with open(cot_file) as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            cot_map[r['id']] = r
print(f"  CoT rows: {len(cot_map)}")

# ═══════════════════════════════════════════════════════════════════
# 2. Build merged dataset
# ═══════════════════════════════════════════════════════════════════

output_rows = []

# --- Part A: Compact (all 8121 matched, thinking = DSL) ---
compact_count = 0
for rid, row in matched.items():
    dsl = dsl_map.get(rid, row.get('solution_process', ''))
    output_rows.append({
        'id': rid,
        'prompt': row['prompt'],
        'answer': row['answer'],
        'thinking': dsl,
        'source': 'compact',
        'type': row['type'],
    })
    compact_count += 1
print(f"\nPart A - Compact: {compact_count}")

# --- Part B: Full CoT (type-aware quality filter) ---
COT_THRESHOLDS = {
    'numeral':   999,   # skip (score never reaches 999)
    'unit_conv': 999,   # skip
    'gravity':   96,
    'symbol':    85,
    'cipher':    85,
    'bit_ops':   80,
}

cot_count = 0
cot_by_type = Counter()
for rid, cot in cot_map.items():
    if not cot.get('correct', False):
        continue
    ptype = cot['type']
    thresh = COT_THRESHOLDS.get(ptype, 999)
    if cot['best_score'] >= thresh:
        row = matched.get(rid)
        if row:
            output_rows.append({
                'id': rid,
                'prompt': row['prompt'],
                'answer': row['answer'],
                'thinking': cot['thinking'],
                'source': 'full_cot',
                'type': ptype,
            })
            cot_count += 1
            cot_by_type[ptype] += 1

print(f"Part B - Full CoT: {cot_count}")
for t in sorted(cot_by_type):
    print(f"  {t:12s}: {cot_by_type[t]}")

# --- Part C: Answer-only (random 4000 from matched) ---
ao_ids = random.sample(list(matched.keys()), min(4000, len(matched)))
ao_count = 0
for rid in ao_ids:
    row = matched[rid]
    output_rows.append({
        'id': rid,
        'prompt': row['prompt'],
        'answer': row['answer'],
        'thinking': '',
        'source': 'answer_only',
        'type': row['type'],
    })
    ao_count += 1
print(f"Part C - Answer-only: {ao_count}")

# ═══════════════════════════════════════════════════════════════════
# 3. Shuffle and write
# ═══════════════════════════════════════════════════════════════════
random.shuffle(output_rows)

# Write CSV (columns expected by training script: id, prompt, answer, thinking)
out_path = os.path.join(DATA, 'sft_merged_v1.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking'])
    w.writeheader()
    for r in output_rows:
        w.writerow({
            'id': r['id'],
            'prompt': r['prompt'],
            'answer': r['answer'],
            'thinking': r['thinking'],
        })

# ═══════════════════════════════════════════════════════════════════
# 4. Summary
# ═══════════════════════════════════════════════════════════════════
total = len(output_rows)
by_source = Counter(r['source'] for r in output_rows)
by_type = Counter(r['type'] for r in output_rows)

print(f"\n{'='*60}")
print(f"MERGED DATASET: {out_path}")
print(f"Total rows: {total}")
print(f"\nBy source:")
for s in ['compact', 'full_cot', 'answer_only']:
    print(f"  {s:15s}: {by_source[s]:5d} ({by_source[s]/total*100:.1f}%)")
print(f"\nBy type:")
for t in sorted(by_type):
    n = by_type[t]
    print(f"  {t:12s}: {n:5d}")

# Also check thinking length stats
has_thinking = sum(1 for r in output_rows if r['thinking'])
empty_thinking = total - has_thinking
print(f"\nWith thinking: {has_thinking}, Without (answer-only): {empty_thinking}")
print(f"Done!")
