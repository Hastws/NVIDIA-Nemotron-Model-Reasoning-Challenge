#!/usr/bin/env python3
"""Analyze full DSL generation results."""
import json
from collections import defaultdict, Counter

with open('data/train_dsl_rules.jsonl') as f:
    rows = [json.loads(l) for l in f]

total = len(rows)
has_dsl = [r for r in rows if r.get('dsl') and r['score'] > 0]
no_dsl = [r for r in rows if not r.get('dsl') or r['score'] <= 0]

print(f"Total rows: {total}")
print(f"With DSL: {len(has_dsl)}")
print(f"Without DSL: {len(no_dsl)}")
print()

# Score distribution
scores = [r['score'] for r in has_dsl]
print(f"Score stats: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.1f}, median={sorted(scores)[len(scores)//2]}")

buckets = Counter()
for s in scores:
    if s >= 90: buckets['90+'] += 1
    elif s >= 80: buckets['80-89'] += 1
    elif s >= 70: buckets['70-79'] += 1
    elif s >= 60: buckets['60-69'] += 1
    elif s >= 50: buckets['50-59'] += 1
    else: buckets['<50'] += 1
print(f"Score buckets: {dict(sorted(buckets.items()))}")
print()

# By type
type_info = defaultdict(list)
for r in has_dsl:
    type_info[r['type']].append(r)

print("=== By Type ===")
for t in sorted(type_info):
    rr = type_info[t]
    ss = [r['score'] for r in rr]
    dsl_lens = [len(r['dsl']) for r in rr]
    print(f"  {t}: n={len(rr)}, score={sum(ss)/len(ss):.1f}, dsl_len={sum(dsl_lens)/len(dsl_lens):.0f} chars")

print()

# Sample outputs per type
print("=== Sample DSL per Type ===")
for t in sorted(type_info):
    rr = type_info[t]
    # Show 3 samples: best, median, worst
    rr_sorted = sorted(rr, key=lambda x: -x['score'])
    samples = [rr_sorted[0], rr_sorted[len(rr_sorted)//2], rr_sorted[-1]]
    print(f"\n--- {t} (n={len(rr)}) ---")
    for label, r in zip(['best', 'median', 'worst'], samples):
        dsl_preview = r['dsl'].replace('\n', ' | ')[:150]
        print(f"  [{label}] score={r['score']}: {dsl_preview}")

# DSL format consistency check
print("\n=== Format Quality Check ===")
import re
token_pat = re.compile(r'^\[.+\]$')
for t in sorted(type_info):
    rr = type_info[t]
    all_lines_ok = 0
    total_lines = 0
    for r in rr:
        lines = [l.strip() for l in r['dsl'].strip().split('\n') if l.strip()]
        total_lines += len(lines)
        all_lines_ok += sum(1 for l in lines if token_pat.match(l))
    pct = all_lines_ok / total_lines * 100 if total_lines else 0
    print(f"  {t}: {all_lines_ok}/{total_lines} lines in [TYPE:ARGS] format ({pct:.1f}%)")

# Check if symbol type has any
print()
symbol_rows = type_info.get('symbol', [])
if symbol_rows:
    print(f"Symbol type: {len(symbol_rows)} rows with DSL")
    for r in symbol_rows[:5]:
        print(f"  id={r['id']} score={r['score']}: {r['dsl'].replace(chr(10), ' | ')[:150]}")
