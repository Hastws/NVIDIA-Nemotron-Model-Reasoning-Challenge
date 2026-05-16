#!/usr/bin/env python3
"""Analyze CoT score distribution and type breakdown to inform merge strategy."""
import json, os
from collections import defaultdict, Counter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

with open(os.path.join(DATA, 'cot_best4_nobox.jsonl')) as f:
    rows = [json.loads(l) for l in f if l.strip()]

print(f"Total rows: {len(rows)}")
correct = [r for r in rows if r.get('correct')]
print(f"Correct: {len(correct)} ({len(correct)/len(rows)*100:.1f}%)")

# Score distribution
scores = [r['best_score'] for r in rows]
print(f"\nScore distribution:")
for thresh in [98, 97, 96, 95, 94, 93, 92, 90, 85, 80, 0]:
    n = sum(1 for s in scores if s >= thresh)
    print(f"  score >= {thresh:3d}: {n:5d} ({n/len(rows)*100:.1f}%)")

# Per-type analysis
print(f"\nPer-type score stats:")
type_scores = defaultdict(list)
for r in rows:
    type_scores[r['type']].append(r['best_score'])

print(f"{'Type':12s} {'Count':>6s} {'Correct':>8s} {'AvgScore':>9s} {'>=95':>6s} {'>=90':>6s} {'<90':>5s}")
for t in sorted(type_scores):
    sc = type_scores[t]
    correct_t = sum(1 for r in rows if r['type']==t and r.get('correct'))
    ge95 = sum(1 for s in sc if s >= 95)
    ge90 = sum(1 for s in sc if s >= 90)
    lt90 = sum(1 for s in sc if s < 90)
    avg = sum(sc)/len(sc)
    print(f"{t:12s} {len(sc):6d} {correct_t:8d} {avg:9.1f} {ge95:6d} {ge90:6d} {lt90:5d}")

# Word count per type
print(f"\nPer-type word count (thinking field):")
type_words = defaultdict(list)
for r in rows:
    type_words[r['type']].append(len(r.get('thinking','').split()))
for t in sorted(type_words):
    ws = type_words[t]
    print(f"  {t:12s}: min={min(ws):3d} avg={sum(ws)//len(ws):3d} max={max(ws):3d}")

# If we filter score >= 95
print(f"\n--- Filtering analysis ---")
for thresh in [95, 93, 90]:
    filtered = [r for r in rows if r['best_score'] >= thresh]
    type_counts = Counter(r['type'] for r in filtered)
    print(f"\nscore >= {thresh}: {len(filtered)} rows")
    for t in sorted(type_counts):
        total_t = len(type_scores[t])
        print(f"  {t:12s}: {type_counts[t]:5d}/{total_t} ({type_counts[t]/total_t*100:.0f}%)")
