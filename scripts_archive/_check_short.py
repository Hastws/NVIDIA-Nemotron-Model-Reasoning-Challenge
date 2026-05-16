#!/usr/bin/env python3
"""Quick check on short thinking entries."""
import csv, os
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
with open(os.path.join(DATA, 'sft_merged_v1.csv')) as f:
    rows = list(csv.DictReader(f))

short = [r for r in rows if r['thinking'].strip() and len(r['thinking'].split()) < 5]
print(f"Short thinking (<5 words): {len(short)}")
print("\nSamples:")
for r in short[:15]:
    print(f"  [{len(r['thinking'].split())}w] {r['thinking'][:150]}")

print(f"\nWord count distribution (all non-empty thinking):")
wc = Counter()
for r in rows:
    t = r['thinking'].strip()
    if t:
        w = len(t.split())
        bucket = f"{(w//10)*10}-{(w//10)*10+9}" if w >= 10 else str(w)
        wc[w if w < 10 else (w//10)*10] += 1

for k in sorted(wc):
    label = str(k) if k < 10 else f"{k}-{k+9}"
    print(f"  {label:>6s} words: {wc[k]:5d}")
