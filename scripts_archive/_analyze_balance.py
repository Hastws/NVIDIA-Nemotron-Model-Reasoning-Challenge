#!/usr/bin/env python3
"""Analyze data balance across types in CoT v2 hybrid dataset."""
import csv, json
from collections import Counter

# 1. Hybrid dataset balance
thinking_lens = {}
type_counts = Counter()
cot_counts = Counter()
ao_counts = Counter()

with open('data/sft_cot_v2_hybrid.csv') as f:
    for r in csv.DictReader(f):
        t = r['type']
        type_counts[t] += 1
        if r['thinking'].strip():
            cot_counts[t] += 1
            thinking_lens.setdefault(t, []).append(len(r['thinking']))
        else:
            ao_counts[t] += 1

print('=== sft_cot_v2_hybrid.csv Data Balance ===')
print(f'{"Type":<12} {"Total":>6} {"w/CoT":>6} {"AO":>6} {"CoT%":>8}')
print('-' * 42)
for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    total = type_counts[t]
    cot = cot_counts[t]
    ao = ao_counts[t]
    rate = cot / total * 100 if total else 0
    print(f'{t:<12} {total:6d} {cot:6d} {ao:6d} {rate:7.1f}%')
print(f'{"TOTAL":<12} {sum(type_counts.values()):6d} {sum(cot_counts.values()):6d} {sum(ao_counts.values()):6d}')

# 2. CoT length stats
print('\n=== CoT Length (chars) ===')
for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    if t in thinking_lens:
        lens = thinking_lens[t]
        print(f'{t:<12} avg={sum(lens)/len(lens):5.0f}  min={min(lens):4d}  max={max(lens):4d}  n={len(lens)}')

# 3. Token estimate per type
print('\n=== Prompt Token Estimates ===')
with open('data/cot_v2.jsonl') as f:
    records = [json.loads(l) for l in f]

for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    recs = [r for r in records if r['type'] == t]
    if recs:
        prompt_lens = [len(r['prompt']) for r in recs]
        total_chars = [len(r['prompt']) + len(r['answer']) for r in recs]
        token_est = [c // 3 for c in total_chars]  # ~3 chars per token for this style
        print(f'{t:<12} prompt_avg={sum(prompt_lens)//len(prompt_lens):5d} chars  '
              f'total_est≈{sum(token_est)//len(token_est):5d} tokens  n={len(recs)}')

# 4. Compare with E1 data distribution
print('\n=== E1 Baseline (600 samples, random) ===')
import pandas as pd
try:
    e1 = pd.read_csv('data/sft_ao_7741.csv')
    # detect type from prompt
    def detect_type(p):
        if 'bit manipulation' in p.lower(): return 'bit_ops'
        if 'gravity' in p.lower() or 'free-fall' in p.lower(): return 'gravity'
        if 'unit' in p.lower() and 'conversion' in p.lower(): return 'unit_conv'
        if 'cipher' in p.lower() or 'encrypt' in p.lower() or 'decrypt' in p.lower(): return 'cipher'
        if 'numeral' in p.lower() or 'roman' in p.lower(): return 'numeral'
        return 'symbol'
    
    e1['type'] = e1['prompt'].apply(detect_type)
    print(f'ao_7741 distribution:')
    print(e1['type'].value_counts().to_string())
except Exception as ex:
    print(f'Could not load E1 data: {ex}')
