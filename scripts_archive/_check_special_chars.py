#!/usr/bin/env python3
import json
from collections import Counter

data = [json.loads(l) for l in open('data/train_cot_v3.jsonl')]

special = Counter()
for d in data:
    for c in d['cot']:
        if ord(c) > 127:
            special[c] += 1

print('Non-ASCII chars in CoT:')
for ch, cnt in special.most_common():
    print(f'  U+{ord(ch):04X}  {repr(ch):6s}  {ch}  count={cnt}')

# Show example contexts for × 
print('\n--- Examples with special x (×) ---')
for d in data[:5]:
    cot = d['cot']
    for i, c in enumerate(cot):
        if c == '\u00d7':  # ×
            start = max(0, i-15)
            end = min(len(cot), i+15)
            print(f'  ...{cot[start:end]}...')
            break
