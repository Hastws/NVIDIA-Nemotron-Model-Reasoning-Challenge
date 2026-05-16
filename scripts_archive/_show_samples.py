#!/usr/bin/env python3
import csv
from collections import defaultdict

rows = list(csv.DictReader(open('competition_data/train.csv')))

types = defaultdict(list)
for r in rows:
    p = r['prompt'].lower()
    if 'bit manipulation' in p or 'bit shift' in p:
        types['bit_ops'].append(r)
    elif 'gravitational' in p or 'gravity' in p:
        types['gravity'].append(r)
    elif 'unit conversion' in p or 'conversion factor' in p:
        types['unit_conv'].append(r)
    elif 'cipher' in p or 'encrypt' in p:
        types['cipher'].append(r)
    elif 'numeral' in p or ('base' in p and 'convert' in p):
        types['numeral'].append(r)
    elif 'symbol' in p or 'equation' in p:
        types['symbol'].append(r)
    else:
        types['unknown'].append(r)

for t in ['gravity', 'unit_conv', 'numeral', 'cipher', 'bit_ops', 'symbol']:
    print(f'=== {t} ({len(types[t])}) ===')
    r = types[t][0]
    print(f'Answer: {r["answer"]}')
    print(r['prompt'][:500])
    print('---')
    print()
