#!/usr/bin/env python3
import csv

rows = [r for r in csv.DictReader(open('competition_data/train.csv'))
        if 'symbol' in r['prompt'].lower() or 'equation' in r['prompt'].lower()]

for r in rows[:8]:
    p = r['prompt']
    print(f'Answer: {r["answer"]}')
    lines = p.strip().split('\n')
    for l in lines:
        l = l.strip()
        if '=' in l or 'determine' in l.lower() or 'result' in l.lower():
            print(f'  {l}')
    print()
