#!/usr/bin/env python3
"""Quick check data overlap."""
import csv, json
from collections import Counter

# train_annotated: has solution_process (rules)
with open('data/train_annotated.csv') as f:
    ann = {r['id']: r for r in csv.DictReader(f)}

matched = {k:v for k,v in ann.items() if v['match']=='True' and v.get('solution_process','').strip()}
print(f'train_annotated: {len(ann)} total, {len(matched)} matched with solution_process')
print(f'Types: {Counter(v["type"] for v in matched.values())}')

# train_dsl_rules: has DSL
with open('data/train_dsl_rules.jsonl') as f:
    dsl = {}
    for line in f:
        r = json.loads(line)
        if r.get('dsl_score', 0) > 0:
            dsl[r['id']] = r
print(f'train_dsl_rules: {len(dsl)} with valid DSL')

# train_cot: existing CoT
with open('data/train_cot.jsonl') as f:
    cot = {}
    for line in f:
        r = json.loads(line)
        cot[r['id']] = r
print(f'train_cot: {len(cot)} rows, {sum(1 for r in cot.values() if r.get("passed"))} passed')

# Overlap
both = set(matched.keys()) & set(dsl.keys())
print(f'\nHas BOTH solution_process AND DSL: {len(both)}')
print(f'Types: {Counter(ann[k]["type"] for k in both)}')

# Show structure
for k in list(both)[:2]:
    d = dsl[k]
    a = ann[k]
    print(f'\n  [{d["type"]}] id={k}')
    print(f'    DSL rule: {d.get("dsl_rule","")[:120]}')
    print(f'    solution_process: {a["solution_process"][:120]}')
    print(f'    DSL keys: {[x for x in d.keys() if x not in ("prompt","id")]}')
