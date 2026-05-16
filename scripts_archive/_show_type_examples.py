#!/usr/bin/env python3
"""Show detailed examples per type for CoT prompt design."""
import csv, json, random
random.seed(42)

with open('data/train_annotated.csv') as f:
    ann = {r['id']: r for r in csv.DictReader(f)}

dsl_data = {}
with open('data/train_dsl_rules.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        dsl_data[obj['id']] = obj

types = ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']
for t in types:
    candidates = [k for k, v in dsl_data.items() if v['type'] == t and v.get('dsl') and v['score'] > 0]
    if not candidates:
        print(f'=== {t}: NO DSL ===\n')
        continue
    rid = random.choice(candidates)
    a = ann[rid]
    d = dsl_data[rid]
    print(f'=== {t} (id={rid}) ===')
    print(f'PROMPT (first 600 chars):')
    print(a['prompt'][:600])
    if len(a['prompt']) > 600:
        print('...[truncated]...')
    print(f'\nANSWER: {a["answer"]}')
    print(f'\nSOLVER PROCESS: {a["solution_process"][:400]}')
    print(f'\nDSL: {d["dsl"]}')
    print(f'\n{"="*80}\n')
