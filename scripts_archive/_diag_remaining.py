#!/usr/bin/env python3
"""Diagnose the 1 failed unit_conv and 14 failed bit_ops."""
import csv
import json
import re

# Load solved IDs
solved = {}
for line in open('data/cot_v2.jsonl'):
    r = json.loads(line)
    solved[r['id']] = r['type']

rows = list(csv.DictReader(open('competition_data/train.csv')))

def detect_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or 'bit shift' in p: return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p: return 'gravity'
    if 'unit conversion' in p or 'conversion factor' in p: return 'unit_conv'
    if 'cipher' in p or 'encrypt' in p: return 'cipher'
    if 'numeral' in p or ('base' in p and 'convert' in p): return 'numeral'
    if 'symbol' in p or 'equation' in p: return 'symbol'
    return 'unknown'

print("=== Failed unit_conv ===")
for r in rows:
    if detect_type(r['prompt']) == 'unit_conv' and r['id'] not in solved:
        print(f"ID: {r['id']}")
        print(f"Answer: {r['answer']}")
        # Try to parse
        prompt = r['prompt']
        pairs = re.findall(r'([\d.]+)\s*\w*\s+becomes\s+([\d.]+)', prompt)
        query_m = re.search(r'convert.*?:\s*([\d.]+)', prompt, re.I)
        if not query_m:
            after = prompt.split('Now')[-1] if 'Now' in prompt else ''
            query_m = re.search(r'([\d.]+)\s*\w', after)
        
        if pairs:
            print(f"Pairs found: {len(pairs)}")
            for p in pairs:
                print(f"  {p[0]} -> {p[1]}, factor={float(p[1])/float(p[0]):.6f}")
        else:
            print("NO PAIRS FOUND")
            print(prompt[:300])
        
        if query_m:
            print(f"Query: {query_m.group(1)}")
            if pairs:
                factors = [float(p[1])/float(p[0]) for p in pairs]
                avg = sum(factors)/len(factors)
                result = avg * float(query_m.group(1))
                print(f"Computed: {result:.2f}, Gold: {r['answer']}")
                print(f"Diff: {abs(result - float(r['answer'])):.4f}")
        else:
            print("NO QUERY FOUND")
            print(prompt[-200:])

print("\n=== Failed bit_ops (14) ===")
count = 0
for r in rows:
    if detect_type(r['prompt']) == 'bit_ops' and r['id'] not in solved:
        count += 1
        print(f"\n{count}. ID: {r['id']}, Gold: {r['answer']}")
        # Count examples
        lines = r['prompt'].strip().split('\n')
        examples = [(l.split(' -> ')[0].strip(), l.split(' -> ')[1].strip()) 
                     for l in lines if ' -> ' in l 
                     and len(l.split(' -> ')[0].strip()) == 8 
                     and len(l.split(' -> ')[1].strip()) == 8]
        print(f"  Examples: {len(examples)}")
