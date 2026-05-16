#!/usr/bin/env python3
"""Diagnose bit_ops failures in detail."""
import csv
import json
import sys
sys.path.insert(0, 'scripts')
from generate_cot_v2 import detect_type, parse_bit_ops, enumerate_bit_functions, eval_bit_function

rows = [r for r in csv.DictReader(open('competition_data/train.csv')) 
        if detect_type(r['prompt']) == 'bit_ops']
solved_ids = set()
for line in open('data/cot_v2.jsonl'):
    rec = json.loads(line)
    if rec['type'] == 'bit_ops':
        solved_ids.add(rec['id'])

failed = [r for r in rows if r['id'] not in solved_ids]
print(f"Total bit_ops: {len(rows)}, Solved: {len(solved_ids)}, Failed: {len(failed)}")

no_match_bits = []
mismatch_count = 0
few_examples = 0

for r in failed[:30]:  # sample 30
    examples, target = parse_bit_ops(r['prompt'])
    gold = r['answer'].strip()
    
    if not examples or not target:
        print(f"  {r['id']}: PARSE FAILURE")
        continue
    
    if len(examples) < 4:
        few_examples += 1
        continue
    
    n = len(examples)
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    gold_bits = [int(gold[i]) for i in range(8)]
    
    no_match_for_this = []
    mismatch_for_this = []
    
    for obit in range(8):
        funcs = enumerate_bit_functions(inputs, outputs, n, obit)
        if not funcs:
            no_match_for_this.append(obit)
        else:
            preds = set()
            for f in funcs:
                p = eval_bit_function(f, target_bits)
                if p is not None:
                    preds.add(p)
            
            gb = gold_bits[obit]
            gold_funcs = [f for f in funcs if eval_bit_function(f, target_bits) == gb]
            if not gold_funcs:
                mismatch_for_this.append(obit)
    
    if no_match_for_this:
        no_match_bits.extend(no_match_for_this)
        if len(no_match_for_this) <= 2:
            # Show detail for these bits
            out_cols = {}
            for obit in no_match_for_this:
                out_col = [outputs[e][obit] for e in range(n)]
                out_cols[obit] = out_col
            print(f"  {r['id']}: NO MATCH for bits {no_match_for_this}, n_examples={n}")
            for obit in no_match_for_this:
                out_col = [outputs[e][obit] for e in range(n)]
                # Check if it might be a function of >4 input bits
                # Count how many 1s and 0s in output
                ones = sum(out_col)
                print(f"    bit {obit}: {ones}/{n} ones")
    elif mismatch_for_this:
        mismatch_count += 1

print(f"\nSummary (first 30 fails):")
print(f"  No-match: {len([1 for b in no_match_bits])} bit failures across problems")
print(f"  Mismatch: {mismatch_count} problems with all bits matched but wrong gold")
print(f"  Few examples: {few_examples}")
