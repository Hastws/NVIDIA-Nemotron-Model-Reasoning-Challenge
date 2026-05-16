#!/usr/bin/env python3
"""
Analyze what functions could match the no-match bits.
Try: AND-XOR composites, 2-input with NOT on one, etc.
"""
import csv
import json
import sys
from itertools import combinations
sys.path.insert(0, 'scripts')
from generate_cot_v2 import detect_type, parse_bit_ops, enumerate_bit_functions

rows = [r for r in csv.DictReader(open('competition_data/train.csv'))
        if detect_type(r['prompt']) == 'bit_ops']
solved_ids = set()
for line in open('data/cot_v2.jsonl'):
    rec = json.loads(line)
    if rec['type'] == 'bit_ops':
        solved_ids.add(rec['id'])

failed = [r for r in rows if r['id'] not in solved_ids]

# For each failed problem, try to find what function matches the no-match bits
# by trying more complex function templates

def try_extended_functions(inputs, outputs, n, obit, target_bits, gold_bit):
    """Try extended function set beyond level 9."""
    out_col = [outputs[e][obit] for e in range(n)]
    matches = []
    
    # NOT-AND: NOT(a) AND b
    for j in range(8):
        for k in range(8):
            if j == k: continue
            col = [(1-inputs[e][j]) & inputs[e][k] for e in range(n)]
            if col == out_col:
                pred = (1-target_bits[j]) & target_bits[k]
                matches.append(('not_and', (j,k), f"NOT(in[{j}]) AND in[{k}]", pred))
    
    # NOT-OR: NOT(a) OR b  
    for j in range(8):
        for k in range(8):
            if j == k: continue
            col = [(1-inputs[e][j]) | inputs[e][k] for e in range(n)]
            if col == out_col:
                pred = (1-target_bits[j]) | target_bits[k]
                matches.append(('not_or', (j,k), f"NOT(in[{j}]) OR in[{k}]", pred))
    
    # AND-XOR: (a AND b) XOR c
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(8):
                if l == j or l == k: continue
                col = [(inputs[e][j] & inputs[e][k]) ^ inputs[e][l] for e in range(n)]
                if col == out_col:
                    pred = (target_bits[j] & target_bits[k]) ^ target_bits[l]
                    matches.append(('and_xor', (j,k,l), f"(in[{j}] AND in[{k}]) XOR in[{l}]", pred))

    # OR-XOR: (a OR b) XOR c
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(8):
                if l == j or l == k: continue
                col = [(inputs[e][j] | inputs[e][k]) ^ inputs[e][l] for e in range(n)]
                if col == out_col:
                    pred = (target_bits[j] | target_bits[k]) ^ target_bits[l]
                    matches.append(('or_xor', (j,k,l), f"(in[{j}] OR in[{k}]) XOR in[{l}]", pred))
    
    # XOR-AND: (a XOR b) AND c
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(8):
                if l == j or l == k: continue
                col = [(inputs[e][j] ^ inputs[e][k]) & inputs[e][l] for e in range(n)]
                if col == out_col:
                    pred = (target_bits[j] ^ target_bits[k]) & target_bits[l]
                    matches.append(('xor_and', (j,k,l), f"(in[{j}] XOR in[{k}]) AND in[{l}]", pred))

    # XOR-OR: (a XOR b) OR c
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(8):
                if l == j or l == k: continue
                col = [(inputs[e][j] ^ inputs[e][k]) | inputs[e][l] for e in range(n)]
                if col == out_col:
                    pred = (target_bits[j] ^ target_bits[k]) | target_bits[l]
                    matches.append(('xor_or', (j,k,l), f"(in[{j}] XOR in[{k}]) OR in[{l}]", pred))
    
    # NOT-XOR: NOT(a XOR b) AND c (= XNOR AND c)
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(8):
                if l == j or l == k: continue
                col = [(1 - (inputs[e][j] ^ inputs[e][k])) & inputs[e][l] for e in range(n)]
                if col == out_col:
                    pred = (1 - (target_bits[j] ^ target_bits[k])) & target_bits[l]
                    matches.append(('xnor_and', (j,k,l), f"XNOR(in[{j}],in[{k}]) AND in[{l}]", pred))
    
    # NOT-XOR-OR: NOT(a XOR b) OR c
    for j in range(8):
        for k in range(j+1, 8):
            for l in range(8):
                if l == j or l == k: continue
                col = [(1 - (inputs[e][j] ^ inputs[e][k])) | inputs[e][l] for e in range(n)]
                if col == out_col:
                    pred = (1 - (target_bits[j] ^ target_bits[k])) | target_bits[l]
                    matches.append(('xnor_or', (j,k,l), f"XNOR(in[{j}],in[{k}]) OR in[{l}]", pred))

    # a AND b AND NOT c
    for j in range(8):
        for k in range(j+1,8):
            for l in range(8):
                if l == j or l == k: continue
                col = [inputs[e][j] & inputs[e][k] & (1-inputs[e][l]) for e in range(n)]
                if col == out_col:
                    pred = target_bits[j] & target_bits[k] & (1-target_bits[l])
                    matches.append(('and_not', (j,k,l), f"in[{j}] AND in[{k}] AND NOT in[{l}]", pred))

    # a OR b OR NOT c  
    for j in range(8):
        for k in range(j+1,8):
            for l in range(8):
                if l == j or l == k: continue
                col = [inputs[e][j] | inputs[e][k] | (1-inputs[e][l]) for e in range(n)]
                if col == out_col:
                    pred = target_bits[j] | target_bits[k] | (1-target_bits[l])
                    matches.append(('or_not', (j,k,l), f"in[{j}] OR in[{k}] OR NOT in[{l}]", pred))

    # Filter by gold bit
    gold_matches = [m for m in matches if m[3] == gold_bit]
    return gold_matches, matches

total_rescued = 0
total_no_match = 0
rescue_by_type = {}

for r in failed:
    examples, target = parse_bit_ops(r['prompt'])
    gold = r['answer'].strip()
    
    if not examples or not target or len(examples) < 4:
        continue
    if len(gold) != 8 or not all(c in '01' for c in gold):
        continue
    
    n = len(examples)
    inputs_mat = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs_mat = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    gold_bits = [int(gold[i]) for i in range(8)]
    
    # Find no-match bits
    no_match = []
    for obit in range(8):
        funcs = enumerate_bit_functions(inputs_mat, outputs_mat, n, obit)
        if not funcs:
            no_match.append(obit)
    
    if not no_match:
        continue  # This failed for other reasons
    
    total_no_match += 1
    
    # Try extended functions for each no-match bit
    all_rescued = True
    for obit in no_match:
        gold_matches, all_matches = try_extended_functions(
            inputs_mat, outputs_mat, n, obit, target_bits, gold_bits[obit])
        if not gold_matches:
            all_rescued = False
        else:
            for m in gold_matches[:1]:
                rt = m[0]
                rescue_by_type[rt] = rescue_by_type.get(rt, 0) + 1
    
    if all_rescued:
        total_rescued += 1

print(f"\nTotal no-match problems: {total_no_match}")
print(f"Rescued with extended functions: {total_rescued}")
print(f"\nRescue by function type:")
for t, c in sorted(rescue_by_type.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")
