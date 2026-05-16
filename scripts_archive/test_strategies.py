#!/usr/bin/env python3
"""Test multiple selection strategies on all bit_ops problems."""
import csv, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from gen_thinking import _enumerate_bit_functions, _eval_bit_function

def classify(p):
    p = p.lower()
    if 'bit manipulation' in p or 'bit shift' in p: return 'bit_ops'
    return 'other'

def parse_bit_ops(prompt):
    lines = prompt.strip().split('\n')
    examples, target = [], None
    for line in lines:
        line = line.strip()
        if ' -> ' in line:
            parts = line.split(' -> ')
            if len(parts)==2:
                inp,out = parts[0].strip(), parts[1].strip()
                if len(inp)==8 and len(out)==8 and all(c in '01' for c in inp+out):
                    examples.append((inp,out))
        if 'determine' in line.lower() and ':' in line:
            t = line.split(':')[-1].strip()
            if len(t)==8 and all(c in '01' for c in t):
                target = t
    return examples, target

def func_n_inputs(f):
    fname = f[0]
    if fname == 'const': return 0
    if fname in ('copy', 'not'): return 1
    if fname in ('xor2','xnor2','and','or','nand','nor','not_and','not_or'): return 2
    return 3

def key_current(f):
    return (len(f[2]), f[2])

def key_prefer_1input(f):
    ni = func_n_inputs(f)
    rank = {1: 0, 0: 1, 2: 2, 3: 3}[ni]
    return (rank, len(f[2]), f[2])

def key_prefer_xor(f):
    priority = {'copy': 0, 'not': 1, 'xor2': 2, 'xnor2': 3, 'const': 4,
        'and': 5, 'or': 6, 'nand': 7, 'nor': 8, 'not_and': 9, 'not_or': 10}
    return (priority.get(f[0], 20), len(f[2]), f[2])

def find_perm(all_cands):
    copy_cands = [[(f, f[1]) for f in cands if f[0]=='copy'] for cands in all_cands]
    if not all(len(c)>0 for c in copy_cands): return None
    available = [set(ibit for _, ibit in cands) for cands in copy_cands]
    perm = [None]*8; used = set()
    def bt(ob):
        if ob==8: return True
        for ib in sorted(available[ob]):
            if ib not in used:
                perm[ob]=ib; used.add(ib)
                if bt(ob+1): return True
                used.remove(ib); perm[ob]=None
        return False
    return list(perm) if bt(0) else None

rows = []
with open(os.path.join(os.path.dirname(__file__), '..', 'competition_data', 'train.csv')) as f:
    for r in csv.DictReader(f):
        if classify(r['prompt']) == 'bit_ops':
            rows.append(r)

n_parse = 0
n_any = 0
n_chain_perm = 0
n_adaptive = 0
results = {'current': 0, 'prefer_1input': 0, 'prefer_xor': 0}
t0 = time.time()

for i, row in enumerate(rows):
    examples, target = parse_bit_ops(row['prompt'])
    gold = row['answer']
    if not examples or not target or len(examples)<4 or len(gold)!=8: continue
    n_parse += 1
    n = len(examples)
    inputs = [[int(ex[0][j]) for j in range(8)] for ex in examples]
    outputs = [[int(ex[1][j]) for j in range(8)] for ex in examples]
    target_bits = [int(target[j]) for j in range(8)]
    
    all_cands = []
    ok = True
    for obit in range(8):
        c = _enumerate_bit_functions(inputs, outputs, n, obit)
        if not c: ok = False; break
        all_cands.append(c)
    if not ok: continue
    
    solved_any = False
    for sname, key_fn in [('current', key_current), ('prefer_1input', key_prefer_1input), ('prefer_xor', key_prefer_xor)]:
        selected = [sorted(c, key=key_fn)[0] for c in all_cands]
        bits = [_eval_bit_function(f, target_bits) for f in selected]
        if None in bits: continue
        answer = ''.join(str(b) for b in bits)
        if answer == gold:
            results[sname] += 1
            solved_any = True
    if solved_any: n_any += 1
    
    # Chain: perm → current
    selected_current = [sorted(c, key=key_current)[0] for c in all_cands]
    bits_current = [_eval_bit_function(f, target_bits) for f in selected_current]
    current_answer = ''.join(str(b) for b in bits_current) if None not in bits_current else ''
    
    perm = find_perm(all_cands)
    chain_ok = False
    if perm is not None:
        perm_answer = ''.join(str(target_bits[perm[j]]) for j in range(8))
        if perm_answer == gold: chain_ok = True
    if not chain_ok and current_answer == gold: chain_ok = True
    if chain_ok: n_chain_perm += 1
    
    # Adaptive
    n_1input = sum(1 for f in selected_current if f[0] in ('copy', 'not'))
    if n_1input >= 6:
        selected_adapt = [sorted(c, key=key_prefer_1input)[0] for c in all_cands]
    else:
        selected_adapt = selected_current
    bits_adapt = [_eval_bit_function(f, target_bits) for f in selected_adapt]
    if None not in bits_adapt and ''.join(str(b) for b in bits_adapt) == gold:
        n_adaptive += 1

elapsed = time.time() - t0
print(f"Parseable: {n_parse}, Time: {elapsed:.1f}s")
print(f"\n{'Strategy':35s} {'Correct':>7s} {'Coverage':>8s}")
print('-' * 55)
for sname in ['current', 'prefer_1input', 'prefer_xor']:
    n = results[sname]
    print(f"{sname:35s} {n:5d}/{n_parse:4d}  {n/n_parse*100:5.1f}%")
print('-' * 55)
print(f"{'union (any strategy)':35s} {n_any:5d}/{n_parse:4d}  {n_any/n_parse*100:5.1f}%")
print(f"{'chain: perm → current':35s} {n_chain_perm:5d}/{n_parse:4d}  {n_chain_perm/n_parse*100:5.1f}%")
print(f"{'adaptive: n_1input≥6':35s} {n_adaptive:5d}/{n_parse:4d}  {n_adaptive/n_parse*100:5.1f}%")
