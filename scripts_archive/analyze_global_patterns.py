#!/usr/bin/env python3
"""Analyze global structure patterns in bit_ops problems."""
import csv, sys, os
sys.path.insert(0, os.path.dirname(__file__))

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

def find_l1_candidates(inputs, outputs, n, obit):
    """Find const/copy/not candidates only."""
    out_col = [outputs[e][obit] for e in range(n)]
    matches = []
    if all(x==0 for x in out_col): matches.append(('const', 0, '0'))
    if all(x==1 for x in out_col): matches.append(('const', 1, '1'))
    for i in range(8):
        ic = [inputs[e][i] for e in range(n)]
        if ic == out_col: matches.append(('copy', i, f"in[{i}]"))
        if [1-x for x in ic] == out_col: matches.append(('not', i, f"NOT in[{i}]"))
    return matches

def find_l2_candidates(inputs, outputs, n, obit):
    """Find 2-input candidates."""
    out_col = [outputs[e][obit] for e in range(n)]
    matches = []
    for j in range(8):
        for k in range(j+1, 8):
            xor = [inputs[e][j] ^ inputs[e][k] for e in range(n)]
            if xor == out_col: matches.append(('xor2', (j,k), f"in[{j}] XOR in[{k}]"))
            if [1-x for x in xor] == out_col: matches.append(('xnor2', (j,k), f"XNOR(in[{j}],in[{k}])"))
            for op_name, op_fn in [('AND', lambda a,b: a&b), ('OR', lambda a,b: a|b),
                                   ('NAND', lambda a,b: 1-(a&b)), ('NOR', lambda a,b: 1-(a|b))]:
                col = [op_fn(inputs[e][j], inputs[e][k]) for e in range(n)]
                if col == out_col: matches.append((op_name.lower(), (j,k), f"in[{j}] {op_name} in[{k}]"))
    for j in range(8):
        for k in range(8):
            if j==k: continue
            na = [(1-inputs[e][j]) & inputs[e][k] for e in range(n)]
            if na == out_col: matches.append(('not_and', (j,k), f"NOT(in[{j}]) AND in[{k}]"))
            no = [(1-inputs[e][j]) | inputs[e][k] for e in range(n)]
            if no == out_col: matches.append(('not_or', (j,k), f"NOT(in[{j}]) OR in[{k}]"))
    return matches

def eval_fn(f, tb):
    fn, args, _ = f
    if fn=='const': return args
    if fn=='copy': return tb[args]
    if fn=='not': return 1-tb[args]
    if fn=='xor2': return tb[args[0]] ^ tb[args[1]]
    if fn=='xnor2': return 1 - (tb[args[0]] ^ tb[args[1]])
    if fn=='and': return tb[args[0]] & tb[args[1]]
    if fn=='or': return tb[args[0]] | tb[args[1]]
    if fn=='nand': return 1 - (tb[args[0]] & tb[args[1]])
    if fn=='nor': return 1 - (tb[args[0]] | tb[args[1]])
    if fn=='not_and': return (1 - tb[args[0]]) & tb[args[1]]
    if fn=='not_or': return (1 - tb[args[0]]) | tb[args[1]]
    return None

def find_perm(copy_cands):
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

def find_cn_perm(cn_cands):
    available = [[(ibit, ftype=='not') for _, ibit, ftype in cands] for cands in cn_cands]
    perm = [None]*8; used = set()
    def bt(ob):
        if ob==8: return True
        for ibit, is_not in sorted(available[ob], key=lambda x: (x[0],x[1])):
            if ibit not in used:
                perm[ob]=(ibit,is_not); used.add(ibit)
                if bt(ob+1): return True
                used.remove(ibit); perm[ob]=None
        return False
    return list(perm) if bt(0) else None

# Load data
rows = []
with open('competition_data/train.csv') as f:
    for r in csv.DictReader(f):
        if classify(r['prompt']) == 'bit_ops':
            rows.append(r)
print(f'Total bit_ops: {len(rows)}')

n_parse = 0
n_pbs_l1_ok = 0
n_has_perm = 0; n_perm_ok = 0
n_has_cn = 0; n_cn_ok = 0
n_l1_all = 0
n_needs_l2 = 0
n_pbs_fail_perm_ok = 0
n_pbs_fail_cn_ok = 0
n_coherent_l1_ok = 0  # prefer copy/not over const when perm-like
n_pbs_l12_ok = 0  # per-bit simplest with l1+l2

for row in rows:
    examples, target = parse_bit_ops(row['prompt'])
    gold = row['answer']
    if not examples or not target or len(examples)<4 or len(gold)!=8:
        continue
    n_parse += 1
    n = len(examples)
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    
    l1 = [find_l1_candidates(inputs, outputs, n, obit) for obit in range(8)]
    all_have_l1 = all(len(c)>0 for c in l1)
    
    if all_have_l1:
        n_l1_all += 1
        # Per-bit simplest (level 1)
        pbs = [sorted(c, key=lambda f: (len(f[2]),f[2]))[0] for c in l1]
        pbs_bits = [eval_fn(f, target_bits) for f in pbs]
        pbs_answer = ''.join(str(b) for b in pbs_bits)
        pbs_correct = pbs_answer == gold
        if pbs_correct: n_pbs_l1_ok += 1
        
        # Permutation
        copy_cands = [[(f, f[1]) for f in cands if f[0]=='copy'] for cands in l1]
        if all(len(c)>0 for c in copy_cands):
            perm = find_perm(copy_cands)
            if perm is not None:
                n_has_perm += 1
                perm_answer = ''.join(str(target_bits[perm[i]]) for i in range(8))
                if perm_answer == gold:
                    n_perm_ok += 1
                    if not pbs_correct: n_pbs_fail_perm_ok += 1
        
        # Copy/NOT perm
        cn_cands = [[(f, f[1], f[0]) for f in cands if f[0] in ('copy','not')] for cands in l1]
        if all(len(c)>0 for c in cn_cands):
            cn_perm = find_cn_perm(cn_cands)
            if cn_perm is not None:
                n_has_cn += 1
                cn_bits = []
                for obit in range(8):
                    ibit, is_not = cn_perm[obit]
                    v = 1-target_bits[ibit] if is_not else target_bits[ibit]
                    cn_bits.append(v)
                cn_answer = ''.join(str(b) for b in cn_bits)
                if cn_answer == gold:
                    n_cn_ok += 1
                    if not pbs_correct: n_pbs_fail_cn_ok += 1
    else:
        n_needs_l2 += 1
        # Try l1+l2 combined
        l12 = []
        for obit in range(8):
            c = l1[obit] if l1[obit] else find_l2_candidates(inputs, outputs, n, obit)
            l12.append(c)
        if all(len(c)>0 for c in l12):
            pbs2 = [sorted(c, key=lambda f: (len(f[2]),f[2]))[0] for c in l12]
            pbs2_bits = [eval_fn(f, target_bits) for f in pbs2]
            if None not in pbs2_bits:
                pbs2_answer = ''.join(str(b) for b in pbs2_bits)
                if pbs2_answer == gold:
                    n_pbs_l12_ok += 1

print(f"\nParseable: {n_parse}")
print(f"\n=== Level-1 Analysis ===")
print(f"All bits have l1: {n_l1_all}/{n_parse} ({n_l1_all/n_parse*100:.1f}%)")
print(f"Per-bit simplest correct: {n_pbs_l1_ok}/{n_l1_all} ({n_pbs_l1_ok/n_l1_all*100:.1f}% of l1-solvable)")
print(f"Needs l2 for some bits: {n_needs_l2}/{n_parse} ({n_needs_l2/n_parse*100:.1f}%)")
print(f"\n=== Global Patterns ===")
print(f"Pure perm found: {n_has_perm}/{n_parse} ({n_has_perm/n_parse*100:.1f}%)")
print(f"Perm correct: {n_perm_ok}/{max(n_has_perm,1)} ({n_perm_ok/max(n_has_perm,1)*100:.1f}%)")
print(f"Copy/NOT perm found: {n_has_cn}/{n_parse} ({n_has_cn/n_parse*100:.1f}%)")
print(f"Copy/NOT perm correct: {n_cn_ok}/{max(n_has_cn,1)} ({n_cn_ok/max(n_has_cn,1)*100:.1f}%)")
print(f"\n=== Recovery ===")
print(f"PBS fails, perm fixes: {n_pbs_fail_perm_ok}")
print(f"PBS fails, CN perm fixes: {n_pbs_fail_cn_ok}")
print(f"PBS l1+l2 correct (needs l2): {n_pbs_l12_ok}/{n_needs_l2}")
total_best = n_pbs_l1_ok + n_pbs_fail_perm_ok + n_pbs_fail_cn_ok + n_pbs_l12_ok
print(f"\n=== Best Estimate ===")
print(f"Current (PBS l1 only):  {n_pbs_l1_ok}/{n_parse} ({n_pbs_l1_ok/n_parse*100:.1f}%)")
print(f"With global patterns:   {total_best}/{n_parse} ({total_best/n_parse*100:.1f}%)")
print(f"Improvement:            +{total_best - n_pbs_l1_ok} problems")
