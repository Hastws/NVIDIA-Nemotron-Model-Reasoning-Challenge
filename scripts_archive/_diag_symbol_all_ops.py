#!/usr/bin/env python3
"""
Quick test: for '*' symbol problems, is the pattern 'remove the char at operator position'?
E.g., \(*[# → remove char at position of '*' → \([#
"""
import csv
from collections import defaultdict

rows = [r for r in csv.DictReader(open('competition_data/train.csv'))
        if 'symbol' in r['prompt'].lower() or 'equation' in r['prompt'].lower()]

OP_CHARS = set('+-*/|\\^&')

def split_by_op(expr):
    for i, c in enumerate(expr):
        if c in OP_CHARS and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i+1:]
    return None

def parse_examples(prompt):
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if 'determine the result for:' in line.lower():
            query = line.split(':')[-1].strip()
        elif '=' in line and 'alice' not in line.lower() and 'equation' not in line.lower() \
                and 'transformation' not in line.lower() and 'determine' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                if lhs and rhs:
                    examples.append((lhs, rhs))
    return examples, query

# Test hypothesis: for each operator, one consistent b94 operation per PROBLEM
# (different problems may have different rules for same operator)
B = 33
R = 94

def s2b(s):
    v = 0
    for c in s:
        v = v * R + (ord(c) - B)
    return v

def b2s(v):
    if v == 0: return chr(B)
    if v < 0: v = v % (R**10)
    chars = []
    while v > 0:
        chars.append(chr((v % R) + B))
        v //= R
    return ''.join(reversed(chars)) if chars else chr(B)

# Charwise operations
def cw_op(fn, l, r):
    if len(l) != len(r):
        return None
    return ''.join(chr(fn(ord(a)-B, ord(b)-B) % R + B) for a, b in zip(l, r))

ALL_OPS = [
    ('concat', lambda l, r: l + r),
    ('concat_rev', lambda l, r: r + l),
    ('b94_add', lambda l, r: b2s(s2b(l) + s2b(r))),
    ('b94_sub', lambda l, r: b2s(s2b(l) - s2b(r))),
    ('b94_sub_rev', lambda l, r: b2s(s2b(r) - s2b(l))),
    ('b94_mul', lambda l, r: b2s(s2b(l) * s2b(r))),
    ('cw_add', lambda l, r: cw_op(lambda a,b: a+b, l, r)),
    ('cw_sub', lambda l, r: cw_op(lambda a,b: a-b, l, r)),
    ('cw_sub_rev', lambda l, r: cw_op(lambda a,b: b-a, l, r)),
    ('cw_xor', lambda l, r: cw_op(lambda a,b: a^b, l, r)),
    ('cw_mul', lambda l, r: cw_op(lambda a,b: a*b, l, r)),
    ('cw_and', lambda l, r: cw_op(lambda a,b: a&b, l, r)),
    ('cw_or', lambda l, r: cw_op(lambda a,b: a|b, l, r)),
    ('cw_min', lambda l, r: cw_op(lambda a,b: min(a,b), l, r)),
    ('cw_max', lambda l, r: cw_op(lambda a,b: max(a,b), l, r)),
]

total = 0
solved = 0
solved_by = defaultdict(int)

for r in rows:
    examples, query = parse_examples(r['prompt'])
    if not examples or not query:
        continue
    
    query_split = split_by_op(query)
    if not query_split:
        continue
    
    ql, qo, qr = query_split
    gold = r['answer'].strip()
    
    # Group by operator
    op_groups = defaultdict(list)
    for lhs, rhs in examples:
        sp = split_by_op(lhs)
        if sp:
            op_groups[sp[1]].append((sp[0], sp[2], rhs))
    
    if qo not in op_groups:
        continue
    
    group = op_groups[qo]
    total += 1
    
    found = False
    for op_name, fn in ALL_OPS:
        all_match = True
        for left, right, result in group:
            try:
                pred = fn(left, right)
                if pred is None or pred != result:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                pred = fn(ql, qr)
            except:
                continue
            if pred == gold:
                solved += 1
                solved_by[op_name] += 1
                found = True
                break
    
    if not found and total <= solved + 15:
        # Show unsolved for inspection
        pass

print(f"Total with operators: {total}")
print(f"Solved: {solved} ({solved/total*100:.1f}%)")
print(f"\nBy operation:")
for op, count in sorted(solved_by.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count}")
