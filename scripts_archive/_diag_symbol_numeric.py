#!/usr/bin/env python3
"""Deep analysis of numeric symbol problems to find more solvable rules."""
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

# Categorize numeric vs string
numeric_fails = []
for r in rows:
    examples, query = parse_examples(r['prompt'])
    if not examples or not query:
        continue
    
    query_split = split_by_op(query)
    if not query_split:
        continue
    
    ql, qo, qr = query_split
    
    # Check if numeric
    all_numeric = True
    for lhs, rhs in examples:
        sp = split_by_op(lhs)
        if not sp:
            all_numeric = False
            break
        l, o, rv = sp
        if not all(c.isdigit() for c in l) or not all(c.isdigit() for c in rv):
            all_numeric = False
            break
        if not all(c.isdigit() for c in rhs):
            all_numeric = False
            break
    
    if all_numeric and all(c.isdigit() for c in ql) and all(c.isdigit() for c in qr):
        numeric_fails.append((r, examples, query, ql, qo, qr))

print(f"Total numeric symbol problems with operators: {len(numeric_fails)}")

# For each, try various rules
OPS = [
    ('a+b', lambda a, b: str(a + b)),
    ('a-b', lambda a, b: str(a - b)),
    ('b-a', lambda a, b: str(b - a)),
    ('a*b', lambda a, b: str(a * b)),
    ('a//b', lambda a, b: str(a // b) if b != 0 else None),
    ('b//a', lambda a, b: str(b // a) if a != 0 else None),
    ('a%b', lambda a, b: str(a % b) if b != 0 else None),
    ('concat', lambda a, b: str(a) + str(b)),
    ('concat_rev', lambda a, b: str(b) + str(a)),
    ('a**2+b**2', lambda a, b: str(a**2 + b**2)),
    ('a**2-b**2', lambda a, b: str(a**2 - b**2)),
    ('(a+b)**2', lambda a, b: str((a + b)**2)),
    ('a**2*b', lambda a, b: str(a**2 * b)),
    ('digit_interleave', lambda a, b: ''.join(x+y for x,y in zip(str(a).zfill(2), str(b).zfill(2)))),
    ('reverse_digits_concat', lambda a, b: str(a)[::-1] + str(b)[::-1]),
    ('abs_diff', lambda a, b: str(abs(a - b))),
    ('max', lambda a, b: str(max(a, b))),
    ('min', lambda a, b: str(min(a, b))),
    ('a*10+b', lambda a, b: str(a * 10 + b)),
    ('a+b*10', lambda a, b: str(a + b * 10)),
    ('digit_sum', lambda a, b: str(sum(int(d) for d in str(a) + str(b)))),
    ('a^b', lambda a, b: str(a ^ b)),  # XOR
    ('a&b', lambda a, b: str(a & b)),  # AND
    ('a|b', lambda a, b: str(a | b)),  # OR
]

solved_by_rule = defaultdict(int)
not_solved = 0

for r, examples, query, ql, qo, qr in numeric_fails:
    gold = r['answer'].strip()
    
    # Group by operator
    op_groups = defaultdict(list)
    for lhs, rhs in examples:
        sp = split_by_op(lhs)
        if sp:
            op_groups[sp[1]].append((int(sp[0]), int(sp[2]), rhs))
    
    if qo not in op_groups:
        not_solved += 1
        continue
    
    group = op_groups[qo]
    found = False
    
    for op_name, op_fn in OPS:
        all_match = True
        for a, b, result in group:
            try:
                pred = op_fn(a, b)
                if pred is None or pred != result:
                    all_match = False
                    break
            except:
                all_match = False
                break
        
        if all_match:
            try:
                pred = op_fn(int(ql), int(qr))
            except:
                continue
            if pred == gold:
                solved_by_rule[op_name] += 1
                found = True
                break
    
    if not found:
        not_solved += 1
        if not_solved <= 10:
            print(f"\nUnsolved: {r['id']}, gold={gold}, query={query}")
            for a, b, result in group[:3]:
                print(f"  {a} {qo} {b} = {result}")

print(f"\nSolved: {sum(solved_by_rule.values())}/{len(numeric_fails)}")
print(f"Not solved: {not_solved}")
print(f"\nBy rule:")
for rule, count in sorted(solved_by_rule.items(), key=lambda x: -x[1]):
    print(f"  {rule}: {count}")
