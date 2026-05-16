"""Analyze mixed/arithmetic symbol problems to understand operator rules."""
import polars as pl
import re
from collections import Counter

df = pl.read_csv('competition_data/train.csv')
sym = df.filter(pl.col('prompt').str.contains('transformation rules'))

count = 0
patterns = Counter()

for i in range(len(sym)):
    row = sym.row(i, named=True)
    prompt = row['prompt']
    answer = row['answer']
    
    block = prompt.split('Now,')[0]
    lines = block.strip().split('\n')
    examples = []
    for line in lines:
        line = line.strip()
        if '=' in line and 'transformation' not in line.lower() and 'alice' not in line.lower() and 'secret' not in line.lower() and 'examples' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                examples.append((parts[0].strip(), parts[1].strip()))
    
    if not examples:
        continue
    
    has_digit = any(c.isdigit() for c in examples[0][0])
    if not has_digit:
        continue
    
    ops_found = set()
    for lhs, rhs in examples:
        m = re.match(r'^(\d+)([^\d])(\d+)$', lhs)
        if m:
            ops_found.add(m.group(2))
    
    query_match = re.search(r'result for:\s*(.+)$', prompt, re.MULTILINE)
    query = query_match.group(1).strip() if query_match else '?'
    
    if len(ops_found) == 1:
        patterns['single_op'] += 1
    elif len(ops_found) > 1:
        patterns['multi_op'] += 1
    else:
        patterns['no_op'] += 1
    
    if count < 10:
        print(f'#{i}: ops={ops_found} query={query} gold={answer}')
        for lhs, rhs in examples:
            m = re.match(r'^(\d+)([^\d])(\d+)$', lhs)
            if m:
                a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
                r = rhs
                checks = {}
                checks['a+b'] = a + b
                checks['a-b'] = a - b
                checks['a*b'] = a * b
                checks['concat_ab'] = str(a) + str(b)
                checks['concat_ba'] = str(b) + str(a)
                checks['a_xor_b'] = a ^ b
                checks['a_or_b'] = a | b
                checks['a_and_b'] = a & b
                if b != 0:
                    checks['a//b'] = a // b
                    checks['a%b'] = a % b
                # Check reversal
                checks['rev_a+b'] = str(a+b)[::-1]
                checks['rev_a-b'] = str(a-b)[::-1]
                checks['rev_a*b'] = str(a*b)[::-1]
                # abs
                checks['abs(a-b)'] = abs(a-b)
                # digit stuff
                checks['digit_sum'] = sum(int(d) for d in str(a)) + sum(int(d) for d in str(b))
                
                match = [k for k, v in checks.items() if str(v) == r]
                print(f'  {lhs} = {rhs}  (a={a} op="{op}" b={b}) matches={match}')
            else:
                print(f'  {lhs} = {rhs}  (not standard)')
        print()
        count += 1

print(f'\nOperator patterns: {patterns}')

# Now try to solve mixed type systematically
print("\n\nSYSTEMATIC MIXED SOLVER:")
# The key hypothesis: the visual operator is "remapped" to a different operation
# E.g., "/" might mean "concatenate", "*" might mean "+", etc.

solved = 0
total_mixed = 0
method_counts = Counter()

for i in range(len(sym)):
    row = sym.row(i, named=True)
    prompt = row['prompt']
    gold = row['answer']
    
    block = prompt.split('Now,')[0]
    lines = block.strip().split('\n')
    examples = []
    for line in lines:
        line = line.strip()
        if '=' in line and 'transformation' not in line.lower() and 'alice' not in line.lower() and 'secret' not in line.lower() and 'examples' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                examples.append((parts[0].strip(), parts[1].strip()))
    
    if not examples or not any(c.isdigit() for c in examples[0][0]):
        continue
    
    total_mixed += 1
    query_match = re.search(r'result for:\s*(.+)$', prompt, re.MULTILINE)
    query = query_match.group(1).strip() if query_match else None
    if not query:
        continue
    
    # Parse all examples and query
    parsed = []
    for lhs, rhs in examples:
        m = re.match(r'^(\d+)([^\d])(\d+)$', lhs)
        if m:
            parsed.append((int(m.group(1)), m.group(2), int(m.group(3)), rhs))
    
    qm = re.match(r'^(\d+)([^\d])(\d+)$', query)
    if not qm or not parsed:
        continue
    
    qa, qop, qb = int(qm.group(1)), qm.group(2), int(qm.group(3))
    
    # Group by operator
    op_groups = {}
    for a, op, b, r in parsed:
        if op not in op_groups:
            op_groups[op] = []
        op_groups[op].append((a, b, r))
    
    # For each operator, try to figure out what operation it maps to
    operations = [
        ('add', lambda a, b: str(a + b)),
        ('sub', lambda a, b: str(a - b)),
        ('mul', lambda a, b: str(a * b)),
        ('concat', lambda a, b: str(a) + str(b)),
        ('rconcat', lambda a, b: str(b) + str(a)),
        ('xor', lambda a, b: str(a ^ b)),
        ('or', lambda a, b: str(a | b)),
        ('and', lambda a, b: str(a & b)),
        ('div', lambda a, b: str(a // b) if b != 0 else 'X'),
        ('mod', lambda a, b: str(a % b) if b != 0 else 'X'),
        ('abs_sub', lambda a, b: str(abs(a - b))),
        ('rev_add', lambda a, b: str(a + b)[::-1]),
        ('rev_sub', lambda a, b: str(a - b)[::-1]),
        ('rev_mul', lambda a, b: str(a * b)[::-1]),
        ('pow', lambda a, b: str(a ** b) if b < 20 else 'X'),
    ]
    
    op_map = {}
    for op, group in op_groups.items():
        for op_name, op_func in operations:
            if all(op_func(a, b) == r for a, b, r in group):
                op_map[op] = (op_name, op_func)
                break
    
    if qop in op_map:
        pred = op_map[qop][1](qa, qb)
        if pred == gold:
            solved += 1
            method_counts[op_map[qop][0]] += 1

print(f'\nMixed type: Solved {solved}/{total_mixed} ({100*solved/total_mixed:.1f}%)')
print(f'Method distribution:')
for m, c in method_counts.most_common():
    print(f'  {m}: {c}')
