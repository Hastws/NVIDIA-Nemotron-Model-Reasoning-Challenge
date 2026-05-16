#!/usr/bin/env python3
"""
Symbol solver v5: Expanded operator detection + extended rules.
Key insight: OP_CHARS only has +-*/|\\^& but problems use $, `, ), (, >, <, !, @, #, etc.
"""
import csv
import re
from collections import defaultdict, Counter

CHAR_BASE = 33
CHAR_RANGE = 94

def str_to_b94(s):
    val = 0
    for c in s:
        val = val * CHAR_RANGE + (ord(c) - CHAR_BASE)
    return val

def b94_to_str(val):
    if val == 0:
        return chr(CHAR_BASE)
    if val < 0:
        val = val % (CHAR_RANGE ** 10)
    chars = []
    while val > 0:
        chars.append(chr((val % CHAR_RANGE) + CHAR_BASE))
        val //= CHAR_RANGE
    return ''.join(reversed(chars)) if chars else chr(CHAR_BASE)

# ── Parse with ALL possible operators ──

# Standard OP_CHARS from current code
STD_OPS = set('+-*/|\\^&')

# ALL printable non-alnum chars that could be operators  
ALL_NONALNUM = set()
for i in range(33, 127):
    c = chr(i)
    if not c.isalnum() and c != '=' and c != ' ':
        ALL_NONALNUM.add(c)
print(f"All potential op chars: {sorted(ALL_NONALNUM)}")
print(f"Non-standard ops: {sorted(ALL_NONALNUM - STD_OPS)}")

def split_by_any_op(expr, op_set):
    """Split expr by any single char in op_set. Try all positions."""
    results = []
    for i, c in enumerate(expr):
        if c in op_set and i > 0 and i < len(expr) - 1:
            left = expr[:i]
            right = expr[i+1:]
            results.append((left, c, right))
    return results

def split_by_op_first(expr, op_set):
    """Split by first matching op char."""
    for i, c in enumerate(expr):
        if c in op_set and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i+1:]
    return None

def is_numeric(s):
    try:
        int(s)
        return True
    except:
        return False

# ── Load data ──
input_path = 'data/train_annotated.csv'
rows = list(csv.DictReader(open(input_path)))
symbol_rows = [r for r in rows if r['type'] == 'symbol']
unsolved = [r for r in symbol_rows if r['match'] != 'True']
print(f"\nSymbol: {len(symbol_rows)} total, {len(unsolved)} unsolved\n")

# ── NUMERIC RULES (comprehensive) ──
NUMERIC_RULES = [
    ('add',        lambda a, b: str(a + b)),
    ('sub',        lambda a, b: str(a - b)),
    ('sub_rev',    lambda a, b: str(b - a)),
    ('mul',        lambda a, b: str(a * b)),
    ('div',        lambda a, b: str(a // b) if b != 0 else None),
    ('div_rev',    lambda a, b: str(b // a) if a != 0 else None),
    ('mod',        lambda a, b: str(a % b) if b != 0 else None),
    ('mod_rev',    lambda a, b: str(b % a) if a != 0 else None),
    ('pow',        lambda a, b: str(a ** b) if 0 <= b <= 20 and a ** b < 10**15 else None),
    ('pow_rev',    lambda a, b: str(b ** a) if 0 <= a <= 20 and b ** a < 10**15 else None),
    ('concat',     lambda a, b: str(a) + str(b)),
    ('concat_rev', lambda a, b: str(b) + str(a)),
    ('xor',        lambda a, b: str(a ^ b)),
    ('bitor',      lambda a, b: str(a | b)),
    ('bitand',     lambda a, b: str(a & b)),
    ('abs_diff',   lambda a, b: str(abs(a - b))),
    ('max',        lambda a, b: str(max(a, b))),
    ('min',        lambda a, b: str(min(a, b))),
    ('add1',       lambda a, b: str(a + b + 1)),
    ('sub1',       lambda a, b: str(a + b - 1)),
    ('mul_add1',   lambda a, b: str(a * b + 1)),
    ('mul_sub1',   lambda a, b: str(a * b - 1)),
    ('mul_add_a',  lambda a, b: str(a * b + a)),
    ('mul_add_b',  lambda a, b: str(a * b + b)),
    ('mul_sub_a',  lambda a, b: str(a * b - a)),
    ('mul_sub_b',  lambda a, b: str(a * b - b)),
    ('sum_sq',     lambda a, b: str(a*a + b*b)),
    ('sq_sum',     lambda a, b: str((a + b) ** 2)),
    ('sq_diff',    lambda a, b: str((a - b) ** 2)),
    ('a_sq_b',     lambda a, b: str(a ** 2 + b)),
    ('a_b_sq',     lambda a, b: str(a + b ** 2)),
    ('a_sq_mul_b', lambda a, b: str(a ** 2 * b)),
    ('a_mul_b_sq', lambda a, b: str(a * b ** 2)),
    ('2a_b',       lambda a, b: str(2 * a + b)),
    ('a_2b',       lambda a, b: str(a + 2 * b)),
    ('2a_2b',      lambda a, b: str(2 * a + 2 * b)),
    ('a2_b2',      lambda a, b: str(a ** 2 - b ** 2)),
    # Digit-level operations for 2-digit numbers
    ('d_cross',    lambda a, b: str((a // 10) * (b % 10) + (a % 10) * (b // 10)) if 10 <= a <= 99 and 10 <= b <= 99 else None),
    ('d_cross2',   lambda a, b: str((a // 10) * (b % 10) * 10 + (a % 10) * (b // 10)) if 10 <= a <= 99 and 10 <= b <= 99 else None),
    ('d_dot',      lambda a, b: str((a // 10) * (b // 10) + (a % 10) * (b % 10)) if 10 <= a <= 99 and 10 <= b <= 99 else None),
    ('d_mul_dig',  lambda a, b: str((a // 10) * (b // 10)) + str((a % 10) * (b % 10)) if 10 <= a <= 99 and 10 <= b <= 99 else None),
    ('d_add_dig',  lambda a, b: str((a // 10) + (b // 10)) + str((a % 10) + (b % 10)) if 10 <= a <= 99 and 10 <= b <= 99 else None),
    ('d_sub_dig',  lambda a, b: str((a // 10) - (b // 10)) + str((a % 10) - (b % 10)) if 10 <= a <= 99 and 10 <= b <= 99 else None),
]

# ── SYMBOLIC RULES ──
SYMBOL_OPS = [
    ('concat',     lambda l, r: l + r),
    ('concat_rev', lambda l, r: r + l),
    ('b94_add',    lambda l, r: b94_to_str(str_to_b94(l) + str_to_b94(r))),
    ('b94_sub',    lambda l, r: b94_to_str(str_to_b94(l) - str_to_b94(r))),
    ('b94_sub_rev',lambda l, r: b94_to_str(str_to_b94(r) - str_to_b94(l))),
    ('b94_mul',    lambda l, r: b94_to_str(str_to_b94(l) * str_to_b94(r))),
    ('b94_xor',    lambda l, r: b94_to_str(str_to_b94(l) ^ str_to_b94(r))),
    ('b94_and',    lambda l, r: b94_to_str(str_to_b94(l) & str_to_b94(r))),
    ('b94_or',     lambda l, r: b94_to_str(str_to_b94(l) | str_to_b94(r))),
]

CHARWISE_OPS = [
    ('cw_add',     lambda a, b: chr(((ord(a)-33) + (ord(b)-33)) % 94 + 33)),
    ('cw_sub',     lambda a, b: chr(((ord(a)-33) - (ord(b)-33)) % 94 + 33)),
    ('cw_sub_rev', lambda a, b: chr(((ord(b)-33) - (ord(a)-33)) % 94 + 33)),
    ('cw_xor',     lambda a, b: chr(((ord(a)-33) ^ (ord(b)-33)) % 94 + 33)),
    ('cw_mul',     lambda a, b: chr(((ord(a)-33) * (ord(b)-33)) % 94 + 33)),
    ('cw_and',     lambda a, b: chr(((ord(a)-33) & (ord(b)-33)) % 94 + 33)),
    ('cw_or',      lambda a, b: chr(((ord(a)-33) | (ord(b)-33)) % 94 + 33)),
    ('cw_max',     lambda a, b: chr(max(ord(a)-33, ord(b)-33) % 94 + 33)),
    ('cw_min',     lambda a, b: chr(min(ord(a)-33, ord(b)-33) % 94 + 33)),
]

def try_cw(fn, left, right, result):
    if len(left) != len(right) or len(left) != len(result):
        return False
    try:
        return ''.join(fn(a, b) for a, b in zip(left, right)) == result
    except:
        return False

def solve_with_expanded_ops(prompt, gold):
    """Try to solve a symbol problem with expanded operator detection."""
    lines = prompt.strip().split('\n')
    examples = []
    query_line = None
    
    for line in lines:
        line = line.strip()
        if 'determine the result for:' in line.lower():
            query_line = line.split(':')[-1].strip()
        elif '=' in line and 'alice' not in line.lower() and 'equation' not in line.lower() \
                and 'transformation' not in line.lower() and 'determine' not in line.lower() \
                and 'below' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                if lhs and rhs:
                    examples.append((lhs, rhs))
    
    if not examples or not query_line:
        return None, 'parse_fail'
    
    # ── Strategy 1: Try ALL non-alnum chars as operators ──
    # First, detect which chars appear as operators in examples
    # For numeric: any non-digit char is an operator
    # For symbolic: need to find consistent operator positions
    
    # Check if this is numeric (all non-op chars are digits)
    all_digits_mode = True
    for lhs, rhs in examples:
        # Check RHS
        for c in rhs:
            if not c.isdigit() and c not in ('-', '.'):
                all_digits_mode = False
                break
        if not all_digits_mode:
            break
    
    if all_digits_mode:
        return solve_numeric_expanded(examples, query_line, gold)
    else:
        return solve_symbolic_expanded(examples, query_line, gold)

def solve_numeric_expanded(examples, query, gold):
    """Solve numeric equation with expanded operator set."""
    # Parse: any single non-digit char between digits is an operator
    def parse_numeric(expr):
        for i, c in enumerate(expr):
            if not c.isdigit() and i > 0 and i < len(expr) - 1:
                left = expr[:i]
                right = expr[i+1:]
                if left.isdigit() and right.isdigit():
                    return left, c, right
        return None
    
    q_parsed = parse_numeric(query)
    if not q_parsed:
        return None, 'no_op_in_query'
    
    q_left_s, q_op, q_right_s = q_parsed
    
    # Group examples by operator
    op_groups = defaultdict(list)
    for lhs, rhs in examples:
        p = parse_numeric(lhs)
        if p:
            op_groups[p[1]].append((p[0], p[2], rhs))
    
    if q_op not in op_groups:
        return None, f'op_not_in_examples(op={q_op})'
    
    group = op_groups[q_op]
    q_a, q_b = int(q_left_s), int(q_right_s)
    
    # Try all numeric rules
    for rule_name, fn in NUMERIC_RULES:
        all_match = True
        for ls, rs, result in group:
            try:
                a, b = int(ls), int(rs)
                pred = fn(a, b)
                if pred is None or pred != result:
                    all_match = False
                    break
            except:
                all_match = False
                break
        
        if all_match:
            try:
                answer = fn(q_a, q_b)
                if answer is not None and answer == gold:
                    return rule_name, 'solved'
            except:
                pass
    
    return None, 'no_numeric_rule'

def solve_symbolic_expanded(examples, query, gold):
    """Solve symbolic equation with expanded operator set."""
    all_op_chars = set()
    for lhs, rhs in examples:
        for c in lhs:
            if c in ALL_NONALNUM:
                all_op_chars.add(c)
    
    # Also check query
    query_op_chars = set()
    for c in query:
        if c in ALL_NONALNUM:
            query_op_chars.add(c)
    
    # Try each potential operator char
    for op_char in query_op_chars:
        # Try to split query by this char
        splits = split_by_any_op(query, {op_char})
        if not splits:
            continue
        
        for q_left, q_op, q_right in splits:
            # Group examples by this operator
            op_groups = defaultdict(list)
            for lhs, rhs in examples:
                ex_splits = split_by_any_op(lhs, {op_char})
                for el, eo, er in ex_splits:
                    op_groups[eo].append((el, er, rhs))
            
            if q_op not in op_groups:
                continue
            
            group = op_groups[q_op]
            if not group:
                continue
            
            # Check if all operands are numeric
            all_numeric = True
            for l, r, res in group:
                if not is_numeric(l) or not is_numeric(r):
                    all_numeric = False
                    break
            
            if all_numeric and is_numeric(q_left) and is_numeric(q_right):
                # Try numeric rules
                q_a, q_b = int(q_left), int(q_right)
                for rule_name, fn in NUMERIC_RULES:
                    all_match = True
                    for ls, rs, result in group:
                        try:
                            pred = fn(int(ls), int(rs))
                            if pred is None or pred != result:
                                all_match = False
                                break
                        except:
                            all_match = False
                            break
                    if all_match:
                        try:
                            answer = fn(q_a, q_b)
                            if answer is not None and answer == gold:
                                return f'num:{rule_name}(op={op_char})', 'solved'
                        except:
                            pass
            
            # Try symbolic ops
            for op_name, fn in SYMBOL_OPS:
                try:
                    all_match = all(fn(l, r) == res for l, r, res in group)
                    if all_match:
                        pred = fn(q_left, q_right)
                        if pred == gold:
                            return f'sym:{op_name}(op={op_char})', 'solved'
                except:
                    continue
            
            # Try charwise ops
            if len(q_left) == len(q_right):
                for op_name, fn in CHARWISE_OPS:
                    try:
                        all_match = all(try_cw(fn, l, r, res) for l, r, res in group)
                        if all_match:
                            pred = ''.join(fn(a, b) for a, b in zip(q_left, q_right))
                            if pred == gold:
                                return f'cw:{op_name}(op={op_char})', 'solved'
                    except:
                        continue
    
    return None, 'no_rule'

# ── Run on all unsolved ──
solved_rules = Counter()
solved_samples = defaultdict(list)
still_unsolved = []
fail_reasons = Counter()

for r in unsolved:
    rule, status = solve_with_expanded_ops(r['prompt'], r['answer'])
    if status == 'solved':
        solved_rules[rule] += 1
        if len(solved_samples[rule]) < 3:
            solved_samples[rule].append(r['id'])
    else:
        still_unsolved.append(r)
        fail_reasons[status] += 1

total_new = sum(solved_rules.values())
print("=" * 70)
print(f"EXPANDED SOLVER RESULTS")
print(f"=" * 70)
print(f"Previously unsolved: {len(unsolved)}")
print(f"Now solvable: {total_new}")
print(f"Still unsolved: {len(still_unsolved)}")
print(f"Improvement: +{total_new}")
print()

print("Rules that work:")
for rule, cnt in solved_rules.most_common():
    samples = solved_samples.get(rule, [])
    print(f"  {rule}: {cnt}  (e.g. {samples[:2]})")

print()
print("Remaining fail reasons:")
for reason, cnt in fail_reasons.most_common():
    print(f"  {reason}: {cnt}")

# ── Detailed analysis of no_rule failures ──
print()
print("=" * 70)
print("DEEPER ANALYSIS of remaining no_rule failures")
print("=" * 70)

no_rule = [r for r in still_unsolved if True]  # all

# Break down: how many have potential operators but no rule fits?
has_op_no_rule = 0
multi_op_types = 0
single_op_only = 0

for r in no_rule[:300]:
    prompt = r['prompt']
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if 'determine the result for:' in line.lower():
            query = line.split(':')[-1].strip()
        elif '=' in line and 'alice' not in line.lower() and 'equation' not in line.lower() \
                and 'transformation' not in line.lower() and 'determine' not in line.lower() \
                and 'below' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                if lhs and rhs:
                    examples.append((lhs, rhs))
    
    if not query:
        continue
    
    # Count distinct operators in examples
    op_chars_found = set()
    for lhs, rhs in examples:
        for c in lhs:
            if c in ALL_NONALNUM:
                op_chars_found.add(c)
    
    if len(op_chars_found) > 1:
        multi_op_types += 1
    elif len(op_chars_found) == 1:
        single_op_only += 1
    else:
        has_op_no_rule += 1

print(f"  Multi-operator problems: {multi_op_types}")
print(f"  Single-operator problems: {single_op_only}")
print(f"  No operator found: {has_op_no_rule}")

# ── Manual inspection of still-unsolved with single operator ──
print()
print("=" * 70)
print("SINGLE-OPERATOR unsolved samples (should be more solvable)")
print("=" * 70)

shown = 0
for r in still_unsolved:
    if shown >= 8:
        break
    prompt = r['prompt']
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if 'determine the result for:' in line.lower():
            query = line.split(':')[-1].strip()
        elif '=' in line and 'alice' not in line.lower() and 'equation' not in line.lower() \
                and 'transformation' not in line.lower() and 'determine' not in line.lower() \
                and 'below' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                if lhs and rhs:
                    examples.append((lhs, rhs))
    
    if not query or not examples:
        continue
    
    # Show only single-op or interesting cases
    op_chars_found = set()
    for lhs, rhs in examples:
        for c in lhs:
            if c in ALL_NONALNUM:
                op_chars_found.add(c)
    
    q_ops = set(c for c in query if c in ALL_NONALNUM)
    
    print(f"\n--- id={r['id']}, ops_in_ex={sorted(op_chars_found)}, ops_in_q={sorted(q_ops)} ---")
    print(f"Gold: {r['answer']}")
    print(f"Query: {query}")
    for lhs, rhs in examples[:6]:
        print(f"  {lhs} = {rhs}")
    
    # Try to manually deduce: for each op in query, show what the examples say
    for op_c in q_ops:
        if op_c in op_chars_found:
            print(f"  → Op '{op_c}' examples:")
            for lhs, rhs in examples:
                parts_list = split_by_any_op(lhs, {op_c})
                for l, o, r_val in parts_list:
                    if is_numeric(l) and is_numeric(r_val) and is_numeric(rhs):
                        a, b, res = int(l), int(r_val), int(rhs)
                        print(f"    {a} {op_c} {b} = {res}  (a+b={a+b}, a-b={a-b}, a*b={a*b}, a^b={a**b if b<20 else '?'})")
    
    shown += 1

# ── Check numeric problems where we found pow ──
print()
print("=" * 70)
print("VERIFY pow rule: Show some pow-solvable examples")
print("=" * 70)

pow_count = 0
for r in unsolved:
    rule, status = solve_with_expanded_ops(r['prompt'], r['answer'])
    if status == 'solved' and 'pow' in str(rule):
        if pow_count < 3:
            print(f"\n--- id={r['id']} rule={rule} ---")
            print(f"Gold: {r['answer']}")
            lines = r['prompt'].strip().split('\n')
            for l in lines[:8]:
                print(f"  | {l}")
        pow_count += 1
print(f"\nTotal pow-solvable: {pow_count}")
