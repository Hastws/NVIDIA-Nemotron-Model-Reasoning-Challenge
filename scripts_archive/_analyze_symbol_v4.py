#!/usr/bin/env python3
"""
Deep analysis of unsolved symbol problems.
Classify into Numeric vs Symbolic, analyze operator patterns,
and attempt to discover new solvable rules.
"""
import csv
import re
from collections import defaultdict, Counter

input_path = 'data/train_annotated.csv'
rows = list(csv.DictReader(open(input_path)))

# Filter symbol problems
symbol_rows = [r for r in rows if r['type'] == 'symbol']
unsolved = [r for r in symbol_rows if r['match'] != 'True']
solved = [r for r in symbol_rows if r['match'] == 'True']

print(f"Symbol: {len(symbol_rows)} total, {len(solved)} solved, {len(unsolved)} unsolved")
print()

# ── Classify subtypes ──
def classify_symbol(prompt):
    """Classify symbol problem into subtypes based on content."""
    lines = prompt.strip().split('\n')
    
    # Check for key phrases
    has_equation = any('equation' in l.lower() for l in lines)
    has_transformation = any('transformation' in l.lower() for l in lines)
    has_alice = any('alice' in l.lower() for l in lines)
    
    # Check operand types in examples
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
    
    # Check if operands are purely numeric
    all_numeric = True
    has_op_in_lhs = False
    op_chars = set('+-*/|\\^&')
    
    for lhs, rhs in examples:
        # Check for operators in lhs
        for c in lhs:
            if c in op_chars:
                has_op_in_lhs = True
                break
        # Check if all non-op chars are digits
        for c in lhs:
            if c not in op_chars and not c.isdigit() and c != ' ':
                all_numeric = False
                break
        for c in rhs:
            if not c.isdigit() and c not in ('-', ' ', '.'):
                all_numeric = False
                break
    
    if not examples:
        return 'no_examples', examples, query
    
    if all_numeric and has_op_in_lhs:
        return 'numeric_equation', examples, query
    elif has_op_in_lhs:
        return 'symbolic_equation', examples, query
    else:
        return 'transformation', examples, query

# ── Classify all ──
subtype_counts = Counter()
subtype_solved = Counter()
subtype_unsolved = Counter()
subtype_samples = defaultdict(list)

for r in symbol_rows:
    subtype, examples, query = classify_symbol(r['prompt'])
    is_solved = r['match'] == 'True'
    subtype_counts[subtype] += 1
    if is_solved:
        subtype_solved[subtype] += 1
    else:
        subtype_unsolved[subtype] += 1
    if not is_solved and len(subtype_samples[subtype]) < 5:
        subtype_samples[subtype].append((r, examples, query))

print("=" * 70)
print("SUBTYPE CLASSIFICATION")
print("=" * 70)
for st in sorted(subtype_counts.keys()):
    total = subtype_counts[st]
    sol = subtype_solved[st]
    unsol = subtype_unsolved[st]
    rate = sol / total * 100 if total > 0 else 0
    print(f"  {st:25s}: {total:5d} total, {sol:5d} solved ({rate:.1f}%), {unsol:5d} unsolved")
print()

# ── Show samples per subtype ──
for st, samples in sorted(subtype_samples.items()):
    print("=" * 70)
    print(f"SUBTYPE: {st} — Sample unsolved problems")
    print("=" * 70)
    for idx, (r, examples, query) in enumerate(samples[:3]):
        print(f"\n--- Sample {idx+1} (id={r['id']}, fail={r['fail_reason']}) ---")
        print(f"Gold answer: {r['answer']}")
        print(f"Examples ({len(examples)}):")
        for lhs, rhs in examples[:5]:
            print(f"  {lhs} = {rhs}")
        if len(examples) > 5:
            print(f"  ... ({len(examples)-5} more)")
        print(f"Query: {query}")
        # Show first 10 lines of prompt for context
        prompt_lines = r['prompt'].strip().split('\n')
        print(f"Prompt first lines:")
        for pl in prompt_lines[:8]:
            print(f"  | {pl}")
        if len(prompt_lines) > 8:
            print(f"  | ... ({len(prompt_lines)-8} more lines)")
    print()

# ── Analyze numeric_equation in detail ──
print("=" * 70)
print("NUMERIC EQUATION — Detailed operator analysis")
print("=" * 70)

op_chars_set = set('+-*/|\\^&')

def parse_op_from_lhs(lhs):
    """Extract operator and operands from LHS."""
    for i, c in enumerate(lhs):
        if c in op_chars_set and i > 0 and i < len(lhs) - 1:
            left = lhs[:i].strip()
            right = lhs[i+1:].strip()
            return left, c, right
    return None

numeric_eq_unsolved = []
for r in symbol_rows:
    subtype, examples, query = classify_symbol(r['prompt'])
    if subtype == 'numeric_equation' and r['match'] != 'True':
        numeric_eq_unsolved.append((r, examples, query))

print(f"\nNumeric equation unsolved: {len(numeric_eq_unsolved)}")

# Try to find patterns in numeric equations
numeric_op_patterns = Counter()
for r, examples, query in numeric_eq_unsolved[:200]:
    if not query:
        numeric_op_patterns['no_query'] += 1
        continue
    parsed = parse_op_from_lhs(query)
    if parsed:
        _, op, _ = parsed
        # Analyze what operations might work
        ops_tried = []
        for lhs, rhs in examples:
            p = parse_op_from_lhs(lhs)
            if p:
                a, op_c, b = p
                try:
                    a_int, b_int, r_int = int(a), int(b), int(rhs)
                    diff = r_int - a_int - b_int
                    ratio = r_int / (a_int * b_int) if a_int * b_int != 0 else None
                    ops_tried.append((a_int, b_int, r_int, op_c))
                except:
                    pass
        if ops_tried:
            # Try to deduce the operation
            a0, b0, r0, op0 = ops_tried[0]
            candidates = []
            # a+b
            if all(a+b == r for a, b, r, _ in ops_tried): candidates.append('add')
            # a-b
            if all(a-b == r for a, b, r, _ in ops_tried): candidates.append('sub')
            # b-a
            if all(b-a == r for a, b, r, _ in ops_tried): candidates.append('sub_rev')
            # a*b
            if all(a*b == r for a, b, r, _ in ops_tried): candidates.append('mul')
            # a//b
            if all(b != 0 and a//b == r for a, b, r, _ in ops_tried): candidates.append('div')
            # b//a
            if all(a != 0 and b//a == r for a, b, r, _ in ops_tried): candidates.append('div_rev')
            # a%b
            if all(b != 0 and a%b == r for a, b, r, _ in ops_tried): candidates.append('mod')
            # b%a
            if all(a != 0 and b%a == r for a, b, r, _ in ops_tried): candidates.append('mod_rev')
            # a**b
            if all(a**b == r for a, b, r, _ in ops_tried if b < 20): candidates.append('pow')
            # a*b+1
            if all(a*b+1 == r for a, b, r, _ in ops_tried): candidates.append('mul_plus1')
            # a*b-1
            if all(a*b-1 == r for a, b, r, _ in ops_tried): candidates.append('mul_minus1')
            # a+b+1
            if all(a+b+1 == r for a, b, r, _ in ops_tried): candidates.append('add_plus1')
            # a+b-1
            if all(a+b-1 == r for a, b, r, _ in ops_tried): candidates.append('add_minus1')
            # a*b+a
            if all(a*b+a == r for a, b, r, _ in ops_tried): candidates.append('mul_add_a')
            # a*b+b
            if all(a*b+b == r for a, b, r, _ in ops_tried): candidates.append('mul_add_b')
            # a*b-a
            if all(a*b-a == r for a, b, r, _ in ops_tried): candidates.append('mul_sub_a')
            # a*b-b
            if all(a*b-b == r for a, b, r, _ in ops_tried): candidates.append('mul_sub_b')
            # a+b*2
            if all(a+b*2 == r for a, b, r, _ in ops_tried): candidates.append('a_plus_2b')
            # a*2+b
            if all(a*2+b == r for a, b, r, _ in ops_tried): candidates.append('2a_plus_b')
            # a^b (XOR)
            if all((a^b) == r for a, b, r, _ in ops_tried): candidates.append('xor')
            # a|b (OR)
            if all((a|b) == r for a, b, r, _ in ops_tried): candidates.append('or')
            # a&b (AND)
            if all((a&b) == r for a, b, r, _ in ops_tried): candidates.append('and')
            # max(a,b)
            if all(max(a,b) == r for a, b, r, _ in ops_tried): candidates.append('max')
            # min(a,b)
            if all(min(a,b) == r for a, b, r, _ in ops_tried): candidates.append('min')
            # abs(a-b)
            if all(abs(a-b) == r for a, b, r, _ in ops_tried): candidates.append('abs_diff')
            # a*a+b*b (sum of squares)
            if all(a*a+b*b == r for a, b, r, _ in ops_tried): candidates.append('sum_sq')
            # (a+b)^2
            if all((a+b)**2 == r for a, b, r, _ in ops_tried): candidates.append('sq_sum')
            # a^2+b
            if all(a**2+b == r for a, b, r, _ in ops_tried): candidates.append('a_sq_plus_b')
            # a+b^2
            if all(a+b**2 == r for a, b, r, _ in ops_tried): candidates.append('a_plus_b_sq')
            # digit concatenation: str(a)+str(b=
            # a*b+a+b
            if all(a*b+a+b == r for a, b, r, _ in ops_tried): candidates.append('mul_add_both')
            # (a+1)*(b+1)-1 = a*b+a+b
            # a*b*2
            if all(a*b*2 == r for a, b, r, _ in ops_tried): candidates.append('mul2')
            
            if candidates:
                for c in candidates:
                    numeric_op_patterns[f'found:{c}'] += 1
            else:
                numeric_op_patterns['no_rule_found'] += 1
                # Print some for manual analysis
    else:
        numeric_op_patterns['no_op_in_query'] += 1

print(f"\nNumeric equation patterns:")
for pat, cnt in numeric_op_patterns.most_common(30):
    print(f"  {pat}: {cnt}")

# ── Analyze no_operator_in_query ──
print()
print("=" * 70)
print("NO OPERATOR IN QUERY — What does the query look like?")
print("=" * 70)

no_op_queries = []
for r in symbol_rows:
    if r['fail_reason'] == 'no_operator_in_query':
        subtype, examples, query = classify_symbol(r['prompt'])
        no_op_queries.append((r, examples, query, subtype))

print(f"Total: {len(no_op_queries)}")

# Classify what the query contains
query_patterns = Counter()
for r, examples, query, subtype in no_op_queries:
    if query is None:
        query_patterns['None'] += 1
    elif all(c.isdigit() for c in query):
        query_patterns['pure_digits'] += 1
    elif all(c.isalpha() or c in '!@#$%^&()_' for c in query):
        query_patterns['pure_symbols'] += 1
    else:
        query_patterns[f'mixed'] += 1

print("Query patterns:")
for p, c in query_patterns.most_common():
    print(f"  {p}: {c}")

# Show samples
print("\nSamples of no_operator_in_query:")
for i, (r, examples, query, subtype) in enumerate(no_op_queries[:5]):
    print(f"\n--- Sample {i+1} (id={r['id']}, subtype={subtype}) ---")
    print(f"Gold: {r['answer']}")
    print(f"Query: '{query}'")
    print(f"Examples ({len(examples)}):")
    for lhs, rhs in examples[:5]:
        print(f"  {lhs} = {rhs}")
    prompt_lines = r['prompt'].strip().split('\n')
    print(f"Full prompt:")
    for pl in prompt_lines[:15]:
        print(f"  | {pl}")

# ── Deep dive into symbolic_equation ──
print()
print("=" * 70)
print("SYMBOLIC EQUATION — Detailed analysis")
print("=" * 70)

sym_eq_unsolved = []
for r in symbol_rows:
    subtype, examples, query = classify_symbol(r['prompt'])
    if subtype == 'symbolic_equation' and r['match'] != 'True':
        sym_eq_unsolved.append((r, examples, query))

print(f"Symbolic equation unsolved: {len(sym_eq_unsolved)}")

# Analyze by operator and operand lengths
op_len_patterns = Counter()
for r, examples, query in sym_eq_unsolved:
    if not query:
        continue
    parsed = parse_op_from_lhs(query)
    if not parsed:
        op_len_patterns['no_op'] += 1
        continue
    left, op, right = parsed
    ll, rl = len(left), len(right)
    # Check example patterns
    ex_lens = []
    for lhs, rhs in examples:
        p = parse_op_from_lhs(lhs)
        if p:
            ex_lens.append((len(p[0]), len(p[2]), len(rhs)))
    if ex_lens:
        # Check if all same length
        if all(l1 == l2 == rl2 for l1, l2, rl2 in ex_lens):
            op_len_patterns[f'same_len({ex_lens[0][0]})_op={op}'] += 1
        elif all(l1 == l2 for l1, l2, _ in ex_lens):
            op_len_patterns[f'equal_ops_op={op}'] += 1
        else:
            op_len_patterns[f'varied_len_op={op}'] += 1

print("\nOperand length patterns:")
for p, c in op_len_patterns.most_common(20):
    print(f"  {p}: {c}")

# ── Try more advanced rules on symbolic equations ──
print()
print("=" * 70)
print("TRYING ADVANCED RULES on all unsolved")
print("=" * 70)

CHAR_BASE = 33
CHAR_RANGE = 94

def try_charwise(fn, left, right, result):
    if len(left) != len(right) or len(left) != len(result):
        return False
    try:
        pred = ''.join(fn(a, b) for a, b in zip(left, right))
        return pred == result
    except:
        return False

def try_charwise_unequal(fn, left, right, result):
    """Try charwise op where result length may differ."""
    try:
        pred = fn(left, right)
        return pred == result
    except:
        return False

# Extended charwise operations
EXTENDED_CW_OPS = [
    ('cw_add', lambda a, b: chr(((ord(a)-33) + (ord(b)-33)) % 94 + 33)),
    ('cw_sub', lambda a, b: chr(((ord(a)-33) - (ord(b)-33)) % 94 + 33)),
    ('cw_sub_rev', lambda a, b: chr(((ord(b)-33) - (ord(a)-33)) % 94 + 33)),
    ('cw_xor', lambda a, b: chr(((ord(a)-33) ^ (ord(b)-33)) % 94 + 33)),
    ('cw_mul', lambda a, b: chr(((ord(a)-33) * (ord(b)-33)) % 94 + 33)),
    ('cw_and', lambda a, b: chr(((ord(a)-33) & (ord(b)-33)) % 94 + 33)),
    ('cw_or', lambda a, b: chr(((ord(a)-33) | (ord(b)-33)) % 94 + 33)),
    ('cw_avg', lambda a, b: chr(((ord(a)-33) + (ord(b)-33)) // 2 % 94 + 33)),
    ('cw_max', lambda a, b: chr(max(ord(a)-33, ord(b)-33) % 94 + 33)),
    ('cw_min', lambda a, b: chr(min(ord(a)-33, ord(b)-33) % 94 + 33)),
    ('cw_add1', lambda a, b: chr(((ord(a)-33) + (ord(b)-33) + 1) % 94 + 33)),
    ('cw_sub1', lambda a, b: chr(((ord(a)-33) + (ord(b)-33) - 1) % 94 + 33)),
    ('cw_mul_add1', lambda a, b: chr(((ord(a)-33) * (ord(b)-33) + 1) % 94 + 33)),
]

# Extended b94 operations
def str_to_b94(s):
    val = 0
    for c in s:
        val = val * 94 + (ord(c) - 33)
    return val

def b94_to_str(val):
    if val == 0:
        return chr(33)
    if val < 0:
        val = val % (94 ** 10)
    chars = []
    while val > 0:
        chars.append(chr((val % 94) + 33))
        val //= 94
    return ''.join(reversed(chars)) if chars else chr(33)

EXTENDED_B94_OPS = [
    ('b94_add', lambda l, r: b94_to_str(str_to_b94(l) + str_to_b94(r))),
    ('b94_sub', lambda l, r: b94_to_str(str_to_b94(l) - str_to_b94(r))),
    ('b94_sub_rev', lambda l, r: b94_to_str(str_to_b94(r) - str_to_b94(l))),
    ('b94_mul', lambda l, r: b94_to_str(str_to_b94(l) * str_to_b94(r))),
    ('b94_xor', lambda l, r: b94_to_str(str_to_b94(l) ^ str_to_b94(r))),
    ('b94_and', lambda l, r: b94_to_str(str_to_b94(l) & str_to_b94(r))),
    ('b94_or', lambda l, r: b94_to_str(str_to_b94(l) | str_to_b94(r))),
    ('b94_add1', lambda l, r: b94_to_str(str_to_b94(l) + str_to_b94(r) + 1)),
    ('b94_sub1', lambda l, r: b94_to_str(str_to_b94(l) + str_to_b94(r) - 1)),
    ('b94_mul_add1', lambda l, r: b94_to_str(str_to_b94(l) * str_to_b94(r) + 1)),
    ('b94_mul_sub1', lambda l, r: b94_to_str(str_to_b94(l) * str_to_b94(r) - 1)),
    ('concat', lambda l, r: l + r),
    ('concat_rev', lambda l, r: r + l),
]

# Count new discoveries
new_solved = Counter()
new_solved_detail = defaultdict(list)

for r in unsolved:
    prompt = r['prompt']
    gold = r['answer']
    
    examples_raw = []
    query = None
    lines = prompt.strip().split('\n')
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
                    examples_raw.append((lhs, rhs))
    
    if not query or not examples_raw:
        continue
    
    # Parse operators from LHS
    op_chars_set_local = set('+-*/|\\^&')
    def local_split(expr):
        for i, c in enumerate(expr):
            if c in op_chars_set_local and i > 0 and i < len(expr) - 1:
                return expr[:i], c, expr[i+1:]
        return None
    
    q_split = local_split(query)
    
    if q_split:
        q_left, q_op, q_right = q_split
        
        # Group examples by operator
        op_groups = defaultdict(list)
        for lhs, rhs in examples_raw:
            sp = local_split(lhs)
            if sp:
                op_groups[sp[1]].append((sp[0], sp[2], rhs))
        
        if q_op in op_groups:
            group = op_groups[q_op]
            
            # Try extended b94 ops
            for op_name, fn in EXTENDED_B94_OPS:
                try:
                    all_match = all(fn(l, r_op) == res for l, r_op, res in group)
                    if all_match:
                        pred = fn(q_left, q_right)
                        if pred == gold:
                            new_solved[f'b94:{op_name}'] += 1
                            if len(new_solved_detail[f'b94:{op_name}']) < 2:
                                new_solved_detail[f'b94:{op_name}'].append(r['id'])
                            break
                except:
                    continue
            else:
                # Try extended charwise ops
                if len(q_left) == len(q_right):
                    for op_name, fn in EXTENDED_CW_OPS:
                        try:
                            all_match = all(
                                try_charwise(fn, l, r_op, res) 
                                for l, r_op, res in group
                            )
                            if all_match:
                                pred = ''.join(fn(a, b) for a, b in zip(q_left, q_right))
                                if pred == gold:
                                    new_solved[f'cw:{op_name}'] += 1
                                    if len(new_solved_detail[f'cw:{op_name}']) < 2:
                                        new_solved_detail[f'cw:{op_name}'].append(r['id'])
                                    break
                        except:
                            continue
    else:
        # No operator in query → transformation type
        # Try to find mapping from input to output
        # Check if examples define a char-level substitution
        if len(examples_raw) >= 2:
            # Build char mapping from all examples
            char_map = {}
            consistent = True
            for lhs, rhs in examples_raw:
                if len(lhs) != len(rhs):
                    consistent = False
                    break
                for c_in, c_out in zip(lhs, rhs):
                    if c_in in char_map:
                        if char_map[c_in] != c_out:
                            consistent = False
                            break
                    else:
                        char_map[c_in] = c_out
                if not consistent:
                    break
            
            if consistent and char_map and query:
                try:
                    pred = ''.join(char_map.get(c, c) for c in query)
                    if pred == gold:
                        new_solved['transform:char_map'] += 1
                        if len(new_solved_detail['transform:char_map']) < 3:
                            new_solved_detail['transform:char_map'].append(r['id'])
                except:
                    pass
            
            # Try constant shift
            if not consistent or not char_map:
                shifts = set()
                for lhs, rhs in examples_raw:
                    if len(lhs) != len(rhs):
                        break
                    for c_in, c_out in zip(lhs, rhs):
                        shifts.add((ord(c_out) - ord(c_in)) % 94)
                if len(shifts) == 1:
                    shift = shifts.pop()
                    try:
                        pred = ''.join(chr((ord(c) - 33 + shift) % 94 + 33) for c in query)
                        if pred == gold:
                            new_solved['transform:const_shift'] += 1
                            if len(new_solved_detail['transform:const_shift']) < 3:
                                new_solved_detail['transform:const_shift'].append(r['id'])
                    except:
                        pass
            
            # Try reverse
            for lhs, rhs in examples_raw:
                if lhs[::-1] == rhs:
                    continue
                else:
                    break
            else:
                if query and query[::-1] == gold:
                    new_solved['transform:reverse'] += 1
            
            # Try b94 mapping on whole strings
            for lhs, rhs in examples_raw[:1]:
                try:
                    lv = str_to_b94(lhs)
                    rv = str_to_b94(rhs)
                    diff = rv - lv
                    # Check if constant diff
                    all_same_diff = all(
                        str_to_b94(rr) - str_to_b94(ll) == diff 
                        for ll, rr in examples_raw
                    )
                    if all_same_diff and query:
                        pred = b94_to_str(str_to_b94(query) + diff)
                        if pred == gold:
                            new_solved['transform:b94_shift'] += 1
                            if len(new_solved_detail['transform:b94_shift']) < 3:
                                new_solved_detail['transform:b94_shift'].append(r['id'])
                except:
                    pass

print(f"\nNew discoveries (would be solvable with extended rules):")
total_new = 0
for rule, cnt in new_solved.most_common():
    total_new += cnt
    sample_ids = new_solved_detail.get(rule, [])
    print(f"  {rule}: {cnt}  (sample ids: {sample_ids})")
print(f"\nTotal new solvable: {total_new}")
print(f"Remaining unsolved: {len(unsolved) - total_new}")

# ── Show some hard unsolved for manual inspection ──
print()
print("=" * 70)
print("STILL UNSOLVED — Random samples for manual analysis")
print("=" * 70)

still_unsolved_ids = set()
for rule, detail in new_solved_detail.items():
    still_unsolved_ids.update(detail)

shown = 0
for r in unsolved:
    if r['id'] in still_unsolved_ids:
        continue
    if shown >= 10:
        break
    subtype, examples, query = classify_symbol(r['prompt'])
    print(f"\n--- id={r['id']}, subtype={subtype}, fail={r['fail_reason']} ---")
    print(f"Gold: {r['answer']}")
    if query:
        print(f"Query: {query}")
    print(f"Examples ({len(examples)}):")
    for lhs, rhs in examples[:6]:
        print(f"  {lhs} = {rhs}")
    if len(examples) > 6:
        print(f"  ... ({len(examples)-6} more)")
    shown += 1
