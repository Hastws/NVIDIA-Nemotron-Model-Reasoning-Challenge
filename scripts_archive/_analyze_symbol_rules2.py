"""
Symbol problem solver v2.
Key discovery: operators map to operations on 2-char operands treated as base-94 numbers.
"""
import pandas as pd
import re

df = pd.read_csv('competition_data/train.csv')

def classify(p):
    pl = p.lower()
    if 'cipher' in pl or 'encrypt' in pl or 'decrypt' in pl or 'encode' in pl or 'decode' in pl: return 'cipher'
    if 'gravit' in pl or 'planet' in pl: return 'gravity'
    if any(x in pl for x in ['base 2', 'base 8', 'base 10', 'base 16', 'binary', 'octal', 'hexadecimal']): return 'numeral'
    if any(x in pl for x in ['meter', 'mile', 'gallon', 'liter', 'pound', 'kilogram', 'inch', 'foot', 'feet', 'yard', 'ounce', 'celsius', 'fahrenheit', 'kelvin']): return 'unit_conv'
    if 'bit' in pl or 'xor' in pl or 'bitwise' in pl: return 'bit_ops'
    return 'symbol'

df['type'] = df['prompt'].apply(classify)
symbol_df = df[df['type'] == 'symbol'].copy()

def parse_examples(prompt):
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if '=' in line and 'determine' not in line.lower() and 'result' not in line.lower() and 'transformation' not in line.lower() and 'alice' not in line.lower() and 'equation' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        if 'determine the result for:' in line.lower():
            query = line.split(':')[-1].strip()
    return examples, query

CHAR_BASE = 33
CHAR_RANGE = 94

def str_to_base94(s):
    """Convert string to base-94 number."""
    val = 0
    for c in s:
        val = val * CHAR_RANGE + (ord(c) - CHAR_BASE)
    return val

def base94_to_str(val, min_len=0):
    """Convert base-94 number back to string."""
    if val < 0:
        val = val % (CHAR_RANGE ** 10)  # large wrap
    if val == 0:
        return chr(CHAR_BASE) * max(1, min_len)
    chars = []
    while val > 0:
        chars.append(chr((val % CHAR_RANGE) + CHAR_BASE))
        val //= CHAR_RANGE
    result = ''.join(reversed(chars))
    while len(result) < min_len:
        result = chr(CHAR_BASE) + result
    return result

# Define all candidate operations
def make_ops():
    return [
        ('concat_lr', lambda l, r: l + r),
        ('concat_rl', lambda l, r: r + l),
        ('b94_add', lambda l, r: base94_to_str(str_to_base94(l) + str_to_base94(r))),
        ('b94_sub', lambda l, r: base94_to_str(str_to_base94(l) - str_to_base94(r))),
        ('b94_sub_rev', lambda l, r: base94_to_str(str_to_base94(r) - str_to_base94(l))),
        ('b94_mul', lambda l, r: base94_to_str(str_to_base94(l) * str_to_base94(r))),
        # Charwise mod-94
        ('cw_add', lambda l, r: ''.join(chr(((ord(a)-CHAR_BASE)+(ord(b)-CHAR_BASE)) % CHAR_RANGE + CHAR_BASE) for a,b in zip(l,r)) if len(l)==len(r) else None),
        ('cw_sub', lambda l, r: ''.join(chr(((ord(a)-CHAR_BASE)-(ord(b)-CHAR_BASE)) % CHAR_RANGE + CHAR_BASE) for a,b in zip(l,r)) if len(l)==len(r) else None),
        ('cw_sub_rev', lambda l, r: ''.join(chr(((ord(b)-CHAR_BASE)-(ord(a)-CHAR_BASE)) % CHAR_RANGE + CHAR_BASE) for a,b in zip(l,r)) if len(l)==len(r) else None),
        ('cw_mul', lambda l, r: ''.join(chr(((ord(a)-CHAR_BASE)*(ord(b)-CHAR_BASE)) % CHAR_RANGE + CHAR_BASE) for a,b in zip(l,r)) if len(l)==len(r) else None),
        ('cw_xor', lambda l, r: ''.join(chr(((ord(a)-CHAR_BASE)^(ord(b)-CHAR_BASE)) % CHAR_RANGE + CHAR_BASE) for a,b in zip(l,r)) if len(l)==len(r) else None),
    ]

OPS = make_ops()

def find_rule(group):
    """Given a list of (left, right, result), find the matching rule."""
    for name, fn in OPS:
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
            return name, fn
    return None, None

def apply_rule(fn, left, right):
    try:
        return fn(left, right)
    except:
        return None

# ===== MAIN ANALYSIS =====
total_problems = 0
solved_problems = 0
rule_dist = {}
wrong_examples = []

for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if len(examples) < 2 or not query:
        continue
    
    # Parse examples into (left, op, right, result)
    parsed = []
    for lhs, rhs in examples:
        for op in ['+', '-', '*']:
            if op in lhs:
                parts = lhs.split(op, 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    parsed.append((parts[0], op, parts[1], rhs))
                    break
    
    if not parsed:
        # No-operator type: the entire LHS is the "input"
        # Try as single-operand transformation
        continue
    
    # Group by operator
    op_groups = {}
    for left, op, right, result in parsed:
        if op not in op_groups:
            op_groups[op] = []
        op_groups[op].append((left, right, result))
    
    # Find rules for each operator
    op_rules = {}
    for op, group in op_groups.items():
        rule_name, rule_fn = find_rule(group)
        if rule_name:
            op_rules[op] = (rule_name, rule_fn)
            rule_dist[rule_name] = rule_dist.get(rule_name, 0) + 1
    
    # Parse query and try to predict
    for op in ['+', '-', '*']:
        if op in query:
            qparts = query.split(op, 1)
            if len(qparts) == 2 and qparts[0] and qparts[1]:
                total_problems += 1
                if op in op_rules:
                    rule_name, rule_fn = op_rules[op]
                    predicted = apply_rule(rule_fn, qparts[0], qparts[1])
                    actual = str(row['answer'])
                    if predicted == actual:
                        solved_problems += 1
                    else:
                        if len(wrong_examples) < 15:
                            wrong_examples.append({
                                'rule': rule_name,
                                'predicted': predicted,
                                'actual': actual,
                                'query': query,
                                'examples': [(l, o, r, res) for l,o,r,res in parsed if o == op],
                            })
            break

print(f"Total symbol problems with operators: {total_problems}")
print(f"Correctly solved: {solved_problems} ({solved_problems/max(total_problems,1)*100:.1f}%)")
print(f"\nRule distribution:")
for rule, count in sorted(rule_dist.items(), key=lambda x: -x[1]):
    print(f"  {rule}: {count}")

print(f"\n=== WRONG PREDICTIONS ({len(wrong_examples)}) ===")
for i, w in enumerate(wrong_examples):
    print(f"\n--- Wrong {i+1} (rule={w['rule']}) ---")
    print(f"  Query: {w['query']} => predicted='{w['predicted']}' actual='{w['actual']}'")
    for l, o, r, res in w['examples']:
        print(f"  Example: '{l}' {o} '{r}' = '{res}'")
        # Show char codes
        print(f"    codes: {[ord(c) for c in l]} {o} {[ord(c) for c in r]} = {[ord(c) for c in res]}")

# ===== NO-OP SYMBOLIC: try single-operand transformations =====
print(f"\n{'='*60}")
print("NO-OPERATOR PROBLEMS: SINGLE TRANSFORM ANALYSIS")
print(f"{'='*60}")

no_op_total = 0
no_op_solved = 0

for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if len(examples) < 2 or not query:
        continue
    
    # Check if any example has operators
    has_ops = False
    for lhs, rhs in examples:
        for op in ['+', '-', '*']:
            if op in lhs:
                has_ops = True
                break
    
    if has_ops:
        continue
    
    no_op_total += 1
    
    # Try: char-level substitution cipher
    # Build mapping from each (input_char, position) -> output_char
    # Or just input_char -> output_char
    
    # Try simple char shift: output[i] = chr(ord(input[i]) + k)
    # Or char substitution (each unique char maps to another)
    
    char_map = {}
    consistent = True
    
    for lhs, rhs in examples:
        if len(lhs) != len(rhs):
            consistent = False
            break
        for c_in, c_out in zip(lhs, rhs):
            if c_in in char_map:
                if char_map[c_in] != c_out:
                    consistent = False
                    break
            char_map[c_in] = c_out
    
    if consistent and char_map and query:
        # Can we predict?
        if all(c in char_map for c in query):
            predicted = ''.join(char_map[c] for c in query)
            actual = str(row['answer'])
            if predicted == actual:
                no_op_solved += 1
    
    if no_op_total <= 3 and not consistent:
        print(f"\n--- No-op problem (inconsistent mapping) ---")
        for lhs, rhs in examples:
            print(f"  '{lhs}' => '{rhs}' (len {len(lhs)} => {len(rhs)})")
        print(f"  Query: '{query}' => '{row['answer']}'")

print(f"\nNo-op problems: {no_op_total}")
print(f"Solved by char substitution: {no_op_solved} ({no_op_solved/max(no_op_total,1)*100:.1f}%)")

# ===== NUMERIC SYMBOL PROBLEMS =====
print(f"\n{'='*60}")
print("NUMERIC SYMBOL PROBLEMS")
print(f"{'='*60}")

# These have numeric operands like "55#38 = ..."
# Each symbol-operator maps to an arithmetic operation
numeric_ops = [
    ('add', lambda a,b: str(a+b)),
    ('sub', lambda a,b: str(a-b)),
    ('sub_rev', lambda a,b: str(b-a)),
    ('mul', lambda a,b: str(a*b)),
    ('abs_diff', lambda a,b: str(abs(a-b))),
    ('concat_ab', lambda a,b: str(a)+str(b)),
    ('concat_ba', lambda a,b: str(b)+str(a)),
    ('div', lambda a,b: str(a//b) if b!=0 else None),
    ('mod', lambda a,b: str(a%b) if b!=0 else None),
    ('pow', lambda a,b: str(a**b) if b<20 else None),
    ('a2+b2', lambda a,b: str(a**2+b**2)),
    ('a2-b2', lambda a,b: str(a**2-b**2)),
    ('a2*b', lambda a,b: str(a**2*b)),
    ('(a+b)*2', lambda a,b: str((a+b)*2)),
    ('a*10+b', lambda a,b: str(a*10+b)),
    ('b*10+a', lambda a,b: str(b*10+a)),
    # Digit operations
    ('rev_a', lambda a,b: str(a)[::-1]),
    ('digit_sum_a+b', lambda a,b: str(sum(int(d) for d in str(abs(a))) + b)),
    ('swap_concat', lambda a,b: str(b)+str(a)),
]

num_total = 0
num_solved = 0

for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if len(examples) < 2 or not query:
        continue
    
    # Check if numeric
    has_digits = any(re.search(r'\d', lhs) for lhs, rhs in examples)
    if not has_digits:
        continue
    
    # Parse: "55#38" -> (55, '#', 38)
    parsed = []
    for lhs, rhs in examples:
        m = re.match(r'(\d+)(.)([\d]+)', lhs)
        if m:
            parsed.append((int(m.group(1)), m.group(2), int(m.group(3)), rhs))
    
    if len(parsed) < 2:
        continue
    
    # Group by operator
    op_groups = {}
    for a, op, b, result in parsed:
        if op not in op_groups:
            op_groups[op] = []
        op_groups[op].append((a, b, result))
    
    # Parse query
    qm = re.match(r'(\d+)(.)([\d]+)', query)
    if not qm:
        continue
    qa, qop, qb = int(qm.group(1)), qm.group(2), int(qm.group(3))
    
    num_total += 1
    
    if qop in op_groups:
        group = op_groups[qop]
        for rule_name, fn in numeric_ops:
            all_match = True
            for a, b, result in group:
                try:
                    pred = fn(a, b)
                    if pred is None or pred != result:
                        all_match = False
                        break
                except:
                    all_match = False
                    break
            if all_match:
                try:
                    predicted = fn(qa, qb)
                    if predicted == str(row['answer']):
                        num_solved += 1
                except:
                    pass
                break

print(f"Numeric problems: {num_total}")
print(f"Solved: {num_solved} ({num_solved/max(num_total,1)*100:.1f}%)")

# GRAND TOTAL
print(f"\n{'='*60}")
print(f"GRAND TOTAL")
print(f"{'='*60}")
print(f"Symbol problems attempted: {total_problems + no_op_total + num_total}")
print(f"Solved: {solved_problems + no_op_solved + num_solved}")
print(f"  Char-op solved: {solved_problems}")
print(f"  No-op solved: {no_op_solved}")  
print(f"  Numeric solved: {num_solved}")
