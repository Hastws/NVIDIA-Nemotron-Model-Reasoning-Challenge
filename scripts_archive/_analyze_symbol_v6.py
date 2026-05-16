#!/usr/bin/env python3
"""
Analyze the 283 no_numeric_rule and 869 no_rule cases.
Focus on finding more patterns, especially digit-level operations for 2-digit numbers.
"""
import csv
from collections import defaultdict, Counter

CHAR_BASE = 33
CHAR_RANGE = 94

ALL_NONALNUM = set()
for i in range(33, 127):
    c = chr(i)
    if not c.isalnum() and c != '=' and c != ' ':
        ALL_NONALNUM.add(c)

def split_by_any_op(expr, op_set=ALL_NONALNUM):
    for i, c in enumerate(expr):
        if c in op_set and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i+1:]
    return None

input_path = 'data/train_annotated.csv'
rows = list(csv.DictReader(open(input_path)))
symbol_rows = [r for r in rows if r['type'] == 'symbol']
unsolved = [r for r in symbol_rows if r['match'] != 'True']

# ── Focus on numeric equations that have no rule ──
print("=" * 70)
print("NUMERIC EQUATIONS: Manual pattern analysis")
print("=" * 70)

numeric_no_rule = []
for r in unsolved:
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
    
    # Check if this is numeric
    q_parsed = split_by_any_op(query)
    if not q_parsed:
        continue
    
    q_left, q_op, q_right = q_parsed
    
    # Filter: both operands must be numeric
    try:
        int(q_left)
        int(q_right)
    except:
        continue
    
    # Group by operator
    op_groups = defaultdict(list)
    for lhs, rhs in examples:
        p = split_by_any_op(lhs)
        if p:
            l, o, r_val = p
            try:
                int(l)
                int(r_val)
                op_groups[o].append((int(l), int(r_val), rhs))
            except:
                pass
    
    if q_op not in op_groups:
        continue
    
    group = op_groups[q_op]
    gold = r['answer']
    
    # Check if we didn't already solve this with v5
    # Let me try many more rules
    solved = False
    for rule_name, fn in [
        ('add', lambda a,b: str(a+b)),
        ('sub', lambda a,b: str(a-b)),
        ('sub_rev', lambda a,b: str(b-a)),
        ('mul', lambda a,b: str(a*b)),
        ('pow', lambda a,b: str(a**b) if 0<=b<=20 and a**b < 10**15 else None),
        ('pow_rev', lambda a,b: str(b**a) if 0<=a<=20 and b**a < 10**15 else None),
        ('concat', lambda a,b: str(a)+str(b)),
        ('concat_rev', lambda a,b: str(b)+str(a)),
        ('abs_diff', lambda a,b: str(abs(a-b))),
        ('add1', lambda a,b: str(a+b+1)),
        ('sub1', lambda a,b: str(a+b-1)),
        ('mul_add1', lambda a,b: str(a*b+1)),
        ('mul_sub1', lambda a,b: str(a*b-1)),
    ]:
        all_match = True
        for a, b, res in group:
            try:
                pred = fn(a, b)
                if pred is None or pred != res:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                ans = fn(int(q_left), int(q_right))
                if ans == gold:
                    solved = True
                    break
            except:
                pass
    
    if not solved:
        numeric_no_rule.append((r, group, q_left, q_op, q_right, gold))

print(f"Truly unsolvable numeric equations: {len(numeric_no_rule)}")
print()

# Show detailed examples with manual analysis
shown = 0
for r_info, group, q_l, q_o, q_r, gold in numeric_no_rule[:30]:
    if shown >= 30:
        break
    print(f"\n--- id={r_info['id']}, op='{q_o}', query={q_l}{q_o}{q_r}, gold={gold} ---")
    for a, b, res in group:
        # Try to figure out the formula
        extras = []
        try:
            res_int = int(res)
            extras.append(f"a+b={a+b}")
            extras.append(f"a-b={a-b}")
            extras.append(f"a*b={a*b}")
            if b != 0: extras.append(f"a//b={a//b}")
            if b != 0: extras.append(f"a%b={a%b}")
            extras.append(f"|a-b|={abs(a-b)}")
            if 10<=a<=99 and 10<=b<=99:
                a1,a2 = a//10, a%10
                b1,b2 = b//10, b%10
                extras.append(f"digits:{a1}{a2},{b1}{b2}")
                extras.append(f"d:a1b1={a1*b1},a1b2={a1*b2},a2b1={a2*b1},a2b2={a2*b2}")
                extras.append(f"d:a1+b1={a1+b1},a2+b2={a2+b2}")
                extras.append(f"concat_digits={a1}{b1}{a2}{b2}")
            if 0 <= b <= 10:
                try: extras.append(f"a^b={a**b}")
                except: pass
        except:
            extras.append(f"res_not_int")
        
        print(f"  {a} {q_o} {b} = {res}  ({', '.join(extras)})")
    shown += 1

# ── Now check: how many have result strings with non-digit chars? ──
print()
print("=" * 70)
print("RESULT STRING ANALYSIS")
print("=" * 70)

res_type = Counter()
for r_info, group, q_l, q_o, q_r, gold in numeric_no_rule:
    for a, b, res in group:
        if res.isdigit() or (res.startswith('-') and res[1:].isdigit()):
            res_type['int'] += 1
        else:
            res_type['non_int'] += 1
            break
    else:
        continue
    # If we broke, the result has non-int entries
    
    # Check gold
    if gold.isdigit() or (gold.startswith('-') and gold[1:].isdigit()):
        pass
    else:
        res_type['gold_non_int'] += 1

print("Result types in unsolvable numeric equations:")
for t, c in res_type.most_common():
    print(f"  {t}: {c}")

# ── Count how many examples per operator group ──
print()
print("=" * 70)
print("EXAMPLES PER OPERATOR (for numeric no-rule)")
print("=" * 70)

group_sizes = Counter()
for r_info, group, q_l, q_o, q_r, gold in numeric_no_rule:
    group_sizes[len(group)] += 1

print("Distribution of example count per operator group:")
for size, cnt in sorted(group_sizes.items()):
    print(f"  {size} examples: {cnt} problems")

# ── Single-example groups: check if gold matches ANY rule ──
print()
print("=" * 70)
print("SINGLE-EXAMPLE RULES THAT PREDICT GOLD")
print("=" * 70)

EXTENDED_RULES = [
    ('add', lambda a,b: str(a+b)),
    ('sub', lambda a,b: str(a-b)),
    ('sub_rev', lambda a,b: str(b-a)),
    ('mul', lambda a,b: str(a*b)),
    ('div', lambda a,b: str(a//b) if b else None),
    ('div_rev', lambda a,b: str(b//a) if a else None),
    ('mod', lambda a,b: str(a%b) if b else None),
    ('mod_rev', lambda a,b: str(b%a) if a else None),
    ('abs_diff', lambda a,b: str(abs(a-b))),
    ('max', lambda a,b: str(max(a,b))),
    ('min', lambda a,b: str(min(a,b))),
    ('pow', lambda a,b: str(a**b) if 0<=b<=20 else None),
    ('pow_rev', lambda a,b: str(b**a) if 0<=a<=20 else None),
    ('concat', lambda a,b: str(a)+str(b)),
    ('concat_rev', lambda a,b: str(b)+str(a)),
    ('add1', lambda a,b: str(a+b+1)),
    ('sub1', lambda a,b: str(a+b-1)),
    ('mul_add1', lambda a,b: str(a*b+1)),
    ('mul_sub1', lambda a,b: str(a*b-1)),
    ('mul_add_a', lambda a,b: str(a*b+a)),
    ('mul_add_b', lambda a,b: str(a*b+b)),
    ('mul_sub_a', lambda a,b: str(a*b-a)),
    ('mul_sub_b', lambda a,b: str(a*b-b)),
    ('a2_b', lambda a,b: str(a**2+b)),
    ('a_b2', lambda a,b: str(a+b**2)),
    ('a2_b2', lambda a,b: str(a**2+b**2)),
    ('a2_sub_b2', lambda a,b: str(a**2-b**2)),
    ('2a_b', lambda a,b: str(2*a+b)),
    ('a_2b', lambda a,b: str(a+2*b)),
    ('3a_b', lambda a,b: str(3*a+b)),
    ('a_3b', lambda a,b: str(a+3*b)),
    ('2ab', lambda a,b: str(2*a*b)),
    ('xor', lambda a,b: str(a^b)),
    ('bitor', lambda a,b: str(a|b)),
    ('bitand', lambda a,b: str(a&b)),
    # Digit-level for 2-digit numbers
    ('d_a1b1_a2b2', lambda a,b: str(a//10*( b//10))+str(a%10*(b%10)) if 10<=a<=99 and 10<=b<=99 else None),
    ('d_cross_prod', lambda a,b: str(a//10*(b%10)+a%10*(b//10)) if 10<=a<=99 and 10<=b<=99 else None),
    ('d_cross_prod2', lambda a,b: str(a//10*(b%10))+str(a%10*(b//10)) if 10<=a<=99 and 10<=b<=99 else None), 
    ('d_rev_a_mul_b', lambda a,b: str(int(str(a)[::-1])*b)),
    ('d_a_mul_rev_b', lambda a,b: str(a*int(str(b)[::-1]))),
    ('d_rev_a_add_b', lambda a,b: str(int(str(a)[::-1])+b)),
    ('d_a_add_rev_b', lambda a,b: str(a+int(str(b)[::-1]))),
    ('d_rev_a_sub_b', lambda a,b: str(int(str(a)[::-1])-b)),
    ('d_rev_concat', lambda a,b: str(a)[::-1]+str(b)[::-1]),
    ('d_interleave', lambda a,b: ''.join(x+y for x,y in zip(str(a),str(b))) if len(str(a))==len(str(b)) else None),
]

# For 1-example problems: which rules fit example AND predict gold?
single_ex_rules = Counter()
single_solved = 0
for r_info, group, q_l, q_o, q_r, gold in numeric_no_rule:
    if len(group) != 1:
        continue
    a, b, res = group[0]
    q_a, q_b = int(q_l), int(q_r)
    
    matching_rules = []
    for rule_name, fn in EXTENDED_RULES:
        try:
            pred_ex = fn(a, b)
            if pred_ex is None or pred_ex != res:
                continue
            # Rule fits example — does it predict gold?
            pred_q = fn(q_a, q_b)
            if pred_q == gold:
                matching_rules.append(rule_name)
        except:
            continue
    
    if matching_rules:
        single_solved += 1
        for rule in matching_rules:
            single_ex_rules[rule] += 1

print(f"Single-example problems: {group_sizes.get(1, 0)}")
print(f"Solvable with extended rules: {single_solved}")
print(f"Rules that work:")
for rule, cnt in single_ex_rules.most_common():
    print(f"  {rule}: {cnt}")

# For 2+ example problems: which rules fit ALL examples AND predict gold?
print()
multi_ex_rules = Counter()
multi_solved = 0
for r_info, group, q_l, q_o, q_r, gold in numeric_no_rule:
    if len(group) < 2:
        continue
    q_a, q_b = int(q_l), int(q_r)
    
    matching_rules = []
    for rule_name, fn in EXTENDED_RULES:
        all_match = True
        for a, b, res in group:
            try:
                pred = fn(a, b)
                if pred is None or pred != res:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                pred_q = fn(q_a, q_b)
                if pred_q == gold:
                    matching_rules.append(rule_name)
            except:
                pass
    
    if matching_rules:
        multi_solved += 1
        for rule in matching_rules:
            multi_ex_rules[rule] += 1

print(f"Multi-example (2+) problems: {sum(c for s,c in group_sizes.items() if s >= 2)}")
print(f"Solvable with extended rules: {multi_solved}")
print(f"Rules that work:")  
for rule, cnt in multi_ex_rules.most_common():
    print(f"  {rule}: {cnt}")
