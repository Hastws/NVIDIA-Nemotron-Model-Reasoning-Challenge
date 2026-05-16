#!/usr/bin/env python3
"""
深入分析 symbol 类型题目的结构性 pattern。
目标：发现新的运算规则类别。
"""
import csv
import re
import string
from collections import defaultdict, Counter

CHAR_BASE = 33
CHAR_RANGE = 94
OP_CHARS = set('+-*/|\\^&')

def split_by_op(expr):
    for i, c in enumerate(expr):
        if c in OP_CHARS and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i + 1:]
    return None

def parse_symbol(prompt):
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
    return examples, query

def main():
    rows = list(csv.DictReader(open('data/train_annotated.csv')))
    symbol_rows = [r for r in rows if r['type'] == 'symbol']
    print(f"Symbol total: {len(symbol_rows)}")

    # ═══════════════════════════════════════════════════════
    # 1. Query operator distribution
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("1. QUERY OPERATOR DISTRIBUTION")
    print("=" * 80)
    
    query_op_chars = Counter()
    query_patterns = Counter()
    no_standard_op = []
    
    for r in symbol_rows:
        examples, query = parse_symbol(r['prompt'])
        if not query:
            query_patterns['NO_QUERY'] += 1
            continue
        
        sp = split_by_op(query)
        if sp:
            query_op_chars[sp[1]] += 1
            left, op, right = sp
            left_type = 'num' if all(c.isdigit() for c in left) else 'sym'
            right_type = 'num' if all(c.isdigit() for c in right) else 'sym'
            query_patterns[f'{left_type}{op}{right_type}'] += 1
        else:
            potential_ops = set()
            for i, c in enumerate(query):
                if i > 0 and i < len(query) - 1 and c not in '0123456789':
                    potential_ops.add(c)
            query_patterns['no_std_op'] += 1
            no_standard_op.append((r, query, examples, potential_ops))
    
    print("Query patterns:")
    for p, cnt in query_patterns.most_common():
        print(f"  {p}: {cnt}")
    print("\nStandard op distribution:")
    for op, cnt in query_op_chars.most_common():
        print(f"  '{op}': {cnt}")

    # ═══════════════════════════════════════════════════════
    # 2. Try non-standard characters as operators
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. NON-STANDARD OPERATOR ANALYSIS")
    print("=" * 80)
    
    # Collect ALL unique non-alphanumeric chars in queries
    all_query_chars = Counter()
    for r, query, examples, pot_ops in no_standard_op:
        for c in query:
            if not c.isalnum():
                all_query_chars[c] += 1
    
    print("All non-alnum chars in no_std_op queries:")
    for c, cnt in all_query_chars.most_common(30):
        print(f"  '{c}' (ord={ord(c)}): {cnt}")

    # ═══════════════════════════════════════════════════════
    # 3. For each non-standard char, try as separator
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. NON-STANDARD SEPARATOR DISCOVERY")
    print("=" * 80)
    
    # Expand OP_CHARS to include ALL possible single-char separators
    ALL_POSSIBLE_OPS = set()
    for c_ord in range(33, 127):
        c = chr(c_ord)
        if not c.isalnum():
            ALL_POSSIBLE_OPS.add(c)
    
    sep_success = Counter()
    
    for r in symbol_rows:
        if r['solvable'] == 'True':
            continue
        examples, query = parse_symbol(r['prompt'])
        if not examples or not query:
            continue
        
        for sep in ALL_POSSIBLE_OPS:
            if sep == '=':
                continue
            parts = query.split(sep)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            
            # Check if at least 1 example also splits by this sep
            matched_ex = 0
            for lhs, rhs in examples:
                lhs_parts = lhs.split(sep)
                if len(lhs_parts) == 2 and lhs_parts[0] and lhs_parts[1]:
                    matched_ex += 1
            
            if matched_ex >= 1:
                sep_success[sep] += 1
    
    print("Separators that work (query splits into 2 + >=1 example matches):")
    for sep, cnt in sep_success.most_common(30):
        print(f"  '{sep}' (ord={ord(sep)}): matches {cnt} problems")

    # ═══════════════════════════════════════════════════════
    # 4. Discover numeric rules for ALL operators (including non-standard)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. NUMERIC RULE DISCOVERY (ALL OPERATORS)")
    print("=" * 80)
    
    extended_ops = OP_CHARS | ALL_POSSIBLE_OPS - {'='}
    
    def try_split_any(expr):
        for i, c in enumerate(expr):
            if c in extended_ops and i > 0 and i < len(expr) - 1:
                return expr[:i], c, expr[i+1:]
        return None
    
    numeric_rules = Counter()
    unknown_numeric = []
    solved_by_ext = 0
    
    RULE_FNS = [
        ('a+b', lambda a,b: a+b),
        ('a-b', lambda a,b: a-b),
        ('b-a', lambda a,b: b-a),
        ('a*b', lambda a,b: a*b),
        ('a//b', lambda a,b: a//b if b!=0 else None),
        ('b//a', lambda a,b: b//a if a!=0 else None),
        ('a%b', lambda a,b: a%b if b!=0 else None),
        ('b%a', lambda a,b: b%a if a!=0 else None),
        ('a^b_xor', lambda a,b: a^b),
        ('a&b', lambda a,b: a&b),
        ('a|b', lambda a,b: a|b),
        ('concat_ab', lambda a,b: int(str(abs(a))+str(abs(b)))),
        ('concat_ba', lambda a,b: int(str(abs(b))+str(abs(a)))),
        ('abs(a-b)', lambda a,b: abs(a-b)),
        ('max', lambda a,b: max(a,b)),
        ('min', lambda a,b: min(a,b)),
        ('a**2+b**2', lambda a,b: a*a+b*b),
        ('a*b+a+b', lambda a,b: a*b+a+b),
        ('(a+b)**2', lambda a,b: (a+b)**2),
        ('a*b+a', lambda a,b: a*b+a),
        ('a*b+b', lambda a,b: a*b+b),
        ('a*b-a', lambda a,b: a*b-a),
        ('a*b-b', lambda a,b: a*b-b),
        ('a*(a+b)', lambda a,b: a*(a+b)),
        ('b*(a+b)', lambda a,b: b*(a+b)),
        ('a*(a-b)', lambda a,b: a*(a-b)),
        ('b*(a-b)', lambda a,b: b*(a-b)),
        ('a**2-b', lambda a,b: a*a-b),
        ('a**2+b', lambda a,b: a*a+b),
        ('a-b**2', lambda a,b: a-b*b),
        ('a+b**2', lambda a,b: a+b*b),
        ('a**2*b', lambda a,b: a*a*b),
        ('a*b**2', lambda a,b: a*b*b),
        ('a**2', lambda a,b: a*a),
        ('b**2', lambda a,b: b*b),
        ('a**3', lambda a,b: a**3),
        ('digit_sum', lambda a,b: sum(int(d) for d in str(abs(a))) + sum(int(d) for d in str(abs(b)))),
    ]
    
    for r in symbol_rows:
        if r['solvable'] == 'True':
            continue
        examples, query = parse_symbol(r['prompt'])
        if not examples or not query:
            continue
        
        # Try split with any non-alnum char
        sp = try_split_any(query)
        if not sp:
            continue
        
        q_left, q_op, q_right = sp
        
        # Collect same-op examples
        op_group = []
        for lhs, rhs in examples:
            esp = None
            for i, c in enumerate(lhs):
                if c == q_op and i > 0 and i < len(lhs) - 1:
                    esp = (lhs[:i], c, lhs[i+1:])
                    break
            if esp:
                op_group.append((esp[0], esp[2], rhs))
        
        if not op_group:
            continue
        
        # Check if all numeric
        try:
            num_pairs = []
            for l, ri, res in op_group:
                nl, nri, nres = int(l), int(ri), int(res)
                num_pairs.append((nl, nri, nres))
            nql, nqr = int(q_left), int(q_right)
        except:
            continue
        
        found = False
        for rn, fn in RULE_FNS:
            if all(fn(a,b) is not None and fn(a,b) == res for a,b,res in num_pairs):
                numeric_rules[f'{q_op}:{rn}'] += 1
                pred = fn(nql, nqr)
                if pred is not None and str(pred) == r['answer']:
                    solved_by_ext += 1
                found = True
                break
        
        if not found:
            unknown_numeric.append((q_op, num_pairs, r['answer'], q_left, q_right))
    
    print(f"Discovered rules (would correctly solve {solved_by_ext} more):")
    for rule, cnt in numeric_rules.most_common():
        print(f"  {rule}: {cnt}")
    
    print(f"\nStill unknown numeric: {len(unknown_numeric)}")
    for op, pairs, gold, ql, qr in unknown_numeric[:20]:
        print(f"  op='{op}', query={ql}{op}{qr}, gold={gold}")
        for a, b, res in pairs[:3]:
            print(f"    {a} {op} {b} = {res}")

    # ═══════════════════════════════════════════════════════
    # 5. Non-numeric: charwise ops with extended operators
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. NON-NUMERIC CHARWISE OPS (EXTENDED)")
    print("=" * 80)
    
    cw_rules = Counter()
    cw_unknown = []
    cw_solved = 0
    
    CW_OPS = [
        ('cw_add', lambda a,b: chr(((ord(a)-33)+(ord(b)-33))%94+33)),
        ('cw_sub', lambda a,b: chr(((ord(a)-33)-(ord(b)-33))%94+33)),
        ('cw_sub_rev', lambda a,b: chr(((ord(b)-33)-(ord(a)-33))%94+33)),
        ('cw_xor', lambda a,b: chr(((ord(a)-33)^(ord(b)-33))%94+33)),
        ('cw_mul', lambda a,b: chr(((ord(a)-33)*(ord(b)-33))%94+33)),
        ('cw_and', lambda a,b: chr(((ord(a)-33)&(ord(b)-33))%94+33)),
        ('cw_or', lambda a,b: chr(((ord(a)-33)|(ord(b)-33))%94+33)),
        ('cw_max', lambda a,b: chr(max(ord(a)-33, ord(b)-33)%94+33)),
        ('cw_min', lambda a,b: chr(min(ord(a)-33, ord(b)-33)%94+33)),
        ('cw_avg', lambda a,b: chr(((ord(a)-33+ord(b)-33)//2)%94+33)),
    ]
    
    def try_cw(fn, left, right, result):
        if len(left) != len(right) or len(left) != len(result):
            return False
        try:
            return ''.join(fn(a,b) for a,b in zip(left,right)) == result
        except:
            return False
    
    for r in symbol_rows:
        if r['solvable'] == 'True':
            continue
        examples, query = parse_symbol(r['prompt'])
        if not examples or not query:
            continue
        
        sp = try_split_any(query)
        if not sp:
            continue
        
        q_left, q_op, q_right = sp
        
        # Skip numeric
        if all(c.isdigit() for c in q_left) and all(c.isdigit() for c in q_right):
            continue
        
        # Collect same-op examples
        op_group = []
        for lhs, rhs in examples:
            for i, c in enumerate(lhs):
                if c == q_op and i > 0 and i < len(lhs) - 1:
                    op_group.append((lhs[:i], lhs[i+1:], rhs))
                    break
        
        if not op_group:
            continue
        
        found = False
        for cw_name, cw_fn in CW_OPS:
            if all(try_cw(cw_fn, l, ri, res) for l, ri, res in op_group):
                cw_rules[f'{q_op}:{cw_name}'] += 1
                if len(q_left) == len(q_right):
                    try:
                        pred = ''.join(cw_fn(a,b) for a,b in zip(q_left, q_right))
                        if pred == r['answer']:
                            cw_solved += 1
                    except:
                        pass
                found = True
                break
        
        if not found:
            # Check concat patterns
            concat_match = all(l + ri == res for l, ri, res in op_group)
            concat_rev_match = all(ri + l == res for l, ri, res in op_group)
            if concat_match:
                cw_rules[f'{q_op}:concat'] += 1
                if q_left + q_right == r['answer']:
                    cw_solved += 1
                found = True
            elif concat_rev_match:
                cw_rules[f'{q_op}:concat_rev'] += 1
                if q_right + q_left == r['answer']:
                    cw_solved += 1
                found = True
        
        if not found and len(op_group) >= 1:
            cw_unknown.append((q_op, op_group[:3], r['answer'], q_left, q_right))
    
    print(f"Discovered charwise rules (correctly solves {cw_solved} more):")
    for rule, cnt in cw_rules.most_common():
        print(f"  {rule}: {cnt}")
    
    print(f"\nStill unknown non-numeric: {len(cw_unknown)}")
    for op, group, gold, ql, qr in cw_unknown[:15]:
        print(f"  op='{op}', query={ql} [{op}] {qr}, gold={gold}")
        for l, ri, res in group[:2]:
            print(f"    {l} [{op}] {ri} = {res}")

    # ═══════════════════════════════════════════════════════
    # 6. Coverage summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("6. POTENTIAL COVERAGE IMPROVEMENT")
    print("=" * 80)
    total_unsolved = sum(1 for r in symbol_rows if r['solvable'] != 'True')
    print(f"Currently unsolved: {total_unsolved}")
    print(f"Numeric rules found: {sum(numeric_rules.values())}")
    print(f"Charwise rules found: {sum(cw_rules.values())}")
    print(f"Correctly solvable now: {solved_by_ext + cw_solved}")
    print(f"Potential coverage: {(134 + solved_by_ext + cw_solved) / 1555 * 100:.1f}%")


if __name__ == '__main__':
    main()
