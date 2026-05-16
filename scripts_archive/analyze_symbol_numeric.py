#!/usr/bin/env python3
"""
专门分析 symbol 中的 unknown numeric 题目，尝试发现更复杂的规则。
同时分析 gravity 精度问题。
"""
import csv
import re
from collections import defaultdict, Counter

OP_CHARS_EXT = set()
for c_ord in range(33, 127):
    c = chr(c_ord)
    if not c.isalnum() and c != '=':
        OP_CHARS_EXT.add(c)

def try_split_any(expr):
    for i, c in enumerate(expr):
        if c in OP_CHARS_EXT and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i+1:]
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

    # ═══════════════════════════════════════════════════════
    # 1. 深度分析 unknown numeric —— 按位数分析
    # ═══════════════════════════════════════════════════════
    print("=" * 80)
    print("1. UNKNOWN NUMERIC: DIGIT-LEVEL PATTERN ANALYSIS")
    print("=" * 80)
    
    digit_patterns = []
    
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
        
        # Collect same-op examples
        op_group = []
        for lhs, rhs in examples:
            for i, c in enumerate(lhs):
                if c == q_op and i > 0 and i < len(lhs) - 1:
                    op_group.append((lhs[:i], lhs[i+1:], rhs))
                    break
        
        if not op_group:
            continue
        
        # Check if all numeric
        try:
            for l, ri, res in op_group:
                int(l); int(ri); int(res)
            int(q_left); int(q_right)
        except:
            continue
        
        digit_patterns.append((q_op, op_group, r['answer'], q_left, q_right, r['id']))
    
    print(f"Total unknown numeric symbol problems: {len(digit_patterns)}")
    
    # Group by number of digits in operands vs result
    len_patterns = Counter()
    for q_op, group, gold, ql, qr, pid in digit_patterns:
        for l, ri, res in group:
            pattern = f"len({len(l)},{len(ri)})→{len(res)}"
            len_patterns[pattern] += 1
    
    print("\nDigit length patterns (operand1, operand2 → result):")
    for pat, cnt in len_patterns.most_common(20):
        print(f"  {pat}: {cnt}")
    
    # ─── Try digit-level operations ───
    print("\n--- DIGIT-LEVEL RULE DISCOVERY ---")
    
    # Rule: digit-by-digit operation (e.g. 34-73: d1=f(3,7), d2=f(4,3))
    DIGIT_OPS = [
        ('d_add', lambda a, b: (a + b) % 10),
        ('d_sub', lambda a, b: (a - b) % 10),
        ('d_sub_rev', lambda a, b: (b - a) % 10),
        ('d_mul', lambda a, b: (a * b) % 10),
        ('d_xor', lambda a, b: a ^ b),
        ('d_add_no_mod', lambda a, b: a + b),
        ('d_mul_no_mod', lambda a, b: a * b),
        ('d_max', lambda a, b: max(a, b)),
        ('d_min', lambda a, b: min(a, b)),
    ]
    
    digit_rule_counts = Counter()
    digit_solved = 0
    
    for q_op, group, gold, ql, qr, pid in digit_patterns:
        # Require same-length operands
        same_len_group = [(l, ri, res) for l, ri, res in group if len(l) == len(ri)]
        if not same_len_group:
            continue
        
        for drn, dfn in DIGIT_OPS:
            all_match = True
            for l, ri, res in same_len_group:
                if len(l) != len(ri):
                    all_match = False
                    break
                try:
                    pred_digits = [dfn(int(l[i]), int(ri[i])) for i in range(len(l))]
                    pred = ''.join(str(d) for d in pred_digits)
                    if pred != res:
                        all_match = False
                        break
                except:
                    all_match = False
                    break
            
            if all_match and len(ql) == len(qr):
                digit_rule_counts[f'{q_op}:{drn}'] += 1
                try:
                    pred_digits = [dfn(int(ql[i]), int(qr[i])) for i in range(len(ql))]
                    pred = ''.join(str(d) for d in pred_digits)
                    if pred == gold:
                        digit_solved += 1
                except:
                    pass
                break
    
    print(f"Digit-level rules found: {sum(digit_rule_counts.values())}")
    print(f"Correctly solved: {digit_solved}")
    for rule, cnt in digit_rule_counts.most_common():
        print(f"  {rule}: {cnt}")

    # ─── Concatenation of digit results ───
    print("\n--- CONCATENATION-OF-DIGIT-PRODUCTS RULE DISCOVERY ---")
    
    CONCAT_DIGIT_OPS = [
        ('dconcat_mul', lambda a, b: str(a * b)),
        ('dconcat_add', lambda a, b: str(a + b)),
        ('dconcat_sub', lambda a, b: str(a - b)),
        ('dconcat_sub_rev', lambda a, b: str(b - a)),
        ('dconcat_pow', lambda a, b: str(a ** b) if b < 10 else None),
    ]
    
    concat_digit_counts = Counter()
    concat_digit_solved = 0
    
    for q_op, group, gold, ql, qr, pid in digit_patterns:
        same_len = [(l, ri, res) for l, ri, res in group if len(l) == len(ri)]
        if not same_len:
            continue
        
        for drn, dfn in CONCAT_DIGIT_OPS:
            all_match = True
            for l, ri, res in same_len:
                try:
                    pred = ''.join(dfn(int(l[i]), int(ri[i])) for i in range(len(l)))
                    if pred != res:
                        all_match = False
                        break
                except:
                    all_match = False
                    break
            
            if all_match and len(ql) == len(qr):
                concat_digit_counts[f'{q_op}:{drn}'] += 1
                try:
                    pred = ''.join(dfn(int(ql[i]), int(qr[i])) for i in range(len(ql)))
                    if pred == gold:
                        concat_digit_solved += 1
                except:
                    pass
                break
    
    print(f"Concat-digit rules found: {sum(concat_digit_counts.values())}")
    print(f"Correctly solved: {concat_digit_solved}")
    for rule, cnt in concat_digit_counts.most_common():
        print(f"  {rule}: {cnt}")

    # ═══════════════════════════════════════════════════════
    # 2. 按每道题的完整 examples 看看能否发现 multi-op 规则
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("2. MULTI-OP CROSS-REFERENCE ANALYSIS")
    print("=" * 80)
    
    # Some problems have multiple different operators in examples
    multi_op_count = 0
    single_op_count = 0
    
    for r in symbol_rows:
        examples, query = parse_symbol(r['prompt'])
        if not examples:
            continue
        
        ops_in_examples = set()
        for lhs, rhs in examples:
            sp = try_split_any(lhs)
            if sp:
                ops_in_examples.add(sp[1])
        
        if len(ops_in_examples) > 1:
            multi_op_count += 1
        else:
            single_op_count += 1
    
    print(f"Problems with single operator type in examples: {single_op_count}")
    print(f"Problems with multiple operator types: {multi_op_count}")

    # ═══════════════════════════════════════════════════════
    # 3. 具体看几道 unknown 题在想什么
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. DETAILED UNKNOWN NUMERIC EXAMPLES")
    print("=" * 80)
    
    shown = 0
    for q_op, group, gold, ql, qr, pid in digit_patterns:
        if shown >= 20:
            break
        # Only show the ones that are really puzzling
        try:
            pairs = [(int(l), int(ri), int(res)) for l, ri, res in group]
        except:
            continue
        
        # Skip if digit-level already solved
        # Just show raw data
        print(f"\nID={pid}, op='{q_op}', query={ql}{q_op}{qr}, gold={gold}")
        for a, b, res in pairs:
            # Show various computations
            diffs = []
            diffs.append(f"a+b={a+b}")
            diffs.append(f"a-b={a-b}")
            diffs.append(f"a*b={a*b}")
            if b != 0:
                diffs.append(f"a//b={a//b}")
            diffs.append(f"a^b={a^b}")
            print(f"  {a} {q_op} {b} = {res}  [{', '.join(diffs)}]")
        shown += 1

    # ═══════════════════════════════════════════════════════
    # 4. Gravity 精度问题
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. GRAVITY PRECISION ANALYSIS")
    print("=" * 80)
    
    grav_mismatches = [r for r in rows if r['type'] == 'gravity' and r['match'] == 'False']
    print(f"Total gravity mismatches: {len(grav_mismatches)}")
    
    diffs = []
    for r in grav_mismatches:
        try:
            diff = float(r['answer']) - float(r['solver_answer'])
            diffs.append(diff)
        except:
            pass
    
    if diffs:
        print(f"Diff stats: min={min(diffs):.4f}, max={max(diffs):.4f}, mean={sum(diffs)/len(diffs):.4f}")
        diff_counts = Counter()
        for d in diffs:
            diff_counts[f"{d:.4f}"] += 1
        print("Diff distribution:")
        for d, cnt in diff_counts.most_common():
            print(f"  {d}: {cnt}")

    # ═══════════════════════════════════════════════════════
    # 5. symbol 全量 coverage summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. SYMBOL COVERAGE BREAKDOWN")
    print("=" * 80)
    
    categories = Counter()
    for r in symbol_rows:
        if r['solvable'] == 'True' and r['match'] == 'True':
            categories['solved_correct'] += 1
        elif r['solvable'] == 'True' and r['match'] == 'False':
            categories['solved_wrong'] += 1
        elif 'no_operator_in_query' in r.get('fail_reason', ''):
            categories['no_std_op'] += 1
        elif 'op_not_in_examples' in r.get('fail_reason', ''):
            categories['op_missing_in_examples'] += 1
        elif 'no_matching_rule' in r.get('fail_reason', ''):
            categories['rule_not_found'] += 1
        elif 'parse_fail' in r.get('fail_reason', ''):
            categories['parse_fail'] += 1
        else:
            categories['other'] += 1
    
    print(f"Total symbol: {len(symbol_rows)}")
    for cat, cnt in categories.most_common():
        print(f"  {cat}: {cnt} ({cnt/len(symbol_rows)*100:.1f}%)")
    
    print(f"\n--- Bottom line ---")
    print(f"Rule-solvable (current): {categories['solved_correct']} ({categories['solved_correct']/len(symbol_rows)*100:.1f}%)")
    print(f"Requires LLM reasoning: {len(symbol_rows) - categories['solved_correct']} ({(len(symbol_rows) - categories['solved_correct'])/len(symbol_rows)*100:.1f}%)")


if __name__ == '__main__':
    main()
