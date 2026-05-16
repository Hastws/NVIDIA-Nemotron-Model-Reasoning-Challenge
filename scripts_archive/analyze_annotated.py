#!/usr/bin/env python3
"""
深入分析 train_annotated.csv 的失败和不匹配样本。
"""
import csv
import re
from collections import defaultdict, Counter

def main():
    rows = list(csv.DictReader(open('data/train_annotated.csv')))
    print(f"Total rows: {len(rows)}")

    # ─── 1. Mismatch 分析 (solver 给了答案但不匹配 gold) ───
    print("\n" + "=" * 80)
    print("MISMATCH ANALYSIS (solver_answer != gold)")
    print("=" * 80)
    mismatches = [r for r in rows if r['solvable'] == 'True' and r['match'] == 'False']
    print(f"Total mismatches: {len(mismatches)}")
    
    by_type = defaultdict(list)
    for r in mismatches:
        by_type[r['type']].append(r)
    
    for t in sorted(by_type):
        items = by_type[t]
        print(f"\n--- {t}: {len(items)} mismatches ---")
        for r in items[:5]:
            print(f"  ID={r['id']}")
            print(f"    gold={r['answer'][:80]}")
            print(f"    solver={r['solver_answer'][:80]}")
            # 数值差
            try:
                diff = abs(float(r['solver_answer']) - float(r['answer']))
                print(f"    numeric_diff={diff:.6f}")
            except:
                print(f"    (not numeric)")
            print(f"    process={r['solution_process'][:120]}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")

    # ─── 2. Symbol 失败原因细分 ───
    print("\n" + "=" * 80)
    print("SYMBOL FAILURE ANALYSIS")
    print("=" * 80)
    symbol_unsolved = [r for r in rows if r['type'] == 'symbol' and r['solvable'] != 'True']
    print(f"Symbol unsolved: {len(symbol_unsolved)}")
    
    reason_counts = Counter()
    for r in symbol_unsolved:
        reason = r['fail_reason']
        # Normalize
        if 'no_matching_rule' in reason:
            reason_counts['no_matching_rule'] += 1
        elif 'no_operator_in_query' in reason:
            reason_counts['no_operator_in_query'] += 1
        elif 'op_not_in_examples' in reason:
            m = re.search(r'op=(.)', reason)
            reason_counts[f'op_not_in_examples(op={m.group(1) if m else "?"})'] += 1
        elif 'parse_fail' in reason:
            reason_counts['parse_fail'] += 1
        else:
            reason_counts[reason] += 1
    
    for reason, cnt in reason_counts.most_common():
        print(f"  {reason}: {cnt}")

    # ─── 3. no_operator_in_query：看看 query 长什么样 ───
    print("\n" + "=" * 80)
    print("SYMBOL: no_operator_in_query 样本 (前 10)")
    print("=" * 80)
    no_op = [r for r in symbol_unsolved if 'no_operator_in_query' in r['fail_reason']]
    for r in no_op[:10]:
        # 提取 query
        prompt = r['prompt']
        lines = prompt.strip().split('\n')
        query_line = ''
        for line in lines:
            if 'determine' in line.lower() or 'result for' in line.lower():
                query_line = line.strip()
                break
        if not query_line:
            query_line = lines[-1].strip() if lines else '?'
        
        # 提取 examples
        example_lines = [l.strip() for l in lines if '=' in l and 'determine' not in l.lower()
                        and 'transformation' not in l.lower() and 'equation' not in l.lower()
                        and 'alice' not in l.lower() and 'below' not in l.lower()]
        
        print(f"  ID={r['id']}, gold={r['answer']}")
        print(f"    query: {query_line[:120]}")
        if example_lines:
            print(f"    first example: {example_lines[0][:120]}")
            if len(example_lines) > 1:
                print(f"    second example: {example_lines[1][:120]}")
        print()

    # ─── 4. op_not_in_examples：看什么情况 ───
    print("\n" + "=" * 80)
    print("SYMBOL: op_not_in_examples 样本 (前 5)")
    print("=" * 80)
    op_missing = [r for r in symbol_unsolved if 'op_not_in_examples' in r['fail_reason']]
    for r in op_missing[:5]:
        prompt = r['prompt']
        lines = prompt.strip().split('\n')
        eq_lines = [l.strip() for l in lines if '=' in l and 'determine' not in l.lower()
                    and 'transformation' not in l.lower() and 'equation' not in l.lower()
                    and 'alice' not in l.lower() and 'below' not in l.lower()]
        query_line = ''
        for line in lines:
            if 'determine' in line.lower():
                query_line = line.split(':')[-1].strip() if ':' in line else line.strip()
                break
        
        print(f"  ID={r['id']}, gold={r['answer']}, reason={r['fail_reason']}")
        print(f"    query: {query_line[:120]}")
        for el in eq_lines[:3]:
            print(f"    example: {el[:120]}")
        print()

    # ─── 5. no_matching_rule：看看 examples 和 query 的结构 ───
    print("\n" + "=" * 80)
    print("SYMBOL: no_matching_rule 样本 (前 10)")
    print("=" * 80)
    no_rule = [r for r in symbol_unsolved if 'no_matching_rule' in r['fail_reason']]
    for r in no_rule[:10]:
        prompt = r['prompt']
        lines = prompt.strip().split('\n')
        eq_lines = [l.strip() for l in lines if '=' in l and 'determine' not in l.lower()
                    and 'transformation' not in l.lower() and 'equation' not in l.lower()
                    and 'alice' not in l.lower() and 'below' not in l.lower()]
        query_line = ''
        for line in lines:
            if 'determine' in line.lower():
                query_line = line.split(':')[-1].strip() if ':' in line else line.strip()
                break
        
        print(f"  ID={r['id']}, gold={r['answer']}, reason={r['fail_reason']}")
        print(f"    query: {query_line[:80]}")
        for el in eq_lines[:3]:
            print(f"    eq: {el[:120]}")
        print()

    # ─── 6. parse_fail 样本 ───
    print("\n" + "=" * 80)
    print("SYMBOL: parse_fail 样本")
    print("=" * 80)
    pf = [r for r in symbol_unsolved if 'parse_fail' in r['fail_reason']]
    for r in pf[:5]:
        lines = r['prompt'].strip().split('\n')
        print(f"  ID={r['id']}, gold={r['answer']}")
        for l in lines[:8]:
            print(f"    {l[:120]}")
        print()

    # ─── 7. Gravity mismatch 分析 ───
    print("\n" + "=" * 80)
    print("GRAVITY MISMATCH ANALYSIS")
    print("=" * 80)
    grav_mm = [r for r in rows if r['type'] == 'gravity' and r['match'] == 'False']
    print(f"Total gravity mismatches: {len(grav_mm)}")
    for r in grav_mm[:5]:
        try:
            diff = abs(float(r['solver_answer']) - float(r['answer']))
        except:
            diff = -1
        print(f"  ID={r['id']}, gold={r['answer']}, solver={r['solver_answer']}, diff={diff:.4f}")
        print(f"    process: {r['solution_process'][:150]}")

    # ─── 8. Bit_ops 分析 ───
    print("\n" + "=" * 80)
    print("BIT_OPS FAILURE ANALYSIS")
    print("=" * 80)
    bo_fail = [r for r in rows if r['type'] == 'bit_ops' and (r['match'] == 'False' or r['solvable'] != 'True')]
    print(f"Total bit_ops failures: {len(bo_fail)}")
    for r in bo_fail[:10]:
        print(f"  ID={r['id']}, solvable={r['solvable']}, gold={r['answer']}, solver={r['solver_answer'][:40]}")
        print(f"    reason={r['fail_reason']}")

    # ─── 9. Symbol solved but mismatch ───
    print("\n" + "=" * 80)
    print("SYMBOL: solved but mismatch (前 10)")
    print("=" * 80)
    sym_mm = [r for r in rows if r['type'] == 'symbol' and r['solvable'] == 'True' and r['match'] == 'False']
    print(f"Total: {len(sym_mm)}")
    for r in sym_mm[:10]:
        print(f"  ID={r['id']}, gold={r['answer'][:60]}, solver={r['solver_answer'][:60]}")
        print(f"    process: {r['solution_process'][:120]}")

    # ─── 10. Overall per-type summary with sub-stats ───
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
        type_rows = [r for r in rows if r['type'] == t]
        total = len(type_rows)
        solved = sum(1 for r in type_rows if r['solvable'] == 'True')
        matched = sum(1 for r in type_rows if r['match'] == 'True')
        
        # Gold answer length stats
        gold_lens = [len(r['answer']) for r in type_rows]
        avg_gold_len = sum(gold_lens) / len(gold_lens) if gold_lens else 0
        max_gold_len = max(gold_lens) if gold_lens else 0
        
        print(f"{t}: {total} total, {solved} solved ({solved/total*100:.1f}%), {matched} matched ({matched/total*100:.1f}%)")
        print(f"  gold answer len: avg={avg_gold_len:.1f}, max={max_gold_len}")

if __name__ == '__main__':
    main()
