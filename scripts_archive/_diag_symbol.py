#!/usr/bin/env python3
"""诊断 symbol 类型失败：分类失败原因 — 有运算符/无运算符/数字型。"""
import csv
import json
import re
from collections import defaultdict

def detect_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or 'bit shift' in p: return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p: return 'gravity'
    if 'unit conversion' in p or 'conversion factor' in p: return 'unit_conv'
    if 'cipher' in p or 'encrypt' in p: return 'cipher'
    if 'numeral' in p or ('base' in p and 'convert' in p): return 'numeral'
    if 'symbol' in p or 'equation' in p: return 'symbol'
    return 'unknown'

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
                and 'transformation' not in line.lower() and 'determine' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                if lhs and rhs:
                    examples.append((lhs, rhs))
    return examples, query

def diagnose_symbol(prompt, gold):
    """对 symbol 题进行分类诊断。"""
    examples, query = parse_symbol(prompt)

    if not examples:
        return 'no_examples_parsed', {}
    if not query:
        return 'no_query_found', {'num_examples': len(examples)}

    # Check if examples have operators
    has_op_examples = []
    no_op_examples = []
    for lhs, rhs in examples:
        sp = split_by_op(lhs)
        if sp:
            has_op_examples.append((sp[0], sp[1], sp[2], rhs))
        else:
            no_op_examples.append((lhs, rhs))

    query_split = split_by_op(query)
    has_query_op = query_split is not None

    # Classify
    if not has_op_examples and not has_query_op:
        # Pure transformation (no operators at all)
        # Check if it looks numeric
        all_numeric = all(
            all(c.isdigit() for c in lhs.strip()) and all(c.isdigit() for c in rhs.strip())
            for lhs, rhs in examples
        )
        if all_numeric:
            return 'pure_transform_numeric', {
                'num_examples': len(examples),
                'sample_lhs': examples[0][0] if examples else '',
                'sample_rhs': examples[0][1] if examples else '',
                'query': query,
                'gold': gold,
            }
        else:
            return 'pure_transform_string', {
                'num_examples': len(examples),
                'sample_lhs': examples[0][0] if examples else '',
                'sample_rhs': examples[0][1] if examples else '',
                'query': query,
                'gold': gold,
            }

    if has_op_examples:
        # Has operators
        ops_used = set(sp[1] for sp in has_op_examples)
        all_lhs_numeric = all(
            all(c.isdigit() for c in sp[0]) and all(c.isdigit() for c in sp[2])
            for sp in has_op_examples
        )
        all_rhs_numeric = all(
            all(c.isdigit() for c in sp[3])
            for sp in has_op_examples
        )
        is_numeric = all_lhs_numeric and all_rhs_numeric

        if is_numeric:
            return 'has_op_numeric_failed', {
                'num_examples': len(has_op_examples),
                'ops': list(ops_used),
                'sample': has_op_examples[0],
                'query': query,
                'gold': gold,
            }
        else:
            return 'has_op_string_failed', {
                'num_examples': len(has_op_examples),
                'ops': list(ops_used),
                'sample': has_op_examples[0],
                'query': query,
                'gold': gold,
            }

    # Has query op but no example ops (weird)
    return 'query_op_no_example_op', {
        'num_examples': len(examples),
        'query': query,
        'gold': gold,
    }


def main():
    train = list(csv.DictReader(open('competition_data/train.csv')))
    solved_ids = set()
    with open('data/cot_v2.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            solved_ids.add(rec['id'])

    symbol_all = [(r['id'], r['prompt'], r['answer'].strip()) for r in train if detect_type(r['prompt']) == 'symbol']
    symbol_failed = [(id_, prompt, gold) for id_, prompt, gold in symbol_all if id_ not in solved_ids]

    print('=' * 70)
    print(f'Symbol Analysis: {len(symbol_all)} total, {len(symbol_all) - len(symbol_failed)} solved, {len(symbol_failed)} failed')
    print('=' * 70)

    reason_counts = defaultdict(int)
    reason_items = defaultdict(list)

    for id_, prompt, gold in symbol_failed:
        reason, info = diagnose_symbol(prompt, gold)
        reason_counts[reason] += 1
        reason_items[reason].append((id_, info))

    print('\nFailure category breakdown:')
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f'  {reason}: {count}')

    # Show samples for each category
    for reason in sorted(reason_counts.keys()):
        items = reason_items[reason]
        print(f'\n--- {reason} ({len(items)} items) — samples ---')
        for i, (id_, info) in enumerate(items[:3]):
            print(f'  [{i+1}] ID={id_}')
            for k, v in info.items():
                val_str = str(v)
                if len(val_str) > 100:
                    val_str = val_str[:100] + '...'
                print(f'      {k}: {val_str}')

    # Deep dive: for has_op_string_failed, what operators are most common?
    if reason_items.get('has_op_string_failed'):
        print(f'\n{"="*70}')
        print('has_op_string_failed — operator distribution:')
        op_freq = defaultdict(int)
        for _, info in reason_items['has_op_string_failed']:
            for op in info['ops']:
                op_freq[op] += 1
        for op, freq in sorted(op_freq.items(), key=lambda x: -x[1]):
            print(f'  "{op}": {freq}')

    if reason_items.get('has_op_numeric_failed'):
        print(f'\n{"="*70}')
        print('has_op_numeric_failed — operator distribution:')
        op_freq = defaultdict(int)
        for _, info in reason_items['has_op_numeric_failed']:
            for op in info['ops']:
                op_freq[op] += 1
        for op, freq in sorted(op_freq.items(), key=lambda x: -x[1]):
            print(f'  "{op}": {freq}')

    # Analyze pure transforms
    for cat in ['pure_transform_numeric', 'pure_transform_string']:
        if reason_items.get(cat):
            print(f'\n{"="*70}')
            print(f'{cat} — examples:')
            for i, (id_, info) in enumerate(reason_items[cat][:5]):
                print(f'  [{i+1}] {info.get("sample_lhs","?")} → {info.get("sample_rhs","?")} | query={info.get("query","?")} | gold={info.get("gold","?")}')


if __name__ == '__main__':
    main()
