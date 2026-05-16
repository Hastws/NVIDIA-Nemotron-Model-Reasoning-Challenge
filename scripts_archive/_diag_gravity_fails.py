#!/usr/bin/env python3
"""诊断 gravity 类型失败的题目：找出未被 cot_v2.jsonl 覆盖的 gravity 题并分析原因。"""
import csv
import json
import re

def detect_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or 'bit shift' in p: return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p: return 'gravity'
    if 'unit conversion' in p or 'conversion factor' in p: return 'unit_conv'
    if 'cipher' in p or 'encrypt' in p: return 'cipher'
    if 'numeral' in p or ('base' in p and 'convert' in p): return 'numeral'
    if 'symbol' in p or 'equation' in p: return 'symbol'
    return 'unknown'

def attempt_solve_gravity(prompt, gold):
    """尝试解 gravity，返回诊断信息。"""
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)

    query_m = re.search(r'for\s+t\s*=\s*([\d.]+)\s*s\s*given', prompt)
    if not query_m:
        after = prompt.split('determine')[-1] if 'determine' in prompt else ''
        query_m = re.search(r't\s*=\s*([\d.]+)\s*s', after)

    if not pairs:
        return {'reason': 'no_pairs_extracted', 'pairs': [], 'query_t': None, 'g_values': [], 'g_avg': None, 'result': None, 'error': None}
    if not query_m:
        return {'reason': 'no_query_t_found', 'pairs': pairs, 'query_t': None, 'g_values': [], 'g_avg': None, 'result': None, 'error': None}

    query_t = float(query_m.group(1))

    g_values = []
    pair_data = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        if t == 0:
            return {'reason': 't_is_zero', 'pairs': pairs, 'query_t': query_t, 'g_values': [], 'g_avg': None, 'result': None, 'error': None}
        g = 2.0 * d / (t * t)
        g_values.append(g)
        pair_data.append((t, d, g))

    g_avg = sum(g_values) / len(g_values)
    result = 0.5 * g_avg * query_t * query_t

    try:
        gold_val = float(gold)
    except:
        return {'reason': 'gold_not_float', 'pairs': pairs, 'query_t': query_t, 'g_values': g_values, 'g_avg': g_avg, 'result': result, 'error': None}

    result_display = f"{result:.2f}"
    error = abs(float(result_display) - gold_val)

    if error > 0.02:
        return {'reason': f'error_too_large ({error:.6f})', 'pairs': pairs, 'query_t': query_t, 'g_values': g_values, 'g_avg': g_avg, 'result': result, 'error': error}

    return {'reason': 'should_have_succeeded(?)', 'pairs': pairs, 'query_t': query_t, 'g_values': g_values, 'g_avg': g_avg, 'result': result, 'error': error}


def main():
    # Load train.csv
    train = list(csv.DictReader(open('competition_data/train.csv')))

    # Load cot_v2.jsonl solved IDs
    solved_ids = set()
    with open('data/cot_v2.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            solved_ids.add(rec['id'])

    # Find gravity questions
    gravity_all = [(r['id'], r['prompt'], r['answer'].strip()) for r in train if detect_type(r['prompt']) == 'gravity']
    gravity_failed = [(id_, prompt, gold) for id_, prompt, gold in gravity_all if id_ not in solved_ids]

    print('=' * 70)
    print(f'Gravity Analysis: {len(gravity_all)} total, {len(gravity_all) - len(gravity_failed)} solved, {len(gravity_failed)} failed')
    print('=' * 70)

    # Group by failure reason
    reasons = {}
    for id_, prompt, gold in gravity_failed:
        diag = attempt_solve_gravity(prompt, gold)
        reason = diag['reason']
        if reason not in reasons:
            reasons[reason] = []
        reasons[reason].append((id_, prompt, gold, diag))

    print(f'\nFailure reasons:')
    for reason, items in sorted(reasons.items(), key=lambda x: -len(x[1])):
        print(f'  {reason}: {len(items)} items')

    print(f'\n{"="*70}')
    print('Detailed per-item diagnostics:')
    print('='*70)

    for i, (id_, prompt, gold) in enumerate(gravity_failed):
        diag = attempt_solve_gravity(prompt, gold)
        print(f'\n--- [{i+1}/{len(gravity_failed)}] ID: {id_} ---')
        print(f'Prompt (first 200): {prompt[:200]}...')
        print(f'Gold answer: {gold}')
        print(f'Failure reason: {diag["reason"]}')
        print(f'  Pairs extracted: {len(diag["pairs"])} → {diag["pairs"]}')
        print(f'  query_t: {diag["query_t"]}')
        if diag['g_values']:
            print(f'  g values: {[f"{g:.6f}" for g in diag["g_values"]]}')
            print(f'  g_avg: {diag["g_avg"]:.6f}')
        if diag['result'] is not None:
            print(f'  Computed result: {diag["result"]:.6f}')
            print(f'  Result (2dp): {diag["result"]:.2f}')
        if diag['error'] is not None:
            print(f'  Error vs gold: {diag["error"]:.6f}')

    # Summary: error distribution for "error_too_large"
    error_items = [(id_, prompt, gold, diag) for id_, prompt, gold, diag in
                   [(id_, p, g, attempt_solve_gravity(p, g)) for id_, p, g in gravity_failed]
                   if diag['error'] is not None and diag['error'] > 0.02]
    if error_items:
        print(f'\n{"="*70}')
        print(f'Error distribution for {len(error_items)} items with error > 0.02:')
        errors = [d['error'] for _, _, _, d in error_items]
        print(f'  Min error:  {min(errors):.6f}')
        print(f'  Max error:  {max(errors):.6f}')
        print(f'  Mean error: {sum(errors)/len(errors):.6f}')
        # How many could be rescued with a looser threshold?
        for threshold in [0.05, 0.1, 0.5, 1.0, 5.0]:
            count = sum(1 for e in errors if e <= threshold)
            print(f'  Error ≤ {threshold:.2f}: {count}/{len(errors)}')

if __name__ == '__main__':
    main()
