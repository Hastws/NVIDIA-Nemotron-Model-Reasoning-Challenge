#!/usr/bin/env python3
"""诊断 bit_ops 类型失败：统计失败原因（无匹配函数 vs 答案不匹配 vs 解析失败）。"""
import csv
import json
import re
from collections import defaultdict
from itertools import combinations

def detect_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or 'bit shift' in p: return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p: return 'gravity'
    if 'unit conversion' in p or 'conversion factor' in p: return 'unit_conv'
    if 'cipher' in p or 'encrypt' in p: return 'cipher'
    if 'numeral' in p or ('base' in p and 'convert' in p): return 'numeral'
    if 'symbol' in p or 'equation' in p: return 'symbol'
    return 'unknown'

def parse_bit_ops(prompt):
    lines = prompt.strip().split('\n')
    examples = []
    target = None
    for line in lines:
        line = line.strip()
        if ' -> ' in line:
            parts = line.split(' -> ')
            if len(parts) == 2:
                inp, out = parts[0].strip(), parts[1].strip()
                if len(inp) == 8 and len(out) == 8 and all(c in '01' for c in inp + out):
                    examples.append((inp, out))
        if 'determine' in line.lower() and ':' in line:
            t = line.split(':')[-1].strip()
            if len(t) == 8 and all(c in '01' for c in t):
                target = t
    return examples, target

def enumerate_bit_functions(inputs, outputs, n, obit):
    out_col = [outputs[e][obit] for e in range(n)]
    matches = []

    for i in range(8):
        ic = [inputs[e][i] for e in range(n)]
        if ic == out_col:
            matches.append(('copy', i, f"in[{i}]"))
        if [1 - x for x in ic] == out_col:
            matches.append(('not', i, f"NOT in[{i}]"))

    for j in range(8):
        for k in range(j + 1, 8):
            xor = [inputs[e][j] ^ inputs[e][k] for e in range(n)]
            if xor == out_col: matches.append(('xor2', (j, k), f"XOR"))
            if [1 - x for x in xor] == out_col: matches.append(('xnor2', (j, k), f"XNOR"))

    for j in range(8):
        for k in range(j + 1, 8):
            for op_name, op_fn in [('AND', lambda a, b: a & b), ('OR', lambda a, b: a | b),
                                   ('NAND', lambda a, b: 1 - (a & b)), ('NOR', lambda a, b: 1 - (a | b))]:
                col = [op_fn(inputs[e][j], inputs[e][k]) for e in range(n)]
                if col == out_col: matches.append((op_name.lower(), (j, k), op_name))

    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                x3 = [inputs[e][j] ^ inputs[e][k] ^ inputs[e][l] for e in range(n)]
                if x3 == out_col: matches.append(('xor3', (j, k, l), 'XOR3'))
                if [1 - x for x in x3] == out_col: matches.append(('xnor3', (j, k, l), 'XNOR3'))

    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                maj = [(inputs[e][j] & inputs[e][k]) | (inputs[e][k] & inputs[e][l]) | (inputs[e][j] & inputs[e][l])
                       for e in range(n)]
                if maj == out_col: matches.append(('maj', (j, k, l), 'MAJ'))
                if [1 - x for x in maj] == out_col: matches.append(('nmaj', (j, k, l), 'NOT_MAJ'))

    if all(x == 0 for x in out_col): matches.append(('const', 0, '0'))
    if all(x == 1 for x in out_col): matches.append(('const', 1, '1'))

    for j in range(8):
        for k in range(8):
            for l in range(8):
                if j == k or j == l or k == l: continue
                ch = [(inputs[e][j] & inputs[e][k]) | ((1 - inputs[e][j]) & inputs[e][l]) for e in range(n)]
                if ch == out_col: matches.append(('ch', (j, k, l), 'CH'))

    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                a3 = [inputs[e][j] & inputs[e][k] & inputs[e][l] for e in range(n)]
                if a3 == out_col: matches.append(('and3', (j, k, l), 'AND3'))
                o3 = [inputs[e][j] | inputs[e][k] | inputs[e][l] for e in range(n)]
                if o3 == out_col: matches.append(('or3', (j, k, l), 'OR3'))

    for combo in combinations(range(8), 4):
        x4 = [inputs[e][combo[0]] ^ inputs[e][combo[1]] ^ inputs[e][combo[2]] ^ inputs[e][combo[3]] for e in range(n)]
        if x4 == out_col: matches.append(('xor4', combo, 'XOR4'))
        if [1 - x for x in x4] == out_col: matches.append(('xnor4', combo, 'XNOR4'))

    return matches

def eval_bit_function(func, target_bits):
    fname, args, desc = func
    tb = target_bits
    if fname == 'copy': return tb[args]
    if fname == 'not': return 1 - tb[args]
    if fname == 'xor2': return tb[args[0]] ^ tb[args[1]]
    if fname == 'xnor2': return 1 - (tb[args[0]] ^ tb[args[1]])
    if fname == 'and': return tb[args[0]] & tb[args[1]]
    if fname == 'or': return tb[args[0]] | tb[args[1]]
    if fname == 'nand': return 1 - (tb[args[0]] & tb[args[1]])
    if fname == 'nor': return 1 - (tb[args[0]] | tb[args[1]])
    if fname == 'xor3': return tb[args[0]] ^ tb[args[1]] ^ tb[args[2]]
    if fname == 'xnor3': return 1 - (tb[args[0]] ^ tb[args[1]] ^ tb[args[2]])
    if fname == 'maj':
        a, b, c = [tb[i] for i in args]
        return (a & b) | (b & c) | (a & c)
    if fname == 'nmaj':
        a, b, c = [tb[i] for i in args]
        return 1 - ((a & b) | (b & c) | (a & c))
    if fname == 'const': return args
    if fname == 'ch':
        j, k, l = args
        return (tb[j] & tb[k]) | ((1 - tb[j]) & tb[l])
    if fname == 'and3': return tb[args[0]] & tb[args[1]] & tb[args[2]]
    if fname == 'or3': return tb[args[0]] | tb[args[1]] | tb[args[2]]
    if fname in ('xor4', 'xnor4'):
        v = 0
        for i in args: v ^= tb[i]
        return v if fname == 'xor4' else 1 - v
    return None

def diagnose_bitops(prompt, gold):
    """详细诊断 bit_ops 失败原因。"""
    examples, target = parse_bit_ops(prompt)
    if not examples or not target:
        return 'parse_fail', {'num_examples': len(examples) if examples else 0, 'has_target': target is not None}
    if len(examples) < 4:
        return 'too_few_examples', {'num_examples': len(examples)}
    if len(gold) != 8 or not all(c in '01' for c in gold):
        return 'invalid_gold', {'gold': gold}

    n = len(examples)
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    gold_bits = [int(gold[i]) for i in range(8)]

    no_match_bits = []
    ambig_wrong_bits = []
    result_bits = [None] * 8

    for obit in range(8):
        all_funcs = enumerate_bit_functions(inputs, outputs, n, obit)
        if not all_funcs:
            no_match_bits.append(obit)
            continue

        preds = set()
        for f in all_funcs:
            p = eval_bit_function(f, target_bits)
            if p is not None:
                preds.add(p)

        if len(preds) == 1:
            result_bits[obit] = preds.pop()
        else:
            gb = gold_bits[obit]
            gold_funcs = [f for f in all_funcs if eval_bit_function(f, target_bits) == gb]
            if gold_funcs:
                result_bits[obit] = gb
            else:
                ambig_wrong_bits.append(obit)

    if no_match_bits:
        return 'no_match_function', {'bits': no_match_bits, 'num_examples': n}

    answer = ''.join(str(b) if b is not None else '?' for b in result_bits)
    if answer != gold:
        diff_bits = [i for i in range(8) if result_bits[i] != gold_bits[i]]
        return 'answer_mismatch', {'computed': answer, 'gold': gold, 'diff_bits': diff_bits}

    return 'unknown_should_succeed', {'computed': answer}


def main():
    train = list(csv.DictReader(open('competition_data/train.csv')))
    solved_ids = set()
    with open('data/cot_v2.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            solved_ids.add(rec['id'])

    bitops_all = [(r['id'], r['prompt'], r['answer'].strip()) for r in train if detect_type(r['prompt']) == 'bit_ops']
    bitops_failed = [(id_, prompt, gold) for id_, prompt, gold in bitops_all if id_ not in solved_ids]

    print('=' * 70)
    print(f'Bit_ops Analysis: {len(bitops_all)} total, {len(bitops_all) - len(bitops_failed)} solved, {len(bitops_failed)} failed')
    print('=' * 70)

    # Diagnose each
    reason_counts = defaultdict(int)
    reason_items = defaultdict(list)
    no_match_bit_freq = defaultdict(int)

    for id_, prompt, gold in bitops_failed:
        reason, info = diagnose_bitops(prompt, gold)
        reason_counts[reason] += 1
        reason_items[reason].append((id_, info))

        if reason == 'no_match_function':
            for b in info['bits']:
                no_match_bit_freq[b] += 1

    print('\nFailure reason breakdown:')
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f'  {reason}: {count}')

    if no_match_bit_freq:
        print(f'\n"no_match_function" — which output bit has no matching rule:')
        for bit in range(8):
            if bit in no_match_bit_freq:
                print(f'  bit {bit}: {no_match_bit_freq[bit]} failures')

        # How many bits fail per question?
        bits_per_q = [len(info['bits']) for _, info in reason_items['no_match_function']]
        print(f'\nBits without match per question:')
        from collections import Counter
        for cnt, freq in sorted(Counter(bits_per_q).items()):
            print(f'  {cnt} bits failed: {freq} questions')

        # Example count distribution
        ex_counts = [info['num_examples'] for _, info in reason_items['no_match_function']]
        print(f'\nExample count distribution for no_match questions:')
        for cnt, freq in sorted(Counter(ex_counts).items()):
            print(f'  {cnt} examples: {freq} questions')

    if reason_items.get('answer_mismatch'):
        print(f'\n"answer_mismatch" — computed differs from gold:')
        for i, (id_, info) in enumerate(reason_items['answer_mismatch'][:5]):
            print(f'  [{i+1}] ID={id_}: computed={info["computed"]}, gold={info["gold"]}, diff_bits={info["diff_bits"]}')

    if reason_items.get('parse_fail'):
        print(f'\n"parse_fail" details:')
        for id_, info in reason_items['parse_fail'][:5]:
            print(f'  ID={id_}: examples={info["num_examples"]}, has_target={info["has_target"]}')


if __name__ == '__main__':
    main()
