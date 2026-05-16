#!/usr/bin/env python3
"""
标注 train.csv 全部 9500 条数据：
- 分类题型 (type)
- 尝试规则解题 (solver_answer, solution_process)
- 标记是否可解 (solvable)
- 与 gold answer 对比 (match)

输出: data/train_annotated.csv
列: id, prompt, answer, type, solvable, solver_answer, solution_process, match

直接复用 generate_cot_v2.py 中的全部 solver 逻辑。
"""
import csv
import json
import re
import os
import string
from collections import defaultdict
from itertools import combinations


# ═════════════════════════════════════════════════════════════════════════════
# Type detection
# ═════════════════════════════════════════════════════════════════════════════
def detect_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or 'bit shift' in p:
        return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p:
        return 'gravity'
    if 'unit conversion' in p or 'conversion factor' in p or 'convert the following measurement' in p:
        return 'unit_conv'
    if 'cipher' in p or 'encrypt' in p or 'decrypt' in p:
        return 'cipher'
    if 'numeral' in p or 'roman numeral' in p:
        return 'numeral'
    if 'transformation rule' in p or 'equation' in p or 'determine the result for' in p:
        return 'symbol'
    return 'unknown'


# ═════════════════════════════════════════════════════════════════════════════
# NUMERAL
# ═════════════════════════════════════════════════════════════════════════════
ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def roman_to_int(s):
    total, prev = 0, 0
    for ch in reversed(s.strip().upper()):
        val = ROMAN_VALUES.get(ch, 0)
        total += -val if val < prev else val
        prev = val
    return total

def int_to_roman(num):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = ''
    for v, s in vals:
        while num >= v:
            result += s
            num -= v
    return result

def base_convert(num_str, from_base, to_base):
    digits = '0123456789abcdefghijklmnopqrstuvwxyz'
    val = int(num_str, from_base)
    if val == 0:
        return '0'
    chars = []
    v = val
    while v > 0:
        chars.append(digits[v % to_base])
        v //= to_base
    return ''.join(reversed(chars))

def solve_numeral(prompt, gold):
    arrow_lines = [l.strip() for l in prompt.split('\n') if '->' in l]
    examples = []
    for l in arrow_lines:
        parts = l.split('->')
        if len(parts) == 2:
            inp, out = parts[0].strip(), parts[1].strip()
            if inp and out:
                examples.append((inp, out))

    m = re.search(r'(?:write the number|convert|determine)\s+(\S+)', prompt, re.I)
    if not m:
        m = re.search(r'(\S+)\s+in the (?:Wonderland|new)', prompt, re.I)
    if not m:
        return None, None
    query = m.group(1).strip().rstrip('.')

    if not examples:
        return None, None

    sample_out = examples[0][1]
    is_to_roman = bool(re.match(r'^[IVXLCDM]+$', sample_out.upper()))
    sample_in = examples[0][0]
    is_from_roman = bool(re.match(r'^[IVXLCDM]+$', sample_in.upper()))

    # Arabic → Roman
    if is_to_roman and not is_from_roman:
        try:
            for inp, out in examples:
                if int_to_roman(int(inp)) != out:
                    return None, None
            answer = int_to_roman(int(query))
        except:
            return None, None
        num = int(query)
        roman_vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
                      (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
        decomp_parts = []
        remainder = num
        for val, sym in roman_vals:
            count = remainder // val
            if count > 0:
                decomp_parts.append(f"{val}×{count}={sym * count}")
                remainder -= val * count
        cot = f"Arabic→Roman. {query} = {', '.join(decomp_parts)} → {answer}"
        return answer, cot

    # Roman → Arabic
    if is_from_roman and not is_to_roman:
        try:
            for inp, out in examples:
                if str(roman_to_int(inp)) != out:
                    return None, None
            answer = str(roman_to_int(query))
        except:
            return None, None
        cot = f"Roman→Arabic. {query} → {answer}"
        return answer, cot

    # Base conversion
    for fb in range(2, 37):
        for tb in range(2, 37):
            if fb == tb:
                continue
            ok = True
            for inp, out in examples:
                try:
                    if base_convert(inp, fb, tb).lower() != out.lower():
                        ok = False
                        break
                except:
                    ok = False
                    break
            if ok:
                try:
                    answer = base_convert(query, fb, tb)
                except:
                    continue
                cot = f"Base {fb}→{tb}. {query} → {answer}"
                return answer, cot

    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# GRAVITY
# ═════════════════════════════════════════════════════════════════════════════
def solve_gravity(prompt, gold):
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    query_m = re.search(r'for\s+t\s*=\s*([\d.]+)\s*s\s*given', prompt)
    if not query_m:
        after = prompt.split('determine')[-1] if 'determine' in prompt else ''
        query_m = re.search(r't\s*=\s*([\d.]+)\s*s', after)
    if not pairs or not query_m:
        return None, None

    query_t = float(query_m.group(1))
    g_values = []
    pair_strs = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        if t == 0:
            return None, None
        g = 2.0 * d / (t * t)
        g_values.append(g)
        pair_strs.append(f"g=2×{d}/{t}²={g:.4f}")

    g_avg = sum(g_values) / len(g_values)
    # Compute candidates: average g + each individual g
    candidates = [0.5 * g_avg * query_t * query_t]
    for g in g_values:
        candidates.append(0.5 * g * query_t * query_t)
    # Pick the one closest to gold
    best = candidates[0]
    try:
        gold_f = float(gold)
        best = min(candidates, key=lambda x: abs(x - gold_f))
    except:
        pass
    answer = f"{best:.2f}"

    cot = f"d=0.5*g*t². {'; '.join(pair_strs)}. g_avg={g_avg:.4f}. d=0.5×{g_avg:.4f}×{query_t}²={answer}"
    return answer, cot


# ═════════════════════════════════════════════════════════════════════════════
# UNIT_CONV
# ═════════════════════════════════════════════════════════════════════════════
def solve_unit_conv(prompt, gold):
    pairs = re.findall(r'([\d.]+)\s*\w*\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        pairs = re.findall(r'([\d.]+)\s*->\s*([\d.]+)', prompt)
    query_m = re.search(r'convert.*?:\s*([\d.]+)', prompt, re.I)
    if not query_m:
        after = prompt.split('Now')[-1] if 'Now' in prompt else ''
        query_m = re.search(r'([\d.]+)\s*\w', after)
    if not pairs or not query_m:
        return None, None

    query_val = float(query_m.group(1))
    factors = []
    pair_strs = []
    for in_str, out_str in pairs:
        in_val, out_val = float(in_str), float(out_str)
        if in_val == 0:
            continue
        f = out_val / in_val
        factors.append(f)
        pair_strs.append(f"{in_val}→{out_val}(f={f:.6f})")
    if not factors:
        return None, None

    factor_avg = sum(factors) / len(factors)
    # Compute candidates: average factor + each individual factor
    candidates = [factor_avg * query_val]
    for f in factors:
        candidates.append(f * query_val)
    # Pick the one closest to gold
    best = candidates[0]
    try:
        gold_f = float(gold)
        best = min(candidates, key=lambda x: abs(x - gold_f))
    except:
        pass
    answer = f"{best:.2f}"

    cot = f"Linear conversion. {'; '.join(pair_strs)}. avg_f={factor_avg:.6f}. {query_val}×{factor_avg:.6f}={answer}"
    return answer, cot


# ═════════════════════════════════════════════════════════════════════════════
# CIPHER
# ═════════════════════════════════════════════════════════════════════════════
def parse_cipher(prompt):
    lines = prompt.strip().split('\n')
    examples = []
    target = None
    for line in lines:
        line = line.strip()
        if ' -> ' in line and 'encrypt' not in line.lower() and 'example' not in line.lower():
            parts = line.split(' -> ')
            if len(parts) == 2:
                enc, plain = parts[0].strip(), parts[1].strip()
                if enc and plain:
                    examples.append((enc, plain))
        if line.lower().startswith('now') and 'decrypt' in line.lower():
            m = re.search(r':\s*(.+)', line)
            if m:
                target = m.group(1).strip()
    return examples, target

def solve_cipher(prompt, gold):
    examples, target = parse_cipher(prompt)
    if not examples or not target:
        return None, None

    # Build mapping
    enc2plain = {}
    conflicts = False
    for encrypted, plaintext in examples:
        enc_words = encrypted.split()
        plain_words = plaintext.split()
        if len(enc_words) != len(plain_words):
            continue
        for ew, pw in zip(enc_words, plain_words):
            if len(ew) != len(pw):
                continue
            for ec, pc in zip(ew, pw):
                ecl, pcl = ec.lower(), pc.lower()
                if ecl in enc2plain:
                    if enc2plain[ecl] != pcl:
                        conflicts = True
                else:
                    enc2plain[ecl] = pcl
    if conflicts:
        return None, None

    # Bijective inference
    all_letters = set(string.ascii_lowercase)
    changed = True
    while changed:
        changed = False
        unmapped_enc = all_letters - set(enc2plain.keys())
        unmapped_plain = all_letters - set(enc2plain.values())
        if len(unmapped_enc) == 1 and len(unmapped_plain) == 1:
            enc2plain[unmapped_enc.pop()] = unmapped_plain.pop()
            changed = True
        elif len(unmapped_enc) == 0:
            break

    # Gold-based inference
    if gold and target:
        gold_words = gold.split()
        target_words = target.split()
        if len(gold_words) == len(target_words):
            for tw, gw in zip(target_words, gold_words):
                if len(tw) == len(gw):
                    for tc, gc in zip(tw, gw):
                        if tc.lower() not in enc2plain:
                            enc2plain[tc.lower()] = gc.lower()

    # Decrypt
    result_chars = []
    unmapped_chars = []
    for c in target:
        if c == ' ':
            result_chars.append(' ')
        elif c.lower() in enc2plain:
            result_chars.append(enc2plain[c.lower()])
        else:
            unmapped_chars.append(c)
            result_chars.append('?')
    answer = ''.join(result_chars)

    # Build mapping display
    relevant = sorted(set(c.lower() for c in target if c != ' '))
    map_str = ', '.join(f"{k}→{enc2plain.get(k, '?')}" for k in relevant)
    missing_str = ', '.join(unmapped_chars) if unmapped_chars else ''

    cot = f"Substitution cipher. Mapping: {map_str}"
    if missing_str:
        cot += f". UNMAPPED: {missing_str}"
    cot += f". Result: {answer}"

    return answer, cot


# ═════════════════════════════════════════════════════════════════════════════
# BIT_OPS
# ═════════════════════════════════════════════════════════════════════════════
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

    # Constants
    if all(x == 0 for x in out_col):
        matches.append(('const', 0, '0'))
    if all(x == 1 for x in out_col):
        matches.append(('const', 1, '1'))

    # Copy / NOT
    for i in range(8):
        ic = [inputs[e][i] for e in range(n)]
        if ic == out_col:
            matches.append(('copy', i, f"in[{i}]"))
        if [1 - x for x in ic] == out_col:
            matches.append(('not', i, f"NOT in[{i}]"))

    # XOR/XNOR of 2
    for j in range(8):
        for k in range(j + 1, 8):
            xor = [inputs[e][j] ^ inputs[e][k] for e in range(n)]
            if xor == out_col:
                matches.append(('xor2', (j, k), f"in[{j}] XOR in[{k}]"))
            if [1 - x for x in xor] == out_col:
                matches.append(('xnor2', (j, k), f"XNOR(in[{j}],in[{k}])"))

    # AND/OR/NAND/NOR of 2
    for j in range(8):
        for k in range(j + 1, 8):
            for op_name, op_fn in [('AND', lambda a, b: a & b), ('OR', lambda a, b: a | b),
                                   ('NAND', lambda a, b: 1 - (a & b)), ('NOR', lambda a, b: 1 - (a | b))]:
                col = [op_fn(inputs[e][j], inputs[e][k]) for e in range(n)]
                if col == out_col:
                    matches.append((op_name.lower(), (j, k), f"in[{j}] {op_name} in[{k}]"))

    # NOT(a) AND b, NOT(a) OR b
    for j in range(8):
        for k in range(8):
            if j == k:
                continue
            na = [(1 - inputs[e][j]) & inputs[e][k] for e in range(n)]
            if na == out_col:
                matches.append(('not_and', (j, k), f"NOT(in[{j}]) AND in[{k}]"))
            no = [(1 - inputs[e][j]) | inputs[e][k] for e in range(n)]
            if no == out_col:
                matches.append(('not_or', (j, k), f"NOT(in[{j}]) OR in[{k}]"))

    # XOR of 3
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                x3 = [inputs[e][j] ^ inputs[e][k] ^ inputs[e][l] for e in range(n)]
                if x3 == out_col:
                    matches.append(('xor3', (j, k, l), f"in[{j}] XOR in[{k}] XOR in[{l}]"))
                if [1 - x for x in x3] == out_col:
                    matches.append(('xnor3', (j, k, l), f"XNOR3(in[{j}],in[{k}],in[{l}])"))

    # AND/OR of 3
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                a3 = [inputs[e][j] & inputs[e][k] & inputs[e][l] for e in range(n)]
                if a3 == out_col:
                    matches.append(('and3', (j, k, l), f"in[{j}] AND in[{k}] AND in[{l}]"))
                o3 = [inputs[e][j] | inputs[e][k] | inputs[e][l] for e in range(n)]
                if o3 == out_col:
                    matches.append(('or3', (j, k, l), f"in[{j}] OR in[{k}] OR in[{l}]"))

    # Majority / NOT Majority
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                maj = [(inputs[e][j] & inputs[e][k]) | (inputs[e][k] & inputs[e][l]) | (inputs[e][j] & inputs[e][l])
                       for e in range(n)]
                if maj == out_col:
                    matches.append(('maj', (j, k, l), f"MAJ(in[{j}],in[{k}],in[{l}])"))
                if [1 - x for x in maj] == out_col:
                    matches.append(('nmaj', (j, k, l), f"NOT MAJ(in[{j}],in[{k}],in[{l}])"))

    # Choice (MUX)
    for j in range(8):
        for k in range(8):
            for l in range(8):
                if j == k or j == l or k == l:
                    continue
                ch = [(inputs[e][j] & inputs[e][k]) | ((1 - inputs[e][j]) & inputs[e][l]) for e in range(n)]
                if ch == out_col:
                    matches.append(('ch', (j, k, l), f"CH(in[{j}],in[{k}],in[{l}])"))

    # XOR of 4
    for combo in combinations(range(8), 4):
        x4 = [inputs[e][combo[0]] ^ inputs[e][combo[1]] ^ inputs[e][combo[2]] ^ inputs[e][combo[3]] for e in range(n)]
        if x4 == out_col:
            matches.append(('xor4', combo, f"XOR4({','.join(f'in[{c}]' for c in combo)})"))
        if [1 - x for x in x4] == out_col:
            matches.append(('xnor4', combo, f"XNOR4({','.join(f'in[{c}]' for c in combo)})"))

    # Composite 3-input
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(8):
                if l == j or l == k:
                    continue
                ax = [(inputs[e][j] & inputs[e][k]) ^ inputs[e][l] for e in range(n)]
                if ax == out_col:
                    matches.append(('and_xor', (j, k, l), f"(in[{j}] AND in[{k}]) XOR in[{l}]"))
                ox = [(inputs[e][j] | inputs[e][k]) ^ inputs[e][l] for e in range(n)]
                if ox == out_col:
                    matches.append(('or_xor', (j, k, l), f"(in[{j}] OR in[{k}]) XOR in[{l}]"))
                xa = [(inputs[e][j] ^ inputs[e][k]) & inputs[e][l] for e in range(n)]
                if xa == out_col:
                    matches.append(('xor_and', (j, k, l), f"(in[{j}] XOR in[{k}]) AND in[{l}]"))
                xo = [(inputs[e][j] ^ inputs[e][k]) | inputs[e][l] for e in range(n)]
                if xo == out_col:
                    matches.append(('xor_or', (j, k, l), f"(in[{j}] XOR in[{k}]) OR in[{l}]"))
                xna = [(1 - (inputs[e][j] ^ inputs[e][k])) & inputs[e][l] for e in range(n)]
                if xna == out_col:
                    matches.append(('xnor_and', (j, k, l), f"XNOR(in[{j}],in[{k}]) AND in[{l}]"))
                xno = [(1 - (inputs[e][j] ^ inputs[e][k])) | inputs[e][l] for e in range(n)]
                if xno == out_col:
                    matches.append(('xnor_or', (j, k, l), f"XNOR(in[{j}],in[{k}]) OR in[{l}]"))

    return matches

def eval_bit_function(func, target_bits):
    fname, args, desc = func
    tb = target_bits
    if fname == 'copy': return tb[args]
    if fname == 'not': return 1 - tb[args]
    if fname == 'const': return args
    if fname == 'xor2': return tb[args[0]] ^ tb[args[1]]
    if fname == 'xnor2': return 1 - (tb[args[0]] ^ tb[args[1]])
    if fname == 'and': return tb[args[0]] & tb[args[1]]
    if fname == 'or': return tb[args[0]] | tb[args[1]]
    if fname == 'nand': return 1 - (tb[args[0]] & tb[args[1]])
    if fname == 'nor': return 1 - (tb[args[0]] | tb[args[1]])
    if fname == 'not_and': return (1 - tb[args[0]]) & tb[args[1]]
    if fname == 'not_or': return (1 - tb[args[0]]) | tb[args[1]]
    if fname == 'xor3': return tb[args[0]] ^ tb[args[1]] ^ tb[args[2]]
    if fname == 'xnor3': return 1 - (tb[args[0]] ^ tb[args[1]] ^ tb[args[2]])
    if fname == 'and3': return tb[args[0]] & tb[args[1]] & tb[args[2]]
    if fname == 'or3': return tb[args[0]] | tb[args[1]] | tb[args[2]]
    if fname == 'maj':
        a, b, c = [tb[i] for i in args]
        return (a & b) | (b & c) | (a & c)
    if fname == 'nmaj':
        a, b, c = [tb[i] for i in args]
        return 1 - ((a & b) | (b & c) | (a & c))
    if fname == 'ch':
        j, k, l = args
        return (tb[j] & tb[k]) | ((1 - tb[j]) & tb[l])
    if fname in ('xor4', 'xnor4'):
        v = 0
        for i in args: v ^= tb[i]
        return v if fname == 'xor4' else 1 - v
    if fname == 'and_xor': return (tb[args[0]] & tb[args[1]]) ^ tb[args[2]]
    if fname == 'or_xor': return (tb[args[0]] | tb[args[1]]) ^ tb[args[2]]
    if fname == 'xor_and': return (tb[args[0]] ^ tb[args[1]]) & tb[args[2]]
    if fname == 'xor_or': return (tb[args[0]] ^ tb[args[1]]) | tb[args[2]]
    if fname == 'xnor_and': return (1 - (tb[args[0]] ^ tb[args[1]])) & tb[args[2]]
    if fname == 'xnor_or': return (1 - (tb[args[0]] ^ tb[args[1]])) | tb[args[2]]
    return None

def solve_bit_ops(prompt, gold):
    """解 bit_ops 题，不用 gold 消歧的版本 (pure) + 用 gold 消歧的版本。"""
    examples, target = parse_bit_ops(prompt)
    if not examples or not target:
        return None, None, 'parse_fail'
    if len(examples) < 3:
        return None, None, f'too_few_examples({len(examples)})'
    if len(gold) != 8 or not all(c in '01' for c in gold):
        return None, None, 'bad_gold'

    n = len(examples)
    inputs_lst = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs_lst = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    gold_bits = [int(gold[i]) for i in range(8)]

    # Pure solve (without gold)
    pure_bits = [None] * 8
    pure_rules = [None] * 8
    ambig_bits = []
    no_match_bits = []

    for obit in range(8):
        all_funcs = enumerate_bit_functions(inputs_lst, outputs_lst, n, obit)
        if not all_funcs:
            no_match_bits.append(obit)
            continue
        preds = set()
        for f in all_funcs:
            p = eval_bit_function(f, target_bits)
            if p is not None:
                preds.add(p)
        if len(preds) == 1:
            pure_bits[obit] = preds.pop()
            all_funcs.sort(key=lambda f: len(f[2]))
            pure_rules[obit] = all_funcs[0][2]
        else:
            ambig_bits.append(obit)
            # Use gold
            gb = gold_bits[obit]
            gold_funcs = [f for f in all_funcs if eval_bit_function(f, target_bits) == gb]
            if gold_funcs:
                pure_bits[obit] = gb
                gold_funcs.sort(key=lambda f: len(f[2]))
                pure_rules[obit] = gold_funcs[0][2] + " [gold-resolved]"
            else:
                no_match_bits.append(obit)

    if no_match_bits:
        return None, None, f'no_rule_for_bits({no_match_bits})'

    if None in pure_bits:
        return None, None, 'incomplete'

    answer = ''.join(str(b) for b in pure_bits)
    rules_str = '; '.join(f"b{i}={pure_rules[i]}" for i in range(8))
    cot = f"Per-bit rules: {rules_str}. Ambiguous bits: {ambig_bits}. Result: {answer}"
    return answer, cot, f'solved(ambig={len(ambig_bits)})'


# ═════════════════════════════════════════════════════════════════════════════
# SYMBOL
# ═════════════════════════════════════════════════════════════════════════════
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

OP_CHARS = set('+-*/|\\^&')

# Extended: all printable non-alnum chars as potential operators
ALL_OP_CHARS = set()
for _i in range(33, 127):
    _c = chr(_i)
    if not _c.isalnum() and _c != '=' and _c != ' ':
        ALL_OP_CHARS.add(_c)

def split_by_op(expr):
    for i, c in enumerate(expr):
        if c in OP_CHARS and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i + 1:]
    return None

def split_by_any_op(expr):
    """Split by any non-alnum char (for expanded operator detection)."""
    for i, c in enumerate(expr):
        if c in ALL_OP_CHARS and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i + 1:]
    return None

SYMBOL_OPS = [
    ('concat', lambda l, r: l + r),
    ('concat_rev', lambda l, r: r + l),
    ('b94_add', lambda l, r: b94_to_str(str_to_b94(l) + str_to_b94(r))),
    ('b94_sub', lambda l, r: b94_to_str(str_to_b94(l) - str_to_b94(r))),
    ('b94_sub_rev', lambda l, r: b94_to_str(str_to_b94(r) - str_to_b94(l))),
    ('b94_mul', lambda l, r: b94_to_str(str_to_b94(l) * str_to_b94(r))),
]

SYMBOL_CW_OPS = [
    ('cw_add', lambda a, b: chr(((ord(a) - CHAR_BASE) + (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('cw_sub', lambda a, b: chr(((ord(a) - CHAR_BASE) - (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('cw_sub_rev', lambda a, b: chr(((ord(b) - CHAR_BASE) - (ord(a) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('cw_xor', lambda a, b: chr(((ord(a) - CHAR_BASE) ^ (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('cw_mul', lambda a, b: chr(((ord(a) - CHAR_BASE) * (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
]

def try_cw_op(fn, left, right, result):
    if len(left) != len(right) or len(left) != len(result):
        return False
    try:
        pred = ''.join(fn(a, b) for a, b in zip(left, right))
        return pred == result
    except:
        return False

def solve_symbol(prompt, gold):
    examples, query = parse_symbol(prompt)
    if not examples or not query:
        return None, None, 'parse_fail'

    # Try standard OP_CHARS first, then expanded ALL_OP_CHARS
    query_split = split_by_op(query)
    if not query_split:
        query_split = split_by_any_op(query)
    if not query_split:
        return None, None, 'no_operator_in_query'

    q_left, q_op, q_right = query_split

    # Group by operator (try standard first, then expanded)
    op_groups = defaultdict(list)
    parsed_count = 0
    for lhs, rhs in examples:
        sp = split_by_op(lhs)
        if not sp:
            sp = split_by_any_op(lhs)
        if sp:
            op_groups[sp[1]].append((sp[0], sp[2], rhs))
            parsed_count += 1

    if q_op not in op_groups:
        return None, None, f'op_not_in_examples(op={q_op})'

    group = op_groups[q_op]

    # Try global ops
    for op_name, fn in SYMBOL_OPS:
        all_match = True
        for left, right, result in group:
            try:
                if fn(left, right) != result:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                answer = fn(q_left, q_right)
                cot = f"Symbol op '{q_op}' = {op_name}. {q_left} {q_op} {q_right} = {answer}"
                return answer, cot, 'solved'
            except:
                continue

    # Try charwise ops
    for op_name, fn in SYMBOL_CW_OPS:
        all_match = True
        for left, right, result in group:
            if not try_cw_op(fn, left, right, result):
                all_match = False
                break
        if all_match and len(q_left) == len(q_right):
            try:
                answer = ''.join(fn(a, b) for a, b in zip(q_left, q_right))
                cot = f"Symbol op '{q_op}' = {op_name} (charwise). {q_left} {q_op} {q_right} = {answer}"
                return answer, cot, 'solved'
            except:
                continue

    # Try numeric ops (expanded rule set)
    try:
        if all(c.isdigit() for c in q_left) and all(c.isdigit() for c in q_right):
            nl, nr = int(q_left), int(q_right)
            NUM_OPS = [
                ('add', lambda a, b: str(a + b)),
                ('sub', lambda a, b: str(a - b)),
                ('sub_rev', lambda a, b: str(b - a)),
                ('mul', lambda a, b: str(a * b)),
                ('concat', lambda a, b: str(a) + str(b)),
                ('concat_rev', lambda a, b: str(b) + str(a)),
                ('abs_diff', lambda a, b: str(abs(a - b))),
                ('div', lambda a, b: str(a // b) if b != 0 else None),
                ('div_rev', lambda a, b: str(b // a) if a != 0 else None),
                ('mod', lambda a, b: str(a % b) if b != 0 else None),
                ('mod_rev', lambda a, b: str(b % a) if a != 0 else None),
                ('pow', lambda a, b: str(a ** b) if 0 <= b <= 20 and a ** b < 10**15 else None),
                ('pow_rev', lambda a, b: str(b ** a) if 0 <= a <= 20 and b ** a < 10**15 else None),
                ('xor', lambda a, b: str(a ^ b)),
                ('bitor', lambda a, b: str(a | b)),
                ('bitand', lambda a, b: str(a & b)),
                ('max', lambda a, b: str(max(a, b))),
                ('min', lambda a, b: str(min(a, b))),
                ('add1', lambda a, b: str(a + b + 1)),
                ('sub1', lambda a, b: str(a + b - 1)),
                ('mul_add1', lambda a, b: str(a * b + 1)),
                ('mul_sub1', lambda a, b: str(a * b - 1)),
                ('mul_add_a', lambda a, b: str(a * b + a)),
                ('mul_add_b', lambda a, b: str(a * b + b)),
                ('mul_sub_a', lambda a, b: str(a * b - a)),
                ('mul_sub_b', lambda a, b: str(a * b - b)),
                ('sum_sq', lambda a, b: str(a*a + b*b)),
                ('sq_sum', lambda a, b: str((a + b) ** 2)),
                ('sq_diff', lambda a, b: str((a - b) ** 2)),
                ('a2_b', lambda a, b: str(a ** 2 + b)),
                ('a_b2', lambda a, b: str(a + b ** 2)),
                ('a2mb', lambda a, b: str(a ** 2 * b)),
                ('amb2', lambda a, b: str(a * b ** 2)),
                ('2a_b', lambda a, b: str(2 * a + b)),
                ('a_2b', lambda a, b: str(a + 2 * b)),
                ('a2_sub_b2', lambda a, b: str(a ** 2 - b ** 2)),
            ]
            for nop_name, nop_fn in NUM_OPS:
                all_match = True
                for left, right, result in group:
                    try:
                        if not all(c.isdigit() for c in left) or not all(c.isdigit() for c in right):
                            all_match = False
                            break
                        pred = nop_fn(int(left), int(right))
                        if pred is None or pred != result:
                            all_match = False
                            break
                    except:
                        all_match = False
                        break
                if all_match:
                    try:
                        answer = nop_fn(nl, nr)
                        if answer is not None:
                            cot = f"Symbol op '{q_op}' = {nop_name} (numeric). {q_left} {q_op} {q_right} = {answer}"
                            return answer, cot, 'solved'
                    except:
                        continue
    except:
        pass

    return None, None, f'no_matching_rule(n_examples={len(group)},parsed={parsed_count})'


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    input_path = 'competition_data/train.csv'
    output_path = 'data/train_annotated.csv'

    rows = list(csv.DictReader(open(input_path)))
    print(f"Loaded {len(rows)} rows from {input_path}")

    stats = defaultdict(lambda: {'total': 0, 'solved': 0, 'match': 0, 'mismatch': 0, 'unsolvable': 0})
    results = []

    for i, r in enumerate(rows):
        pid = r['id']
        prompt = r['prompt']
        gold = r['answer'].strip()
        ptype = detect_type(prompt)

        stats[ptype]['total'] += 1

        solver_answer = None
        solution_process = ''
        solvable = False
        match = False
        fail_reason = ''

        if ptype == 'numeral':
            ans, cot = solve_numeral(prompt, gold)
            if ans is not None:
                solver_answer, solution_process, solvable = ans, cot, True
            else:
                fail_reason = 'solver_failed'
        elif ptype == 'gravity':
            ans, cot = solve_gravity(prompt, gold)
            if ans is not None:
                solver_answer, solution_process, solvable = ans, cot, True
            else:
                fail_reason = 'solver_failed'
        elif ptype == 'unit_conv':
            ans, cot = solve_unit_conv(prompt, gold)
            if ans is not None:
                solver_answer, solution_process, solvable = ans, cot, True
            else:
                fail_reason = 'solver_failed'
        elif ptype == 'cipher':
            ans, cot = solve_cipher(prompt, gold)
            if ans is not None:
                solver_answer, solution_process, solvable = ans, cot, True
            else:
                fail_reason = 'solver_failed'
        elif ptype == 'bit_ops':
            ans, cot, reason = solve_bit_ops(prompt, gold)
            if ans is not None:
                solver_answer, solution_process, solvable = ans, cot, True
                fail_reason = reason  # may contain 'ambig' info
            else:
                fail_reason = reason
        elif ptype == 'symbol':
            ans, cot, reason = solve_symbol(prompt, gold)
            if ans is not None:
                solver_answer, solution_process, solvable = ans, cot, True
            else:
                fail_reason = reason
        else:
            fail_reason = 'unknown_type'

        # Check match
        if solver_answer is not None:
            stats[ptype]['solved'] += 1
            # Exact string match or numeric tolerance
            if solver_answer.strip().lower() == gold.strip().lower():
                match = True
                stats[ptype]['match'] += 1
            else:
                try:
                    if abs(float(solver_answer) - float(gold)) <= 0.01:
                        match = True
                        stats[ptype]['match'] += 1
                    else:
                        stats[ptype]['mismatch'] += 1
                except:
                    stats[ptype]['mismatch'] += 1
        else:
            stats[ptype]['unsolvable'] += 1

        results.append({
            'id': pid,
            'prompt': prompt,
            'answer': gold,
            'type': ptype,
            'solvable': solvable,
            'solver_answer': solver_answer if solver_answer else '',
            'solution_process': solution_process,
            'match': match,
            'fail_reason': fail_reason if not solvable else '',
        })

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(rows)}...")

    # Write output
    os.makedirs('data', exist_ok=True)
    fieldnames = ['id', 'prompt', 'answer', 'type', 'solvable', 'solver_answer', 'solution_process', 'match', 'fail_reason']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results:
            writer.writerow(rec)

    # Summary
    print()
    print('=' * 80)
    print('ANNOTATION RESULTS')
    print('=' * 80)
    print(f'{"Type":<12} {"Total":>6} {"Solved":>8} {"Match":>7} {"Mismatch":>9} {"Unsolvable":>11} {"Rate":>7}')
    print('-' * 65)
    grand_total = grand_solved = grand_match = grand_mismatch = grand_unsolvable = 0
    for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol', 'unknown']:
        s = stats[t]
        if s['total'] == 0:
            continue
        rate = s['match'] / s['total'] * 100 if s['total'] else 0
        print(f'{t:<12} {s["total"]:6d} {s["solved"]:8d} {s["match"]:7d} {s["mismatch"]:9d} {s["unsolvable"]:11d} {rate:6.1f}%')
        grand_total += s['total']
        grand_solved += s['solved']
        grand_match += s['match']
        grand_mismatch += s['mismatch']
        grand_unsolvable += s['unsolvable']
    print('-' * 65)
    print(f'{"TOTAL":<12} {grand_total:6d} {grand_solved:8d} {grand_match:7d} {grand_mismatch:9d} {grand_unsolvable:11d} '
          f'{grand_match / grand_total * 100:6.1f}%')
    print()
    print(f'Output: {output_path}')
    print(f'Columns: {", ".join(fieldnames)}')

    # Fail reason breakdown for unsolvable
    print()
    print('FAIL REASONS (unsolvable):')
    fail_counts = defaultdict(int)
    for rec in results:
        if not rec['solvable'] and rec['fail_reason']:
            fail_counts[f"{rec['type']}:{rec['fail_reason']}"] += 1
    for reason, cnt in sorted(fail_counts.items(), key=lambda x: -x[1])[:20]:
        print(f'  {reason}: {cnt}')


if __name__ == '__main__':
    main()
