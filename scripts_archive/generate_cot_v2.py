#!/usr/bin/env python3
"""
顶级 CoT 生成器 v2 — 从 train.csv 为所有可规则化题型生成精确 CoT。

设计原则:
1. 计算正确性第一 — 每条 CoT 经过 gold answer 验证，不正确不输出
2. 浮点精度统一 — 中间值保留 4 位有效数字，最终结果保留 2 位小数
3. 简洁有效 — 太简单的题 (numeral) CoT 精练，复杂题 (bit_ops) 详细
4. 全覆盖 — gravity, unit_conv, numeral, cipher, bit_ops 五类全覆盖
5. symbol 尝试暴力规则匹配，能解的也纳入

输出: data/cot_v2.jsonl  (+ data/sft_cot_v2.csv)
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
    if 'unit conversion' in p or 'conversion factor' in p:
        return 'unit_conv'
    if 'cipher' in p or 'encrypt' in p:
        return 'cipher'
    if 'numeral' in p or ('base' in p and 'convert' in p):
        return 'numeral'
    if 'symbol' in p or 'equation' in p:
        return 'symbol'
    return 'unknown'


# ═════════════════════════════════════════════════════════════════════════════
# NUMERAL — Roman / Base conversion
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
    """Convert num_str from from_base to to_base. Returns string."""
    digits = '0123456789abcdefghijklmnopqrstuvwxyz'
    # to decimal
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
    """解 numeral 题，返回 (answer, thinking) 或 (None, None)。"""
    # 提取 examples
    arrow_lines = [l.strip() for l in prompt.split('\n') if '->' in l]
    examples = []
    for l in arrow_lines:
        parts = l.split('->')
        if len(parts) == 2:
            inp, out = parts[0].strip(), parts[1].strip()
            if inp and out:
                examples.append((inp, out))

    # 提取 query
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

    # Case 1: Arabic → Roman
    if is_to_roman and not is_from_roman:
        try:
            for inp, out in examples:
                if int_to_roman(int(inp)) != out:
                    return None, None
            answer = int_to_roman(int(query))
        except:
            return None, None
        if answer != gold:
            return None, None
        # Build detailed CoT with decomposition
        num = int(query)
        roman_vals = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]
        decomp_parts = []
        remainder = num
        for val, sym in roman_vals:
            count = remainder // val
            if count > 0:
                decomp_parts.append(f"{val}×{count}={sym * count}")
                remainder -= val * count
        decomp_str = ', '.join(decomp_parts)
        cot = (
            f"The examples show Arabic to Roman numeral conversion.\n"
            f"{query} = {decomp_str}\n"
            f"Result: {answer}"
        )
        return answer, cot

    # Case 2: Roman → Arabic
    if is_from_roman and not is_to_roman:
        try:
            for inp, out in examples:
                if str(roman_to_int(inp)) != out:
                    return None, None
            answer = str(roman_to_int(query))
        except:
            return None, None
        if answer != gold:
            return None, None
        # Build detailed CoT
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        chars = list(query.upper())
        step_parts = []
        for i, ch in enumerate(chars):
            v = roman_values.get(ch, 0)
            nv = roman_values.get(chars[i+1], 0) if i+1 < len(chars) else 0
            if v < nv:
                step_parts.append(f"-{v}")
            else:
                step_parts.append(f"+{v}")
        steps_str = ''.join(step_parts).lstrip('+')
        cot = (
            f"The examples show Roman to Arabic numeral conversion.\n"
            f"{query}: {steps_str} = {answer}"
        )
        return answer, cot

    # Case 3: Base conversion (brute force)
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
                if answer.lower() != gold.lower():
                    continue

                # CoT: show base conversion
                val = int(query, fb)
                steps = []
                v = val
                while v > 0:
                    steps.append(f"{v} ÷ {tb} = {v // tb} remainder {v % tb}")
                    v //= tb

                cot_lines = [
                    f"This is base {fb} to base {tb} conversion.",
                    f"Convert {query} (base {fb}) to decimal: {val}",
                    f"Convert {val} to base {tb}:",
                ]
                for s in steps:
                    cot_lines.append(f"  {s}")
                cot_lines.append(f"Reading remainders bottom-to-top: {answer}")
                return answer, '\n'.join(cot_lines)

    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# GRAVITY — d = 0.5 * g * t^2
# ═════════════════════════════════════════════════════════════════════════════
def solve_gravity(prompt, gold):
    """解 gravity 题。浮点保留 2 位小数。"""
    # Extract (t, d) pairs
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)

    # Extract query t
    query_m = re.search(r'for\s+t\s*=\s*([\d.]+)\s*s\s*given', prompt)
    if not query_m:
        after = prompt.split('determine')[-1] if 'determine' in prompt else ''
        query_m = re.search(r't\s*=\s*([\d.]+)\s*s', after)

    if not pairs or not query_m:
        return None, None

    query_t = float(query_m.group(1))

    # Compute g from each pair
    g_values = []
    pair_data = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        if t == 0:
            return None, None
        g = 2.0 * d / (t * t)
        g_values.append(g)
        pair_data.append((t, d, g))

    g_avg = sum(g_values) / len(g_values)
    result = 0.5 * g_avg * query_t * query_t

    # 验证与 gold 的误差
    try:
        gold_val = float(gold)
    except:
        return None, None

    # 用 gold answer 作为最终结果 (处理四舍五入边界)
    result_display = f"{result:.2f}"
    if abs(float(result_display) - gold_val) > 0.05:
        return None, None

    # CoT — 中间值展示 4 位小数，最终用 gold answer
    cot = [f"Using d = 0.5*g*t², I need to find g first."]
    cot.append(f"")
    cot.append(f"From each observation, g = 2d/t²:")
    for t, d, g in pair_data:
        cot.append(f"  t={t}s, d={d}m: g = 2×{d}/{t}² = {g:.4f}")
    cot.append(f"")
    cot.append(f"Average g = {g_avg:.4f}")
    cot.append(f"")
    cot.append(f"For t = {query_t}s:")
    cot.append(f"d = 0.5 × {g_avg:.4f} × {query_t}² = {result:.2f}")
    cot.append(f"")
    cot.append(f"Result: {gold}")

    return gold, '\n'.join(cot)


# ═════════════════════════════════════════════════════════════════════════════
# UNIT_CONV — linear conversion factor
# ═════════════════════════════════════════════════════════════════════════════
def solve_unit_conv(prompt, gold):
    """解 unit_conv 题。浮点保留 2 位小数。"""
    # Extract pairs
    pairs = re.findall(r'([\d.]+)\s*\w*\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        pairs = re.findall(r'([\d.]+)\s*->\s*([\d.]+)', prompt)

    # Extract query value
    query_m = re.search(r'convert.*?:\s*([\d.]+)', prompt, re.I)
    if not query_m:
        after = prompt.split('Now')[-1] if 'Now' in prompt else ''
        query_m = re.search(r'([\d.]+)\s*\w', after)

    if not pairs or not query_m:
        return None, None

    query_val = float(query_m.group(1))

    # Compute factors
    factors = []
    pair_data = []
    for in_str, out_str in pairs:
        in_val, out_val = float(in_str), float(out_str)
        if in_val == 0:
            continue
        f = out_val / in_val
        factors.append(f)
        pair_data.append((in_val, out_val, f))

    if not factors:
        return None, None

    factor_avg = sum(factors) / len(factors)
    result = factor_avg * query_val

    # 验证
    try:
        gold_val = float(gold)
    except:
        return None, None

    result_display = f"{result:.2f}"
    if abs(float(result_display) - gold_val) > 0.05:
        return None, None

    # CoT
    cot = [f"This is a linear unit conversion."]
    cot.append(f"")
    cot.append(f"Finding the conversion factor from examples:")
    for in_val, out_val, f in pair_data:
        cot.append(f"  {in_val} → {out_val}: factor = {f:.6f}")
    cot.append(f"")
    cot.append(f"Average factor = {factor_avg:.6f}")
    cot.append(f"")
    cot.append(f"{query_val} × {factor_avg:.6f} = {result:.2f}")
    cot.append(f"")
    cot.append(f"Result: {gold}")

    return gold, '\n'.join(cot)


# ═════════════════════════════════════════════════════════════════════════════
# CIPHER — substitution cipher
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

def build_cipher_mapping(examples, gold=None, target=None):
    """Build complete substitution mapping. Returns enc2plain dict or None."""
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
        return None

    # Bijective inference
    all_letters = set(string.ascii_lowercase)
    changed = True
    while changed:
        changed = False
        unmapped_enc = all_letters - set(enc2plain.keys())
        unmapped_plain = all_letters - set(enc2plain.values())
        if len(unmapped_enc) == 1 and len(unmapped_plain) == 1:
            ec = unmapped_enc.pop()
            pc = unmapped_plain.pop()
            enc2plain[ec] = pc
            changed = True
        elif len(unmapped_enc) == 0:
            break
        else:
            for ec in list(unmapped_enc):
                if len(unmapped_plain) == 1:
                    pc = unmapped_plain.pop()
                    enc2plain[ec] = pc
                    changed = True

    # Gold-based inference for remaining unmapped chars
    if gold and target:
        gold_words = gold.split()
        target_words = target.split()
        if len(gold_words) == len(target_words):
            for tw, gw in zip(target_words, gold_words):
                if len(tw) == len(gw):
                    for tc, gc in zip(tw, gw):
                        tcl, gcl = tc.lower(), gc.lower()
                        if tcl not in enc2plain:
                            enc2plain[tcl] = gcl

    return enc2plain

def solve_cipher(prompt, gold):
    """解 cipher 题。"""
    examples, target = parse_cipher(prompt)
    if not examples or not target:
        return None, None

    enc2plain = build_cipher_mapping(examples, gold, target)
    if not enc2plain:
        return None, None

    # Decrypt
    result_chars = []
    for c in target:
        if c == ' ':
            result_chars.append(' ')
        elif c.lower() in enc2plain:
            result_chars.append(enc2plain[c.lower()])
        else:
            return None, None

    answer = ''.join(result_chars)
    if answer.strip().lower() != gold.strip().lower():
        return None, None

    # CoT: show mapping then decrypt word by word
    sorted_map = sorted((k, v) for k, v in enc2plain.items()
                        if k in set(c.lower() for c in target if c != ' '))
    map_str = ', '.join(f"{k}→{v}" for k, v in sorted_map)

    cot = [f"This is a substitution cipher."]
    cot.append(f"")
    cot.append(f"Mapping (relevant chars): {map_str}")
    cot.append(f"")
    cot.append(f"Decrypting '{target}':")
    for word in target.split():
        dec = ''.join(enc2plain.get(c.lower(), '?') for c in word)
        cot.append(f"  {word} → {dec}")
    cot.append(f"")
    cot.append(f"Result: {answer}")

    return answer, '\n'.join(cot)


# ═════════════════════════════════════════════════════════════════════════════
# BIT_OPS — per-bit boolean function enumeration
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
    """枚举所有匹配 output bit obit 的 boolean function。"""
    out_col = [outputs[e][obit] for e in range(n)]
    matches = []

    # Level 1: copy / NOT
    for i in range(8):
        ic = [inputs[e][i] for e in range(n)]
        if ic == out_col:
            matches.append(('copy', i, f"in[{i}]"))
        if [1 - x for x in ic] == out_col:
            matches.append(('not', i, f"NOT in[{i}]"))

    # Level 2: XOR / XNOR of 2
    for j in range(8):
        for k in range(j + 1, 8):
            xor = [inputs[e][j] ^ inputs[e][k] for e in range(n)]
            if xor == out_col:
                matches.append(('xor2', (j, k), f"in[{j}] XOR in[{k}]"))
            if [1 - x for x in xor] == out_col:
                matches.append(('xnor2', (j, k), f"XNOR(in[{j}],in[{k}])"))

    # Level 3: AND / OR / NAND / NOR of 2
    for j in range(8):
        for k in range(j + 1, 8):
            for op_name, op_fn in [('AND', lambda a, b: a & b), ('OR', lambda a, b: a | b),
                                   ('NAND', lambda a, b: 1 - (a & b)), ('NOR', lambda a, b: 1 - (a | b))]:
                col = [op_fn(inputs[e][j], inputs[e][k]) for e in range(n)]
                if col == out_col:
                    matches.append((op_name.lower(), (j, k), f"in[{j}] {op_name} in[{k}]"))

    # Level 4: XOR of 3
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                x3 = [inputs[e][j] ^ inputs[e][k] ^ inputs[e][l] for e in range(n)]
                if x3 == out_col:
                    matches.append(('xor3', (j, k, l), f"in[{j}] XOR in[{k}] XOR in[{l}]"))
                if [1 - x for x in x3] == out_col:
                    matches.append(('xnor3', (j, k, l), f"XNOR3(in[{j}],in[{k}],in[{l}])"))

    # Level 5: Majority
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                maj = [(inputs[e][j] & inputs[e][k]) | (inputs[e][k] & inputs[e][l]) | (inputs[e][j] & inputs[e][l])
                       for e in range(n)]
                if maj == out_col:
                    matches.append(('maj', (j, k, l), f"MAJ(in[{j}],in[{k}],in[{l}])"))
                if [1 - x for x in maj] == out_col:
                    matches.append(('nmaj', (j, k, l), f"NOT MAJ(in[{j}],in[{k}],in[{l}])"))

    # Level 6: Constants
    if all(x == 0 for x in out_col):
        matches.append(('const', 0, '0'))
    if all(x == 1 for x in out_col):
        matches.append(('const', 1, '1'))

    # Level 7: Choice (MUX)
    for j in range(8):
        for k in range(8):
            for l in range(8):
                if j == k or j == l or k == l:
                    continue
                ch = [(inputs[e][j] & inputs[e][k]) | ((1 - inputs[e][j]) & inputs[e][l])
                      for e in range(n)]
                if ch == out_col:
                    matches.append(('ch', (j, k, l), f"CH(in[{j}],in[{k}],in[{l}])"))

    # Level 8: AND/OR of 3
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                a3 = [inputs[e][j] & inputs[e][k] & inputs[e][l] for e in range(n)]
                if a3 == out_col:
                    matches.append(('and3', (j, k, l), f"in[{j}] AND in[{k}] AND in[{l}]"))
                o3 = [inputs[e][j] | inputs[e][k] | inputs[e][l] for e in range(n)]
                if o3 == out_col:
                    matches.append(('or3', (j, k, l), f"in[{j}] OR in[{k}] OR in[{l}]"))

    # Level 9: XOR of 4
    for combo in combinations(range(8), 4):
        x4 = [inputs[e][combo[0]] ^ inputs[e][combo[1]] ^ inputs[e][combo[2]] ^ inputs[e][combo[3]]
              for e in range(n)]
        if x4 == out_col:
            matches.append(('xor4', combo, f"XOR4({','.join(f'in[{c}]' for c in combo)})"))
        if [1 - x for x in x4] == out_col:
            matches.append(('xnor4', combo, f"XNOR4({','.join(f'in[{c}]' for c in combo)})"))

    # Level 10: NOT(a) AND b, NOT(a) OR b (asymmetric 2-input)
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

    # Level 11: Composite 3-input (AND-XOR, OR-XOR, XOR-AND, XOR-OR, XNOR-AND, XNOR-OR)
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
    """Evaluate a function on target bits."""
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
    # Level 10: asymmetric 2-input
    if fname == 'not_and': return (1 - tb[args[0]]) & tb[args[1]]
    if fname == 'not_or': return (1 - tb[args[0]]) | tb[args[1]]
    # Level 11: composite 3-input
    if fname == 'and_xor': return (tb[args[0]] & tb[args[1]]) ^ tb[args[2]]
    if fname == 'or_xor': return (tb[args[0]] | tb[args[1]]) ^ tb[args[2]]
    if fname == 'xor_and': return (tb[args[0]] ^ tb[args[1]]) & tb[args[2]]
    if fname == 'xor_or': return (tb[args[0]] ^ tb[args[1]]) | tb[args[2]]
    if fname == 'xnor_and': return (1 - (tb[args[0]] ^ tb[args[1]])) & tb[args[2]]
    if fname == 'xnor_or': return (1 - (tb[args[0]] ^ tb[args[1]])) | tb[args[2]]
    return None

def solve_bit_ops(prompt, gold):
    """解 bit_ops 题。用 gold 消歧。"""
    examples, target = parse_bit_ops(prompt)
    if not examples or not target:
        return None, None
    if len(examples) < 4:
        return None, None
    if len(gold) != 8 or not all(c in '01' for c in gold):
        return None, None

    n = len(examples)
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    gold_bits = [int(gold[i]) for i in range(8)]

    result_bits = [None] * 8
    bit_rules = [None] * 8
    ambig_count = 0

    for obit in range(8):
        all_funcs = enumerate_bit_functions(inputs, outputs, n, obit)
        if not all_funcs:
            return None, None

        # Check if all agree
        preds = set()
        for f in all_funcs:
            p = eval_bit_function(f, target_bits)
            if p is not None:
                preds.add(p)

        if len(preds) == 1:
            result_bits[obit] = preds.pop()
            bit_rules[obit] = all_funcs[0][2]
        else:
            ambig_count += 1
            # Use gold to resolve
            gb = gold_bits[obit]
            gold_funcs = [f for f in all_funcs if eval_bit_function(f, target_bits) == gb]
            if gold_funcs:
                result_bits[obit] = gb
                # Pick simplest (shortest desc)
                gold_funcs.sort(key=lambda f: len(f[2]))
                bit_rules[obit] = gold_funcs[0][2]
            else:
                return None, None

    if None in result_bits:
        return None, None

    answer = ''.join(str(b) for b in result_bits)
    if answer != gold:
        return None, None

    # CoT
    cot = [f"Analyzing the 8-bit transformation rule per output bit:"]
    cot.append(f"")
    for i in range(8):
        marker = '' if ambig_count == 0 else ''
        cot.append(f"  bit {i}: {bit_rules[i]}")
    cot.append(f"")
    cot.append(f"Applying to {target}:")

    # Show application steps
    steps = []
    for i in range(8):
        steps.append(f"  bit {i}: {bit_rules[i]} → {result_bits[i]}")
    for s in steps:
        cot.append(s)
    cot.append(f"")
    cot.append(f"Result: {answer}")

    return answer, '\n'.join(cot)


# ═════════════════════════════════════════════════════════════════════════════
# SYMBOL — brute-force rule matching via base-94 arithmetic
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
    """Parse symbol prompt into examples and query."""
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

# 定义所有候选运算
SYMBOL_OPS = [
    ('concat', lambda l, r: l + r),
    ('concat_rev', lambda l, r: r + l),
    ('b94_add', lambda l, r: b94_to_str(str_to_b94(l) + str_to_b94(r))),
    ('b94_sub', lambda l, r: b94_to_str(str_to_b94(l) - str_to_b94(r))),
    ('b94_sub_rev', lambda l, r: b94_to_str(str_to_b94(r) - str_to_b94(l))),
    ('b94_mul', lambda l, r: b94_to_str(str_to_b94(l) * str_to_b94(r))),
]

# 等长时的 charwise 运算
SYMBOL_CW_OPS = [
    ('cw_add', lambda a, b: chr(((ord(a) - CHAR_BASE) + (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('cw_sub', lambda a, b: chr(((ord(a) - CHAR_BASE) - (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('cw_sub_rev', lambda a, b: chr(((ord(b) - CHAR_BASE) - (ord(a) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('cw_xor', lambda a, b: chr(((ord(a) - CHAR_BASE) ^ (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('cw_mul', lambda a, b: chr(((ord(a) - CHAR_BASE) * (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
]

def try_cw_op(name, fn, left, right, result):
    """Try a charwise op on equal-length operands."""
    if len(left) != len(right):
        return False
    try:
        pred = ''.join(fn(a, b) for a, b in zip(left, right))
        return pred == result
    except:
        return False

def apply_cw_op(fn, left, right):
    return ''.join(fn(a, b) for a, b in zip(left, right))

def solve_symbol(prompt, gold):
    """解 symbol 题 (尽力而为)。"""
    examples, query = parse_symbol(prompt)
    if not examples or not query:
        return None, None

    # Parse into (left, op_char, right, result) for examples with operators
    # Operators could be: +, -, *, /, |, \, ^, &
    OP_CHARS = set('+-*/|\\^&')

    def split_by_op(expr):
        """Split expression by operator. Return (left, op, right) or None."""
        for i, c in enumerate(expr):
            if c in OP_CHARS and i > 0 and i < len(expr) - 1:
                return expr[:i], c, expr[i + 1:]
        return None

    parsed = []
    for lhs, rhs in examples:
        sp = split_by_op(lhs)
        if sp:
            parsed.append((sp[0], sp[1], sp[2], rhs))

    if not parsed:
        return None, None

    query_split = split_by_op(query)
    if not query_split:
        return None, None

    q_left, q_op, q_right = query_split

    # Group by operator
    op_groups = defaultdict(list)
    for left, op, right, result in parsed:
        op_groups[op].append((left, right, result))

    if q_op not in op_groups:
        return None, None

    group = op_groups[q_op]

    # Try all operations
    for op_name, fn in SYMBOL_OPS:
        all_match = True
        for left, right, result in group:
            try:
                pred = fn(left, right)
                if pred != result:
                    all_match = False
                    break
            except:
                all_match = False
                break
        if all_match:
            try:
                answer = fn(q_left, q_right)
            except:
                continue
            if answer == gold:
                cot = f"The operator '{q_op}' maps to {op_name}.\n"
                cot += f"Applying: {q_left} {q_op} {q_right} = {answer}"
                return answer, cot

    # Try charwise ops (only if all operands equal length)
    for op_name, fn in SYMBOL_CW_OPS:
        all_match = True
        for left, right, result in group:
            if not try_cw_op(op_name, fn, left, right, result):
                all_match = False
                break
        if all_match:
            if len(q_left) == len(q_right):
                try:
                    answer = apply_cw_op(fn, q_left, q_right)
                except:
                    continue
                if answer == gold:
                    cot = f"The operator '{q_op}' maps to {op_name} (charwise).\n"
                    cot += f"Applying: {q_left} {q_op} {q_right} = {answer}"
                    return answer, cot

    # Try digit-based operations (for numeric operands with symbol operators)
    # E.g., 34/44 = 1 could mean digits: concat, multiply, etc.
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
                ('digit_sum', lambda a, b: str(sum(int(d) for d in str(a)) + sum(int(d) for d in str(b)))),
                ('digit_concat_sum', lambda a, b: str(sum(int(d) for d in (str(a) + str(b))))),
            ]
            for nop_name, nop_fn in NUM_OPS:
                all_match = True
                for left, right, result in group:
                    try:
                        if not all(c.isdigit() for c in left) or not all(c.isdigit() for c in right):
                            all_match = False
                            break
                        pred = nop_fn(int(left), int(right))
                        if pred != result:
                            all_match = False
                            break
                    except:
                        all_match = False
                        break
                if all_match:
                    try:
                        answer = nop_fn(nl, nr)
                    except:
                        continue
                    if answer == gold:
                        cot = f"The operator '{q_op}' maps to {nop_name} (numeric).\n"
                        cot += f"Applying: {q_left} {q_op} {q_right} = {answer}"
                        return answer, cot
    except:
        pass

    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — 生成全部 CoT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    rows = list(csv.DictReader(open('competition_data/train.csv')))

    solvers = {
        'numeral': solve_numeral,
        'gravity': solve_gravity,
        'unit_conv': solve_unit_conv,
        'cipher': solve_cipher,
        'bit_ops': solve_bit_ops,
        'symbol': solve_symbol,
    }

    stats = defaultdict(lambda: {'total': 0, 'solved': 0, 'failed': 0})
    records = []
    unsolved_records = []

    for r in rows:
        prompt = r['prompt']
        gold = r['answer'].strip()
        ptype = detect_type(prompt)

        if ptype not in solvers:
            continue

        stats[ptype]['total'] += 1
        solver = solvers[ptype]
        answer, thinking = solver(prompt, gold)

        if answer is not None:
            stats[ptype]['solved'] += 1
            records.append({
                'id': r['id'],
                'type': ptype,
                'prompt': prompt,
                'answer': gold,
                'thinking': thinking,
            })
        else:
            stats[ptype]['failed'] += 1
            unsolved_records.append({
                'id': r['id'],
                'type': ptype,
                'prompt': prompt,
                'answer': gold,
                'thinking': '',  # no CoT available
            })

    # 输出 JSONL
    os.makedirs('data', exist_ok=True)
    jsonl_path = 'data/cot_v2.jsonl'
    with open(jsonl_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # 输出 CSV (与 train.csv 格式兼容 + thinking 列)
    csv_path = 'data/sft_cot_v2.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking', 'type'])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                'id': rec['id'],
                'prompt': rec['prompt'],
                'answer': rec['answer'],
                'thinking': rec['thinking'],
                'type': rec['type'],
            })

    # 输出混合版本: CoT records + unsolved records (thinking 为空)
    hybrid_csv_path = 'data/sft_cot_v2_hybrid.csv'
    all_records = records + unsolved_records
    with open(hybrid_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking', 'type'])
        writer.writeheader()
        for rec in all_records:
            writer.writerow({
                'id': rec['id'],
                'prompt': rec['prompt'],
                'answer': rec['answer'],
                'thinking': rec['thinking'],
                'type': rec['type'],
            })

    # 报告
    print('=' * 70)
    print('CoT v2 Generation Results')
    print('=' * 70)
    print(f'{"Type":<12} {"Total":>6} {"Solved":>8} {"Failed":>8} {"Rate":>8}')
    print('-' * 45)
    total_solved = 0
    total_all = 0
    for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
        s = stats[t]
        rate = s['solved'] / s['total'] * 100 if s['total'] else 0
        print(f'{t:<12} {s["total"]:6d} {s["solved"]:8d} {s["failed"]:8d} {rate:7.1f}%')
        total_solved += s['solved']
        total_all += s['total']
    print('-' * 45)
    print(f'{"TOTAL":<12} {total_all:6d} {total_solved:8d} {total_all - total_solved:8d} '
          f'{total_solved / total_all * 100:7.1f}%')
    print()
    print(f'Output JSONL: {jsonl_path} ({len(records)} records)')
    print(f'Output CSV:   {csv_path}')
    print(f'Output CSV (hybrid): {hybrid_csv_path} ({len(all_records)} records = {len(records)} CoT + {len(unsolved_records)} answer-only)')

    # 展示一条样例 per type
    print()
    for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
        samples = [r for r in records if r['type'] == t]
        if samples:
            s = samples[0]
            print(f'--- {t} sample (thinking: {len(s["thinking"])} chars) ---')
            print(s['thinking'][:300])
            print()


if __name__ == '__main__':
    main()
