#!/usr/bin/env python3
"""
为 train.csv 中的每道题生成可读的 step-by-step thinking (CoT)。
每条 CoT 都展示真实的计算/推理过程，而非模板空话。

参考: generate_cot_v2.py 的风格，但更简洁可读。

输出: data/sft_thinking.csv (id, prompt, answer, thinking, type)

Usage:
  python3 scripts/gen_thinking.py                  # 默认从 competition_data/train.csv
  python3 scripts/gen_thinking.py --input FILE     # 指定输入
  python3 scripts/gen_thinking.py --sample 5       # 只输出每类 N 个样例到 stdout
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import string
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / 'competition_data' / 'train.csv'
DEFAULT_OUTPUT = PROJECT_DIR / 'data' / 'sft_thinking.csv'

# ═══════════════════════════════════════════════════════════════════════════════
#  Roman numeral utils
# ═══════════════════════════════════════════════════════════════════════════════
ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
ROMAN_DECOMP = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
]

def roman_to_int(s):
    total = prev = 0
    for ch in reversed(s.strip().upper()):
        val = ROMAN_VALUES.get(ch, 0)
        total += -val if val < prev else val
        prev = val
    return total

def int_to_roman(num):
    result = ''
    for v, s in ROMAN_DECOMP:
        while num >= v:
            result += s
            num -= v
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Type detection
# ═══════════════════════════════════════════════════════════════════════════════
_NUM_EQ_PATTERN = re.compile(r'^(\d+)([^\d])(\d+)$')


def _is_numeric_equation(prompt):
    """Check if a symbol/equation prompt is numeric (digit-op-digit) vs symbolic."""
    for line in prompt.strip().split('\n'):
        line = line.strip()
        if ' = ' in line and 'alice' not in line.lower() and 'equation' not in line.lower() \
                and 'transformation' not in line.lower() and 'determine' not in line.lower():
            lhs = line.split(' = ', 1)[0].strip()
            if lhs:
                return bool(_NUM_EQ_PATTERN.match(lhs))
    return False


def classify(prompt):
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
    if 'symbol' in p or 'equation' in p or 'transformation rules' in p:
        if _is_numeric_equation(prompt):
            return 'eq_numeric'
        return 'eq_symbolic'
    return 'unknown'


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMERAL — Roman / Base conversion
# ═══════════════════════════════════════════════════════════════════════════════
def gen_thinking_numeral(prompt, gold):
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
        return None

    query = m.group(1).strip().rstrip('.')
    if not examples:
        return None

    sample_out = examples[0][1]
    sample_in = examples[0][0]
    is_to_roman = bool(re.match(r'^[IVXLCDM]+$', sample_out.upper()))
    is_from_roman = bool(re.match(r'^[IVXLCDM]+$', sample_in.upper()))

    # Case 1: Arabic → Roman
    if is_to_roman and not is_from_roman:
        try:
            for inp, out in examples:
                if int_to_roman(int(inp)) != out:
                    return None
            expected = int_to_roman(int(query))
        except Exception:
            return None
        if expected != gold:
            return None

        num = int(query)
        cot = ["The examples show Arabic to Roman numeral conversion."]
        cot.append(f"")
        cot.append(f"Converting {query}:")
        remaining = num
        for v, s in ROMAN_DECOMP:
            while remaining >= v:
                remaining -= v
                cot.append(f"  {v} fits into {remaining + v} → write '{s}', remaining: {remaining}")
        cot.append(f"")
        cot.append(f"Result: {expected}")
        return '\n'.join(cot)

    # Case 2: Roman → Arabic
    if is_from_roman and not is_to_roman:
        try:
            for inp, out in examples:
                if str(roman_to_int(inp)) != out:
                    return None
            expected = str(roman_to_int(query))
        except Exception:
            return None
        if expected != gold:
            return None

        cot = ["The examples show Roman to Arabic numeral conversion."]
        cot.append(f"")
        cot.append(f"Converting {query}:")
        chars = list(query.upper())
        steps = []
        total = prev = 0
        for ch in reversed(chars):
            val = ROMAN_VALUES.get(ch, 0)
            if val < prev:
                total -= val
                steps.append(f"  {ch} ({val}) < previous → subtract: total = {total}")
            else:
                total += val
                steps.append(f"  {ch} ({val}) → add: total = {total}")
            prev = val
        for s in reversed(steps):
            cot.append(s)
        cot.append(f"")
        cot.append(f"Result: {expected}")
        return '\n'.join(cot)

    # Case 3: Base conversion (brute force)
    for fb in range(2, 37):
        for tb in range(2, 37):
            if fb == tb:
                continue
            ok = True
            for inp, out in examples:
                try:
                    val = int(inp, fb)
                    digits = []
                    v = val if val > 0 else 0
                    if v == 0:
                        conv = "0"
                    else:
                        while v > 0:
                            digits.append("0123456789abcdefghijklmnopqrstuvwxyz"[v % tb])
                            v //= tb
                        conv = "".join(reversed(digits))
                    if conv.lower() != out.lower():
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok:
                try:
                    val = int(query, fb)
                    if val == 0:
                        expected = "0"
                    else:
                        digits = []
                        v = val
                        while v > 0:
                            digits.append("0123456789abcdefghijklmnopqrstuvwxyz"[v % tb])
                            v //= tb
                        expected = "".join(reversed(digits))
                except Exception:
                    continue
                if expected.lower() != gold.lower():
                    continue

                cot = [f"This is base {fb} to base {tb} conversion."]
                cot.append(f"")
                cot.append(f"Step 1: Convert {query} (base {fb}) to decimal:")
                digit_strs = []
                for i, ch in enumerate(reversed(query)):
                    d = int(ch, fb)
                    power = fb ** i
                    digit_strs.append(f"{d}×{fb}^{i}={d * power}")
                cot.append(f"  {' + '.join(reversed(digit_strs))} = {val}")
                cot.append(f"")
                cot.append(f"Step 2: Convert {val} to base {tb}:")
                v = val
                while v > 0:
                    cot.append(f"  {v} ÷ {tb} = {v // tb} remainder {v % tb}")
                    v //= tb
                cot.append(f"  Reading remainders bottom-to-top: {expected}")
                cot.append(f"")
                cot.append(f"Result: {expected}")
                return '\n'.join(cot)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIT_CONV — linear conversion factor
# ═══════════════════════════════════════════════════════════════════════════════
def gen_thinking_unit(prompt, gold):
    pairs = re.findall(r'([\d.]+)\s*\w*\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        pairs = re.findall(r'([\d.]+)\s*->\s*([\d.]+)', prompt)

    query_m = re.search(r'convert.*?:\s*([\d.]+)', prompt, re.I)
    if not query_m:
        after = prompt.split('Now')[-1] if 'Now' in prompt else ''
        query_m = re.search(r'([\d.]+)\s*\w', after)

    if not pairs or not query_m:
        return None

    query_val = float(query_m.group(1))

    factors = []
    pair_data = []
    for in_s, out_s in pairs:
        in_v, out_v = float(in_s), float(out_s)
        if in_v == 0:
            continue
        f = out_v / in_v
        factors.append(f)
        pair_data.append((in_v, out_v, f))

    if not factors:
        return None

    factor_avg = sum(factors) / len(factors)
    result = factor_avg * query_val

    try:
        gold_val = float(gold)
    except ValueError:
        return None
    if abs(float(f"{result:.2f}") - gold_val) > 0.05:
        return None

    cot = ["This is a linear unit conversion."]
    cot.append("")
    cot.append("Finding the conversion factor from examples:")
    for in_v, out_v, f in pair_data:
        cot.append(f"  {in_v} → {out_v}: factor = {f:.6f}")
    cot.append("")
    cot.append(f"Average factor = {factor_avg:.6f}")
    cot.append("")
    cot.append(f"{query_val} × {factor_avg:.6f} = {result:.2f}")
    cot.append("")
    cot.append(f"Result: {gold}")
    return '\n'.join(cot)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAVITY — d = 0.5 * g * t²
# ═══════════════════════════════════════════════════════════════════════════════
def gen_thinking_gravity(prompt, gold):
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    query_m = re.search(r'for\s+t\s*=\s*([\d.]+)\s*s\s*given', prompt)
    if not query_m:
        after = prompt.split('determine')[-1] if 'determine' in prompt else ''
        query_m = re.search(r't\s*=\s*([\d.]+)\s*s', after)

    if not pairs or not query_m:
        return None

    query_t = float(query_m.group(1))

    g_values = []
    pair_data = []
    for t_s, d_s in pairs:
        t, d = float(t_s), float(d_s)
        if t == 0:
            return None
        g = 2.0 * d / (t * t)
        g_values.append(g)
        pair_data.append((t, d, g))

    g_avg = sum(g_values) / len(g_values)
    result = 0.5 * g_avg * query_t * query_t

    try:
        gold_val = float(gold)
    except ValueError:
        return None
    if abs(float(f"{result:.2f}") - gold_val) > 0.05:
        return None

    cot = ["Using d = 0.5*g*t², I need to find g first."]
    cot.append("")
    cot.append("From each observation, g = 2d/t²:")
    for t, d, g in pair_data:
        cot.append(f"  t={t}s, d={d}m: g = 2×{d}/{t}² = {g:.4f}")
    cot.append("")
    cot.append(f"Average g = {g_avg:.4f}")
    cot.append("")
    cot.append(f"For t = {query_t}s:")
    cot.append(f"d = 0.5 × {g_avg:.4f} × {query_t}² = {result:.2f}")
    cot.append("")
    cot.append(f"Result: {gold}")
    return '\n'.join(cot)


# ═══════════════════════════════════════════════════════════════════════════════
#  CIPHER — substitution cipher
# ═══════════════════════════════════════════════════════════════════════════════
def gen_thinking_cipher(prompt, gold):
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
    if not examples or not target:
        return None

    # Build char-level mapping from examples ONLY (no guessing)
    direct_map = {}
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
                if ecl in direct_map and direct_map[ecl] != pcl:
                    return None  # conflict
                direct_map[ecl] = pcl

    # Build full mapping (direct + gold-inferred) for verification only
    enc2plain = dict(direct_map)
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

    # Decrypt and verify
    result_chars = []
    for c in target:
        if c == ' ':
            result_chars.append(' ')
        elif c.lower() in enc2plain:
            result_chars.append(enc2plain[c.lower()])
        else:
            return None
    answer = ''.join(result_chars)
    if answer.strip().lower() != gold.strip().lower():
        return None

    # ── Build CoT with pattern-matching approach ──

    cot = ["This is a substitution cipher.\n"]

    # Step 1: Extract known mappings
    cot.append("Step 1: Extract letter mapping from each word pair:")
    seen = set()
    for encrypted, plaintext in examples:
        enc_words = encrypted.split()
        plain_words = plaintext.split()
        if len(enc_words) != len(plain_words):
            continue
        for ew, pw in zip(enc_words, plain_words):
            if len(ew) != len(pw):
                continue
            new_pairs = []
            for ec, pc in zip(ew, pw):
                ecl, pcl = ec.lower(), pc.lower()
                if ecl not in seen:
                    seen.add(ecl)
                    new_pairs.append(f"{ecl}→{pcl}")
            if new_pairs:
                cot.append(f"  {ew} → {pw}: {', '.join(new_pairs)}")

    cot.append("\nStep 2: Verify each letter maps to exactly one letter (one-to-one).")

    # Step 3: Partial decode — apply known mappings, mark unknowns as _
    target_words = target.split()
    gold_words = gold.split()

    partial_results = []  # (enc_word, partial_str, gold_word, has_gap)
    for tw, gw in zip(target_words, gold_words):
        chars_partial = []
        has_gap = False
        for c in tw:
            cl = c.lower()
            if cl in direct_map:
                chars_partial.append(direct_map[cl])
            else:
                chars_partial.append('_')
                has_gap = True
        partial_str = ''.join(chars_partial)
        partial_results.append((tw, partial_str, gw, has_gap))

    has_any_gap = any(r[3] for r in partial_results)

    cot.append(f"\nStep 3: Apply known mappings to target words{', mark unknowns as _' if has_any_gap else ''}:")
    for tw, partial_str, gw, has_gap in partial_results:
        pairs = []
        for c in tw:
            cl = c.lower()
            if cl in direct_map:
                pairs.append(f"{cl}→{direct_map[cl]}")
            else:
                pairs.append(f"{cl}→?")
        status = "" if not has_gap else ""
        cot.append(f"  {tw}: {', '.join(pairs)} → \"{partial_str}\"")

    # Step 4: Pattern matching for words with gaps
    if has_any_gap:
        cot.append(f"\nStep 4: Resolve unknown letters by pattern matching:")
        for tw, partial_str, gw, has_gap in partial_results:
            if not has_gap:
                continue
            # Show the pattern and resolution
            cot.append(f"  \"{partial_str}\" → matches \"{gw}\"")
            # Derive new mappings
            for i, c in enumerate(tw):
                cl = c.lower()
                if cl not in direct_map:
                    cot.append(f"    {cl}→{gw[i].lower()}")

        # Step 5: Full decryption with all mappings resolved
        cot.append(f"\nStep 5: Decrypt target:")
        for tw, partial_str, gw, has_gap in partial_results:
            pairs = []
            for c in tw:
                cl = c.lower()
                p = enc2plain.get(cl, '?')
                pairs.append(f"{cl}→{p}")
            cot.append(f"  {tw}: {', '.join(pairs)} → \"{gw}\"")

    cot.append(f"\nResult: {answer}")
    return '\n'.join(cot)


# ═══════════════════════════════════════════════════════════════════════════════
#  BIT_OPS — per-bit boolean function enumeration + gold-guided selection
# ═══════════════════════════════════════════════════════════════════════════════
def _enumerate_bit_functions(inputs, outputs, n, obit):
    """枚举所有匹配 output bit obit 的 boolean function。"""
    out_col = [outputs[e][obit] for e in range(n)]
    matches = []

    # Constants
    if all(x == 0 for x in out_col):
        matches.append(('const', 0, '0'))
    if all(x == 1 for x in out_col):
        matches.append(('const', 1, '1'))

    # Level 1: copy / NOT
    for i in range(8):
        ic = [inputs[e][i] for e in range(n)]
        if ic == out_col:
            matches.append(('copy', i, f"in[{i}]"))
        if [1 - x for x in ic] == out_col:
            matches.append(('not', i, f"NOT(in[{i}])"))

    # Level 2: XOR / XNOR of 2
    for j in range(8):
        for k in range(j + 1, 8):
            xor = [inputs[e][j] ^ inputs[e][k] for e in range(n)]
            if xor == out_col:
                matches.append(('xor2', (j, k), f"in[{j}] XOR in[{k}]"))
            if [1 - x for x in xor] == out_col:
                matches.append(('xnor2', (j, k), f"NOT(in[{j}] XOR in[{k}])"))

    # Level 3: AND / OR / NAND / NOR of 2
    for j in range(8):
        for k in range(j + 1, 8):
            for op_name, op_fn in [('AND', lambda a, b: a & b), ('OR', lambda a, b: a | b),
                                   ('NAND', lambda a, b: 1 - (a & b)), ('NOR', lambda a, b: 1 - (a | b))]:
                col = [op_fn(inputs[e][j], inputs[e][k]) for e in range(n)]
                if col == out_col:
                    desc_name = op_name
                    if op_name in ('NAND', 'NOR'):
                        base = 'AND' if op_name == 'NAND' else 'OR'
                        desc = f"NOT(in[{j}] {base} in[{k}])"
                    else:
                        desc = f"in[{j}] {desc_name} in[{k}]"
                    matches.append((op_name.lower(), (j, k), desc))

    # Level 4: XOR of 3
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                x3 = [inputs[e][j] ^ inputs[e][k] ^ inputs[e][l] for e in range(n)]
                if x3 == out_col:
                    matches.append(('xor3', (j, k, l), f"in[{j}] XOR in[{k}] XOR in[{l}]"))

    # Level 5: Asymmetric 2-input
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

    # Level 6: Composite 3-input (AND/OR + XOR)
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

    # Level 7: AND3 / OR3 / NAND3 / NOR3
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                a3 = [inputs[e][j] & inputs[e][k] & inputs[e][l] for e in range(n)]
                if a3 == out_col:
                    matches.append(('and3', (j, k, l), f"in[{j}] AND in[{k}] AND in[{l}]"))
                if [1 - x for x in a3] == out_col:
                    matches.append(('nand3', (j, k, l), f"NOT(in[{j}] AND in[{k}] AND in[{l}])"))
                o3 = [inputs[e][j] | inputs[e][k] | inputs[e][l] for e in range(n)]
                if o3 == out_col:
                    matches.append(('or3', (j, k, l), f"in[{j}] OR in[{k}] OR in[{l}]"))
                if [1 - x for x in o3] == out_col:
                    matches.append(('nor3', (j, k, l), f"NOT(in[{j}] OR in[{k}] OR in[{l}])"))

    # Level 8: (XOR of 2) AND/OR third
    for j in range(8):
        for k in range(j + 1, 8):
            xor_jk = [inputs[e][j] ^ inputs[e][k] for e in range(n)]
            for l in range(8):
                if l == j or l == k:
                    continue
                xa = [xor_jk[e] & inputs[e][l] for e in range(n)]
                if xa == out_col:
                    matches.append(('xor_and', (j, k, l), f"(in[{j}] XOR in[{k}]) AND in[{l}]"))
                if [1 - x for x in xa] == out_col:
                    matches.append(('nxor_and', (j, k, l), f"NOT((in[{j}] XOR in[{k}]) AND in[{l}])"))
                xo = [xor_jk[e] | inputs[e][l] for e in range(n)]
                if xo == out_col:
                    matches.append(('xor_or', (j, k, l), f"(in[{j}] XOR in[{k}]) OR in[{l}]"))
                if [1 - x for x in xo] == out_col:
                    matches.append(('nxor_or', (j, k, l), f"NOT((in[{j}] XOR in[{k}]) OR in[{l}])"))

    # Level 9: (AND of 2) OR/AND third, (OR of 2) AND/OR third
    for j in range(8):
        for k in range(j + 1, 8):
            and_jk = [inputs[e][j] & inputs[e][k] for e in range(n)]
            or_jk = [inputs[e][j] | inputs[e][k] for e in range(n)]
            for l in range(8):
                if l == j or l == k:
                    continue
                ao = [and_jk[e] | inputs[e][l] for e in range(n)]
                if ao == out_col:
                    matches.append(('and_or', (j, k, l), f"(in[{j}] AND in[{k}]) OR in[{l}]"))
                if [1 - x for x in ao] == out_col:
                    matches.append(('nand_or', (j, k, l), f"NOT((in[{j}] AND in[{k}]) OR in[{l}])"))
                oa = [or_jk[e] & inputs[e][l] for e in range(n)]
                if oa == out_col:
                    matches.append(('or_and', (j, k, l), f"(in[{j}] OR in[{k}]) AND in[{l}]"))
                if [1 - x for x in oa] == out_col:
                    matches.append(('nor_and', (j, k, l), f"NOT((in[{j}] OR in[{k}]) AND in[{l}])"))

    # Level 10: XOR of 4
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                for m in range(l + 1, 8):
                    x4 = [inputs[e][j] ^ inputs[e][k] ^ inputs[e][l] ^ inputs[e][m] for e in range(n)]
                    if x4 == out_col:
                        matches.append(('xor4', (j, k, l, m), f"in[{j}] XOR in[{k}] XOR in[{l}] XOR in[{m}]"))
                    if [1 - x for x in x4] == out_col:
                        matches.append(('xnor4', (j, k, l, m), f"NOT(in[{j}] XOR in[{k}] XOR in[{l}] XOR in[{m}])"))

    # Level 11: Majority of 3
    for j in range(8):
        for k in range(j + 1, 8):
            for l in range(k + 1, 8):
                maj = [1 if (inputs[e][j] + inputs[e][k] + inputs[e][l]) >= 2 else 0 for e in range(n)]
                if maj == out_col:
                    matches.append(('maj3', (j, k, l), f"MAJ(in[{j}],in[{k}],in[{l}])"))
                if [1 - x for x in maj] == out_col:
                    matches.append(('nmaj3', (j, k, l), f"NOT(MAJ(in[{j}],in[{k}],in[{l}]))"))

    # Level 12: NOT(a) AND b AND c / NOT(a) OR b OR c
    for j in range(8):
        for k in range(8):
            if k == j:
                continue
            for l in range(k + 1, 8):
                if l == j:
                    continue
                nab = [(1 - inputs[e][j]) & inputs[e][k] & inputs[e][l] for e in range(n)]
                if nab == out_col:
                    matches.append(('not_and2', (j, k, l), f"NOT(in[{j}]) AND in[{k}] AND in[{l}]"))
                if [1 - x for x in nab] == out_col:
                    matches.append(('nnot_and2', (j, k, l), f"NOT(NOT(in[{j}]) AND in[{k}] AND in[{l}])"))
                nob = [(1 - inputs[e][j]) | inputs[e][k] | inputs[e][l] for e in range(n)]
                if nob == out_col:
                    matches.append(('not_or2', (j, k, l), f"NOT(in[{j}]) OR in[{k}] OR in[{l}]"))
                if [1 - x for x in nob] == out_col:
                    matches.append(('nnot_or2', (j, k, l), f"NOT(NOT(in[{j}]) OR in[{k}] OR in[{l}])"))

    return matches


def _eval_bit_function(func, target_bits):
    """Evaluate a function on target bits."""
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
    if fname == 'xor3': return tb[args[0]] ^ tb[args[1]] ^ tb[args[2]]
    if fname == 'not_and': return (1 - tb[args[0]]) & tb[args[1]]
    if fname == 'not_or': return (1 - tb[args[0]]) | tb[args[1]]
    if fname == 'and_xor': return (tb[args[0]] & tb[args[1]]) ^ tb[args[2]]
    if fname == 'or_xor': return (tb[args[0]] | tb[args[1]]) ^ tb[args[2]]
    if fname == 'and3': return tb[args[0]] & tb[args[1]] & tb[args[2]]
    if fname == 'nand3': return 1 - (tb[args[0]] & tb[args[1]] & tb[args[2]])
    if fname == 'or3': return tb[args[0]] | tb[args[1]] | tb[args[2]]
    if fname == 'nor3': return 1 - (tb[args[0]] | tb[args[1]] | tb[args[2]])
    if fname == 'xor_and': return (tb[args[0]] ^ tb[args[1]]) & tb[args[2]]
    if fname == 'nxor_and': return 1 - ((tb[args[0]] ^ tb[args[1]]) & tb[args[2]])
    if fname == 'xor_or': return (tb[args[0]] ^ tb[args[1]]) | tb[args[2]]
    if fname == 'nxor_or': return 1 - ((tb[args[0]] ^ tb[args[1]]) | tb[args[2]])
    if fname == 'and_or': return (tb[args[0]] & tb[args[1]]) | tb[args[2]]
    if fname == 'nand_or': return 1 - ((tb[args[0]] & tb[args[1]]) | tb[args[2]])
    if fname == 'or_and': return (tb[args[0]] | tb[args[1]]) & tb[args[2]]
    if fname == 'nor_and': return 1 - ((tb[args[0]] | tb[args[1]]) & tb[args[2]])
    if fname == 'xor4': return tb[args[0]] ^ tb[args[1]] ^ tb[args[2]] ^ tb[args[3]]
    if fname == 'xnor4': return 1 - (tb[args[0]] ^ tb[args[1]] ^ tb[args[2]] ^ tb[args[3]])
    if fname == 'maj3': return 1 if (tb[args[0]] + tb[args[1]] + tb[args[2]]) >= 2 else 0
    if fname == 'nmaj3': return 0 if (tb[args[0]] + tb[args[1]] + tb[args[2]]) >= 2 else 1
    if fname == 'not_and2': return (1 - tb[args[0]]) & tb[args[1]] & tb[args[2]]
    if fname == 'nnot_and2': return 1 - ((1 - tb[args[0]]) & tb[args[1]] & tb[args[2]])
    if fname == 'not_or2': return (1 - tb[args[0]]) | tb[args[1]] | tb[args[2]]
    if fname == 'nnot_or2': return 1 - ((1 - tb[args[0]]) | tb[args[1]] | tb[args[2]])
    # tt3: arbitrary 3-input truth table
    if fname == 'tt3':
        j, k, l, fid = args
        idx = tb[j] * 4 + tb[k] * 2 + tb[l]
        return (fid >> idx) & 1
    return None


_TYPE_ORDER = {
    'const': (0, 0, 0),
    'copy': (1, 0, 1),
    'not': (1, 1, 2),
    'xor2': (2, 1, 3),
    'and': (2, 1, 4),
    'or': (2, 1, 5),
    'xnor2': (2, 2, 6),
    'nand': (2, 2, 7),
    'nor': (2, 2, 8),
    'not_and': (2, 2, 9),
    'not_or': (2, 2, 10),
    'xor3': (3, 2, 11),
    'and3': (3, 2, 12),
    'or3': (3, 2, 13),
    'and_xor': (3, 2, 14),
    'or_xor': (3, 2, 15),
    'xor_and': (3, 2, 16),
    'xor_or': (3, 2, 17),
    'and_or': (3, 2, 18),
    'or_and': (3, 2, 19),
    'not_and2': (3, 2, 20),
    'not_or2': (3, 2, 21),
    'nand3': (3, 3, 22),
    'nor3': (3, 3, 23),
    'nxor_and': (3, 3, 24),
    'nxor_or': (3, 3, 25),
    'nand_or': (3, 3, 26),
    'nor_and': (3, 3, 27),
    'nnot_and2': (3, 3, 28),
    'nnot_or2': (3, 3, 29),
    'maj3': (3, 3, 30),
    'nmaj3': (3, 3, 31),
    'xor4': (4, 3, 32),
    'xnor4': (4, 4, 33),
    'tt3': (5, 5, 50),
}


def _func_sort_key(func):
    """Deterministic complexity-based sort key."""
    fname, args, _ = func
    n_inputs, n_ops, type_ord = _TYPE_ORDER.get(fname, (6, 6, 99))
    if isinstance(args, int):
        var_key = (args,)
    elif isinstance(args, tuple):
        var_key = args
    else:
        var_key = ()
    return (n_inputs, n_ops, type_ord, var_key)


# --- 2-input gate lookup for tt3 description ---
_GATE2 = {
    0b0000: '0',
    0b1111: '1',
    0b0110: 'a XOR b',
    0b1001: 'NOT(a XOR b)',
    0b0001: 'a AND b',
    0b0111: 'a OR b',
    0b1110: 'NOT(a AND b)',
    0b1000: 'NOT(a OR b)',
    0b0010: 'a AND NOT(b)',
    0b0100: 'NOT(a) AND b',
    0b1011: 'a OR NOT(b)',
    0b1101: 'NOT(a) OR b',
    0b0011: 'a',
    0b1100: 'NOT(a)',
    0b0101: 'b',
    0b1010: 'NOT(b)',
}


def _tt3_description(j, k, l, fid):
    """Generate readable description for arbitrary 3-input truth table.

    Uses Shannon decomposition: f(j,k,l) = MUX(j, f1(k,l), f0(k,l))
    where f0 = f(j=0,...) and f1 = f(j=1,...).
    """
    # Extract f0 (j=0) and f1 (j=1) sub-functions of (k,l)
    f0_bits = fid & 0xF          # rows where j=0: indices 0,1,2,3
    f1_bits = (fid >> 4) & 0xF   # rows where j=1: indices 4,5,6,7

    if f0_bits == f1_bits:
        # Independent of j
        gate = _GATE2.get(f0_bits, f'TT2({f0_bits:04b})')
        if gate in ('0', '1'):
            return gate
        if gate in ('a', 'NOT(a)'):
            return gate.replace('a', f'in[{k}]')
        if gate in ('b', 'NOT(b)'):
            return gate.replace('b', f'in[{l}]')
        return gate.replace('a', f'in[{k}]').replace('b', f'in[{l}]')

    g0 = _GATE2.get(f0_bits, f'TT2({f0_bits:04b})')
    g1 = _GATE2.get(f1_bits, f'TT2({f1_bits:04b})')

    def _fmt_gate(g, kk, ll):
        if g in ('0', '1'):
            return g
        if g in ('a',):
            return f'in[{kk}]'
        if g in ('b',):
            return f'in[{ll}]'
        if g == 'NOT(a)':
            return f'NOT(in[{kk}])'
        if g == 'NOT(b)':
            return f'NOT(in[{ll}])'
        return g.replace('a', f'in[{kk}]').replace('b', f'in[{ll}]')

    s0 = _fmt_gate(g0, k, l)
    s1 = _fmt_gate(g1, k, l)

    # Wrap complex sub-expressions in parens
    def _wrap(s):
        return f'({s})' if ' ' in s else s

    # Simplifications
    if s0 == '0' and s1 == '1':
        return f'in[{j}]'
    if s0 == '1' and s1 == '0':
        return f'NOT(in[{j}])'
    if s0 == '0':
        return f'in[{j}] AND {_wrap(s1)}'
    if s1 == '0':
        return f'NOT(in[{j}]) AND {_wrap(s0)}'
    if s1 == '1':
        return f'in[{j}] OR {_wrap(s0)}'
    if s0 == '1':
        return f'NOT(in[{j}]) OR {_wrap(s1)}'

    return f'MUX(in[{j}], {s1}, {s0})'


def _search_tt3_gold(inputs, outputs, n, obit, target_bits, gold_bit):
    """Brute-force search for 3-input truth table matching examples + gold.

    Searches all ordered triples (j,k,l) and all 256 truth tables.
    Returns (func_tuple) or None.
    """
    out_col = [outputs[e][obit] for e in range(n)]

    for j in range(8):
        for k in range(8):
            if k == j:
                continue
            for l in range(8):
                if l == j or l == k:
                    continue
                # Build example index columns
                idx_col = [inputs[e][j] * 4 + inputs[e][k] * 2 + inputs[e][l] for e in range(n)]
                target_idx = target_bits[j] * 4 + target_bits[k] * 2 + target_bits[l]

                # Build constraints: for each example, fid bit at idx must equal out_col
                # Also: fid bit at target_idx must equal gold_bit
                # Try to find valid fid
                fid = 0
                ok = True
                constraints = {}  # idx -> required_bit
                for e in range(n):
                    idx = idx_col[e]
                    req = out_col[e]
                    if idx in constraints:
                        if constraints[idx] != req:
                            ok = False
                            break
                    else:
                        constraints[idx] = req

                if not ok:
                    continue

                # Check gold constraint
                if target_idx in constraints and constraints[target_idx] != gold_bit:
                    continue

                # Build fid from constraints
                fid = 0
                for idx, bit in constraints.items():
                    if bit:
                        fid |= (1 << idx)
                # Set gold bit
                if gold_bit:
                    fid |= (1 << target_idx)
                # else: already 0

                # Verify
                valid = True
                for e in range(n):
                    idx = idx_col[e]
                    if ((fid >> idx) & 1) != out_col[e]:
                        valid = False
                        break
                if not valid:
                    continue
                # Verify gold
                if ((fid >> target_idx) & 1) != gold_bit:
                    continue

                desc = _tt3_description(j, k, l, fid)
                return ('tt3', (j, k, l, fid), desc)

    return None


def _substitute_values(desc, bits):
    """Replace in[X] with actual bit values in a function description."""
    import re as _re_sub
    return _re_sub.sub(r'in\[(\d+)\]', lambda m: str(bits[int(m.group(1))]), desc)


def _get_input_refs(func):
    """Get the input bit indices referenced by a function."""
    fname, args, _ = func
    if fname == 'const': return []
    if fname in ('copy', 'not'): return [args]
    if isinstance(args, tuple):
        if fname == 'tt3': return list(args[:3])  # (j,k,l,fid)
        return list(args)
    return []


def _get_complexity_level(fname):
    """Return search complexity level: 0=const, 1=single, 2=pair, 3=triple+, 4=tt3."""
    if fname == 'const': return 0
    if fname in ('copy', 'not'): return 1
    if fname in ('xor2', 'xnor2', 'and', 'or', 'nand', 'nor', 'not_and', 'not_or'): return 2
    if fname == 'tt3': return 4
    return 3


def _make_op_desc(fname, args):
    """Generate human-readable description for an operation with specific args."""
    if fname == 'const': return str(args)
    if fname == 'copy': return f"in[{args}]"
    if fname == 'not': return f"NOT(in[{args}])"
    j, k = args[0], args[1]
    if fname == 'xor2': return f"in[{j}] XOR in[{k}]"
    if fname == 'xnor2': return f"NOT(in[{j}] XOR in[{k}])"
    if fname == 'and': return f"in[{j}] AND in[{k}]"
    if fname == 'or': return f"in[{j}] OR in[{k}]"
    if fname == 'nand': return f"NOT(in[{j}] AND in[{k}])"
    if fname == 'nor': return f"NOT(in[{j}] OR in[{k}])"
    if fname == 'not_and': return f"NOT(in[{j}]) AND in[{k}]"
    if fname == 'not_or': return f"NOT(in[{j}]) OR in[{k}]"
    return str(fname)


def _eval_func_col(func, inputs, n):
    """Evaluate a function across all n examples, return column of results."""
    return [_eval_bit_function(func, inputs[e]) for e in range(n)]


def _find_near_misses(inputs, n, out_col, correct_func, max_nm=2):
    """Find near-miss candidates: same op type, different indices.

    Returns list of (desc_str, computed_col, match_count) sorted by match desc.
    """
    fname, args, _ = correct_func
    level = _get_complexity_level(fname)
    if level < 2:
        return []

    near = []

    if fname in ('xor2', 'xnor2', 'and', 'or', 'nand', 'nor'):
        correct_pair = tuple(sorted(args))
        for j in range(8):
            for k in range(j + 1, 8):
                if (j, k) == correct_pair:
                    continue
                cand = (fname, (j, k), _make_op_desc(fname, (j, k)))
                col = _eval_func_col(cand, inputs, n)
                mc = sum(1 for a, b in zip(col, out_col) if a == b)
                if mc > n // 2 and mc < n:  # e.g. 3 or 4 out of 5
                    near.append((_make_op_desc(fname, (j, k)), col, mc))
    elif fname in ('not_and', 'not_or'):
        for j in range(8):
            for k in range(8):
                if j == k:
                    continue
                if isinstance(args, tuple) and (j, k) == tuple(args):
                    continue
                cand = (fname, (j, k), _make_op_desc(fname, (j, k)))
                col = _eval_func_col(cand, inputs, n)
                mc = sum(1 for a, b in zip(col, out_col) if a == b)
                if mc > n // 2 and mc < n:
                    near.append((_make_op_desc(fname, (j, k)), col, mc))

    # Sort: highest match count first, then lexicographic for determinism
    near.sort(key=lambda x: (-x[2], x[0]))
    return near[:max_nm]


def gen_thinking_bit(prompt, gold):
    """Generate thinking CoT for bit_ops — v8 derivation-based reasoning.

    Shows explicit derivation for each output bit:
    1. Target output values across all examples
    2. Near-misses shown for 2+ input ops (teaches rejection via match count)
    3. Correct rule shown matching ALL examples (5/5 ✓)
    4. Application to target with explicit computation
    """
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

    if not examples or not target:
        return None
    if len(examples) < 4:
        return None
    if len(gold) != 8 or not all(c in '01' for c in gold):
        return None

    n = len(examples)
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in examples]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in examples]
    target_bits = [int(target[i]) for i in range(8)]
    gold_bits = [int(gold[i]) for i in range(8)]

    chosen = [None] * 8
    result_bits = [None] * 8

    for obit in range(8):
        gold_bit = gold_bits[obit]
        funcs = _enumerate_bit_functions(inputs, outputs, n, obit)
        correct_funcs = [f for f in funcs if _eval_bit_function(f, target_bits) == gold_bit]

        if correct_funcs:
            correct_funcs.sort(key=_func_sort_key)
            chosen[obit] = correct_funcs[0]
            result_bits[obit] = gold_bit
        else:
            tt3_func = _search_tt3_gold(inputs, outputs, n, obit, target_bits, gold_bit)
            if tt3_func is None:
                return None
            chosen[obit] = tt3_func
            result_bits[obit] = gold_bit

    answer = ''.join(str(b) for b in result_bits)
    assert answer == gold, f"v7 bug: {answer} != {gold}"

    # Build derivation-based CoT:
    # ① Per-bit derivation (show data → search → match)
    # ② Execution on target
    # ③ Conclusion
    cot = []

    cot.append("Analyzing each output bit:")

    for obit in range(8):
        out_col = [outputs[e][obit] for e in range(n)]
        out_str = ','.join(str(v) for v in out_col)
        func = chosen[obit]
        fname, args, desc = func
        level = _get_complexity_level(fname)

        if fname == 'const':
            cot.append(f"bit {obit}: [{out_str}] → const {args}")
            continue

        cot.append(f"bit {obit}: [{out_str}]")

        if level == 1:  # COPY or NOT
            match_col = _eval_func_col(func, inputs, n)
            match_str = ','.join(str(v) for v in match_col)
            cot.append(f"  {desc}: [{match_str}] → match")
        else:  # 2-input or higher
            # Show near-misses (teaches model to reject partial matches)
            near_misses = _find_near_misses(inputs, n, out_col, func)
            for nm_desc, nm_col, nm_match in near_misses:
                nm_str = ','.join(str(v) for v in nm_col)
                cot.append(f"  {nm_desc}: [{nm_str}] {nm_match}/{n} ✗")

            # Show correct match
            correct_col = _eval_func_col(func, inputs, n)
            correct_str = ','.join(str(v) for v in correct_col)
            cot.append(f"  {desc}: [{correct_str}] {n}/{n} ✓")

    # ── ② Execution on target ──
    non_const = [(i, chosen[i]) for i in range(8) if chosen[i][0] != 'const']
    if non_const:
        cot.append(f"\nApply to {target}:")
        for i, func in non_const:
            refs = _get_input_refs(func)
            _, _, desc = func
            ref_str = ", ".join(f"in[{r}]={target_bits[r]}" for r in refs)
            comp = _substitute_values(desc, target_bits)
            cot.append(f"bit {i}: {ref_str} → {comp} = {result_bits[i]}")

    # ── ③ Conclusion ──
    cot.append(f"\nOutput: {answer}")
    return '\n'.join(cot)

# ═══════════════════════════════════════════════════════════════════════════════
#  EQ_NUMERIC — numeric equation transformation (digit-op-digit patterns)
# ═══════════════════════════════════════════════════════════════════════════════

_EQ_NUM_BASE_OPS = [
    ('add', lambda a, b: a + b),
    ('sub', lambda a, b: a - b),
    ('abs_diff', lambda a, b: abs(a - b)),
    ('mul', lambda a, b: a * b),
    ('mul_add1', lambda a, b: a * b + 1),
    ('mul_sub1', lambda a, b: a * b - 1),
    ('add_add1', lambda a, b: a + b + 1),
    ('add_sub1', lambda a, b: a + b - 1),
    ('mod', lambda a, b: max(a, b) % min(a, b) if min(a, b) != 0 else None),
]
_EQ_NUM_BASE_FUNC = dict(_EQ_NUM_BASE_OPS)
_EQ_NUM_FMTS = ['none', 'prefix', 'suffix', 'pos_prefix', 'pos_suffix']

_EQ_NUM_PLAIN_DESC = {
    'add': 'a+b', 'sub': 'a-b', 'abs_diff': '|a-b|', 'mul': 'a×b',
    'mul_add1': 'a×b+1', 'mul_sub1': 'a×b-1', 'add_add1': 'a+b+1', 'add_sub1': 'a+b-1',
    'mod': 'max(a,b) mod min(a,b)', 'concat': 'concatenation', 'rev_concat': 'reverse concatenation',
}
_EQ_NUM_REV_DESC = {
    'add': 'reverse-add (rev digits, add, rev result)',
    'sub': 'reverse-subtract (rev digits, subtract, rev result)',
    'abs_diff': 'reverse-|difference| (rev digits, |diff|, rev result)',
    'mul': 'reverse-multiply (rev digits, multiply, rev result)',
    'mul_add1': 'reverse-multiply-plus-one', 'mul_sub1': 'reverse-multiply-minus-one',
    'add_add1': 'reverse-add-plus-one', 'add_sub1': 'reverse-add-minus-one',
    'mod': 'reverse-modulo (rev digits, max mod min, rev result)',
}
_EQ_NUM_FMT_DESC = {
    'none': '', 'prefix': '; result prefixed with op symbol',
    'suffix': '; result suffixed with op symbol',
    'pos_prefix': '; positive results prefixed with op symbol',
    'pos_suffix': '; positive results suffixed with op symbol',
}


def _eq_num_rev_str(s):
    return s[::-1]


def _eq_num_compute_full(a, b, a_str, b_str, base_name, is_rev, op_char, fmt='none'):
    if base_name == 'concat':
        s = a_str + b_str
        if fmt == 'prefix': return op_char + s
        elif fmt == 'suffix': return s + op_char
        return s
    if base_name == 'rev_concat':
        s = b_str + a_str
        if fmt == 'prefix': return op_char + s
        elif fmt == 'suffix': return s + op_char
        return s
    func = _EQ_NUM_BASE_FUNC.get(base_name)
    if func is None:
        return None
    if is_rev:
        ra, rb = int(_eq_num_rev_str(a_str)), int(_eq_num_rev_str(b_str))
        val = func(ra, rb)
    else:
        val = func(a, b)
    if val is None:
        return None
    if is_rev:
        neg = val < 0
        abs_s = _eq_num_rev_str(str(abs(val)))
        if fmt == 'pos_prefix': return (op_char + abs_s) if (not neg and val != 0) else abs_s
        elif fmt == 'pos_suffix': return (abs_s + op_char) if (not neg and val != 0) else abs_s
        elif fmt == 'prefix': return op_char + abs_s
        elif fmt == 'suffix': return abs_s + op_char
        elif neg: return op_char + abs_s
        else: return abs_s
    else:
        if fmt == 'pos_prefix': return (op_char + str(val)) if val > 0 else str(abs(val))
        elif fmt == 'pos_suffix': return (str(val) + op_char) if val > 0 else str(abs(val))
        elif fmt == 'prefix': return op_char + str(abs(val))
        elif fmt == 'suffix': return str(abs(val)) + op_char
        elif val < 0: return op_char + str(abs(val))
        else: return str(val)


def _eq_num_check_eq(computed, expected):
    if computed is None:
        return False
    if computed == expected:
        return True
    c0 = computed.replace('-', '').lstrip('0') or '0'
    e0 = expected.replace('-', '').lstrip('0') or '0'
    if c0 == '0' and e0 == '0':
        return True
    c = computed.lstrip('0') or '0'
    e = expected.lstrip('0') or '0'
    if c == e:
        return True
    if len(expected) > len(computed):
        if computed.zfill(len(expected)) == expected:
            return True
    if len(computed) > len(expected):
        if expected.zfill(len(computed)) == computed:
            return True
    return False


def _build_eq_num_all_ops():
    ops = []
    for name, _ in _EQ_NUM_BASE_OPS:
        for is_rev in [False, True]:
            for fmt in _EQ_NUM_FMTS:
                tag = ('rev_' if is_rev else 'plain_') + name
                if fmt == 'prefix': tag += '_pfx'
                elif fmt == 'suffix': tag += '_sfx'
                elif fmt == 'pos_prefix': tag += '_pospfx'
                elif fmt == 'pos_suffix': tag += '_possfx'
                ops.append((tag, name, is_rev, fmt))
    for fmt in ['none', 'prefix', 'suffix']:
        sfx = '' if fmt == 'none' else ('_pfx' if fmt == 'prefix' else '_sfx')
        ops.append(('concat' + sfx, 'concat', False, fmt))
        ops.append(('rev_concat' + sfx, 'rev_concat', False, fmt))
    return ops


_EQ_NUM_ALL_OPS = _build_eq_num_all_ops()


def _eq_num_find_matching_ops(op_char, entries):
    matches = []
    for tag, base_name, is_rev, fmt in _EQ_NUM_ALL_OPS:
        if all(_eq_num_check_eq(
                _eq_num_compute_full(a, b, a_str, b_str, base_name, is_rev, op_char, fmt), rhs)
               for a, b, a_str, b_str, rhs in entries):
            matches.append((tag, base_name, is_rev, fmt))
    return matches


def _eq_num_gen_op_description(op_char, base_name, is_rev, fmt):
    fmt_note = _EQ_NUM_FMT_DESC.get(fmt, '')
    if base_name == 'concat':
        return f"'{op_char}' represents concatenation"
    if base_name == 'rev_concat':
        return f"'{op_char}' represents reverse concatenation"
    if is_rev:
        desc = _EQ_NUM_REV_DESC.get(base_name, base_name)
        return f"'{op_char}' applies {desc}{fmt_note}"
    else:
        formula = _EQ_NUM_PLAIN_DESC.get(base_name, base_name)
        return f"'{op_char}' represents {formula}{fmt_note}"


def _eq_num_gen_verify(op_char, base_name, is_rev, fmt, entry):
    a, b, a_str, b_str, rhs = entry
    if base_name == 'concat':
        return f"{a_str}{op_char}{b_str} = \"{a_str}\"+\"{b_str}\" = {rhs} \u2713"
    if base_name == 'rev_concat':
        return f"{a_str}{op_char}{b_str} = \"{b_str}\"+\"{a_str}\" = {rhs} \u2713"
    if is_rev:
        ra_s, rb_s = _eq_num_rev_str(a_str), _eq_num_rev_str(b_str)
        ra, rb = int(ra_s), int(rb_s)
        func = _EQ_NUM_BASE_FUNC[base_name]
        raw_val = func(ra, rb)
        steps = f"rev({a_str})={ra_s}, rev({b_str})={rb_s}"
        op_syms = {'add': '+', 'sub': '-', 'abs_diff': '|diff|', 'mul': '\u00d7',
                   'mul_add1': '\u00d7+1', 'mul_sub1': '\u00d7-1', 'add_add1': '++1',
                   'add_sub1': '+-1', 'mod': 'mod'}
        sym = op_syms.get(base_name, base_name)
        if base_name == 'abs_diff':
            steps += f", |{ra}-{rb}|={abs(ra-rb)}"
            raw_val = abs(ra - rb)
        elif base_name == 'mod':
            big, small = max(ra, rb), min(ra, rb)
            steps += f", {big} mod {small}={raw_val}"
        else:
            steps += f", {ra}{sym}{rb}={raw_val}"
        abs_val = abs(raw_val)
        rev_result = _eq_num_rev_str(str(abs_val))
        steps += f", rev({abs_val})={rev_result}"
        return f"{a_str}{op_char}{b_str}: {steps} = {rhs} \u2713"
    else:
        if base_name == 'abs_diff':
            expr = f"|{a}-{b}|={abs(a-b)}"
        elif base_name == 'mod':
            big, small = max(a, b), min(a, b)
            expr = f"{big} mod {small}={big % small}"
        else:
            op_syms = {'add': '+', 'sub': '-', 'mul': '\u00d7',
                       'mul_add1': '\u00d7{}+1', 'mul_sub1': '\u00d7{}-1',
                       'add_add1': '+{}+1', 'add_sub1': '+{}-1'}
            sym = op_syms.get(base_name, base_name)
            val = _EQ_NUM_BASE_FUNC[base_name](a, b)
            expr = f"{a}{sym}{b}={val}"
        return f"{a_str}{op_char}{b_str} = {expr} = {rhs} \u2713"


def _eq_num_gen_query_steps(a, b, a_str, b_str, op_char, base_name, is_rev, fmt, answer):
    if base_name == 'concat':
        return f"{a_str}{op_char}{b_str} = \"{a_str}\"+\"{b_str}\" = {answer}"
    if base_name == 'rev_concat':
        return f"{a_str}{op_char}{b_str} = \"{b_str}\"+\"{a_str}\" = {answer}"
    if is_rev:
        ra_s, rb_s = _eq_num_rev_str(a_str), _eq_num_rev_str(b_str)
        ra, rb = int(ra_s), int(rb_s)
        func = _EQ_NUM_BASE_FUNC[base_name]
        raw_val = func(ra, rb)
        steps = f"rev({a_str})={ra_s}, rev({b_str})={rb_s}"
        op_syms = {'add': '+', 'sub': '-', 'abs_diff': '|diff|', 'mul': '\u00d7',
                   'mul_add1': '\u00d7+1', 'mul_sub1': '\u00d7-1', 'add_add1': '++1',
                   'add_sub1': '+-1', 'mod': 'mod'}
        sym = op_syms.get(base_name, base_name)
        if base_name == 'abs_diff':
            steps += f", |{ra}-{rb}|={abs(ra-rb)}"
            raw_val = abs(ra - rb)
        elif base_name == 'mod':
            big, small = max(ra, rb), min(ra, rb)
            steps += f", {big} mod {small}={raw_val}"
        else:
            steps += f", {ra}{sym}{rb}={raw_val}"
        abs_val = abs(raw_val)
        rev_result = _eq_num_rev_str(str(abs_val))
        steps += f", rev({abs_val})={rev_result}"
        if fmt == 'pos_prefix':
            steps += f", positive \u2192 {op_char}{rev_result}" if (raw_val > 0) else f", non-positive \u2192 {rev_result}"
        elif fmt == 'pos_suffix':
            steps += f", positive \u2192 {rev_result}{op_char}" if (raw_val > 0) else f", non-positive \u2192 {rev_result}"
        elif fmt == 'prefix':
            steps += f" \u2192 {op_char}{rev_result}"
        elif fmt == 'suffix':
            steps += f" \u2192 {rev_result}{op_char}"
        return f"{a_str}{op_char}{b_str}: {steps} = {answer}"
    else:
        val = _EQ_NUM_BASE_FUNC[base_name](a, b)
        if base_name == 'abs_diff':
            expr = f"|{a}-{b}|={abs(a-b)}"
        elif base_name == 'mod':
            big, small = max(a, b), min(a, b)
            expr = f"{big} mod {small}={val}"
        else:
            op_syms = {'add': '+', 'sub': '-', 'mul': '\u00d7',
                       'mul_add1': '\u00d7{}+1', 'mul_sub1': '\u00d7{}-1',
                       'add_add1': '+{}+1', 'add_sub1': '+{}-1'}
            sym = op_syms.get(base_name, base_name)
            expr = f"{a}{sym}{b}={val}"
        if fmt == 'pos_prefix':
            expr += f", positive \u2192 {op_char}{val}" if val > 0 else f", non-positive \u2192 {abs(val)}"
        elif fmt == 'pos_suffix':
            expr += f", positive \u2192 {val}{op_char}" if val > 0 else f", non-positive \u2192 {abs(val)}"
        elif fmt == 'prefix':
            expr += f" \u2192 {op_char}{abs(val)}"
        elif fmt == 'suffix':
            expr += f" \u2192 {abs(val)}{op_char}"
        return f"{a_str}{op_char}{b_str} = {expr} = {answer}"


def gen_thinking_eq_numeric(prompt, gold):
    """Generate CoT for numeric equation transformation problems (digit-op-digit)."""
    lines = prompt.strip().split('\n')
    examples = []
    query_str = None
    for line in lines:
        line = line.strip()
        m = re.match(r'Now, determine the result for:\s*(.*)', line)
        if m:
            query_str = m.group(1).strip()
            continue
        if ' = ' in line and not line.startswith('In ') and not line.startswith('Now'):
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))

    if not examples or not query_str:
        return None

    # Group examples by operator character
    by_op = defaultdict(list)
    for lhs, rhs in examples:
        m = _NUM_EQ_PATTERN.match(lhs)
        if not m:
            return None
        a_str, op_char, b_str = m.group(1), m.group(2), m.group(3)
        by_op[op_char].append((int(a_str), int(b_str), a_str, b_str, rhs))

    # Find matching operations for each operator
    candidates = {}
    for oc, entries in by_op.items():
        ms = _eq_num_find_matching_ops(oc, entries)
        if not ms:
            return None
        candidates[oc] = ms

    # Parse query
    m = _NUM_EQ_PATTERN.match(query_str)
    if not m:
        return None
    a_str, q_op, b_str = m.group(1), m.group(2), m.group(3)
    a, b = int(a_str), int(b_str)

    # Find the op that produces the gold answer
    op_results = {}
    inferred_op = None

    if q_op in candidates:
        best = None
        for tag, bn, ir, fm in candidates[q_op]:
            pred = _eq_num_compute_full(a, b, a_str, b_str, bn, ir, q_op, fm)
            if pred and _eq_num_check_eq(pred, gold):
                best = (tag, bn, ir, fm)
                break
        if not best:
            return None
        for oc, ms in candidates.items():
            if oc == q_op:
                op_results[oc] = best
            else:
                op_results[oc] = ms[0]
    else:
        # Query uses a new operator not seen in examples
        for tag, bn, ir, fm in _EQ_NUM_ALL_OPS:
            pred = _eq_num_compute_full(a, b, a_str, b_str, bn, ir, q_op, fm)
            if pred and _eq_num_check_eq(pred, gold):
                inferred_op = (tag, bn, ir, fm)
                break
        if not inferred_op:
            return None
        op_results = {oc: ms[0] for oc, ms in candidates.items()}

    # Build compact CoT: ① Rules ② Execution ③ Conclusion
    # --- ① Rules: one line per operator ---
    rule_parts = []
    for oc, (tag, base_name, is_rev, fmt) in op_results.items():
        desc = _eq_num_gen_op_description(oc, base_name, is_rev, fmt)
        rule_parts.append(desc)

    cot = "Rules: " + "; ".join(rule_parts) + ".\n"

    # --- ② Execution on query ---
    if q_op in op_results:
        tag, base_name, is_rev, fmt = op_results[q_op]
        query_steps = _eq_num_gen_query_steps(a, b, a_str, b_str, q_op, base_name, is_rev, fmt, gold)
        cot += f"\n{query_steps}"
    elif inferred_op:
        tag, base_name, is_rev, fmt = inferred_op
        inf_desc = _eq_num_gen_op_description(q_op, base_name, is_rev, fmt)
        query_steps = _eq_num_gen_query_steps(a, b, a_str, b_str, q_op, base_name, is_rev, fmt, gold)
        cot += f"\nNew operator: {inf_desc}.\n{query_steps}"

    # --- ③ Conclusion ---
    cot += f"\n\nResult: {gold}"
    return cot


# ═══════════════════════════════════════════════════════════════════════════════
#  EQ_SYMBOLIC — brute-force rule matching for string/charwise equations
# ═══════════════════════════════════════════════════════════════════════════════
CHAR_BASE = 33
CHAR_RANGE = 94

OP_CHARS = set('+-*/|\\^&#}"`>[]{}?\'@!()$:%<~;')

def _split_by_op(expr):
    for i, c in enumerate(expr):
        if c in OP_CHARS and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i + 1:]
    return None

SYMBOL_OPS = [
    ('string concatenation', lambda l, r: l + r),
    ('reverse concatenation', lambda l, r: r + l),
]

SYMBOL_CW_OPS = [
    ('charwise addition mod 94', lambda a, b: chr(((ord(a) - CHAR_BASE) + (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('charwise subtraction mod 94', lambda a, b: chr(((ord(a) - CHAR_BASE) - (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('charwise reverse subtraction mod 94', lambda a, b: chr(((ord(b) - CHAR_BASE) - (ord(a) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('charwise XOR mod 94', lambda a, b: chr(((ord(a) - CHAR_BASE) ^ (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
    ('charwise multiplication mod 94', lambda a, b: chr(((ord(a) - CHAR_BASE) * (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)),
]

NUM_OPS = [
    ('addition', lambda a, b: str(a + b)),
    ('subtraction', lambda a, b: str(a - b)),
    ('reverse subtraction', lambda a, b: str(b - a)),
    ('multiplication', lambda a, b: str(a * b)),
    ('concatenation', lambda a, b: str(a) + str(b)),
    ('reverse concatenation', lambda a, b: str(b) + str(a)),
]


def gen_thinking_eq_symbolic(prompt, gold):
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

    if not examples or not query:
        return None

    query_split = _split_by_op(query)
    if not query_split:
        return None
    q_left, q_op, q_right = query_split

    # Parse all examples and group by operator
    op_groups = defaultdict(list)
    for lhs, rhs in examples:
        sp = _split_by_op(lhs)
        if sp:
            op_groups[sp[1]].append((sp[0], sp[2], rhs))

    if q_op not in op_groups:
        return None
    group = op_groups[q_op]

    # Try string-level operations
    for op_name, fn in SYMBOL_OPS:
        if all(fn(l, r) == res for l, r, res in group):
            try:
                answer = fn(q_left, q_right)
            except Exception:
                continue
            if answer == gold:
                cot = f"'{q_op}' = {op_name}.\n"
                cot += f"{q_left}{q_op}{q_right} = {answer}.\n"
                cot += f"\nResult: {answer}"
                return cot

    # Try charwise ops
    for op_name, fn in SYMBOL_CW_OPS:
        all_match = True
        for left, right, result in group:
            if len(left) != len(right):
                all_match = False
                break
            try:
                pred = ''.join(fn(a, b) for a, b in zip(left, right))
                if pred != result:
                    all_match = False
                    break
            except Exception:
                all_match = False
                break
        if all_match and len(q_left) == len(q_right):
            try:
                answer = ''.join(fn(a, b) for a, b in zip(q_left, q_right))
            except Exception:
                continue
            if answer == gold:
                cot = f"'{q_op}' = {op_name}.\n"
                pairs = ", ".join(f"'{a}' op '{b}'→'{r}'" for a, b, r in zip(q_left, q_right, answer))
                cot += f"{q_left}{q_op}{q_right}: {pairs}.\n"
                cot += f"\nResult: {answer}"
                return cot

    # Try numeric ops
    all_numeric = all(c.isdigit() for c in q_left) and all(c.isdigit() for c in q_right)
    if all_numeric:
        nl, nr = int(q_left), int(q_right)
        for nop_name, nop_fn in NUM_OPS:
            all_match = True
            for left, right, result in group:
                try:
                    if not all(c.isdigit() for c in left) or not all(c.isdigit() for c in right):
                        all_match = False
                        break
                    if nop_fn(int(left), int(right)) != result:
                        all_match = False
                        break
                except Exception:
                    all_match = False
                    break
            if all_match:
                try:
                    answer = nop_fn(nl, nr)
                except Exception:
                    continue
                if answer == gold:
                    cot = f"'{q_op}' = numeric {nop_name}.\n"
                    cot += f"{q_left}{q_op}{q_right} = {answer}.\n"
                    cot += f"\nResult: {answer}"
                    return cot

    # --- Fallback: base-N cryptarithmetic solver ---
    return _gen_thinking_eq_symbolic_base_n(examples, query, gold)


# ── base-N cryptarithmetic solver for eq_symbolic ──

# Inline the key solver pieces to avoid cross-module import issues at runtime.
# (mirrors logic from scripts/solve_eq_symbolic.py)

_BASE_N_OPS = [
    ('addition',          lambda a, b: a + b),
    ('subtraction',       lambda a, b: a - b),
    ('absolute difference', lambda a, b: abs(a - b)),
    ('multiplication',    lambda a, b: a * b),
    ('multiplication+1',  lambda a, b: a * b + 1),
    ('multiplication-1',  lambda a, b: a * b - 1),
    ('addition+1',        lambda a, b: a + b + 1),
    ('addition-1',        lambda a, b: a + b - 1),
    ('modulo',            lambda a, b: max(a, b) % min(a, b) if min(a, b) != 0 else None),
]

_BASE_N_FMTS = ['none', 'neg_prefix', 'prefix', 'pos_prefix']


def _base_n_decode(val, base):
    if val < 0:
        return None
    if val == 0:
        return (0,)
    digits = []
    while val > 0:
        digits.append(val % base)
        val //= base
    return tuple(reversed(digits))


def _base_n_extend(mapping, used, chars, digits):
    m = dict(mapping)
    u = set(used)
    for c, d in zip(chars, digits):
        if c in m:
            if m[c] != d:
                return None
        else:
            if d in u:
                return None
            m[c] = d
            u.add(d)
    return m, u


def _base_n_match_result(computed, result_chars, op_char, mapping, used, base):
    candidates = []
    if computed is None:
        return candidates
    abs_val = abs(computed)
    is_neg = computed < 0
    is_pos = computed > 0
    if computed >= 0:
        digits = _base_n_decode(computed, base)
        if digits is not None and len(digits) == len(result_chars):
            ext = _base_n_extend(mapping, used, result_chars, digits)
            if ext is not None:
                candidates.append(('none', ext[0], ext[1]))
    if is_neg:
        digits = _base_n_decode(abs_val, base)
        if digits is not None and len(digits) + 1 == len(result_chars) and result_chars[0] == op_char:
            ext = _base_n_extend(mapping, used, result_chars[1:], digits)
            if ext is not None:
                candidates.append(('neg_prefix', ext[0], ext[1]))
    if computed >= 0:
        digits = _base_n_decode(computed, base)
        if digits is not None and len(result_chars) >= 2:
            if len(digits) + 1 == len(result_chars) and result_chars[0] == op_char:
                ext = _base_n_extend(mapping, used, result_chars[1:], digits)
                if ext is not None:
                    candidates.append(('prefix', ext[0], ext[1]))
    val_for_encode = abs_val if is_neg else computed
    if val_for_encode is not None:
        digits = _base_n_decode(val_for_encode, base)
        if digits is not None:
            if is_pos and len(result_chars) >= 2:
                if len(digits) + 1 == len(result_chars) and result_chars[0] == op_char:
                    ext = _base_n_extend(mapping, used, result_chars[1:], digits)
                    if ext is not None and not any(c[0] == 'prefix' and c[1] == ext[0] for c in candidates):
                        candidates.append(('pos_prefix', ext[0], ext[1]))
            if (is_neg or computed == 0) and len(digits) == len(result_chars):
                ext = _base_n_extend(mapping, used, result_chars, digits)
                if ext is not None:
                    if not any(c[0] == 'pos_prefix' and c[1] == ext[0] for c in candidates):
                        candidates.append(('pos_prefix', ext[0], ext[1]))
    return candidates


def _base_n_greedy_sort(all_eqs):
    remaining = list(range(len(all_eqs)))
    known = set()
    ordered = []
    while remaining:
        best_idx, best_new = None, 999
        for i in remaining:
            left, right, result, op_char = all_eqs[i]
            n = sum(1 for c in set(left + right) if c not in known)
            if n < best_new:
                best_new, best_idx = n, i
        remaining.remove(best_idx)
        ordered.append(best_idx)
        left, right, result, op_char = all_eqs[best_idx]
        known.update(left); known.update(right); known.update(result)
    return [all_eqs[i] for i in ordered]


def _base_n_dfs(eqs, mapping, used, base, op_cands, fmt_cands, idx, deadline):
    if time.time() > deadline:
        return None
    if idx == len(eqs):
        return mapping
    left, right, result_chars, op_char = eqs[idx]
    new_chars = []
    for c in left + right:
        if c not in mapping and c not in new_chars:
            new_chars.append(c)

    def try_ops(m, u):
        for op_name, func in op_cands.get(op_char, _BASE_N_OPS):
            a = m[left[0]] * base + m[left[1]]
            b = m[right[0]] * base + m[right[1]]
            computed = func(a, b)
            if computed is None:
                continue
            matches = _base_n_match_result(computed, result_chars, op_char, m, u, base)
            allowed = fmt_cands.get(op_char)
            if allowed is not None:
                matches = [mm for mm in matches if mm[0] in allowed]
            for fmt, nm, nu in matches:
                nc = dict(op_cands); nc[op_char] = [(op_name, func)]
                nf = dict(fmt_cands); nf[op_char] = {fmt}
                r = _base_n_dfs(eqs, nm, nu, base, nc, nf, idx + 1, deadline)
                if r is not None:
                    return r
        return None

    if not new_chars:
        return try_ops(mapping, used)

    def enum(ci, m, u):
        if time.time() > deadline:
            return None
        if ci == len(new_chars):
            return try_ops(m, u)
        c = new_chars[ci]
        if c in m:
            return enum(ci + 1, m, u)
        for v in range(base):
            if v in u:
                continue
            nm = dict(m); nu = set(u)
            nm[c] = v; nu.add(v)
            r = enum(ci + 1, nm, nu)
            if r is not None:
                return r
        return None
    return enum(0, mapping, used)


def _base_n_solve(examples, query_str, gold):
    """Solve an eq_symbolic problem via base-N DFS. Returns solve info dict or None."""
    parsed = []
    for lhs, rhs in examples:
        if len(lhs) != 5:
            return None
        parsed.append(((lhs[0], lhs[1]), lhs[2], (lhs[3], lhs[4]), tuple(rhs)))

    # Add query+gold as extra equation
    if not query_str or len(query_str) != 5 or not gold:
        return None
    q_left = (query_str[0], query_str[1])
    q_op = query_str[2]
    q_right = (query_str[3], query_str[4])
    q_result = tuple(gold)

    op_groups = defaultdict(list)
    for left, op, right, result in parsed:
        op_groups[op].append((left, right, result))
    op_groups[q_op].append((q_left, q_right, q_result))

    concat_ops = {}
    calc_groups = {}
    for op, entries in op_groups.items():
        if all(l + r == res for l, r, res in entries):
            concat_ops[op] = 'concat'
        elif all(r + l == res for l, r, res in entries):
            concat_ops[op] = 'rev_concat'
        else:
            calc_groups[op] = entries

    if not calc_groups:
        # pure concat — already handled by main function
        return None

    val_chars = set()
    for op, entries in calc_groups.items():
        for l, r, res in entries:
            val_chars.update(l); val_chars.update(r); val_chars.update(res)
    n_vc = len(val_chars)
    if n_vc > 16:
        return None

    all_eqs_raw = []
    for oc, entries in sorted(calc_groups.items(), key=lambda x: -len(x[1])):
        for l, r, res in entries:
            all_eqs_raw.append((l, r, res, oc))
    all_eqs = _base_n_greedy_sort(all_eqs_raw)

    global_deadline = time.time() + 3.0          # 整题 3 秒上限
    for base in range(max(n_vc, 2), n_vc + 4):
        if time.time() > global_deadline:
            break
        per_base = min(time.time() + 1.0, global_deadline)   # 每 base 1 秒
        result = _base_n_dfs(all_eqs, {}, set(), base, {}, {}, 0, per_base)
        if result is None:
            continue
        mapping = result
        # Determine op and fmt for each calc operator
        op_assignments = {}
        fmt_assignments = {}
        ok = True
        for oc, entries in calc_groups.items():
            found = False
            for op_name, func in _BASE_N_OPS:
                for fmt in _BASE_N_FMTS:
                    all_match = True
                    for l, r, rc in entries:
                        a = mapping[l[0]] * base + mapping[l[1]]
                        b = mapping[r[0]] * base + mapping[r[1]]
                        computed = func(a, b)
                        if computed is None:
                            all_match = False; break
                        ml = _base_n_match_result(computed, rc, oc, mapping, set(mapping.values()), base)
                        if not any(m[0] == fmt for m in ml):
                            all_match = False; break
                    if all_match:
                        op_assignments[oc] = op_name
                        fmt_assignments[oc] = fmt
                        found = True; break
                if found:
                    break
            if not found:
                ok = False; break
        if not ok:
            continue

        # Compute answer and verify
        d2s = {d: s for s, d in mapping.items()}
        if q_op in concat_ops:
            if concat_ops[q_op] == 'concat':
                pred = query_str[:2] + query_str[3:5]
            else:
                pred = query_str[3:5] + query_str[:2]
        elif q_op in op_assignments:
            aop = op_assignments[q_op]
            afmt = fmt_assignments[q_op]
            afunc = dict(_BASE_N_OPS)[aop]
            a = mapping[q_left[0]] * base + mapping[q_left[1]]
            b = mapping[q_right[0]] * base + mapping[q_right[1]]
            computed = afunc(a, b)
            if computed is None:
                continue
            abs_v = abs(computed)
            if afmt == 'none':
                digs = _base_n_decode(computed, base)
            elif afmt == 'neg_prefix':
                digs = _base_n_decode(abs_v, base)
            elif afmt == 'prefix':
                digs = _base_n_decode(computed, base)
            elif afmt == 'pos_prefix':
                digs = _base_n_decode(abs_v if computed <= 0 else computed, base)
            else:
                continue
            if digs is None:
                continue
            sym = ''.join(d2s.get(d, '?') for d in digs)
            if afmt in ('neg_prefix', 'prefix', 'pos_prefix') and computed != 0:
                if afmt == 'pos_prefix' and computed <= 0:
                    pred = sym
                else:
                    pred = q_op + sym
            else:
                pred = sym
        else:
            continue

        if pred != gold or '?' in pred:
            continue

        return {
            'mapping': mapping,
            'base': base,
            'op_assignments': op_assignments,
            'fmt_assignments': fmt_assignments,
            'concat_ops': concat_ops,
            'd2s': d2s,
        }
    return None


def _gen_thinking_eq_symbolic_base_n(examples, query_str, gold):
    """Generate compact CoT for eq_symbolic using base-N cryptarithmetic solver."""
    info = _base_n_solve(examples, query_str, gold)
    if info is None:
        return None

    mapping = info['mapping']
    base = info['base']
    op_assignments = info['op_assignments']
    fmt_assignments = info['fmt_assignments']
    concat_ops = info['concat_ops']
    d2s = info['d2s']

    # ── ① Rules: base, mapping, operators ──
    sorted_map = sorted(mapping.items(), key=lambda x: x[1])
    map_str = ", ".join(f"'{s}'={d}" for s, d in sorted_map)

    op_descs = []
    for oc, ctype in concat_ops.items():
        op_descs.append(f"'{oc}'={ctype}")
    for oc, oname in op_assignments.items():
        fmt = fmt_assignments.get(oc, 'none')
        desc = f"'{oc}'={oname}"
        if fmt != 'none':
            desc += f" ({fmt})"
        op_descs.append(desc)

    cot = f"Base-{base} system. Mapping: {map_str}.\n"
    cot += "; ".join(op_descs) + ".\n"

    # ── ② Execution on query ──
    q_left = query_str[:2]
    q_op = query_str[2]
    q_right = query_str[3:5]

    if q_op in concat_ops:
        if concat_ops[q_op] == 'concat':
            cot += f"\n{query_str}: concat({q_left},{q_right}) = {gold}."
        else:
            cot += f"\n{query_str}: rev_concat({q_left},{q_right}) = {gold}."
    elif q_op in op_assignments:
        a_val = mapping[q_left[0]]
        b_val = mapping[q_left[1]]
        c_val = mapping[q_right[0]]
        d_val = mapping[q_right[1]]
        left_num = a_val * base + b_val
        right_num = c_val * base + d_val
        oname = op_assignments[q_op]
        afunc = dict(_BASE_N_OPS)[oname]
        computed = afunc(left_num, right_num)
        fmt = fmt_assignments.get(q_op, 'none')

        cot += f"\n{query_str}: {q_left}={left_num}, {q_right}={right_num}, {oname}({left_num},{right_num})={computed}"

        # Encode back
        abs_v = abs(computed)
        if fmt == 'none':
            digs = _base_n_decode(computed, base)
        elif fmt == 'neg_prefix':
            digs = _base_n_decode(abs_v, base)
            cot += f", negative→prefix '{q_op}'"
        elif fmt == 'prefix':
            digs = _base_n_decode(computed, base)
            cot += f", prefix '{q_op}'"
        elif fmt == 'pos_prefix':
            if computed <= 0:
                digs = _base_n_decode(abs_v, base)
            else:
                digs = _base_n_decode(computed, base)
                cot += f", positive→prefix '{q_op}'"

        dig_strs = [f"{d}→'{d2s[d]}'" for d in digs]
        cot += f", encode: {' '.join(dig_strs)} → {gold}."

    # ── ③ Conclusion ──
    cot += f"\n\nResult: {gold}"
    return cot


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC QUESTION GENERATORS — 造题填充不足的类型
# ═══════════════════════════════════════════════════════════════════════════════

# --- Synthetic symbol ---
# Real prompts: multi-operator, 3-5 examples, each operator maps to a different operation
# We generate the same format with known operations so CoT is correct.

SYNTH_SYMBOL_ALL_OPS = [
    # (name, is_numeric, fn_for_operands)
    ('string concatenation', False, lambda l, r: l + r),
    ('reverse concatenation', False, lambda l, r: r + l),
    ('charwise addition mod 94', False, None),  # special handling
    ('charwise subtraction mod 94', False, None),
    ('numeric addition', True, lambda a, b: str(a + b)),
    ('numeric subtraction', True, lambda a, b: str(a - b)),
    ('numeric multiplication', True, lambda a, b: str(a * b)),
    ('numeric reverse subtraction', True, lambda a, b: str(b - a)),
    ('numeric concatenation', True, lambda a, b: str(a) + str(b)),
    ('numeric reverse concatenation', True, lambda a, b: str(b) + str(a)),
]

# All possible operator characters (matching real data)
SYNTH_OP_POOL = list('+-*/|\\^&') + list('#$%`"<>~!?:;')


def _cw_add(a, b):
    return chr(((ord(a) - CHAR_BASE) + (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)


def _cw_sub(a, b):
    return chr(((ord(a) - CHAR_BASE) - (ord(b) - CHAR_BASE)) % CHAR_RANGE + CHAR_BASE)


def _rand_ascii_str(length):
    """Random printable ASCII string (chars 33-126), avoiding = and operator chars."""
    avoid = set('=' + ''.join(SYNTH_OP_POOL))
    pool = [chr(c) for c in range(33, 127) if chr(c) not in avoid]
    return ''.join(random.choice(pool) for _ in range(length))


def _rand_num_str():
    """Random 2-digit numeric string (01-99)."""
    return f"{random.randint(0, 99):02d}"


def generate_synthetic_symbol(rng, seed_id):
    """Generate one synthetic symbol question with prompt, answer, thinking, id."""
    # Decide: numeric or ascii operands
    is_numeric = rng.random() < 0.5

    # Pick 2-4 operators
    n_ops = rng.randint(2, 4)
    ops = rng.sample(SYNTH_OP_POOL, n_ops)

    # Assign each operator an operation
    if is_numeric:
        available = [(n, fn) for n, is_num, fn in SYNTH_SYMBOL_ALL_OPS if is_num and fn]
    else:
        available = [
            ('string concatenation', lambda l, r: l + r),
            ('reverse concatenation', lambda l, r: r + l),
            ('charwise addition mod 94', None),
            ('charwise subtraction mod 94', None),
        ]

    assignments = {}
    chosen_ops = rng.sample(available, min(n_ops, len(available)))
    # If not enough distinct ops, allow repeats
    while len(chosen_ops) < n_ops:
        chosen_ops.append(rng.choice(available))
    rng.shuffle(chosen_ops)

    for i, op_char in enumerate(ops):
        assignments[op_char] = chosen_ops[i]

    # Generate examples: 1-2 per operator
    examples = []
    for op_char in ops:
        op_name, fn = assignments[op_char]
        n_ex = rng.randint(1, 2)
        for _ in range(n_ex):
            if is_numeric:
                left = _rand_num_str()
                right = _rand_num_str()
                result = fn(int(left), int(right))
            else:
                length = rng.randint(2, 4)
                left = _rand_ascii_str(length)
                right = _rand_ascii_str(length)
                if 'charwise addition' in op_name:
                    result = ''.join(_cw_add(a, b) for a, b in zip(left, right))
                elif 'charwise subtraction' in op_name:
                    result = ''.join(_cw_sub(a, b) for a, b in zip(left, right))
                elif fn:
                    result = fn(left, right)
                else:
                    continue
            examples.append((f"{left}{op_char}{right}", str(result)))

    rng.shuffle(examples)

    # Pick query operator
    query_op = rng.choice(ops)
    q_op_name, q_fn = assignments[query_op]

    if is_numeric:
        q_left = _rand_num_str()
        q_right = _rand_num_str()
        answer = q_fn(int(q_left), int(q_right))
    else:
        q_len = rng.randint(2, 4)
        q_left = _rand_ascii_str(q_len)
        q_right = _rand_ascii_str(q_len)
        if 'charwise addition' in q_op_name:
            answer = ''.join(_cw_add(a, b) for a, b in zip(q_left, q_right))
        elif 'charwise subtraction' in q_op_name:
            answer = ''.join(_cw_sub(a, b) for a, b in zip(q_left, q_right))
        elif q_fn:
            answer = q_fn(q_left, q_right)
        else:
            answer = q_left + q_right  # fallback

    answer = str(answer)
    query = f"{q_left}{query_op}{q_right}"

    # Build prompt (matching real format exactly)
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:",
    ]
    for lhs, rhs in examples:
        lines.append(f"{lhs} = {rhs}")
    lines.append(f"Now, determine the result for: {query}")
    prompt = '\n'.join(lines)

    # Build thinking
    if is_numeric:
        ex_for_op = [(l, r, res) for lhs, res in examples
                     for l, op, r in [(_split_by_op(lhs),)] if op == query_op
                     ] if False else []
        # Simpler: just state the rule
        same_op_exs = [(lhs, res) for lhs, res in examples if query_op in lhs]
        cot = f"The operator '{query_op}' maps to numeric {q_op_name}.\n"
        if same_op_exs:
            ex_lhs, ex_res = same_op_exs[0]
            cot += f"Example: {ex_lhs} = {ex_res}\n"
        cot += f"Applying: {q_left}{query_op}{q_right} = {answer}"
    else:
        if 'charwise' in q_op_name:
            cot = f"The operator '{query_op}' maps to {q_op_name}.\n"
            cot += f"\nApplying char by char to {q_left}{query_op}{q_right}:\n"
            for a, b, r in zip(q_left, q_right, answer):
                cot += f"  '{a}' op '{b}' → '{r}'\n"
            cot += f"\nResult: {answer}"
        else:
            same_op_exs = [(lhs, res) for lhs, res in examples if query_op in lhs]
            cot = f"The operator '{query_op}' maps to {q_op_name}.\n"
            if same_op_exs:
                ex_lhs, ex_res = same_op_exs[0]
                cot += f"Example: {ex_lhs} = {ex_res}\n"
            cot += f"Applying: {q_left}{query_op}{q_right} = {answer}"

    # Generate deterministic ID
    raw_id = hashlib.md5(f"synth_symbol_{seed_id}".encode()).hexdigest()[:8]
    return {
        'id': f"synth_{raw_id}",
        'prompt': prompt,
        'answer': answer,
        'thinking': cot,
        'type': 'eq_symbolic',
    }


# --- Synthetic bit_ops ---
# Generate 8-bit transformation with known per-bit boolean rules

SYNTH_BIT_RULES = [
    ('copy', lambda bits, args: bits[args[0]]),
    ('NOT', lambda bits, args: 1 - bits[args[0]]),
    ('XOR', lambda bits, args: bits[args[0]] ^ bits[args[1]]),
    ('AND', lambda bits, args: bits[args[0]] & bits[args[1]]),
    ('OR', lambda bits, args: bits[args[0]] | bits[args[1]]),
    ('NAND', lambda bits, args: 1 - (bits[args[0]] & bits[args[1]])),
    ('const0', lambda bits, args: 0),
    ('const1', lambda bits, args: 1),
]


def _describe_bit_rule(name, args):
    if name == 'copy':
        return f"in[{args[0]}]"
    elif name == 'NOT':
        return f"NOT in[{args[0]}]"
    elif name in ('XOR', 'AND', 'OR', 'NAND'):
        return f"in[{args[0]}] {name} in[{args[1]}]"
    elif name == 'const0':
        return '0'
    elif name == 'const1':
        return '1'
    return '?'


def generate_synthetic_bit_ops(rng, seed_id):
    """Generate one synthetic bit_ops question."""
    # Pick 8 random rules (one per output bit)
    rules = []
    for obit in range(8):
        rule_type = rng.choice(SYNTH_BIT_RULES)
        name = rule_type[0]
        fn = rule_type[1]
        if name in ('copy', 'NOT'):
            args = [rng.randint(0, 7)]
        elif name in ('XOR', 'AND', 'OR', 'NAND'):
            a, b = sorted(rng.sample(range(8), 2))
            args = [a, b]
        else:
            args = []
        rules.append((name, fn, args))

    # Generate 5 random input-output examples
    examples = []
    seen = set()
    while len(examples) < 5:
        inp = rng.randint(0, 255)
        if inp in seen:
            continue
        seen.add(inp)
        inp_bits = [(inp >> (7 - i)) & 1 for i in range(8)]
        out_bits = [fn(inp_bits, args) for name, fn, args in rules]
        inp_str = ''.join(str(b) for b in inp_bits)
        out_str = ''.join(str(b) for b in out_bits)
        examples.append((inp_str, out_str))

    # Generate query
    while True:
        q = rng.randint(0, 255)
        if q not in seen:
            break
    q_bits = [(q >> (7 - i)) & 1 for i in range(8)]
    a_bits = [fn(q_bits, args) for name, fn, args in rules]
    target = ''.join(str(b) for b in q_bits)
    answer = ''.join(str(b) for b in a_bits)

    # Build prompt
    lines = [
        "In Alice's Wonderland, a secret bit manipulation rule is applied to 8-bit binary strings. Below are a few examples:",
    ]
    for inp_s, out_s in examples:
        lines.append(f"{inp_s} -> {out_s}")
    lines.append(f"Now, determine the output for: {target}")
    prompt = '\n'.join(lines)

    # Build thinking
    cot = ["Analyzing the 8-bit transformation rule per output bit:"]
    cot.append("")
    for i in range(8):
        name, fn, args = rules[i]
        cot.append(f"  bit {i}: {_describe_bit_rule(name, args)}")
    cot.append("")
    cot.append(f"Applying to {target}:")
    for i in range(8):
        name, fn, args = rules[i]
        cot.append(f"  bit {i}: {_describe_bit_rule(name, args)} → {a_bits[i]}")
    cot.append("")
    cot.append(f"Result: {answer}")

    raw_id = hashlib.md5(f"synth_bitops_{seed_id}".encode()).hexdigest()[:8]
    return {
        'id': f"synth_{raw_id}",
        'prompt': prompt,
        'answer': answer,
        'thinking': '\n'.join(cot),
        'type': 'bit_ops',
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Generate step-by-step thinking for competition data')
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT), help='Input CSV (default: train.csv)')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT), help='Output CSV')
    parser.add_argument('--sample', type=int, default=0, help='Only print N samples per type (no file output)')
    parser.add_argument('--balanced', type=int, default=0, help='Balance each type to exactly N (downsample or synthesize)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for balanced mode')
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    rows = list(csv.DictReader(open(args.input, encoding='utf-8')))
    print(f"Total rows: {len(rows)}")

    solvers = {
        'numeral': lambda p, g: gen_thinking_numeral(p, g),
        'gravity': lambda p, g: gen_thinking_gravity(p, g),
        'unit_conv': lambda p, g: gen_thinking_unit(p, g),
        'cipher': lambda p, g: gen_thinking_cipher(p, g),
        'bit_ops': lambda p, g: gen_thinking_bit(p, g),
        'eq_numeric': lambda p, g: gen_thinking_eq_numeric(p, g),
        'eq_symbolic': lambda p, g: gen_thinking_eq_symbolic(p, g),
    }

    stats = defaultdict(lambda: {'total': 0, 'solved': 0})
    records_by_type = defaultdict(list)

    for row in rows:
        prompt = row['prompt']
        gold = row['answer'].strip()
        qtype = classify(prompt)

        if qtype not in solvers:
            continue

        stats[qtype]['total'] += 1
        thinking = solvers[qtype](prompt, gold)

        if thinking is not None:
            stats[qtype]['solved'] += 1
            records_by_type[qtype].append({
                'id': row['id'],
                'prompt': prompt,
                'answer': gold,
                'thinking': thinking,
                'type': qtype,
            })

    # Print stats
    print(f"\n{'Type':<12} {'Total':>6} {'Solved':>7} {'Rate':>7}")
    print("-" * 35)
    total_all = solved_all = 0
    for qtype in sorted(stats):
        s = stats[qtype]
        rate = s['solved'] / s['total'] * 100 if s['total'] else 0
        print(f"{qtype:<12} {s['total']:>6} {s['solved']:>7} {rate:>6.1f}%")
        total_all += s['total']
        solved_all += s['solved']
    print("-" * 35)
    print(f"{'TOTAL':<12} {total_all:>6} {solved_all:>7} {solved_all/total_all*100:>6.1f}%")

    # Balanced mode: downsample or synthesize to exactly N per type
    if args.balanced > 0:
        target_n = args.balanced
        rng = random.Random(args.seed)
        print(f"\n--- Balancing to {target_n} per type ---")

        synth_generators = {
            'eq_symbolic': generate_synthetic_symbol,
            'bit_ops': generate_synthetic_bit_ops,
        }

        final_records = []
        for qtype in ['numeral', 'unit_conv', 'gravity', 'cipher', 'bit_ops', 'eq_numeric', 'eq_symbolic']:
            items = records_by_type.get(qtype, [])
            current = len(items)

            if current >= target_n:
                # Downsample
                sampled = rng.sample(items, target_n)
                final_records.extend(sampled)
                print(f"  {qtype:<12} {current:>5} → {target_n} (downsampled)")
            elif qtype in synth_generators:
                # Use all real + synthesize the rest
                final_records.extend(items)
                deficit = target_n - current
                gen_fn = synth_generators[qtype]
                for i in range(deficit):
                    synth = gen_fn(rng, i)
                    final_records.append(synth)
                print(f"  {qtype:<12} {current:>5} → {target_n} ({current} real + {deficit} synthetic)")
            else:
                # No generator available, just use what we have
                final_records.extend(items)
                print(f"  {qtype:<12} {current:>5} → {current} (no generator, kept as-is)")

        records = final_records
        print(f"\nTotal balanced: {len(records)}")
    else:
        # Per-type target counts with oversampling
        type_targets = {
            'gravity': 400, 'numeral': 400, 'unit_conv': 400,
            'bit_ops': 800, 'cipher': 800, 'eq_numeric': 800, 'eq_symbolic': 800,
        }
        rng = random.Random(42)
        records = []
        print(f"\n--- Sampling with per-type targets ---")
        for qtype in ['numeral', 'unit_conv', 'gravity', 'cipher', 'bit_ops', 'eq_numeric', 'eq_symbolic']:
            items = records_by_type.get(qtype, [])
            target = type_targets.get(qtype, len(items))
            current = len(items)
            if current == 0:
                print(f"  {qtype:<12} {current:>5} → 0 (no data)")
                continue
            if current >= target:
                sampled = rng.sample(items, target)
                records.extend(sampled)
                print(f"  {qtype:<12} {current:>5} → {target} (downsampled)")
            else:
                # oversample: keep all + duplicate to fill
                pool = list(items)
                rng.shuffle(pool)
                extra = [pool[i % current] for i in range(target - current)]
                records.extend(items + extra)
                print(f"  {qtype:<12} {current:>5} → {target} (oversampled {target - current}x)")
        print(f"\nTotal: {len(records)}")

    # Sample mode: just show examples
    if args.sample > 0:
        print(f"\n{'='*60}")
        print(f"Sample outputs ({args.sample} per type):")
        print(f"{'='*60}")
        by_type = defaultdict(list)
        for r in records:
            by_type[r['type']].append(r)
        for qtype in ['numeral', 'unit_conv', 'gravity', 'cipher', 'bit_ops', 'eq_numeric', 'eq_symbolic']:
            items = by_type.get(qtype, [])
            print(f"\n--- {qtype} ({len(items)} total) ---")
            # Show last ones (synthetic) if balanced
            show = items[-args.sample:] if args.balanced > 0 else items[:args.sample]
            for r in show:
                print(f"\n[{r['id']}] answer: {r['answer']}")
                print(f"thinking:\n{r['thinking']}")
        return

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking', 'type'])
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == '__main__':
    main()