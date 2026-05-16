#!/usr/bin/env python3
"""
Generate balanced synthetic bit_ops training data.

Problem: Original training data has severely imbalanced operation distribution:
  COPY=45.1%, NOR=0.5%, NAND=1.5%, XNOR=7%, 3-input=1.5%

This script generates synthetic problems with controlled operation frequencies,
boosting underrepresented operations (NOR, NAND, XNOR, NOT_AND, NOT_OR,
3-input combos like AND_XOR, OR_XOR, XOR3, MAJ3, etc.)

Output format matches sft_thinking.csv: id, prompt, answer, thinking, type
"""

import csv
import hashlib
import random
import re
import sys
import os
from collections import Counter, defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
#  ALL SUPPORTED BIT OPERATIONS (matching _enumerate_bit_functions in gen_thinking.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry: (internal_name, display_desc_fn, eval_fn, n_args, category)
# n_args: 0=constant, 1=single, 2=pair, 3=triple

def _eval_copy(bits, args): return bits[args[0]]
def _eval_not(bits, args): return 1 - bits[args[0]]
def _eval_xor2(bits, args): return bits[args[0]] ^ bits[args[1]]
def _eval_xnor2(bits, args): return 1 - (bits[args[0]] ^ bits[args[1]])
def _eval_and(bits, args): return bits[args[0]] & bits[args[1]]
def _eval_or(bits, args): return bits[args[0]] | bits[args[1]]
def _eval_nand(bits, args): return 1 - (bits[args[0]] & bits[args[1]])
def _eval_nor(bits, args): return 1 - (bits[args[0]] | bits[args[1]])
def _eval_not_and(bits, args): return (1 - bits[args[0]]) & bits[args[1]]
def _eval_not_or(bits, args): return (1 - bits[args[0]]) | bits[args[1]]
def _eval_xor3(bits, args): return bits[args[0]] ^ bits[args[1]] ^ bits[args[2]]
def _eval_and_xor(bits, args): return (bits[args[0]] & bits[args[1]]) ^ bits[args[2]]
def _eval_or_xor(bits, args): return (bits[args[0]] | bits[args[1]]) ^ bits[args[2]]
def _eval_and3(bits, args): return bits[args[0]] & bits[args[1]] & bits[args[2]]
def _eval_or3(bits, args): return bits[args[0]] | bits[args[1]] | bits[args[2]]
def _eval_nand3(bits, args): return 1 - (bits[args[0]] & bits[args[1]] & bits[args[2]])
def _eval_nor3(bits, args): return 1 - (bits[args[0]] | bits[args[1]] | bits[args[2]])
def _eval_maj3(bits, args): return 1 if (bits[args[0]] + bits[args[1]] + bits[args[2]]) >= 2 else 0
def _eval_xor_and(bits, args): return (bits[args[0]] ^ bits[args[1]]) & bits[args[2]]
def _eval_xor_or(bits, args): return (bits[args[0]] ^ bits[args[1]]) | bits[args[2]]
def _eval_and_or(bits, args): return (bits[args[0]] & bits[args[1]]) | bits[args[2]]
def _eval_or_and(bits, args): return (bits[args[0]] | bits[args[1]]) & bits[args[2]]
def _eval_const0(bits, args): return 0
def _eval_const1(bits, args): return 1


def _desc_copy(args): return f"in[{args[0]}]"
def _desc_not(args): return f"NOT(in[{args[0]}])"
def _desc_xor2(args): return f"in[{args[0]}] XOR in[{args[1]}]"
def _desc_xnor2(args): return f"NOT(in[{args[0]}] XOR in[{args[1]}])"
def _desc_and(args): return f"in[{args[0]}] AND in[{args[1]}]"
def _desc_or(args): return f"in[{args[0]}] OR in[{args[1]}]"
def _desc_nand(args): return f"NOT(in[{args[0]}] AND in[{args[1]}])"
def _desc_nor(args): return f"NOT(in[{args[0]}] OR in[{args[1]}])"
def _desc_not_and(args): return f"NOT(in[{args[0]}]) AND in[{args[1]}]"
def _desc_not_or(args): return f"NOT(in[{args[0]}]) OR in[{args[1]}]"
def _desc_xor3(args): return f"in[{args[0]}] XOR in[{args[1]}] XOR in[{args[2]}]"
def _desc_and_xor(args): return f"(in[{args[0]}] AND in[{args[1]}]) XOR in[{args[2]}]"
def _desc_or_xor(args): return f"(in[{args[0]}] OR in[{args[1]}]) XOR in[{args[2]}]"
def _desc_and3(args): return f"in[{args[0]}] AND in[{args[1]}] AND in[{args[2]}]"
def _desc_or3(args): return f"in[{args[0]}] OR in[{args[1]}] OR in[{args[2]}]"
def _desc_nand3(args): return f"NOT(in[{args[0]}] AND in[{args[1]}] AND in[{args[2]}])"
def _desc_nor3(args): return f"NOT(in[{args[0]}] OR in[{args[1]}] OR in[{args[2]}])"
def _desc_maj3(args): return f"MAJ(in[{args[0]}],in[{args[1]}],in[{args[2]}])"
def _desc_xor_and(args): return f"(in[{args[0]}] XOR in[{args[1]}]) AND in[{args[2]}]"
def _desc_xor_or(args): return f"(in[{args[0]}] XOR in[{args[1]}]) OR in[{args[2]}]"
def _desc_and_or(args): return f"(in[{args[0]}] AND in[{args[1]}]) OR in[{args[2]}]"
def _desc_or_and(args): return f"(in[{args[0]}] OR in[{args[1]}]) AND in[{args[2]}]"
def _desc_const0(args): return "0"
def _desc_const1(args): return "1"


# Operation registry: (name, eval_fn, desc_fn, n_inputs, category)
ALL_OPS = [
    # Constants (0-input)
    ('const0', _eval_const0, _desc_const0, 0, 'constant'),
    ('const1', _eval_const1, _desc_const1, 0, 'constant'),
    # 1-input
    ('copy',   _eval_copy,   _desc_copy,   1, '1-input'),
    ('NOT',    _eval_not,    _desc_not,    1, '1-input'),
    # 2-input symmetric
    ('XOR',    _eval_xor2,   _desc_xor2,   2, '2-input'),
    ('XNOR',   _eval_xnor2,  _desc_xnor2,  2, '2-input'),
    ('AND',    _eval_and,    _desc_and,    2, '2-input'),
    ('OR',     _eval_or,     _desc_or,     2, '2-input'),
    ('NAND',   _eval_nand,   _desc_nand,   2, '2-input'),
    ('NOR',    _eval_nor,    _desc_nor,    2, '2-input'),
    # 2-input asymmetric
    ('NOT_AND', _eval_not_and, _desc_not_and, 2, '2-input-asym'),
    ('NOT_OR',  _eval_not_or,  _desc_not_or,  2, '2-input-asym'),
    # 3-input
    ('XOR3',    _eval_xor3,    _desc_xor3,    3, '3-input'),
    ('AND_XOR', _eval_and_xor, _desc_and_xor, 3, '3-input'),
    ('OR_XOR',  _eval_or_xor,  _desc_or_xor,  3, '3-input'),
    ('AND3',    _eval_and3,    _desc_and3,    3, '3-input'),
    ('OR3',     _eval_or3,     _desc_or3,     3, '3-input'),
    ('NAND3',   _eval_nand3,   _desc_nand3,   3, '3-input'),
    ('NOR3',    _eval_nor3,    _desc_nor3,    3, '3-input'),
    ('MAJ3',    _eval_maj3,    _desc_maj3,    3, '3-input'),
    ('XOR_AND', _eval_xor_and, _desc_xor_and, 3, '3-input'),
    ('XOR_OR',  _eval_xor_or,  _desc_xor_or,  3, '3-input'),
    ('AND_OR',  _eval_and_or,  _desc_and_or,  3, '3-input'),
    ('OR_AND',  _eval_or_and,  _desc_or_and,  3, '3-input'),
]

OP_BY_NAME = {op[0]: op for op in ALL_OPS}

# ═══════════════════════════════════════════════════════════════════════════════
#  TARGET DISTRIBUTION — boost underrepresented ops
# ═══════════════════════════════════════════════════════════════════════════════
# Original distribution had COPY at 45%, NOR at 0.5%, etc.
# We want roughly uniform per-category with slight emphasis on what test might have.

# Weights for selecting operation type per bit rule
# Higher = more likely to be selected
# V2: Aligned to test distribution — COPY~35%, XNOR boosted, 3-input reduced
OP_WEIGHTS = {
    'const0':   3,
    'const1':   2,
    'copy':     15,   # V2: test dist has ~44% COPY, bring closer
    'NOT':      8,
    'XOR':      10,
    'XNOR':     14,   # V2: boosted — 2/4 V115 failures were XNOR
    'AND':      10,
    'OR':       8,
    'NAND':     6,    # V2: reduced — was over-represented (3.5% vs test 1.5%)
    'NOR':      6,    # V2: reduced — was over-represented (3.1% vs test 0.5%)
    'NOT_AND':  5,
    'NOT_OR':   5,
    'XOR3':     2,
    'AND_XOR':  2,
    'OR_XOR':   2,
    'AND3':     1,
    'OR3':      1,
    'NAND3':    1,
    'NOR3':     1,
    'MAJ3':     1,
    'XOR_AND':  1,
    'XOR_OR':   1,
    'AND_OR':   1,
    'OR_AND':   1,
}


def _pick_args(rng, n_inputs):
    """Pick random input bit indices for an operation."""
    if n_inputs == 0:
        return []
    elif n_inputs == 1:
        return [rng.randint(0, 7)]
    elif n_inputs == 2:
        a, b = rng.sample(range(8), 2)
        return sorted([a, b])
    else:  # 3
        return sorted(rng.sample(range(8), 3))


def _pick_args_asym(rng, n_inputs):
    """Pick args for asymmetric ops (NOT_AND, NOT_OR) — order matters."""
    if n_inputs == 2:
        a = rng.randint(0, 7)
        b = rng.choice([x for x in range(8) if x != a])
        return [a, b]  # NOT(in[a]) OP in[b] — a is negated
    return _pick_args(rng, n_inputs)


def _find_near_misses_synth(examples, n_examples, obit, correct_rule, max_nm=2):
    """Find near-miss candidates for synthetic problem: same op type, different indices.

    Returns list of (desc_str, computed_col, match_count).
    """
    name, eval_fn, desc_fn, args = correct_rule
    n_inputs = len(args) if name not in ('const0', 'const1', 'copy', 'NOT') else (1 if name in ('copy', 'NOT') else 0)
    if n_inputs < 2:
        return []

    out_col = [int(examples[e][1][obit]) for e in range(n_examples)]
    near = []

    # Get the op's eval function from OP_BY_NAME
    op_entry = OP_BY_NAME.get(name)
    if op_entry is None:
        return []
    _, op_eval, op_desc, op_n_inputs, op_cat = op_entry

    if op_cat == '2-input' and op_n_inputs == 2:
        correct_pair = tuple(sorted(args))
        for j in range(8):
            for k in range(j + 1, 8):
                if (j, k) == correct_pair:
                    continue
                col = [op_eval([int(examples[e][0][b]) for b in range(8)], [j, k]) for e in range(n_examples)]
                mc = sum(1 for a, b in zip(col, out_col) if a == b)
                if mc > n_examples // 2 and mc < n_examples:
                    near.append((op_desc([j, k]), col, mc))
    elif op_cat == '2-input-asym' and op_n_inputs == 2:
        for j in range(8):
            for k in range(8):
                if j == k:
                    continue
                if (j, k) == tuple(args):
                    continue
                col = [op_eval([int(examples[e][0][b]) for b in range(8)], [j, k]) for e in range(n_examples)]
                mc = sum(1 for a, b in zip(col, out_col) if a == b)
                if mc > n_examples // 2 and mc < n_examples:
                    near.append((op_desc([j, k]), col, mc))

    near.sort(key=lambda x: (-x[2], x[0]))
    return near[:max_nm]


def generate_one_problem(rng, seed_id, op_weights=None):
    """Generate one synthetic bit_ops problem with derivation-based CoT.

    Each problem has 8 output bits, each governed by one rule.
    We ensure at least 2 distinct operation types per problem for diversity.
    CoT shows explicit derivation: target values → near-misses → correct match.
    """
    if op_weights is None:
        op_weights = OP_WEIGHTS

    op_names = list(op_weights.keys())
    weights = [op_weights[n] for n in op_names]

    # Pick 8 rules (one per output bit), ensure >= 2 distinct ops
    for _attempt in range(20):
        rules = []
        for obit in range(8):
            chosen_name = rng.choices(op_names, weights=weights, k=1)[0]
            op_entry = OP_BY_NAME[chosen_name]
            name, eval_fn, desc_fn, n_inputs, category = op_entry

            if category == '2-input-asym':
                args = _pick_args_asym(rng, n_inputs)
            else:
                args = _pick_args(rng, n_inputs)

            rules.append((name, eval_fn, desc_fn, args))

        distinct_ops = len(set(r[0] for r in rules))
        if distinct_ops >= 2:
            break

    # Generate 5 random input-output examples
    examples = []
    seen = set()
    while len(examples) < 5:
        inp = rng.randint(0, 255)
        if inp in seen:
            continue
        seen.add(inp)
        inp_bits = [(inp >> (7 - i)) & 1 for i in range(8)]
        out_bits = [eval_fn(inp_bits, args) for _, eval_fn, _, args in rules]
        inp_str = ''.join(str(b) for b in inp_bits)
        out_str = ''.join(str(b) for b in out_bits)
        examples.append((inp_str, out_str))

    n_ex = len(examples)

    # Generate query (not in examples)
    while True:
        q = rng.randint(0, 255)
        if q not in seen:
            break
    q_bits = [(q >> (7 - i)) & 1 for i in range(8)]
    a_bits = [eval_fn(q_bits, args) for _, eval_fn, _, args in rules]
    target = ''.join(str(b) for b in q_bits)
    answer = ''.join(str(b) for b in a_bits)

    # Build prompt (matches original format exactly)
    prompt_lines = [
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
        "and possibly majority or choice functions.",
        "",
        "Here are some examples of input -> output:",
    ]
    for inp_s, out_s in examples:
        prompt_lines.append(f"{inp_s} -> {out_s}")
    prompt_lines.append(f"\nNow, determine the output for: {target}")
    prompt = '\n'.join(prompt_lines)

    # Build derivation-based CoT
    cot = []
    cot.append("Analyzing each output bit:")

    for obit in range(8):
        name, eval_fn, desc_fn, args = rules[obit]
        out_col = [int(examples[e][1][obit]) for e in range(n_ex)]
        out_str = ','.join(str(v) for v in out_col)

        if name in ('const0', 'const1'):
            v = 0 if name == 'const0' else 1
            cot.append(f"bit {obit}: [{out_str}] → const {v}")
            continue

        cot.append(f"bit {obit}: [{out_str}]")
        desc = desc_fn(args)

        if name in ('copy', 'NOT'):
            # Show matching column
            match_col = [eval_fn([int(examples[e][0][b]) for b in range(8)], args) for e in range(n_ex)]
            match_str = ','.join(str(v) for v in match_col)
            cot.append(f"  {desc}: [{match_str}] → match")
        else:
            # 2-input or higher: show near-misses then correct match
            near_misses = _find_near_misses_synth(examples, n_ex, obit, rules[obit])
            for nm_desc, nm_col, nm_match in near_misses:
                nm_str = ','.join(str(v) for v in nm_col)
                cot.append(f"  {nm_desc}: [{nm_str}] {nm_match}/{n_ex} ✗")

            correct_col = [eval_fn([int(examples[e][0][b]) for b in range(8)], args) for e in range(n_ex)]
            correct_str = ','.join(str(v) for v in correct_col)
            cot.append(f"  {desc}: [{correct_str}] {n_ex}/{n_ex} ✓")

    # Apply to target
    non_const = [(i, rules[i]) for i in range(8) if rules[i][0] not in ('const0', 'const1')]
    if non_const:
        cot.append(f"\nApply to {target}:")
        for i, (name, eval_fn, desc_fn, args) in non_const:
            desc = desc_fn(args)
            refs = _get_refs(name, args)
            ref_str = ", ".join(f"in[{r}]={q_bits[r]}" for r in refs)
            comp = re.sub(r'in\[(\d+)\]', lambda m: str(q_bits[int(m.group(1))]), desc)
            cot.append(f"bit {i}: {ref_str} → {comp} = {a_bits[i]}")

    cot.append(f"\nOutput: {answer}")

    raw_id = hashlib.md5(f"synth_bal_bitops_{seed_id}".encode()).hexdigest()[:8]
    return {
        'id': f"synth_{raw_id}",
        'prompt': prompt,
        'answer': answer,
        'thinking': '\n'.join(cot),
        'type': 'bit_ops',
    }


def _get_refs(name, args):
    """Get input bit indices referenced by this operation."""
    if name in ('const0', 'const1'):
        return []
    elif name in ('copy', 'NOT'):
        return [args[0]]
    else:
        return list(args)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPLEXITY-TIERED GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# Tier 1: Force at least 4 out of 8 bits to use 2-input or harder ops
# V2: Focus on 2-input ops esp. XNOR, keep some COPY
TIER_HARD_WEIGHTS = {
    'const0':   2,
    'const1':   2,
    'copy':     8,     # V2: still present
    'NOT':      6,
    'XOR':      12,
    'XNOR':     18,    # V2: XNOR is key weakness
    'AND':      10,
    'OR':       8,
    'NAND':     8,
    'NOR':      8,
    'NOT_AND':  6,
    'NOT_OR':   6,
    'XOR3':     2,
    'AND_XOR':  2,
    'OR_XOR':   2,
    'AND3':     1,
    'OR3':      1,
    'NAND3':    1,
    'NOR3':     1,
    'MAJ3':     1,
    'XOR_AND':  1,
    'XOR_OR':   1,
    'AND_OR':   1,
    'OR_AND':   1,
}

# V2: 3-input tier — still include 3-input but don't dominate
TIER_3INPUT_WEIGHTS = {
    'const0':   1,
    'const1':   1,
    'copy':     5,
    'NOT':      4,
    'XOR':      6,
    'XNOR':     10,    # V2: keep XNOR visible even in 3-input tier
    'AND':      6,
    'OR':       5,
    'NAND':     4,
    'NOR':      4,
    'NOT_AND':  3,
    'NOT_OR':   3,
    'XOR3':     8,     # V2: reduced from 15
    'AND_XOR':  8,
    'OR_XOR':   8,
    'AND3':     5,
    'OR3':      5,
    'NAND3':    5,
    'NOR3':     5,
    'MAJ3':     5,
    'XOR_AND':  4,
    'XOR_OR':   4,
    'AND_OR':   4,
    'OR_AND':   4,
}


def generate_tiered_problems(n_total, seed=42):
    """Generate n_total problems with tiered complexity distribution.

    Distribution V2 (aligned closer to test distribution):
    - 30% easy: copy/NOT dominant (test has ~44% COPY + 10% NOT)
    - 50% medium: 2-input dominant with XNOR boosted (key weakness)
    - 20% hard: 3-input included (test ~2%, keep some exposure)

    V4: derivation-based CoT — no error-correction path.
    """
    rng = random.Random(seed)
    n_easy = int(n_total * 0.30)
    n_hard2 = int(n_total * 0.50)
    n_3input = n_total - n_easy - n_hard2

    problems = []
    idx = 0

    print(f"Generating {n_total} balanced bit_ops problems (derivation CoT):")
    print(f"  Tier 'easy' (1-input dominant): {n_easy}")
    print(f"  Tier 'hard-2input' (NAND/NOR/XNOR boosted): {n_hard2}")
    print(f"  Tier '3-input' (composite ops): {n_3input}")

    # V2 Easy tier: mostly copy/NOT, some XOR/XNOR for mixing
    TIER_EASY_WEIGHTS = {
        'const0': 3, 'const1': 2,
        'copy': 25, 'NOT': 15,     # V2: copy boosted to match test dist
        'XOR': 6, 'XNOR': 8, 'AND': 5, 'OR': 5,  # V2: add XNOR exposure
        'NAND': 2, 'NOR': 2,
        'NOT_AND': 2, 'NOT_OR': 2,
        'XOR3': 0, 'AND_XOR': 0, 'OR_XOR': 0,
        'AND3': 0, 'OR3': 0, 'NAND3': 0, 'NOR3': 0,
        'MAJ3': 0, 'XOR_AND': 0, 'XOR_OR': 0, 'AND_OR': 0, 'OR_AND': 0,
    }
    for i in range(n_easy):
        problems.append(generate_one_problem(rng, idx, TIER_EASY_WEIGHTS))
        idx += 1

    for i in range(n_hard2):
        problems.append(generate_one_problem(rng, idx, TIER_HARD_WEIGHTS))
        idx += 1

    for i in range(n_3input):
        problems.append(generate_one_problem(rng, idx, TIER_3INPUT_WEIGHTS))
        idx += 1

    # Shuffle
    rng.shuffle(problems)
    return problems


def analyze_distribution(problems):
    """Analyze the operation distribution of generated problems."""
    op_counter = Counter()
    category_counter = Counter()
    max_complexity = Counter()  # per problem

    for p in problems:
        thinking = p['thinking']
        problem_ops = set()

        for line in thinking.split('\n'):
            line = line.strip()
            # New derivation format:
            #   "bit X: [...] → const 0/1"
            #   "  desc: [...] → match"        (1-input)
            #   "  desc: [...] N/N ✓"           (2+ input correct)
            if '→ const' in line:
                if 'const 0' in line:
                    op_counter['const0'] += 1
                else:
                    op_counter['const1'] += 1
                problem_ops.add('constant')
            elif '→ match' in line:
                desc = line.split(':')[0].strip()
                op = _classify_op(desc)
                op_counter[op] += 1
                problem_ops.add(_op_category(op))
            elif '✓' in line and '/' in line:
                desc = line.split(':')[0].strip()
                op = _classify_op(desc)
                op_counter[op] += 1
                problem_ops.add(_op_category(op))

        # Track max complexity per problem
        if '3-input' in problem_ops:
            max_complexity['3-input'] += 1
        elif '2-input' in problem_ops:
            max_complexity['2-input'] += 1
        elif '1-input' in problem_ops:
            max_complexity['1-input'] += 1
        else:
            max_complexity['constant'] += 1

    total_ops = sum(op_counter.values())
    print(f"\n{'='*60}")
    print(f"  OPERATION DISTRIBUTION ({total_ops} total bit rules)")
    print(f"{'='*60}")
    for op, cnt in op_counter.most_common():
        print(f"  {op:15s} {cnt:5d} ({cnt/total_ops*100:5.1f}%)")

    print(f"\n  Problem max complexity:")
    for cat in ['constant', '1-input', '2-input', '3-input']:
        cnt = max_complexity.get(cat, 0)
        print(f"    {cat:12s}: {cnt:4d} ({cnt/len(problems)*100:5.1f}%)")


def _classify_op(desc):
    """Classify operation from its description string."""
    if desc == '0': return 'const0'
    if desc == '1': return 'const1'
    if desc.startswith('MAJ('): return 'MAJ3'
    if 'XOR' not in desc and 'AND' not in desc and 'OR' not in desc and 'NOT' not in desc:
        if re.match(r'^in\[\d+\]$', desc):
            return 'COPY'

    # Count input references
    refs = re.findall(r'in\[\d+\]', desc)
    n_refs = len(refs)

    if n_refs == 1:
        if 'NOT' in desc: return 'NOT'
        return 'COPY'

    if 'NOT(' in desc:
        inner = desc[4:-1] if desc.startswith('NOT(') and desc.endswith(')') else desc
        if n_refs == 2:
            if 'XOR' in inner: return 'XNOR'
            if 'AND' in inner: return 'NAND'
            if 'OR' in inner: return 'NOR'
        if n_refs == 3:
            return desc.split('(')[0] + '_' + _classify_op(inner)

    if n_refs == 2:
        if 'NOT(' in desc and ('AND' in desc or 'OR' in desc):
            if 'AND' in desc: return 'NOT_AND'
            if 'OR' in desc: return 'NOT_OR'
        if 'XOR' in desc: return 'XOR'
        if 'AND' in desc: return 'AND'
        if 'OR' in desc: return 'OR'

    if n_refs == 3:
        return '3-input'

    return desc[:20]


def _op_category(op):
    if op in ('const0', 'const1'): return 'constant'
    if op in ('COPY', 'NOT'): return '1-input'
    if op.startswith('3-') or op in ('XOR3', 'AND_XOR', 'OR_XOR', 'AND3', 'OR3',
                                      'NAND3', 'NOR3', 'MAJ3', 'XOR_AND', 'XOR_OR',
                                      'AND_OR', 'OR_AND'):
        return '3-input'
    return '2-input'


# ═══════════════════════════════════════════════════════════════════════════════
#  VERIFY GENERATED PROBLEMS (use gen_thinking.py solver to validate)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_problems(problems, max_verify=100):
    """Verify generated problems by re-solving them with the bit solver."""
    # Import solver from gen_thinking.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    try:
        from gen_thinking import gen_thinking_bit
    except ImportError:
        print("WARNING: Cannot import gen_thinking_bit, skipping verification")
        return True

    n_verify = min(max_verify, len(problems))
    n_pass = 0
    n_fail = 0

    for p in problems[:n_verify]:
        cot = gen_thinking_bit(p['prompt'], p['answer'])
        if cot is not None:
            n_pass += 1
        else:
            n_fail += 1
            if n_fail <= 5:
                print(f"  VERIFY FAIL: id={p['id']}, answer={p['answer']}")
                print(f"    Prompt: {p['prompt'][:200]}...")

    print(f"\nVerification: {n_pass}/{n_verify} passed, {n_fail} failed")
    return n_fail == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate balanced synthetic bit_ops data')
    parser.add_argument('--n', type=int, default=800,
                        help='Number of problems to generate (default: 800)')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: data/synth_balanced_bitops.csv)')
    parser.add_argument('--verify', type=int, default=50,
                        help='Number of problems to verify with solver (0=skip)')
    parser.add_argument('--analyze', action='store_true', default=True,
                        help='Print distribution analysis')
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'synth_balanced_bitops.csv'
    )

    problems = generate_tiered_problems(args.n, seed=args.seed)
    print(f"\nGenerated {len(problems)} problems")

    if args.analyze:
        analyze_distribution(problems)

    if args.verify > 0:
        print(f"\nVerifying {args.verify} problems with gen_thinking solver...")
        verify_problems(problems, args.verify)

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking', 'type'])
        writer.writeheader()
        writer.writerows(problems)

    print(f"\nWritten to: {output_path}")
    print(f"  Rows: {len(problems)}")

    # Show a few samples
    print(f"\n{'='*60}")
    print("  SAMPLE PROBLEMS")
    print(f"{'='*60}")
    for p in problems[:3]:
        print(f"\n--- {p['id']} ---")
        print(f"Answer: {p['answer']}")
        print(f"Thinking:\n{p['thinking']}")
        print()


if __name__ == '__main__':
    main()
