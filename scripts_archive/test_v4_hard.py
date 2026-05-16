"""Test v4-hard (strict pattern) vs v4-soft vs v3.

User's proposal:
1. Global constraint: pattern covers ALL bits, no mixing
2. Executable pattern: out[i] = NOT(in[(i+shift) mod 8])
3. Pattern vs bit-rule: choose ONE, no hybrid
4. Score threshold: coverage ≥ 7/8 (87.5%) → use pattern, else fallback to per-bit
"""
import csv, sys, os, time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from gen_thinking import _enumerate_bit_functions, _eval_bit_function, classify

# ═══════════════════════════════════════════════════════════════════════════════
#  Shared: complexity sort key
# ═══════════════════════════════════════════════════════════════════════════════

_TYPE_ORDER = {
    'const': (0, 0, 0), 'copy': (1, 0, 1), 'not': (1, 1, 2),
    'xor2': (2, 1, 3), 'and': (2, 1, 4), 'or': (2, 1, 5),
    'xnor2': (2, 2, 6), 'nand': (2, 2, 7), 'nor': (2, 2, 8),
    'not_and': (2, 2, 9), 'not_or': (2, 2, 10),
    'xor3': (3, 2, 11), 'and3': (3, 2, 12), 'or3': (3, 2, 13),
    'and_xor': (3, 2, 14), 'or_xor': (3, 2, 15),
    'xor_and': (3, 2, 16), 'xor_or': (3, 2, 17),
    'and_or': (3, 2, 18), 'or_and': (3, 2, 19),
    'not_and2': (3, 2, 20), 'not_or2': (3, 2, 21),
    'nand3': (3, 3, 22), 'nor3': (3, 3, 23),
    'nxor_and': (3, 3, 24), 'nxor_or': (3, 3, 25),
    'nand_or': (3, 3, 26), 'nor_and': (3, 3, 27),
    'nnot_and2': (3, 3, 28), 'nnot_or2': (3, 3, 29),
    'maj3': (3, 3, 30), 'nmaj3': (3, 3, 31),
    'xor4': (4, 3, 32), 'xnor4': (4, 4, 33),
}

def _func_sort_key(func):
    fname, args, _ = func
    n_inputs, n_ops, type_ord = _TYPE_ORDER.get(fname, (5, 5, 99))
    if isinstance(args, int): var_key = (args,)
    elif isinstance(args, tuple): var_key = args
    else: var_key = ()
    return (n_inputs, n_ops, type_ord, var_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pattern detection (shared)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_patterns(all_candidates):
    """Detect ALL candidate patterns with coverage info."""
    patterns = []

    # Shift patterns
    for shift in range(8):
        copy_map, not_map = {}, {}
        for obit in range(8):
            src = (obit + shift) % 8
            for f in all_candidates[obit]:
                if f[0] == 'copy' and f[1] == src and obit not in copy_map:
                    copy_map[obit] = f
                if f[0] == 'not' and f[1] == src and obit not in not_map:
                    not_map[obit] = f
        if len(copy_map) >= 6:
            patterns.append(('shift_copy', shift, copy_map, len(copy_map)))
        if len(not_map) >= 6:
            patterns.append(('shift_not', shift, not_map, len(not_map)))

    # Uniform operation
    for op_type in ['copy', 'not', 'xor2', 'and', 'or']:
        op_map = {}
        for obit in range(8):
            typed = [f for f in all_candidates[obit] if f[0] == op_type]
            if typed:
                typed.sort(key=_func_sort_key)
                op_map[obit] = typed[0]
        if len(op_map) >= 6:
            patterns.append(('uniform_' + op_type, None, op_map, len(op_map)))

    # Sort by coverage descending
    patterns.sort(key=lambda p: -p[3])
    return patterns


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy: v3 (baseline, len+lex)
# ═══════════════════════════════════════════════════════════════════════════════

def select_v3(all_cands, target_bits):
    chosen = [sorted(c, key=lambda f: (len(f[2]), f[2]))[0] for c in all_cands]
    bits = [_eval_bit_function(chosen[i], target_bits) for i in range(8)]
    return ''.join(str(b) for b in bits), 'v3', None


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy: v4-soft (current, pattern as soft preference)
# ═══════════════════════════════════════════════════════════════════════════════

def select_v4_soft(all_cands, target_bits):
    patterns = _detect_patterns(all_cands)
    chosen = [None] * 8
    pat_used = None

    if patterns:
        best_pat = patterns[0]  # highest coverage
        pname, pdetail, pmap, pcov = best_pat
        any_applied = False
        for obit in range(8):
            simplest = sorted(all_cands[obit], key=_func_sort_key)[0]
            s_level = _func_sort_key(simplest)[0]
            if obit in pmap:
                p_level = _func_sort_key(pmap[obit])[0]
                if p_level <= s_level:
                    chosen[obit] = pmap[obit]
                    any_applied = True
                else:
                    chosen[obit] = simplest
            else:
                chosen[obit] = simplest
        if any_applied:
            pat_used = pname

    if pat_used is None:
        for obit in range(8):
            chosen[obit] = sorted(all_cands[obit], key=_func_sort_key)[0]

    bits = [_eval_bit_function(chosen[i], target_bits) for i in range(8)]
    return ''.join(str(b) for b in bits), 'v4-soft', pat_used


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy: v4-hard (user's proposal — strict pattern, no mixing)
# ═══════════════════════════════════════════════════════════════════════════════

def select_v4_hard(all_cands, target_bits, threshold=7):
    """Strict: use pattern for ALL 8 bits if coverage ≥ threshold, else pure per-bit."""
    patterns = _detect_patterns(all_cands)
    chosen = [None] * 8

    # Try pattern with ≥ threshold coverage
    pat_used = None
    for pname, pdetail, pmap, pcov in patterns:
        if pcov >= threshold:
            # Check if we can fill remaining bits with same-type candidates
            full_map = dict(pmap)
            can_fill = True
            for obit in range(8):
                if obit not in full_map:
                    # For uncovered bits, try to find a candidate of the same type
                    pat_type = list(pmap.values())[0][0]  # e.g. 'copy', 'not'
                    typed = [f for f in all_cands[obit] if f[0] == pat_type]
                    if typed:
                        typed.sort(key=_func_sort_key)
                        full_map[obit] = typed[0]
                    else:
                        can_fill = False
                        break
            if can_fill and len(full_map) == 8:
                chosen = [full_map[i] for i in range(8)]
                pat_used = pname
                break
            elif pcov == 8:
                chosen = [pmap[i] for i in range(8)]
                pat_used = pname
                break

    # Fallback: pure per-bit
    if pat_used is None:
        for obit in range(8):
            chosen[obit] = sorted(all_cands[obit], key=_func_sort_key)[0]

    bits = [_eval_bit_function(chosen[i], target_bits) for i in range(8)]
    return ''.join(str(b) for b in bits), 'v4-hard', pat_used


def select_v4_hard_8only(all_cands, target_bits):
    """Even stricter: only use pattern if it covers ALL 8 bits."""
    return select_v4_hard(all_cands, target_bits, threshold=8)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_bit_ops(prompt):
    lines = prompt.strip().split('\n')
    examples, target = [], None
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


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'competition_data', 'train.csv')
    rows = []
    with open(data_path) as f:
        for r in csv.DictReader(f):
            if classify(r['prompt']) == 'bit_ops':
                rows.append(r)

    strategies = [
        ('v3 (len,lex)', select_v3),
        ('v4-soft (mix)', select_v4_soft),
        ('v4-hard (≥7)', lambda c, t: select_v4_hard(c, t, 7)),
        ('v4-hard (=8)', select_v4_hard_8only),
        ('v4-hard (≥6)', lambda c, t: select_v4_hard(c, t, 6)),
    ]

    print("=" * 70)
    print("  v3 vs v4-soft vs v4-hard Selection Comparison")
    print("=" * 70)

    t0 = time.time()
    results = {name: {'correct': 0, 'total': 0, 'pat_used': Counter()} for name, _ in strategies}
    
    # Track per-problem results for pairwise comparison
    per_problem = []

    for row in rows:
        examples, target = parse_bit_ops(row['prompt'])
        gold = row['answer']
        if not examples or not target or len(examples) < 4 or len(gold) != 8:
            continue

        n = len(examples)
        inputs = [[int(ex[0][j]) for j in range(8)] for ex in examples]
        outputs = [[int(ex[1][j]) for j in range(8)] for ex in examples]
        target_bits = [int(target[j]) for j in range(8)]

        all_cands = []
        ok = True
        for obit in range(8):
            c = _enumerate_bit_functions(inputs, outputs, n, obit)
            if not c:
                ok = False; break
            all_cands.append(c)
        if not ok:
            continue

        problem_results = {}
        for name, fn in strategies:
            ans, _, pat = fn(all_cands, target_bits)
            is_correct = (ans == gold)
            results[name]['correct'] += is_correct
            results[name]['total'] += 1
            if pat:
                results[name]['pat_used'][pat] += 1
            problem_results[name] = is_correct
        
        per_problem.append(problem_results)

    elapsed = time.time() - t0

    n_total = results['v3 (len,lex)']['total']
    print(f"\nParsed: {n_total} problems, Time: {elapsed:.1f}s\n")

    print(f"{'Strategy':20s} {'Correct':>8s} {'Rate':>8s} {'vs v3':>8s}")
    print(f"{'-'*48}")
    v3_correct = results['v3 (len,lex)']['correct']
    for name, _ in strategies:
        r = results[name]
        delta = r['correct'] - v3_correct
        sign = '+' if delta > 0 else ''
        print(f"{name:20s} {r['correct']:>8d} {r['correct']/n_total*100:>7.1f}% {sign}{delta:>6d}")

    # Pairwise: v4-soft vs v4-hard
    print(f"\n{'='*70}")
    print(f"  Pairwise Comparison")
    print(f"{'='*70}")

    for hard_name in ['v4-hard (≥7)', 'v4-hard (=8)', 'v4-hard (≥6)']:
        soft_name = 'v4-soft (mix)'
        both = sum(1 for p in per_problem if p[soft_name] and p[hard_name])
        soft_only = sum(1 for p in per_problem if p[soft_name] and not p[hard_name])
        hard_only = sum(1 for p in per_problem if not p[soft_name] and p[hard_name])
        neither = sum(1 for p in per_problem if not p[soft_name] and not p[hard_name])
        print(f"\n  {soft_name} vs {hard_name}:")
        print(f"    Both correct: {both}")
        print(f"    Soft only:    {soft_only}")
        print(f"    Hard only:    {hard_only}")
        print(f"    Neither:      {neither}")

    # Pattern usage
    print(f"\n{'='*70}")
    print(f"  Pattern Usage by Strategy")
    print(f"{'='*70}")
    for name, _ in strategies:
        pats = results[name]['pat_used']
        if pats:
            print(f"\n  {name}:")
            for p, cnt in pats.most_common():
                print(f"    {p:25s}: {cnt:4d}")
        else:
            print(f"\n  {name}: no patterns used")


if __name__ == '__main__':
    main()
