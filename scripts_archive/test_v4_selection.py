"""Test v4 bit_ops selection improvements vs current v3.

Improvements:
1. Deterministic tie-break: (n_inputs, n_ops, type_order, var_indices)
2. Canonical descriptions: XNOR→NOT(XOR), NAND→NOT(AND), etc.
3. Global structure detection: shift/uniform patterns across 8 bits
4. Structure-aware selection: prefer globally consistent patterns
"""
import csv, sys, os, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from gen_thinking import _enumerate_bit_functions, _eval_bit_function, classify


# ═══════════════════════════════════════════════════════════════════════════════
#  Improvement 1: Deterministic complexity-based tie-break
# ═══════════════════════════════════════════════════════════════════════════════

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
}

def _func_sort_key(func):
    """Deterministic complexity-based tie-break key."""
    fname, args, _ = func
    n_inputs, n_ops, type_ord = _TYPE_ORDER.get(fname, (5, 5, 99))
    if isinstance(args, int):
        var_key = (args,)
    elif isinstance(args, tuple):
        var_key = args
    else:
        var_key = ()
    return (n_inputs, n_ops, type_ord, var_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  Improvement 2: Canonical descriptions
# ═══════════════════════════════════════════════════════════════════════════════

def _canonical_desc(func):
    """Canonical form — decompose XNOR/NAND/NOR into NOT(base_op)."""
    fname, args, _ = func
    if fname == 'const': return f"const({args})"
    if fname == 'copy': return f"in[{args}]"
    if fname == 'not': return f"NOT(in[{args}])"
    if fname == 'xor2': return f"in[{args[0]}] XOR in[{args[1]}]"
    if fname == 'and': return f"in[{args[0]}] AND in[{args[1]}]"
    if fname == 'or': return f"in[{args[0]}] OR in[{args[1]}]"
    if fname == 'xnor2': return f"NOT(in[{args[0]}] XOR in[{args[1]}])"
    if fname == 'nand': return f"NOT(in[{args[0]}] AND in[{args[1]}])"
    if fname == 'nor': return f"NOT(in[{args[0]}] OR in[{args[1]}])"
    if fname == 'not_and': return f"NOT(in[{args[0]}]) AND in[{args[1]}]"
    if fname == 'not_or': return f"NOT(in[{args[0]}]) OR in[{args[1]}]"
    if fname == 'xor3': return f"in[{args[0]}] XOR in[{args[1]}] XOR in[{args[2]}]"
    if fname == 'and3': return f"in[{args[0]}] AND in[{args[1]}] AND in[{args[2]}]"
    if fname == 'or3': return f"in[{args[0]}] OR in[{args[1]}] OR in[{args[2]}]"
    if fname == 'nand3': return f"NOT(in[{args[0]}] AND in[{args[1]}] AND in[{args[2]}])"
    if fname == 'nor3': return f"NOT(in[{args[0]}] OR in[{args[1]}] OR in[{args[2]}])"
    if fname == 'and_xor': return f"(in[{args[0]}] AND in[{args[1]}]) XOR in[{args[2]}]"
    if fname == 'or_xor': return f"(in[{args[0]}] OR in[{args[1]}]) XOR in[{args[2]}]"
    if fname == 'xor_and': return f"(in[{args[0]}] XOR in[{args[1]}]) AND in[{args[2]}]"
    if fname == 'xor_or': return f"(in[{args[0]}] XOR in[{args[1]}]) OR in[{args[2]}]"
    if fname == 'and_or': return f"(in[{args[0]}] AND in[{args[1]}]) OR in[{args[2]}]"
    if fname == 'or_and': return f"(in[{args[0]}] OR in[{args[1]}]) AND in[{args[2]}]"
    if fname == 'nxor_and': return f"NOT((in[{args[0]}] XOR in[{args[1]}]) AND in[{args[2]}])"
    if fname == 'nxor_or': return f"NOT((in[{args[0]}] XOR in[{args[1]}]) OR in[{args[2]}])"
    if fname == 'nand_or': return f"NOT((in[{args[0]}] AND in[{args[1]}]) OR in[{args[2]}])"
    if fname == 'nor_and': return f"NOT((in[{args[0]}] OR in[{args[1]}]) AND in[{args[2]}])"
    if fname == 'not_and2': return f"NOT(in[{args[0]}]) AND in[{args[1]}] AND in[{args[2]}]"
    if fname == 'nnot_and2': return f"NOT(NOT(in[{args[0]}]) AND in[{args[1]}] AND in[{args[2]}])"
    if fname == 'not_or2': return f"NOT(in[{args[0]}]) OR in[{args[1]}] OR in[{args[2]}]"
    if fname == 'nnot_or2': return f"NOT(NOT(in[{args[0]}]) OR in[{args[1]}] OR in[{args[2]}])"
    if fname == 'xor4': return f"in[{args[0]}] XOR in[{args[1]}] XOR in[{args[2]}] XOR in[{args[3]}]"
    if fname == 'xnor4': return f"NOT(in[{args[0]}] XOR in[{args[1]}] XOR in[{args[2]}] XOR in[{args[3]}])"
    if fname == 'maj3': return f"MAJ(in[{args[0]}],in[{args[1]}],in[{args[2]}])"
    if fname == 'nmaj3': return f"NOT(MAJ(in[{args[0]}],in[{args[1]}],in[{args[2]}]))"
    return func[2]


# ═══════════════════════════════════════════════════════════════════════════════
#  Improvement 3+4: Global structure detection & structure-aware selection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_global_pattern(all_candidates):
    """Detect structural patterns across 8 output bits.
    Returns (pattern_name, detail, bit_map) or None.
    bit_map: {obit: candidate_func} for bits this pattern covers.
    """
    best = None
    best_coverage = 0

    # Check shift patterns (copy and NOT)
    for shift in range(8):
        copy_map = {}
        not_map = {}
        for obit in range(8):
            src = (obit + shift) % 8
            for f in all_candidates[obit]:
                if f[0] == 'copy' and f[1] == src and obit not in copy_map:
                    copy_map[obit] = f
                if f[0] == 'not' and f[1] == src and obit not in not_map:
                    not_map[obit] = f

        if len(copy_map) >= 6 and len(copy_map) > best_coverage:
            best = ('shift_copy', shift, copy_map)
            best_coverage = len(copy_map)
        if len(not_map) >= 6 and len(not_map) > best_coverage:
            best = ('shift_not', shift, not_map)
            best_coverage = len(not_map)

    # Check uniform operation
    for op_type in ['copy', 'not', 'xor2', 'and', 'or']:
        op_map = {}
        for obit in range(8):
            # Pick the first candidate of this type (sorted by var index)
            typed = [f for f in all_candidates[obit] if f[0] == op_type]
            if typed:
                typed.sort(key=_func_sort_key)
                op_map[obit] = typed[0]
        if len(op_map) >= 6 and len(op_map) > best_coverage:
            best = ('uniform_' + op_type, None, op_map)
            best_coverage = len(op_map)

    return best


def select_v4(all_candidates, target_bits):
    """v4 selection: global pattern as SOFT preference + complexity tie-break.
    
    Key rule: global pattern ONLY overrides when it's at the same or lower
    complexity level as the per-bit simplest. Never override const/copy with
    a 2-input pattern.
    """
    chosen = [None] * 8

    # Try global pattern first
    pattern = _detect_global_pattern(all_candidates)
    pattern_used = None

    if pattern:
        pname, pdetail, pmap = pattern
        any_pattern_applied = False
        for obit in range(8):
            simplest = sorted(all_candidates[obit], key=_func_sort_key)[0]
            simplest_level = _func_sort_key(simplest)[0]  # n_inputs

            if obit in pmap:
                pat_level = _func_sort_key(pmap[obit])[0]
                if pat_level <= simplest_level:
                    # Pattern is equally or more simple → use it
                    chosen[obit] = pmap[obit]
                    any_pattern_applied = True
                else:
                    # Simpler candidate exists → don't override
                    chosen[obit] = simplest
            else:
                chosen[obit] = simplest
        if any_pattern_applied:
            pattern_used = pname
    
    if pattern_used is None:
        for obit in range(8):
            chosen[obit] = sorted(all_candidates[obit], key=_func_sort_key)[0]

    result_bits = [_eval_bit_function(chosen[i], target_bits) for i in range(8)]
    answer = ''.join(str(b) for b in result_bits)
    return answer, chosen, pattern_used


def select_v3(all_candidates, target_bits):
    """v3 selection: (len(desc), desc) — the current approach."""
    chosen = [None] * 8
    for obit in range(8):
        chosen[obit] = sorted(all_candidates[obit], key=lambda f: (len(f[2]), f[2]))[0]
    result_bits = [_eval_bit_function(chosen[i], target_bits) for i in range(8)]
    answer = ''.join(str(b) for b in result_bits)
    return answer, chosen, None


# ═══════════════════════════════════════════════════════════════════════════════
#  Test
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

    print("=" * 70)
    print("  v3 vs v4 Selection Comparison (1602 bit_ops problems)")
    print("=" * 70)

    t0 = time.time()
    n_parse = 0
    n_skip = 0
    v3_correct = 0
    v4_correct = 0
    both_correct = 0
    v3_only = 0
    v4_only = 0
    neither = 0
    v4_pattern_used = Counter()
    v4_pattern_correct = Counter()

    # Track which functions are chosen differently
    diff_cases = []

    for row in rows:
        examples, target = parse_bit_ops(row['prompt'])
        gold = row['answer']
        rid = row['id']
        if not examples or not target or len(examples) < 4 or len(gold) != 8:
            n_skip += 1
            continue
        n_parse += 1

        n = len(examples)
        inputs = [[int(ex[0][j]) for j in range(8)] for ex in examples]
        outputs = [[int(ex[1][j]) for j in range(8)] for ex in examples]
        target_bits = [int(target[j]) for j in range(8)]

        # Get all candidates
        all_cands = []
        ok = True
        for obit in range(8):
            c = _enumerate_bit_functions(inputs, outputs, n, obit)
            if not c:
                ok = False
                break
            all_cands.append(c)
        if not ok:
            n_skip += 1
            continue

        # Compare
        v3_ans, v3_chosen, _ = select_v3(all_cands, target_bits)
        v4_ans, v4_chosen, v4_pat = select_v4(all_cands, target_bits)

        v3_ok = (v3_ans == gold)
        v4_ok = (v4_ans == gold)

        if v3_ok and v4_ok:
            both_correct += 1
        elif v3_ok:
            v3_only += 1
            diff_cases.append(('v3_only', rid, gold, v3_ans, v4_ans, v4_pat,
                               [(v3_chosen[i][2], v4_chosen[i][2])
                                for i in range(8) if v3_chosen[i][2] != v4_chosen[i][2]]))
        elif v4_ok:
            v4_only += 1
            diff_cases.append(('v4_only', rid, gold, v3_ans, v4_ans, v4_pat,
                               [(v3_chosen[i][2], v4_chosen[i][2])
                                for i in range(8) if v3_chosen[i][2] != v4_chosen[i][2]]))
        else:
            neither += 1

        v3_correct += v3_ok
        v4_correct += v4_ok

        if v4_pat:
            v4_pattern_used[v4_pat] += 1
            if v4_ok:
                v4_pattern_correct[v4_pat] += 1

    elapsed = time.time() - t0

    print(f"\nParsed: {n_parse} problems, skipped: {n_skip}")
    print(f"Time: {elapsed:.1f}s\n")

    print(f"{'Strategy':20s} {'Correct':>8s} {'Rate':>8s}")
    print(f"{'-'*40}")
    print(f"{'v3 (len,lex)':20s} {v3_correct:>8d} {v3_correct/n_parse*100:>7.1f}%")
    print(f"{'v4 (improved)':20s} {v4_correct:>8d} {v4_correct/n_parse*100:>7.1f}%")

    print(f"\nDetailed comparison:")
    print(f"  Both correct:  {both_correct}")
    print(f"  v3 only:       {v3_only} (v4 regressed)")
    print(f"  v4 only:       {v4_only} (v4 improved)")
    print(f"  Neither:       {neither}")
    print(f"  Net change:    {v4_correct - v3_correct:+d}")

    if v4_pattern_used:
        print(f"\nGlobal patterns detected:")
        for pat, cnt in v4_pattern_used.most_common():
            corr = v4_pattern_correct.get(pat, 0)
            print(f"  {pat:20s}: {cnt:4d} problems, {corr:4d} correct ({corr/cnt*100:.1f}%)")

    if diff_cases:
        print(f"\n{'='*70}")
        print(f"  Diff cases (showing first 20)")
        print(f"{'='*70}")
        for case_type, rid, gold, v3a, v4a, pat, diffs in diff_cases[:20]:
            tag = "✅ v4 GAINED" if case_type == 'v4_only' else "❌ v4 LOST"
            print(f"\n  {tag} [{rid}] gold={gold}")
            print(f"    v3={v3a}  v4={v4a}  pattern={pat}")
            for old, new in diffs[:4]:
                print(f"    changed: {old} → {new}")


if __name__ == '__main__':
    main()
