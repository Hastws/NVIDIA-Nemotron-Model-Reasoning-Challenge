#!/usr/bin/env python3
"""Solver for eq_symbolic problems using base-N cryptarithmetic approach.
Adapted from user's DFS solver with greedy equation ordering."""

import csv
import re
import sys
import time
from collections import defaultdict

INPUT_FILE = 'competition_data/train.csv'
NUM_PATTERN = re.compile(r'^(\d+)([^\d])(\d+)$')


def parse_symbol_problem(prompt):
    lines = prompt.split('\n')
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
    return examples, query_str


def is_numeric(examples):
    for lhs, rhs in examples:
        if not NUM_PATTERN.match(lhs):
            return False
    return True


def decode_val(val, base):
    if val < 0:
        return None
    if val == 0:
        return (0,)
    digits = []
    while val > 0:
        digits.append(val % base)
        val //= base
    return tuple(reversed(digits))


ALL_OPS = [
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


def try_extend_mapping(mapping, used, result_chars, digits):
    m = dict(mapping)
    u = set(used)
    for c, d in zip(result_chars, digits):
        if c in m:
            if m[c] != d:
                return None
        else:
            if d in u:
                return None
            m[c] = d
            u.add(d)
    return m, u


def try_match_result(computed, result_chars, op_char, mapping, used, base):
    candidates = []
    if computed is None:
        return candidates
    abs_val = abs(computed)
    is_neg = computed < 0
    is_pos = computed > 0

    if computed >= 0:
        digits = decode_val(computed, base)
        if digits is not None and len(digits) == len(result_chars):
            ext = try_extend_mapping(mapping, used, result_chars, digits)
            if ext is not None:
                candidates.append(('none', ext[0], ext[1]))

    if is_neg:
        digits = decode_val(abs_val, base)
        if digits is not None and len(digits) + 1 == len(result_chars) and result_chars[0] == op_char:
            ext = try_extend_mapping(mapping, used, result_chars[1:], digits)
            if ext is not None:
                candidates.append(('neg_prefix', ext[0], ext[1]))

    if computed >= 0:
        digits = decode_val(computed, base)
        if digits is not None and len(result_chars) >= 2:
            if len(digits) + 1 == len(result_chars) and result_chars[0] == op_char:
                ext = try_extend_mapping(mapping, used, result_chars[1:], digits)
                if ext is not None:
                    candidates.append(('prefix', ext[0], ext[1]))

    val_for_encode = abs_val if is_neg else computed
    if val_for_encode is not None:
        digits = decode_val(val_for_encode, base)
        if digits is not None:
            if is_pos and len(result_chars) >= 2:
                if len(digits) + 1 == len(result_chars) and result_chars[0] == op_char:
                    ext = try_extend_mapping(mapping, used, result_chars[1:], digits)
                    if ext is not None and not any(c[0] == 'prefix' and c[1] == ext[0] for c in candidates):
                        candidates.append(('pos_prefix', ext[0], ext[1]))
            if (is_neg or computed == 0) and len(digits) == len(result_chars):
                ext = try_extend_mapping(mapping, used, result_chars, digits)
                if ext is not None:
                    if not any(c[0] == 'pos_prefix' and c[1] == ext[0] for c in candidates):
                        candidates.append(('pos_prefix', ext[0], ext[1]))

    return candidates


def greedy_sort_eqs(all_eqs):
    remaining = list(range(len(all_eqs)))
    known_chars = set()
    ordered = []
    while remaining:
        best_idx = None
        best_new = 999
        for i in remaining:
            left, right, result, op_char = all_eqs[i]
            new_count = sum(1 for c in set(left + right) if c not in known_chars)
            if new_count < best_new:
                best_new = new_count
                best_idx = i
        remaining.remove(best_idx)
        ordered.append(best_idx)
        left, right, result, op_char = all_eqs[best_idx]
        known_chars.update(left)
        known_chars.update(right)
        known_chars.update(result)
    return [all_eqs[i] for i in ordered]


def solve_equations_dfs(eqs, mapping, used, base, op_candidates, fmt_candidates, eq_idx, deadline):
    if time.time() > deadline:
        return None
    if eq_idx == len(eqs):
        return mapping

    left, right, result_chars, op_char = eqs[eq_idx]

    new_chars_in_input = []
    for c in left + right:
        if c not in mapping and c not in new_chars_in_input:
            new_chars_in_input.append(c)

    def try_ops_and_recurse(cur_m, cur_u):
        for op_name, func in op_candidates.get(op_char, ALL_OPS):
            a = cur_m[left[0]] * base + cur_m[left[1]]
            b = cur_m[right[0]] * base + cur_m[right[1]]
            computed = func(a, b)
            if computed is None:
                continue
            matches = try_match_result(computed, result_chars, op_char, cur_m, cur_u, base)
            allowed_fmts = fmt_candidates.get(op_char)
            if allowed_fmts is not None:
                matches = [mm for mm in matches if mm[0] in allowed_fmts]
            for fmt, new_m, new_u in matches:
                new_cands = dict(op_candidates)
                new_cands[op_char] = [(op_name, func)]
                new_fmts = dict(fmt_candidates)
                new_fmts[op_char] = {fmt}
                r = solve_equations_dfs(eqs, new_m, new_u, base, new_cands, new_fmts, eq_idx + 1, deadline)
                if r is not None:
                    return r
        return None

    if not new_chars_in_input:
        return try_ops_and_recurse(mapping, used)
    else:
        def enum_new(ci, cur_m, cur_u):
            if time.time() > deadline:
                return None
            if ci == len(new_chars_in_input):
                return try_ops_and_recurse(cur_m, cur_u)
            c = new_chars_in_input[ci]
            if c in cur_m:
                return enum_new(ci + 1, cur_m, cur_u)
            for v in range(base):
                if v in cur_u:
                    continue
                nm = dict(cur_m)
                nu = set(cur_u)
                nm[c] = v
                nu.add(v)
                r = enum_new(ci + 1, nm, nu)
                if r is not None:
                    return r
            return None
        return enum_new(0, mapping, used)


ALL_FMTS = ['none', 'neg_prefix', 'prefix', 'pos_prefix']


def try_solve(examples, query_str, answer):
    parsed = []
    for lhs, rhs in examples:
        if len(lhs) != 5:
            return None
        left = (lhs[0], lhs[1])
        op = lhs[2]
        right = (lhs[3], lhs[4])
        result = tuple(rhs)
        parsed.append((left, op, right, result))

    # Also parse query + gold answer as an additional equation
    query_eq = None
    if query_str and len(query_str) == 5 and answer:
        q_left = (query_str[0], query_str[1])
        q_op = query_str[2]
        q_right = (query_str[3], query_str[4])
        q_result = tuple(answer)
        query_eq = (q_left, q_op, q_right, q_result)

    op_groups = defaultdict(list)
    for left, op, right, result in parsed:
        op_groups[op].append((left, right, result))
    # Add query to its op group too
    if query_eq:
        op_groups[query_eq[1]].append((query_eq[0], query_eq[2], query_eq[3]))

    concat_ops = {}
    calc_op_groups = {}
    for op, entries in op_groups.items():
        is_concat = all(left + right == result for left, right, result in entries)
        is_rev_concat = all(right + left == result for left, right, result in entries)
        if is_concat:
            concat_ops[op] = 'concat'
        elif is_rev_concat:
            concat_ops[op] = 'rev_concat'
        else:
            calc_op_groups[op] = entries

    if not calc_op_groups:
        return ('all_concat', concat_ops, None, None, None, None)

    value_chars = set()
    for op, entries in calc_op_groups.items():
        for left, right, result in entries:
            value_chars.update(left)
            value_chars.update(right)
            value_chars.update(result)
    n_vc = len(value_chars)
    if n_vc > 16:
        return None

    # Build equations including query+gold as constraint
    all_eqs_raw = []
    for op_char, entries in sorted(calc_op_groups.items(), key=lambda x: -len(x[1])):
        for left, right, result in entries:
            all_eqs_raw.append((left, right, result, op_char))
    all_eqs = greedy_sort_eqs(all_eqs_raw)

    for base in range(max(n_vc, 2), n_vc + 8):
        deadline = time.time() + 5.0
        result = solve_equations_dfs(all_eqs, {}, set(), base, {}, {}, 0, deadline)
        if result is not None:
            mapping = result
            op_assignments = {}
            fmt_assignments = {}
            ok = True
            for op_char, entries in calc_op_groups.items():
                found = False
                for op_name, func in ALL_OPS:
                    for fmt in ALL_FMTS:
                        all_match = True
                        for left, right, res_chars in entries:
                            a = mapping[left[0]] * base + mapping[left[1]]
                            b = mapping[right[0]] * base + mapping[right[1]]
                            computed = func(a, b)
                            if computed is None:
                                all_match = False
                                break
                            mlist = try_match_result(computed, res_chars, op_char, mapping, set(mapping.values()), base)
                            if not any(mi[0] == fmt for mi in mlist):
                                all_match = False
                                break
                        if all_match:
                            op_assignments[op_char] = op_name
                            fmt_assignments[op_char] = fmt
                            found = True
                            break
                    if found:
                        break
                if not found:
                    ok = False
                    break
            if ok:
                return ('solved', concat_ops, mapping, op_assignments, base, fmt_assignments)

    return None


def compute_answer(result_tuple, query_str):
    """Given a successful solve result, compute the answer for the query."""
    if result_tuple is None:
        return None
    tag = result_tuple[0]
    if tag == 'all_concat':
        concat_ops = result_tuple[1]
        if len(query_str) != 5:
            return None
        op_char = query_str[2]
        if op_char not in concat_ops:
            return None
        left = query_str[:2]
        right = query_str[3:5]
        if concat_ops[op_char] == 'concat':
            return left + right
        else:
            return right + left
    elif tag == 'solved':
        _, concat_ops, mapping, op_assignments, base, fmt_assignments = result_tuple
        if len(query_str) != 5:
            return None
        op_char = query_str[2]
        left = (query_str[0], query_str[1])
        right = (query_str[3], query_str[4])

        # Check if this op is a concat op
        if op_char in concat_ops:
            l = query_str[:2]
            r = query_str[3:5]
            if concat_ops[op_char] == 'concat':
                return l + r
            else:
                return r + l

        if op_char not in op_assignments:
            return None
        op_name = op_assignments[op_char]
        fmt = fmt_assignments[op_char]
        func = dict(ALL_OPS)[op_name]

        if left[0] not in mapping or left[1] not in mapping:
            return None
        if right[0] not in mapping or right[1] not in mapping:
            return None

        a = mapping[left[0]] * base + mapping[left[1]]
        b = mapping[right[0]] * base + mapping[right[1]]
        computed = func(a, b)
        if computed is None:
            return None

        d2s = {d: s for s, d in mapping.items()}

        abs_val = abs(computed)
        if fmt == 'none':
            digits = decode_val(computed, base)
            if digits is None:
                return None
            return ''.join(d2s.get(d, '?') for d in digits)
        elif fmt == 'neg_prefix':
            digits = decode_val(abs_val, base)
            if digits is None:
                return None
            return op_char + ''.join(d2s.get(d, '?') for d in digits)
        elif fmt == 'prefix':
            digits = decode_val(computed, base)
            if digits is None:
                return None
            return op_char + ''.join(d2s.get(d, '?') for d in digits)
        elif fmt == 'pos_prefix':
            if computed < 0 or computed == 0:
                digits = decode_val(abs_val, base)
                if digits is None:
                    return None
                return ''.join(d2s.get(d, '?') for d in digits)
            else:
                digits = decode_val(computed, base)
                if digits is None:
                    return None
                return op_char + ''.join(d2s.get(d, '?') for d in digits)
    return None


def main():
    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    sym_rows = [r for r in rows if 'transformation rules' in r['prompt']]
    symbolic_rows = []
    for row in sym_rows:
        examples, query_str = parse_symbol_problem(row['prompt'])
        if not examples or is_numeric(examples):
            continue
        symbolic_rows.append(row)

    print(f"Total symbolic: {len(symbolic_rows)}")

    test_count = int(sys.argv[1]) if len(sys.argv) > 1 else len(symbolic_rows)
    solved = 0
    failed = 0
    all_concat_cnt = 0
    solved_op_freq = defaultdict(int)
    solved_fmt_freq = defaultdict(int)
    timeout_count = 0
    answer_match = 0
    answer_mismatch = 0

    for i, row in enumerate(symbolic_rows[:test_count]):
        examples, query_str = parse_symbol_problem(row['prompt'])
        answer = row['answer'].strip()
        t0 = time.time()

        if (i+1) % 50 == 0 or i < 5:
            print(f"[{i+1}/{test_count}]...", flush=True)

        result = try_solve(examples, query_str, answer)
        elapsed = time.time() - t0
        if result is None:
            failed += 1
            if elapsed > 10:
                timeout_count += 1
            if failed <= 10:
                print(f"  ❌ id={row['id']} gold={answer} (None, {elapsed:.1f}s)")
        elif result[0] == 'all_concat':
            all_concat_cnt += 1
            # Verify answer
            pred = compute_answer(result, query_str)
            if pred == answer:
                answer_match += 1
            else:
                answer_mismatch += 1
                print(f"  ⚠️ concat mismatch id={row['id']} gold={answer} pred={pred}")
        elif result[0] == 'solved':
            solved += 1
            _, _, _, op_assign, base, fmt_assign = result
            for oc, on in op_assign.items():
                solved_op_freq[on] += 1
            for oc, fn in fmt_assign.items():
                solved_fmt_freq[fn] += 1
            # Verify answer
            pred = compute_answer(result, query_str)
            if pred == answer:
                answer_match += 1
                if solved <= 20:
                    print(f"  ✅ id={row['id']} gold={answer} base={base} ({elapsed:.1f}s)")
            else:
                answer_mismatch += 1
                print(f"  ⚠️ MISMATCH id={row['id']} gold={answer} pred={pred} base={base}")

        if (i+1) % 100 == 0:
            total = solved + all_concat_cnt + failed
            print(f"  Progress: {i+1}/{test_count}, Solved={solved}, Concat={all_concat_cnt}, Failed={failed}, Timeouts={timeout_count}", flush=True)

    print(f"\n{'='*50}")
    print(f"Solved: {solved}, Concat: {all_concat_cnt}, Failed: {failed}")
    print(f"Timeouts (>10s): {timeout_count}")
    print(f"Answer match: {answer_match}, Answer mismatch: {answer_mismatch}")
    total_success = solved + all_concat_cnt
    print(f"Success rate: {100*total_success/max(test_count,1):.1f}% ({total_success}/{test_count})")

    print(f"\n--- Solved: Op frequency ---")
    for op, cnt in sorted(solved_op_freq.items(), key=lambda x: -x[1]):
        print(f"  {op}: {cnt}")

    print(f"\n--- Solved: Fmt frequency ---")
    for fmt, cnt in sorted(solved_fmt_freq.items(), key=lambda x: -x[1]):
        print(f"  {fmt}: {cnt}")


if __name__ == '__main__':
    main()
