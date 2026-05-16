#!/usr/bin/env python3
import csv
import re
from collections import defaultdict

INPUT_FILE = '/home/SENSETIME/quchunzhi/ws/develop/senseauto-perception-camera/perception_camera_sdk/train(1).csv'
NUM_PATTERN = re.compile(r'^(\d+)([^\d])(\d+)$')

BASE_OPS = [
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

BASE_FUNC = dict(BASE_OPS)

FMTS = ['none', 'prefix', 'suffix', 'pos_prefix', 'pos_suffix']


def rev_str(s):
    return s[::-1]


def compute_raw(a, b, a_str, b_str, base_name, is_rev):
    if base_name == 'concat':
        return a_str + b_str
    if base_name == 'rev_concat':
        return b_str + a_str
    func = BASE_FUNC.get(base_name)
    if func is None:
        return None
    if is_rev:
        ra, rb = int(rev_str(a_str)), int(rev_str(b_str))
        return func(ra, rb)
    else:
        return func(a, b)


def compute_full(a, b, a_str, b_str, base_name, is_rev, op_char, fmt='none'):
    val = compute_raw(a, b, a_str, b_str, base_name, is_rev)
    if val is None:
        return None

    if base_name in ('concat', 'rev_concat'):
        s = val
        if fmt == 'prefix':
            return op_char + s
        elif fmt == 'suffix':
            return s + op_char
        return s

    if is_rev:
        neg = val < 0
        abs_s = rev_str(str(abs(val)))
        if fmt == 'pos_prefix':
            return (op_char + abs_s) if (not neg and val != 0) else abs_s
        elif fmt == 'pos_suffix':
            return (abs_s + op_char) if (not neg and val != 0) else abs_s
        elif fmt == 'prefix':
            return op_char + abs_s
        elif fmt == 'suffix':
            return abs_s + op_char
        elif neg:
            return op_char + abs_s
        else:
            return abs_s
    else:
        if fmt == 'pos_prefix':
            if val > 0:
                return op_char + str(val)
            return str(abs(val))
        elif fmt == 'pos_suffix':
            if val > 0:
                return str(val) + op_char
            return str(abs(val))
        elif fmt == 'prefix':
            return op_char + str(abs(val)) if val < 0 else op_char + str(val)
        elif fmt == 'suffix':
            return str(abs(val)) + op_char if val < 0 else str(val) + op_char
        else:
            if val < 0:
                return op_char + str(abs(val))
            return str(val)


def check_eq(computed, expected):
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


def build_ops():
    ops = []
    for name, _ in BASE_OPS:
        for is_rev in [False, True]:
            for fmt in FMTS:
                tag = ('rev_' if is_rev else 'plain_') + name
                if fmt == 'prefix':
                    tag += '_pfx'
                elif fmt == 'suffix':
                    tag += '_sfx'
                elif fmt == 'pos_prefix':
                    tag += '_pospfx'
                elif fmt == 'pos_suffix':
                    tag += '_possfx'
                ops.append((tag, name, is_rev, fmt))
    for fmt in ['none', 'prefix', 'suffix']:
        sfx = '' if fmt == 'none' else ('_pfx' if fmt == 'prefix' else '_sfx')
        ops.append(('concat' + sfx, 'concat', False, fmt))
        ops.append(('rev_concat' + sfx, 'rev_concat', False, fmt))
    return ops


ALL_OPS = build_ops()


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


def classify_and_group(examples):
    by_op = defaultdict(list)
    all_numeric = True
    for lhs, rhs in examples:
        m = NUM_PATTERN.match(lhs)
        if not m:
            all_numeric = False
            break
        a_str, op_char, b_str = m.group(1), m.group(2), m.group(3)
        a, b = int(a_str), int(b_str)
        by_op[op_char].append((a, b, a_str, b_str, rhs))
    return all_numeric, by_op


def find_matching_ops(op_char, entries):
    matches = []
    for tag, base_name, is_rev, fmt in ALL_OPS:
        ok = True
        for a, b, a_str, b_str, rhs in entries:
            computed = compute_full(a, b, a_str, b_str, base_name, is_rev, op_char, fmt)
            if not check_eq(computed, rhs):
                ok = False
                break
        if ok:
            matches.append((tag, base_name, is_rev, fmt))
    return matches


def main():
    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Total rows: {len(rows)}")

    sym_rows = [r for r in rows if 'transformation rules' in r['prompt']]
    print(f"Symbol transform: {len(sym_rows)}")

    numeric = 0
    symbolic = 0
    solved = 0
    verified = 0
    inferred = 0
    failed = []
    unsolved_ops = []
    op_freq = defaultdict(int)

    for row in sym_rows:
        prompt = row['prompt']
        answer = row['answer'].strip()
        examples, query_str = parse_symbol_problem(prompt)
        if not examples:
            continue
        all_num, by_op = classify_and_group(examples)
        if not all_num:
            symbolic += 1
            continue
        numeric += 1

        candidates = {}
        all_found = True
        for oc, entries in by_op.items():
            ms = find_matching_ops(oc, entries)
            if ms:
                candidates[oc] = ms
            else:
                all_found = False
                unsolved_ops.append((row['id'], oc, entries[:2]))
        if not all_found:
            continue

        if not query_str:
            solved += 1
            for oc, ms in candidates.items():
                op_freq[ms[0][0]] += 1
            continue

        m = NUM_PATTERN.match(query_str)
        if not m:
            solved += 1
            continue
        a_str, q_op, b_str = m.group(1), m.group(2), m.group(3)
        a, b = int(a_str), int(b_str)

        if q_op in candidates:
            found = False
            for tag, bn, ir, fm in candidates[q_op]:
                pred = compute_full(a, b, a_str, b_str, bn, ir, q_op, fm)
                if pred and check_eq(pred, answer):
                    solved += 1
                    verified += 1
                    found = True
                    candidates[q_op] = [(tag, bn, ir, fm)]
                    for oc, ms in candidates.items():
                        op_freq[ms[0][0]] += 1
                    break
            if not found:
                preds = []
                for _, bn, ir, fm in candidates[q_op]:
                    preds.append(compute_full(a, b, a_str, b_str, bn, ir, q_op, fm))
                failed.append((row['id'], query_str, answer, preds, 'wrong'))
        else:
            found = False
            for tag, bn, ir, fm in ALL_OPS:
                pred = compute_full(a, b, a_str, b_str, bn, ir, q_op, fm)
                if pred and check_eq(pred, answer):
                    solved += 1
                    inferred += 1
                    found = True
                    for oc, ms in candidates.items():
                        op_freq[ms[0][0]] += 1
                    op_freq[tag] += 1
                    break
            if not found:
                failed.append((row['id'], query_str, answer, None, 'unknown'))

    total = verified + inferred
    print(f"\n{'='*50}")
    print(f"Numeric: {numeric}, Symbolic: {symbolic}")
    print(f"Solved: {solved}/{numeric} ({100*solved/max(numeric,1):.1f}%)")
    print(f"  Verified: {verified}, Inferred: {inferred}")
    print(f"  Total correct: {total}/{numeric} ({100*total/max(numeric,1):.1f}%)")
    print(f"  Failed: {len(failed)}")
    print(f"\nOp frequency (top 30):")
    for n, c in sorted(op_freq.items(), key=lambda x: -x[1])[:30]:
        print(f"  {n}: {c}")
    if failed:
        print(f"\nFailed (first 15):")
        for rid, q, ans, pred, reason in failed[:15]:
            print(f"  [{reason}] {rid}: {q} -> exp={ans}, pred={pred}")
    if unsolved_ops:
        print(f"\nUnsolved ops (first 20):")
        for rid, oc, entries in unsolved_ops[:20]:
            print(f"  {rid} op='{oc}':")
            for a, b, a_s, b_s, r in entries:
                print(f"    {a_s}{oc}{b_s} = {r}")


if __name__ == '__main__':
    main()