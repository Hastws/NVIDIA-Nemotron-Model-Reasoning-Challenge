#!/usr/bin/env python3
import csv
import re
from collections import defaultdict

INPUT_FILE = '/home/SENSETIME/quchunzhi/ws/develop/senseauto-perception-camera/perception_camera_sdk/train(1).csv'
OUTPUT_FILE = '/home/SENSETIME/quchunzhi/ws/develop/senseauto-perception-camera/perception_camera_sdk/numeric_eq_cot.csv'
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

PLAIN_DESC = {
    'add': 'a+b',
    'sub': 'a-b',
    'abs_diff': '|a-b|',
    'mul': 'a×b',
    'mul_add1': 'a×b+1',
    'mul_sub1': 'a×b-1',
    'add_add1': 'a+b+1',
    'add_sub1': 'a+b-1',
    'mod': 'max(a,b) mod min(a,b)',
    'concat': 'concatenation of a and b',
    'rev_concat': 'reverse concatenation (b then a)',
}

REV_DESC = {
    'add': 'reverse-add: reverse each number, add, reverse result',
    'sub': 'reverse-subtract: reverse each number, subtract, reverse result',
    'abs_diff': 'reverse-absolute-difference: reverse each number, take |difference|, reverse result',
    'mul': 'reverse-multiply: reverse each number, multiply, reverse result',
    'mul_add1': 'reverse-multiply-plus-one: reverse each number, multiply and add 1, reverse result',
    'mul_sub1': 'reverse-multiply-minus-one: reverse each number, multiply and subtract 1, reverse result',
    'add_add1': 'reverse-add-plus-one: reverse each number, add and plus 1, reverse result',
    'add_sub1': 'reverse-add-minus-one: reverse each number, add and subtract 1, reverse result',
    'mod': 'reverse-modulo: reverse each number, take max mod min, reverse result',
}

FMT_DESC = {
    'none': '',
    'prefix': '; result is prefixed with the operator symbol',
    'suffix': '; result is suffixed with the operator symbol',
    'pos_prefix': '; positive results are prefixed with the operator symbol, negative/zero results show absolute value only',
    'pos_suffix': '; positive results are suffixed with the operator symbol, negative/zero results show absolute value only',
}


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
        if isinstance(val, int):
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
        return str(val)
    else:
        if isinstance(val, int):
            if fmt == 'pos_prefix':
                return (op_char + str(val)) if val > 0 else str(abs(val))
            elif fmt == 'pos_suffix':
                return (str(val) + op_char) if val > 0 else str(abs(val))
            elif fmt == 'prefix':
                return op_char + str(abs(val))
            elif fmt == 'suffix':
                return str(abs(val)) + op_char
            elif val < 0:
                return op_char + str(abs(val))
            else:
                return str(val)
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


def gen_op_description(op_char, base_name, is_rev, fmt):
    if base_name == 'concat':
        return f"'{op_char}' represents concatenation of a and b"
    if base_name == 'rev_concat':
        return f"'{op_char}' represents reverse concatenation (b followed by a)"
    fmt_note = FMT_DESC.get(fmt, '')
    if is_rev:
        desc = REV_DESC.get(base_name, base_name)
        return f"'{op_char}' applies {desc}{fmt_note}"
    else:
        formula = PLAIN_DESC.get(base_name, base_name)
        return f"'{op_char}' represents {formula}{fmt_note}"


def gen_example_verification(op_char, base_name, is_rev, fmt, entries):
    entry = entries[0]
    a, b, a_str, b_str, rhs = entry
    if base_name == 'concat':
        return f"{a_str}{op_char}{b_str} = \"{a_str}\"+\"{b_str}\" = {rhs} ✓"
    if base_name == 'rev_concat':
        return f"{a_str}{op_char}{b_str} = \"{b_str}\"+\"{a_str}\" = {rhs} ✓"
    if is_rev:
        ra_s = rev_str(a_str)
        rb_s = rev_str(b_str)
        ra, rb = int(ra_s), int(rb_s)
        func = BASE_FUNC[base_name]
        raw_val = func(ra, rb)
        steps = f"rev({a_str})={ra_s}, rev({b_str})={rb_s}"
        if base_name == 'add':
            steps += f", {ra}+{rb}={raw_val}"
        elif base_name == 'sub':
            steps += f", {ra}-{rb}={raw_val}"
        elif base_name == 'abs_diff':
            steps += f", |{ra}-{rb}|={abs(ra - rb)}"
            raw_val = abs(ra - rb)
        elif base_name == 'mul':
            steps += f", {ra}×{rb}={raw_val}"
        elif base_name == 'mul_add1':
            steps += f", {ra}×{rb}+1={raw_val}"
        elif base_name == 'mul_sub1':
            steps += f", {ra}×{rb}-1={raw_val}"
        elif base_name == 'add_add1':
            steps += f", {ra}+{rb}+1={raw_val}"
        elif base_name == 'add_sub1':
            steps += f", {ra}+{rb}-1={raw_val}"
        elif base_name == 'mod':
            big, small = max(ra, rb), min(ra, rb)
            steps += f", {big} mod {small}={raw_val}"
        neg = raw_val < 0
        abs_val = abs(raw_val)
        rev_result = rev_str(str(abs_val))
        steps += f", rev({abs_val})={rev_result}"
        if fmt == 'pos_prefix':
            if not neg and raw_val != 0:
                steps += f", positive → {op_char}{rev_result}"
            else:
                steps += f", negative/zero → {rev_result}"
        elif fmt == 'pos_suffix':
            if not neg and raw_val != 0:
                steps += f", positive → {rev_result}{op_char}"
            else:
                steps += f", negative/zero → {rev_result}"
        return f"{a_str}{op_char}{b_str}: {steps} = {rhs} ✓"
    else:
        if base_name == 'add':
            expr = f"{a}+{b}={a + b}"
        elif base_name == 'sub':
            expr = f"{a}-{b}={a - b}"
        elif base_name == 'abs_diff':
            expr = f"|{a}-{b}|={abs(a - b)}"
        elif base_name == 'mul':
            expr = f"{a}×{b}={a * b}"
        elif base_name == 'mul_add1':
            expr = f"{a}×{b}+1={a * b + 1}"
        elif base_name == 'mul_sub1':
            expr = f"{a}×{b}-1={a * b - 1}"
        elif base_name == 'add_add1':
            expr = f"{a}+{b}+1={a + b + 1}"
        elif base_name == 'add_sub1':
            expr = f"{a}+{b}-1={a + b - 1}"
        elif base_name == 'mod':
            big, small = max(a, b), min(a, b)
            expr = f"{big} mod {small}={big % small}"
        else:
            expr = f"{a},{b}"
        val = BASE_FUNC[base_name](a, b)
        if fmt == 'pos_prefix':
            if val > 0:
                expr += f", positive → {op_char}{val}"
            else:
                expr += f", negative/zero → {abs(val)}"
        elif fmt == 'pos_suffix':
            if val > 0:
                expr += f", positive → {val}{op_char}"
            else:
                expr += f", negative/zero → {abs(val)}"
        return f"{a_str}{op_char}{b_str} = {expr} = {rhs} ✓"


def gen_query_steps(a, b, a_str, b_str, op_char, base_name, is_rev, fmt, answer):
    if base_name == 'concat':
        return f"{a_str}{op_char}{b_str} = \"{a_str}\"+\"{b_str}\" = {answer}"
    if base_name == 'rev_concat':
        return f"{a_str}{op_char}{b_str} = \"{b_str}\"+\"{a_str}\" = {answer}"
    if is_rev:
        ra_s = rev_str(a_str)
        rb_s = rev_str(b_str)
        ra, rb = int(ra_s), int(rb_s)
        func = BASE_FUNC[base_name]
        raw_val = func(ra, rb)
        steps = f"rev({a_str})={ra_s}, rev({b_str})={rb_s}"
        if base_name == 'add':
            steps += f", {ra}+{rb}={raw_val}"
        elif base_name == 'sub':
            steps += f", {ra}-{rb}={raw_val}"
        elif base_name == 'abs_diff':
            steps += f", |{ra}-{rb}|={abs(ra - rb)}"
            raw_val = abs(ra - rb)
        elif base_name == 'mul':
            steps += f", {ra}×{rb}={raw_val}"
        elif base_name == 'mul_add1':
            steps += f", {ra}×{rb}+1={raw_val}"
        elif base_name == 'mul_sub1':
            steps += f", {ra}×{rb}-1={raw_val}"
        elif base_name == 'add_add1':
            steps += f", {ra}+{rb}+1={raw_val}"
        elif base_name == 'add_sub1':
            steps += f", {ra}+{rb}-1={raw_val}"
        elif base_name == 'mod':
            big, small = max(ra, rb), min(ra, rb)
            steps += f", {big} mod {small}={raw_val}"
        neg = raw_val < 0
        abs_val = abs(raw_val)
        rev_result = rev_str(str(abs_val))
        steps += f", rev({abs_val})={rev_result}"
        if fmt == 'pos_prefix':
            if not neg and raw_val != 0:
                steps += f", positive → {op_char}{rev_result}"
            else:
                steps += f", negative/zero → {rev_result}"
        elif fmt == 'pos_suffix':
            if not neg and raw_val != 0:
                steps += f", positive → {rev_result}{op_char}"
            else:
                steps += f", negative/zero → {rev_result}"
        elif fmt == 'prefix':
            steps += f" → {op_char}{rev_result}"
        elif fmt == 'suffix':
            steps += f" → {rev_result}{op_char}"
        return f"{a_str}{op_char}{b_str}: {steps} = {answer}"
    else:
        val = BASE_FUNC[base_name](a, b)
        if base_name == 'add':
            expr = f"{a}+{b}={val}"
        elif base_name == 'sub':
            expr = f"{a}-{b}={val}"
        elif base_name == 'abs_diff':
            expr = f"|{a}-{b}|={abs(a - b)}"
        elif base_name == 'mul':
            expr = f"{a}×{b}={val}"
        elif base_name == 'mul_add1':
            expr = f"{a}×{b}+1={val}"
        elif base_name == 'mul_sub1':
            expr = f"{a}×{b}-1={val}"
        elif base_name == 'add_add1':
            expr = f"{a}+{b}+1={val}"
        elif base_name == 'add_sub1':
            expr = f"{a}+{b}-1={val}"
        elif base_name == 'mod':
            big, small = max(a, b), min(a, b)
            expr = f"{big} mod {small}={val}"
        else:
            expr = str(val)
        if fmt == 'pos_prefix':
            if val > 0:
                expr += f", positive → {op_char}{val}"
            else:
                expr += f", negative/zero → {abs(val)}"
        elif fmt == 'pos_suffix':
            if val > 0:
                expr += f", positive → {val}{op_char}"
            else:
                expr += f", negative/zero → {abs(val)}"
        elif fmt == 'prefix':
            expr += f" → {op_char}{abs(val)}"
        elif fmt == 'suffix':
            expr += f" → {abs(val)}{op_char}"
        return f"{a_str}{op_char}{b_str} = {expr} = {answer}"


def gen_cot(examples, query_str, answer, op_results, by_op, inferred_op=None):
    parts = ["Analyzing operators:"]
    for oc, (tag, base_name, is_rev, fmt) in op_results.items():
        desc = gen_op_description(oc, base_name, is_rev, fmt)
        entries = by_op[oc]
        verify = gen_example_verification(oc, base_name, is_rev, fmt, entries)
        parts.append(f"{desc}. {verify}")

    if not query_str:
        return " ".join(parts)

    m = NUM_PATTERN.match(query_str)
    if not m:
        parts.append(f"Query: {query_str} = {answer}")
        return " ".join(parts)

    a_str, q_op, b_str = m.group(1), m.group(2), m.group(3)
    a, b = int(a_str), int(b_str)

    if q_op in op_results:
        tag, base_name, is_rev, fmt = op_results[q_op]
        query_steps = gen_query_steps(a, b, a_str, b_str, q_op, base_name, is_rev, fmt, answer)
        parts.append(f"Query: {query_steps}")
    elif inferred_op:
        tag, base_name, is_rev, fmt = inferred_op
        inf_desc = gen_op_description(q_op, base_name, is_rev, fmt)
        query_steps = gen_query_steps(a, b, a_str, b_str, q_op, base_name, is_rev, fmt, answer)
        parts.append(f"Query uses '{q_op}' (new operator): {inf_desc}. {query_steps}")
    else:
        parts.append(f"Query: {query_str} = {answer}")

    return " ".join(parts)


def main():
    rows = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Total rows: {len(rows)}")

    sym_rows = [r for r in rows if 'transformation rules' in r['prompt']]
    print(f"Symbol transform: {len(sym_rows)}")

    results = []
    solved = 0
    failed = 0

    for row in sym_rows:
        prompt = row['prompt']
        answer = row['answer'].strip()
        rid = row['id']
        examples, query_str = parse_symbol_problem(prompt)
        if not examples:
            continue

        all_num, by_op = classify_and_group(examples)
        if not all_num:
            continue

        candidates = {}
        all_found = True
        for oc, entries in by_op.items():
            ms = find_matching_ops(oc, entries)
            if ms:
                candidates[oc] = ms
            else:
                all_found = False
        if not all_found:
            continue

        if not query_str:
            op_results = {oc: ms[0] for oc, ms in candidates.items()}
            cot = gen_cot(examples, query_str, answer, op_results, by_op)
            results.append((rid, prompt, answer, cot))
            solved += 1
            continue

        m = NUM_PATTERN.match(query_str)
        if not m:
            op_results = {oc: ms[0] for oc, ms in candidates.items()}
            cot = gen_cot(examples, query_str, answer, op_results, by_op)
            results.append((rid, prompt, answer, cot))
            solved += 1
            continue

        a_str, q_op, b_str = m.group(1), m.group(2), m.group(3)
        a, b = int(a_str), int(b_str)

        inferred_op = None
        if q_op in candidates:
            best = None
            for tag, bn, ir, fm in candidates[q_op]:
                pred = compute_full(a, b, a_str, b_str, bn, ir, q_op, fm)
                if pred and check_eq(pred, answer):
                    best = (tag, bn, ir, fm)
                    break
            if best:
                op_results = {}
                for oc, ms in candidates.items():
                    if oc == q_op:
                        op_results[oc] = best
                    else:
                        op_results[oc] = ms[0]
                cot = gen_cot(examples, query_str, answer, op_results, by_op)
                results.append((rid, prompt, answer, cot))
                solved += 1
            else:
                failed += 1
        else:
            found = False
            for tag, bn, ir, fm in ALL_OPS:
                pred = compute_full(a, b, a_str, b_str, bn, ir, q_op, fm)
                if pred and check_eq(pred, answer):
                    inferred_op = (tag, bn, ir, fm)
                    found = True
                    break
            if found:
                op_results = {oc: ms[0] for oc, ms in candidates.items()}
                cot = gen_cot(examples, query_str, answer, op_results, by_op, inferred_op)
                results.append((rid, prompt, answer, cot))
                solved += 1
            else:
                failed += 1

    print(f"Solved with CoT: {solved}, Failed: {failed}")

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prompt', 'answer', 'thinking'])
        for rid, prompt, answer, cot in results:
            writer.writerow([rid, prompt, answer, cot])
    print(f"Written to {OUTPUT_FILE}")

    print(f"\nSample CoTs:")
    for rid, prompt, answer, cot in results[:5]:
        print(f"\n[{rid}] answer={answer}")
        print(f"  CoT: {cot}")


if __name__ == '__main__':
    main()
