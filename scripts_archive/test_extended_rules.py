#!/usr/bin/env python3
"""测试扩展规则集能额外解多少 symbol 题"""
import csv
from collections import defaultdict, Counter

OP_EXT = set(chr(c) for c in range(33, 127) if not chr(c).isalnum() and chr(c) != '=')

def try_split(expr):
    for i, c in enumerate(expr):
        if c in OP_EXT and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i + 1:]
    return None

def parse_symbol(prompt):
    lines = prompt.strip().split('\n')
    examples, query = [], None
    for line in lines:
        line = line.strip()
        if 'determine the result for:' in line.lower():
            query = line.split(':')[-1].strip()
        elif ('=' in line and 'alice' not in line.lower()
              and 'equation' not in line.lower()
              and 'transformation' not in line.lower()
              and 'determine' not in line.lower()
              and 'below' not in line.lower()):
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                if lhs and rhs:
                    examples.append((lhs, rhs))
    return examples, query

RULES = [
    ('a+b', lambda a, b: a + b),
    ('a-b', lambda a, b: a - b),
    ('b-a', lambda a, b: b - a),
    ('a*b', lambda a, b: a * b),
    ('a*b-1', lambda a, b: a * b - 1),
    ('a*b+1', lambda a, b: a * b + 1),
    ('a//b', lambda a, b: a // b if b else None),
    ('b//a', lambda a, b: b // a if a else None),
    ('a%b', lambda a, b: a % b if b else None),
    ('b%a', lambda a, b: b % a if a else None),
    ('a^b', lambda a, b: a ^ b),
    ('a&b', lambda a, b: a & b),
    ('a|b', lambda a, b: a | b),
    ('concat_ab', lambda a, b: int(str(abs(a)) + str(abs(b)))),
    ('concat_ba', lambda a, b: int(str(abs(b)) + str(abs(a)))),
    ('abs(a-b)', lambda a, b: abs(a - b)),
    ('(a+b)//2', lambda a, b: (a + b) // 2),
    ('a*b-a', lambda a, b: a * b - a),
    ('a*b-b', lambda a, b: a * b - b),
    ('a*b+a', lambda a, b: a * b + a),
    ('a*b+b', lambda a, b: a * b + b),
    ('a*b-a-b', lambda a, b: a * b - a - b),
    ('a*b+a+b', lambda a, b: a * b + a + b),
    ('a*b-a+b', lambda a, b: a * b - a + b),
    ('a*b+a-b', lambda a, b: a * b + a - b),
    ('a**2+b**2', lambda a, b: a * a + b * b),
    ('a**2-b**2', lambda a, b: a * a - b * b),
    ('(a-b)**2', lambda a, b: (a - b) ** 2),
    ('(a+b)**2', lambda a, b: (a + b) ** 2),
    ('a*(a+b)', lambda a, b: a * (a + b)),
    ('b*(a+b)', lambda a, b: b * (a + b)),
    ('a*(a-b)', lambda a, b: a * (a - b)),
    ('a*(b-a)', lambda a, b: a * (b - a)),
    ('b*(a-b)', lambda a, b: b * (a - b)),
    ('b*(b-a)', lambda a, b: b * (b - a)),
    ('a**2', lambda a, b: a * a),
    ('b**2', lambda a, b: b * b),
    ('a**2+b', lambda a, b: a * a + b),
    ('a**2-b', lambda a, b: a * a - b),
    ('b**2+a', lambda a, b: b * b + a),
    ('b**2-a', lambda a, b: b * b - a),
    ('a*2+b', lambda a, b: a * 2 + b),
    ('a+b*2', lambda a, b: a + b * 2),
    ('a*2-b', lambda a, b: a * 2 - b),
    ('a-b*2', lambda a, b: a - b * 2),
    ('(a+b)*2', lambda a, b: (a + b) * 2),
    ('(a-b)*2', lambda a, b: (a - b) * 2),
    ('a*2', lambda a, b: a * 2),
    ('b*2', lambda a, b: b * 2),
    ('max(a,b)', lambda a, b: max(a, b)),
    ('min(a,b)', lambda a, b: min(a, b)),
]

DIGIT_PAIR_OPS = [
    ('dpair_mul', lambda a, b: str(a * b)),
    ('dpair_add', lambda a, b: str(a + b)),
    ('dpair_sub', lambda a, b: str(a - b)),
    ('dpair_sub_rev', lambda a, b: str(b - a)),
    ('dpair_mul-1', lambda a, b: str(a * b - 1)),
    ('dpair_mul+1', lambda a, b: str(a * b + 1)),
    ('dpair_a2+b', lambda a, b: str(a * a + b)),
    ('dpair_a+b2', lambda a, b: str(a + b * b)),
    ('dpair_a2-b', lambda a, b: str(a * a - b)),
]

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

B94_OPS = [
    ('b94_add', lambda l, r: b94_to_str(str_to_b94(l) + str_to_b94(r))),
    ('b94_sub', lambda l, r: b94_to_str(str_to_b94(l) - str_to_b94(r))),
    ('b94_sub_rev', lambda l, r: b94_to_str(str_to_b94(r) - str_to_b94(l))),
    ('b94_mul', lambda l, r: b94_to_str(str_to_b94(l) * str_to_b94(r))),
    ('concat', lambda l, r: l + r),
    ('concat_rev', lambda l, r: r + l),
]

CW_OPS = [
    ('cw_add', lambda a, b: chr(((ord(a) - 33) + (ord(b) - 33)) % 94 + 33)),
    ('cw_sub', lambda a, b: chr(((ord(a) - 33) - (ord(b) - 33)) % 94 + 33)),
    ('cw_sub_rev', lambda a, b: chr(((ord(b) - 33) - (ord(a) - 33)) % 94 + 33)),
    ('cw_xor', lambda a, b: chr(((ord(a) - 33) ^ (ord(b) - 33)) % 94 + 33)),
    ('cw_mul', lambda a, b: chr(((ord(a) - 33) * (ord(b) - 33)) % 94 + 33)),
    ('cw_and', lambda a, b: chr(((ord(a) - 33) & (ord(b) - 33)) % 94 + 33)),
    ('cw_or', lambda a, b: chr(((ord(a) - 33) | (ord(b) - 33)) % 94 + 33)),
]


def main():
    rows = list(csv.DictReader(open('data/train_annotated.csv')))
    symbol_unsolved = [r for r in rows if r['type'] == 'symbol' and r['solvable'] != 'True']

    total_newly_solved = 0
    rule_dist = Counter()

    for r in symbol_unsolved:
        examples, query = parse_symbol(r['prompt'])
        if not examples or not query:
            continue
        sp = try_split(query)
        if not sp:
            continue
        ql, qop, qr = sp

        op_group = []
        for lhs, rhs in examples:
            for i, c in enumerate(lhs):
                if c == qop and i > 0 and i < len(lhs) - 1:
                    op_group.append((lhs[:i], lhs[i + 1:], rhs))
                    break
        if not op_group:
            continue

        solved = False

        # --- Whole-number rules ---
        try:
            num_pairs = [(int(l), int(ri), int(res)) for l, ri, res in op_group]
            nql, nqr = int(ql), int(qr)

            for rn, fn in RULES:
                if all(fn(a, b) is not None and fn(a, b) == res for a, b, res in num_pairs):
                    pred = fn(nql, nqr)
                    if pred is not None and str(pred) == r['answer']:
                        total_newly_solved += 1
                        rule_dist[rn] += 1
                        solved = True
                    break
            if solved:
                continue
        except:
            pass

        # --- Digit-pair rules ---
        same_len = [(l, ri, res) for l, ri, res in op_group if len(l) == len(ri)]
        if same_len and len(ql) == len(qr):
            for drn, dfn in DIGIT_PAIR_OPS:
                all_match = True
                for l, ri, res in same_len:
                    try:
                        pred = ''.join(dfn(int(l[i]), int(ri[i])) for i in range(len(l)))
                        if pred != res:
                            all_match = False
                            break
                    except:
                        all_match = False
                        break
                if all_match:
                    try:
                        pred = ''.join(dfn(int(ql[i]), int(qr[i])) for i in range(len(ql)))
                        if pred == r['answer']:
                            total_newly_solved += 1
                            rule_dist[f'digit:{drn}'] += 1
                            solved = True
                    except:
                        pass
                    break
            if solved:
                continue

        # --- B94 / concat / charwise for non-numeric ---
        for op_name, fn in B94_OPS:
            try:
                if all(fn(l, ri) == res for l, ri, res in op_group):
                    pred = fn(ql, qr)
                    if pred == r['answer']:
                        total_newly_solved += 1
                        rule_dist[f'b94:{op_name}'] += 1
                        solved = True
                    break
            except:
                continue
        if solved:
            continue

        # Charwise
        for cw_name, cw_fn in CW_OPS:
            all_match = True
            for l, ri, res in op_group:
                if len(l) != len(ri) or len(l) != len(res):
                    all_match = False
                    break
                try:
                    pred_cw = ''.join(cw_fn(a, b) for a, b in zip(l, ri))
                    if pred_cw != res:
                        all_match = False
                        break
                except:
                    all_match = False
                    break
            if all_match and len(ql) == len(qr):
                try:
                    pred_cw = ''.join(cw_fn(a, b) for a, b in zip(ql, qr))
                    if pred_cw == r['answer']:
                        total_newly_solved += 1
                        rule_dist[f'cw:{cw_name}'] += 1
                        solved = True
                except:
                    pass
                break

    print(f'Additional correctly solvable: {total_newly_solved}')
    print(f'Currently correctly solved: 105')
    print(f'New total: {105 + total_newly_solved} / 1555 = {(105 + total_newly_solved) / 1555 * 100:.1f}%')
    print()
    print('Rule distribution:')
    for r, cnt in rule_dist.most_common():
        print(f'  {r}: {cnt}')


if __name__ == '__main__':
    main()
