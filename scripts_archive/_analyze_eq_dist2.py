#!/usr/bin/env python3
"""Analyze eq_numeric and eq_symbolic operation distribution."""
import csv, re
from collections import Counter

with open('data/sft_thinking.csv') as f:
    rows = list(csv.DictReader(f))

# ===== EQ_NUMERIC =====
eq_num = [r for r in rows if r['type'] == 'eq_numeric']
print(f"{'='*70}")
print(f"  EQ_NUMERIC 操作分布 ({len(eq_num)} rows)")
print(f"{'='*70}")

num_ops = Counter()
num_formats = Counter()
num_n_ops = Counter()

for r in eq_num:
    cot = r['thinking']
    if not cot:
        continue
    ops_found = []
    for m in re.finditer(r"'.' (?:applies|represents) ([^;,\n]+)", cot):
        op = m.group(1).strip().rstrip('.')
        ops_found.append(op)
    for m in re.finditer(r"New operator: '.' applies ([^;,\n]+)", cot):
        op = m.group(1).strip().rstrip('.')
        ops_found.append(op)
    for op in ops_found:
        num_ops[op.lower().strip()] += 1
    num_n_ops[len(ops_found)] += 1
    if 'positive results suffixed' in cot:
        num_formats['pos_suffix'] += 1
    elif 'positive results prefixed' in cot:
        num_formats['pos_prefix'] += 1
    elif 'result suffixed with op symbol' in cot:
        num_formats['suffix'] += 1
    elif 'result prefixed with op symbol' in cot:
        num_formats['prefix'] += 1
    else:
        num_formats['none'] += 1

total_ops = sum(num_ops.values())
print(f"\n  --- Raw 算子 (共 {total_ops} instances) ---")

# Categorize
cat_map = {}
for op in num_ops:
    if 'reverse-subtract' in op: cat_map[op] = 'rev-subtract'
    elif 'reverse-add-plus-one' in op: cat_map[op] = 'rev-add+1'
    elif 'reverse-add-minus-one' in op: cat_map[op] = 'rev-add-1'
    elif 'reverse-multiply-plus-one' in op: cat_map[op] = 'rev-mul+1'
    elif 'reverse-multiply-minus-one' in op: cat_map[op] = 'rev-mul-1'
    elif 'reverse-multiply' in op: cat_map[op] = 'rev-multiply'
    elif 'reverse-add' in op: cat_map[op] = 'rev-add'
    elif 'reverse-|difference|' in op: cat_map[op] = 'rev-|diff|'
    elif 'reverse-modulo' in op: cat_map[op] = 'rev-modulo'
    elif 'reverse concatenation' in op: cat_map[op] = 'rev_concat'
    elif 'concatenation' in op: cat_map[op] = 'concat'
    elif op in ('a+b+1',): cat_map[op] = 'add+1'
    elif op in ('a+b-1',): cat_map[op] = 'add-1'
    elif 'mod' in op: cat_map[op] = 'modulo'
    elif op in ('a+b',): cat_map[op] = 'add'
    elif op in ('a-b',): cat_map[op] = 'subtract'
    elif op in ('|a-b|',): cat_map[op] = 'abs_diff'
    elif 'b' in op and '*' in op or 'mul' in op: cat_map[op] = 'multiply'
    else: cat_map[op] = op  # keep as-is

cat_counts = Counter()
for op, cnt in num_ops.items():
    cat_counts[cat_map.get(op, op)] += cnt

for cat, cnt in cat_counts.most_common():
    pct = cnt / total_ops * 100
    bar = '#' * max(1, int(pct * 2))
    print(f"  {cat:25s} {cnt:4d} ({pct:5.1f}%) {bar}")

plain = sum(c for o, c in num_ops.items() if 'reverse' not in o and 'rev' not in o)
rev = total_ops - plain
print(f"\n  Plain: {plain} ({plain/total_ops*100:.1f}%), Reverse: {rev} ({rev/total_ops*100:.1f}%)")

print(f"\n  --- 格式修饰符 ---")
for k, c in num_formats.most_common():
    print(f"  {k:15s} {c:4d} ({c/len(eq_num)*100:.1f}%)")

print(f"\n  --- 每题算子数 ---")
for n, c in sorted(num_n_ops.items()):
    print(f"  {n} ops/problem: {c:4d} ({c/len(eq_num)*100:.1f}%)")

# ===== EQ_SYMBOLIC =====
eq_sym = [r for r in rows if r['type'] == 'eq_symbolic']
print(f"\n{'='*70}")
print(f"  EQ_SYMBOLIC 操作分布 ({len(eq_sym)} rows)")
print(f"{'='*70}")

sym_ops = Counter()
sym_bases = Counter()
sym_formats = Counter()
sym_n_ops = Counter()

for r in eq_sym:
    cot = r['thinking']
    if not cot:
        continue
    m = re.search(r'Base-(\d+)', cot)
    if m:
        sym_bases[int(m.group(1))] += 1
    lines = cot.strip().split('\n')
    if len(lines) >= 2:
        ops_line = lines[1]
        n = 0
        for m2 in re.finditer(r"'.'=([^;]+?)(?:;|$)", ops_line):
            op = m2.group(1).strip().rstrip('.')
            fmt_match = re.search(r'\((\w+)\)', op)
            if fmt_match:
                sym_formats[fmt_match.group(1)] += 1
                op = re.sub(r'\s*\(\w+\)', '', op).strip()
            else:
                sym_formats['none'] += 1
            sym_ops[op] += 1
            n += 1
        sym_n_ops[n] += 1

total_sym = sum(sym_ops.values())
print(f"\n  --- 算子分布 (共 {total_sym} instances) ---")
for op, cnt in sym_ops.most_common():
    pct = cnt / total_sym * 100
    bar = '#' * max(1, int(pct * 2))
    print(f"  {op:30s} {cnt:4d} ({pct:5.1f}%) {bar}")

print(f"\n  --- Base 分布 ---")
for base, cnt in sym_bases.most_common():
    print(f"  base-{base:2d}: {cnt:4d} ({cnt/len(eq_sym)*100:.1f}%)")

print(f"\n  --- 格式修饰符 ---")
for k, cnt in sym_formats.most_common():
    print(f"  {k:15s} {cnt:4d} ({cnt/total_sym*100:.1f}%)")

print(f"\n  --- 每题算子数 ---")
for n, cnt in sorted(sym_n_ops.items()):
    print(f"  {n} ops/problem: {cnt:4d} ({cnt/len(eq_sym)*100:.1f}%)")
