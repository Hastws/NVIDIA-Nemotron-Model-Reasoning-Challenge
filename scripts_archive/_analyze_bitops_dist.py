"""Analyze bit_ops operation distribution in training data."""
import csv, re
from collections import Counter, defaultdict

with open('data/sft_thinking.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

bit_rows = [r for r in rows if r['type'] == 'bit_ops' and r.get('thinking','').strip()]
print(f"bit_ops rows with thinking: {len(bit_rows)}")

op_counter = Counter()
per_problem_ops = []
SINGLE_RE = re.compile(r'Bit \d+: (in\[\d+\])\.')

for row in bit_rows:
    thinking = row['thinking']
    problem_ops = []
    for line in thinking.split('\n'):
        line = line.strip()
        if not (line.startswith('Bit') or 'always' in line):
            continue
        if 'For input' in line or 'Output:' in line:
            continue
        # skip execution lines like "bit 3: in[4]=0..."
        if re.match(r'^bit \d+:', line):
            continue
        
        op = None
        if 'always 0' in line: op = 'const_0'
        elif 'always 1' in line: op = 'const_1'
        elif re.search(r'NOT\(in\[\d+\] XOR in\[\d+\]\)', line): op = 'XNOR'
        elif re.search(r'NOT\(in\[\d+\] AND in\[\d+\]\)', line): op = 'NAND'
        elif re.search(r'NOT\(in\[\d+\] OR in\[\d+\]\)', line): op = 'NOR'
        elif re.search(r'\(in\[\d+\] AND in\[\d+\]\) XOR in\[\d+\]', line): op = 'AND_XOR'
        elif re.search(r'\(in\[\d+\] OR in\[\d+\]\) XOR in\[\d+\]', line): op = 'OR_XOR'
        elif re.search(r'\(in\[\d+\] AND in\[\d+\]\) OR in\[\d+\]', line): op = 'AND_OR'
        elif re.search(r'\(in\[\d+\] XOR in\[\d+\]\) AND in\[\d+\]', line): op = 'XOR_AND'
        elif re.search(r'\(in\[\d+\] XOR in\[\d+\]\) OR in\[\d+\]', line): op = 'XOR_OR'
        elif re.search(r'NOT\(in\[\d+\]\)', line): op = 'NOT'
        elif re.search(r'in\[\d+\] AND in\[\d+\]', line): op = 'AND'
        elif re.search(r'in\[\d+\] OR in\[\d+\]', line): op = 'OR'
        elif re.search(r'in\[\d+\] XOR in\[\d+\]', line): op = 'XOR'
        elif SINGLE_RE.search(line): op = 'COPY'
        
        if op:
            op_counter[op] += 1
            problem_ops.append(op)
    
    per_problem_ops.append(problem_ops)

total_ops = sum(op_counter.values())
print(f"\n{'='*60}")
print(f"  1. 操作类型总体分布 (共 {total_ops} 个 bit 规则)")
print(f"{'='*60}")
for op, cnt in op_counter.most_common():
    pct = 100*cnt/total_ops
    bar = '█' * int(pct/2)
    print(f"  {op:12s}: {cnt:5d} ({pct:5.1f}%) {bar}")

COMPLEXITY = {
    'const_0':0,'const_1':0,'COPY':1,'NOT':1,
    'AND':2,'OR':2,'XOR':2,'XNOR':2,'NAND':2,'NOR':2,
    'AND_XOR':3,'OR_XOR':3,'AND_OR':3,'XOR_AND':3,'XOR_OR':3,
}
complexity_counter = Counter()
for op, cnt in op_counter.items():
    complexity_counter[COMPLEXITY.get(op,-1)] += cnt

print(f"\n{'='*60}")
print(f"  2. 操作复杂度分布")
print(f"{'='*60}")
for c in sorted(complexity_counter):
    cnt = complexity_counter[c]
    label = {0:'constant',1:'1-input',2:'2-input',3:'3-input'}.get(c,'?')
    print(f"  {label:12s}: {cnt:5d} ({100*cnt/total_ops:5.1f}%)")

print(f"\n{'='*60}")
print(f"  3. 2-input 六种操作对比")
print(f"{'='*60}")
for op in ['AND','OR','XOR','XNOR','NAND','NOR']:
    cnt = op_counter.get(op,0)
    print(f"  {op:6s}: {cnt:5d} ({100*cnt/total_ops:5.1f}%)")

print(f"\n{'='*60}")
print(f"  4. 3-input 组合操作对比")
print(f"{'='*60}")
for op in ['AND_XOR','OR_XOR','AND_OR','XOR_AND','XOR_OR']:
    cnt = op_counter.get(op,0)
    print(f"  {op:10s}: {cnt:5d} ({100*cnt/total_ops:5.1f}%)")

# 每道题操作多样性
unique_per = [len(set(ops)) for ops in per_problem_ops]
max_c_per = [max((COMPLEXITY.get(o,-1) for o in ops), default=-1) for ops in per_problem_ops]

print(f"\n{'='*60}")
print(f"  5. 每道题操作多样性")
print(f"{'='*60}")
for n, cnt in sorted(Counter(unique_per).items()):
    print(f"  {n} 种操作: {cnt} 题 ({100*cnt/len(per_problem_ops):.1f}%)")

print(f"\n  题目最高复杂度:")
for c, cnt in sorted(Counter(max_c_per).items()):
    label = {0:'constant-only',1:'最高1-input',2:'最高2-input',3:'最高3-input'}.get(c,'?')
    print(f"  {label:20s}: {cnt:4d} 题 ({100*cnt/len(per_problem_ops):.1f}%)")

# ═══════════════════════════════════
# 6. 也分析 train.csv 原始数据中的 bit_ops 操作分布
# ═══════════════════════════════════
print(f"\n{'='*60}")
print(f"  6. 训练集 vs 测试集 (train.csv 全量 bit_ops)")
print(f"{'='*60}")

import sys
sys.path.insert(0, 'scripts')
from gen_thinking import gen_thinking_bit

with open('competition_data/train.csv', 'r', encoding='utf-8') as f:
    all_rows = list(csv.DictReader(f))

all_bit = [r for r in all_rows if 'bit manipulation' in r['prompt'].lower() or '8-bit binary' in r['prompt'].lower()]
print(f"  train.csv bit_ops: {len(all_bit)} rows")

# Generate thinking for ALL bit_ops and count ops
all_op_counter = Counter()
success = 0
fail = 0
for row in all_bit:
    cot = gen_thinking_bit(row['prompt'], row['answer'])
    if cot is None:
        fail += 1
        continue
    success += 1
    for line in cot.split('\n'):
        line = line.strip()
        if not (line.startswith('Bit') or 'always' in line):
            continue
        if re.match(r'^bit \d+:', line):
            continue
            
        op = None
        if 'always 0' in line: op = 'const_0'
        elif 'always 1' in line: op = 'const_1'
        elif re.search(r'NOT\(in\[\d+\] XOR in\[\d+\]\)', line): op = 'XNOR'
        elif re.search(r'NOT\(in\[\d+\] AND in\[\d+\]\)', line): op = 'NAND'
        elif re.search(r'NOT\(in\[\d+\] OR in\[\d+\]\)', line): op = 'NOR'
        elif re.search(r'\(in\[\d+\] AND in\[\d+\]\) XOR in\[\d+\]', line): op = 'AND_XOR'
        elif re.search(r'\(in\[\d+\] OR in\[\d+\]\) XOR in\[\d+\]', line): op = 'OR_XOR'
        elif re.search(r'\(in\[\d+\] AND in\[\d+\]\) OR in\[\d+\]', line): op = 'AND_OR'
        elif re.search(r'\(in\[\d+\] XOR in\[\d+\]\) AND in\[\d+\]', line): op = 'XOR_AND'
        elif re.search(r'\(in\[\d+\] XOR in\[\d+\]\) OR in\[\d+\]', line): op = 'XOR_OR'
        elif re.search(r'NOT\(in\[\d+\]\)', line): op = 'NOT'
        elif re.search(r'in\[\d+\] AND in\[\d+\]', line): op = 'AND'
        elif re.search(r'in\[\d+\] OR in\[\d+\]', line): op = 'OR'
        elif re.search(r'in\[\d+\] XOR in\[\d+\]', line): op = 'XOR'
        elif SINGLE_RE.search(line): op = 'COPY'
        
        if op:
            all_op_counter[op] += 1

all_total = sum(all_op_counter.values())
print(f"  Successfully analyzed: {success}/{len(all_bit)}, failed: {fail}")
print(f"  Total bit rules: {all_total}")

print(f"\n  {'Op':12s} {'Training':>10s} {'Full Set':>10s} {'Ratio':>8s}")
print(f"  {'-'*42}")
all_ops = sorted(set(list(op_counter.keys()) + list(all_op_counter.keys())), 
                 key=lambda x: all_op_counter.get(x,0), reverse=True)
for op in all_ops:
    t_cnt = op_counter.get(op, 0)
    a_cnt = all_op_counter.get(op, 0)
    t_pct = 100*t_cnt/total_ops if total_ops else 0
    a_pct = 100*a_cnt/all_total if all_total else 0
    ratio = t_pct/a_pct if a_pct > 0 else float('inf')
    flag = ' ⚠️' if abs(ratio - 1) > 0.3 else ''
    print(f"  {op:12s} {t_pct:8.1f}%  {a_pct:8.1f}%  {ratio:6.2f}x{flag}")
