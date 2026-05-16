#!/usr/bin/env python3
"""Deep recompute verification for gravity and unit_conv."""
import json, random, re
random.seed(456)

records = []
with open('data/cot_v2.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

gravity = [r for r in records if r['type'] == 'gravity']
unit_conv = [r for r in records if r['type'] == 'unit_conv']

print('--- GRAVITY deep recompute (10 samples) ---')
for rec in random.sample(gravity, 10):
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    qm = re.search(r'for t\s*=\s*([\d.]+)s', prompt, re.IGNORECASE)
    if obs and qm:
        qt = float(qm.group(1))
        gs = [2*float(d)/(float(t)**2) for t, d in obs]
        avg_g = sum(gs)/len(gs)
        comp = f'{0.5*avg_g*qt*qt:.2f}'
        flag = 'OK' if comp == answer else 'FAIL'
        print(f'  [{flag}] {rec["id"]}: g_avg={avg_g:.4f}, t={qt}, computed={comp}, answer={answer}')

print()
print('--- UNIT_CONV deep recompute (10 samples) ---')
for rec in random.sample(unit_conv, 10):
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'([\d.]+)\s*m\s*becomes\s*([\d.]+)', prompt)
    qm = re.search(r'convert the following measurement:\s*([\d.]+)', prompt, re.IGNORECASE)
    if obs and qm:
        qv = float(qm.group(1))
        factors = [float(o)/float(i) for i, o in obs]
        avg_f = sum(factors)/len(factors)
        comp = f'{avg_f*qv:.2f}'
        flag = 'OK' if comp == answer else 'FAIL'
        print(f'  [{flag}] {rec["id"]}: factor={avg_f:.6f}, q={qv}, computed={comp}, answer={answer}')

# Full sweep: check ALL gravity and unit_conv
print()
print('--- FULL SWEEP: gravity ---')
g_ok, g_fail = 0, 0
for rec in gravity:
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    qm = re.search(r'for t\s*=\s*([\d.]+)s', prompt, re.IGNORECASE)
    if obs and qm:
        qt = float(qm.group(1))
        gs = [2*float(d)/(float(t)**2) for t, d in obs]
        avg_g = sum(gs)/len(gs)
        comp = f'{0.5*avg_g*qt*qt:.2f}'
        if comp == answer:
            g_ok += 1
        else:
            g_fail += 1
            if g_fail <= 3:
                print(f'  MISMATCH {rec["id"]}: computed={comp}, answer={answer}')
print(f'  gravity: {g_ok} OK, {g_fail} FAIL out of {len(gravity)}')

print()
print('--- FULL SWEEP: unit_conv ---')
u_ok, u_fail, u_skip = 0, 0, 0
for rec in unit_conv:
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'([\d.]+)\s*m\s*becomes\s*([\d.]+)', prompt)
    qm = re.search(r'convert the following measurement:\s*([\d.]+)', prompt, re.IGNORECASE)
    if obs and qm:
        qv = float(qm.group(1))
        factors = [float(o)/float(i) for i, o in obs]
        avg_f = sum(factors)/len(factors)
        comp = f'{avg_f*qv:.2f}'
        if comp == answer:
            u_ok += 1
        else:
            u_fail += 1
            if u_fail <= 3:
                print(f'  MISMATCH {rec["id"]}: computed={comp}, answer={answer}')
    else:
        u_skip += 1
print(f'  unit_conv: {u_ok} OK, {u_fail} FAIL, {u_skip} skipped out of {len(unit_conv)}')

# Full sweep: numeral
print()
print('--- FULL SWEEP: numeral ---')
def to_roman(num):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = ''
    for v, s in vals:
        while num >= v:
            result += s
            num -= v
    return result

numeral = [r for r in records if r['type'] == 'numeral']
n_ok, n_fail = 0, 0
for rec in numeral:
    prompt, answer = rec['prompt'], rec['answer']
    qm = re.search(r'number\s+(\d+)\s+in', prompt)
    if qm:
        num = int(qm.group(1))
        comp = to_roman(num)
        if comp == answer:
            n_ok += 1
        else:
            n_fail += 1
            if n_fail <= 3:
                print(f'  MISMATCH {rec["id"]}: {num} -> computed={comp}, answer={answer}')
print(f'  numeral: {n_ok} OK, {n_fail} FAIL out of {len(numeral)}')
