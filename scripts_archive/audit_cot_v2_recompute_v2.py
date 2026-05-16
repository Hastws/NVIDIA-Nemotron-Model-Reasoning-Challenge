#!/usr/bin/env python3
"""Corrected full recompute verification."""
import json, re
from collections import defaultdict

records = []
with open('data/cot_v2.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

by_type = defaultdict(list)
for r in records:
    by_type[r['type']].append(r)

# ── GRAVITY ─────────────────────────────────────────────────────────────────
print('--- FULL SWEEP: gravity ---')
g_ok, g_fail, g_skip = 0, 0, 0
g_fail_examples = []
for rec in by_type['gravity']:
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    # Fix: match the query t from "determine the falling distance for t = X.XXs"
    qm = re.search(r'determine the falling distance for t\s*=\s*([\d.]+)s', prompt, re.IGNORECASE)
    if obs and qm:
        qt = float(qm.group(1))
        gs = [2*float(d)/(float(t)**2) for t, d in obs]
        avg_g = sum(gs)/len(gs)
        comp = f'{0.5*avg_g*qt*qt:.2f}'
        if comp == answer:
            g_ok += 1
        else:
            g_fail += 1
            if len(g_fail_examples) < 5:
                g_fail_examples.append((rec['id'], comp, answer, avg_g, qt))
    else:
        g_skip += 1
print(f'  OK: {g_ok}, FAIL: {g_fail}, SKIP: {g_skip} / {len(by_type["gravity"])}')
for rid, comp, ans, g, t in g_fail_examples:
    print(f'  MISMATCH {rid}: computed={comp}, answer={ans}, g={g:.4f}, t={t}')

# ── UNIT_CONV ───────────────────────────────────────────────────────────────
print('\n--- FULL SWEEP: unit_conv ---')
u_ok, u_fail, u_skip = 0, 0, 0
u_fail_examples = []
for rec in by_type['unit_conv']:
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
            if len(u_fail_examples) < 5:
                u_fail_examples.append((rec['id'], comp, answer, avg_f, qv))
    else:
        u_skip += 1
print(f'  OK: {u_ok}, FAIL: {u_fail}, SKIP: {u_skip} / {len(by_type["unit_conv"])}')
for rid, comp, ans, f, q in u_fail_examples:
    print(f'  MISMATCH {rid}: computed={comp}, answer={ans}, factor={f:.6f}, q={q}')

# ── NUMERAL ─────────────────────────────────────────────────────────────────
print('\n--- FULL SWEEP: numeral ---')
def to_roman(num):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = ''
    for v, s in vals:
        while num >= v:
            result += s
            num -= v
    return result

n_ok, n_fail = 0, 0
for rec in by_type['numeral']:
    prompt, answer = rec['prompt'], rec['answer']
    qm = re.search(r'number\s+(\d+)\s+in', prompt)
    if qm:
        comp = to_roman(int(qm.group(1)))
        if comp == answer:
            n_ok += 1
        else:
            n_fail += 1
            if n_fail <= 3:
                print(f'  MISMATCH {rec["id"]}: {qm.group(1)} -> computed={comp}, answer={answer}')
print(f'  OK: {n_ok}, FAIL: {n_fail} / {len(by_type["numeral"])}')

# ── CIPHER ──────────────────────────────────────────────────────────────────
print('\n--- FULL SWEEP: cipher (via mapping in thinking) ---')
c_ok, c_fail, c_skip = 0, 0, 0
c_fail_examples = []
for rec in by_type['cipher']:
    thinking, answer = rec['thinking'], rec['answer']
    prompt = rec['prompt']
    mapping_match = re.search(r'Mapping \(relevant chars\):\s*(.*)', thinking)
    decrypt_match = re.search(r'decrypt the following text:\s*(.*)', prompt, re.IGNORECASE)
    if mapping_match and decrypt_match:
        map_str = mapping_match.group(1).strip()
        char_map = {}
        for pair in map_str.split(', '):
            if '\u2192' in pair:
                parts = pair.split('\u2192')
                if len(parts) == 2 and len(parts[0].strip()) == 1 and len(parts[1].strip()) == 1:
                    char_map[parts[0].strip()] = parts[1].strip()
        encrypted = decrypt_match.group(1).strip()
        decrypted = ''
        for ch in encrypted:
            if ch in char_map:
                decrypted += char_map[ch]
            elif ch == ' ':
                decrypted += ' '
            else:
                decrypted += ch
        if decrypted == answer:
            c_ok += 1
        else:
            c_fail += 1
            if len(c_fail_examples) < 5:
                c_fail_examples.append((rec['id'], decrypted, answer))
    else:
        c_skip += 1
print(f'  OK: {c_ok}, FAIL: {c_fail}, SKIP: {c_skip} / {len(by_type["cipher"])}')
for rid, dec, ans in c_fail_examples:
    print(f'  MISMATCH {rid}: decrypted="{dec}", answer="{ans}"')

# ── BIT_OPS ─────────────────────────────────────────────────────────────────
print('\n--- FULL SWEEP: bit_ops (applied bits in thinking match answer) ---')
b_ok, b_fail, b_skip = 0, 0, 0
for rec in by_type['bit_ops']:
    thinking, answer = rec['thinking'], rec['answer']
    # Extract the applied results: "bit X: ... → Y"
    applied = re.findall(r'bit \d: .+ \u2192 (\d)', thinking)
    if len(applied) == 8:
        computed = ''.join(applied)
        if computed == answer:
            b_ok += 1
        else:
            b_fail += 1
            if b_fail <= 3:
                print(f'  MISMATCH {rec["id"]}: computed={computed}, answer={answer}')
    else:
        b_skip += 1
print(f'  OK: {b_ok}, FAIL: {b_fail}, SKIP: {b_skip} / {len(by_type["bit_ops"])}')

# ── SUMMARY ─────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('RECOMPUTE SUMMARY')
print('='*60)
types_results = {
    'gravity': (g_ok, g_fail, g_skip, len(by_type['gravity'])),
    'unit_conv': (u_ok, u_fail, u_skip, len(by_type['unit_conv'])),
    'numeral': (n_ok, n_fail, 0, len(by_type['numeral'])),
    'cipher': (c_ok, c_fail, c_skip, len(by_type['cipher'])),
    'bit_ops': (b_ok, b_fail, b_skip, len(by_type['bit_ops'])),
}
for t, (ok, fail, skip, total) in types_results.items():
    rate = f'{ok/(ok+fail)*100:.1f}%' if (ok+fail) > 0 else 'N/A'
    print(f'  {t:<12}: {ok:>5} OK, {fail:>4} FAIL, {skip:>3} SKIP  ({rate} match rate)')
