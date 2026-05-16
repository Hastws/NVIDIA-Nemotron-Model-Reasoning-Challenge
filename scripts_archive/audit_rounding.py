#!/usr/bin/env python3
"""Check rounding tolerance for gravity and unit_conv."""
import json, re, statistics

records = []
with open('data/cot_v2.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

gravity = [r for r in records if r['type'] == 'gravity']
diffs = []
for rec in gravity:
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    qm = re.search(r'determine the falling distance for t\s*=\s*([\d.]+)s', prompt, re.IGNORECASE)
    if obs and qm:
        qt = float(qm.group(1))
        gs = [2*float(d)/(float(t)**2) for t, d in obs]
        avg_g = sum(gs)/len(gs)
        comp = 0.5*avg_g*qt*qt
        diffs.append(abs(comp - float(answer)))

print("GRAVITY rounding analysis:")
print(f"  Mean diff: {statistics.mean(diffs):.6f}")
print(f"  Max diff:  {max(diffs):.6f}")
print(f"  within 0.005 (rounds same): {sum(1 for d in diffs if d < 0.005)} / {len(diffs)}")
print(f"  within 0.01: {sum(1 for d in diffs if d < 0.01)} / {len(diffs)}")
print(f"  within 0.02: {sum(1 for d in diffs if d < 0.02)} / {len(diffs)}")
print(f"  within 0.05: {sum(1 for d in diffs if d < 0.05)} / {len(diffs)}")

unit_conv = [r for r in records if r['type'] == 'unit_conv']
diffs_u = []
for rec in unit_conv:
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'([\d.]+)\s*m\s*becomes\s*([\d.]+)', prompt)
    qm = re.search(r'convert the following measurement:\s*([\d.]+)', prompt, re.IGNORECASE)
    if obs and qm:
        qv = float(qm.group(1))
        factors = [float(o)/float(i) for i, o in obs]
        avg_f = sum(factors)/len(factors)
        comp = avg_f*qv
        diffs_u.append(abs(comp - float(answer)))

print("\nUNIT_CONV rounding analysis:")
print(f"  Mean diff: {statistics.mean(diffs_u):.6f}")
print(f"  Max diff:  {max(diffs_u):.6f}")
print(f"  within 0.005: {sum(1 for d in diffs_u if d < 0.005)} / {len(diffs_u)}")
print(f"  within 0.01: {sum(1 for d in diffs_u if d < 0.01)} / {len(diffs_u)}")
print(f"  within 0.02: {sum(1 for d in diffs_u if d < 0.02)} / {len(diffs_u)}")

# Check: the CoT thinking shows g with 4 decimals, then uses that rounded value
# Let's simulate: round avg_g to 4 decimals, then compute
print("\n--- Gravity: simulating with rounded g (4 decimal places) ---")
g_ok_rounded = 0
for rec in gravity:
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    qm = re.search(r'determine the falling distance for t\s*=\s*([\d.]+)s', prompt, re.IGNORECASE)
    if obs and qm:
        qt = float(qm.group(1))
        gs = [round(2*float(d)/(float(t)**2), 4) for t, d in obs]
        avg_g = round(sum(gs)/len(gs), 4)
        comp = round(0.5 * avg_g * qt * qt, 2)
        comp_str = f"{comp:.2f}" if '.' in answer else str(int(comp))
        # Strip trailing zero after decimal if answer does that
        if answer.endswith('.0') or (not answer.endswith('0') and comp_str.endswith('0')):
            pass
        if comp_str == answer or f"{comp:.2f}" == f"{float(answer):.2f}":
            g_ok_rounded += 1

print(f"  Matches with rounded g: {g_ok_rounded} / {len(gravity)}")

# Try the exact approach the thinking uses
print("\n--- Gravity: exact approach from thinking (round each g to 4dp, avg to 4dp) ---")
g_exact = 0
g_fail_ex = []
for rec in gravity:
    prompt, answer = rec['prompt'], rec['answer']
    obs = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    qm = re.search(r'determine the falling distance for t\s*=\s*([\d.]+)s', prompt, re.IGNORECASE)
    if obs and qm:
        qt = float(qm.group(1))
        gs = [round(2*float(d)/(float(t)**2), 4) for t, d in obs]
        avg_g = round(sum(gs)/len(gs), 4)
        comp_val = 0.5 * avg_g * qt * qt
        comp_str = f"{comp_val:.2f}"
        # Also try matching with float comparison
        if comp_str == answer:
            g_exact += 1
        elif abs(comp_val - float(answer)) < 0.005:
            g_exact += 1
        else:
            if len(g_fail_ex) < 3:
                g_fail_ex.append(f"  {rec['id']}: gs={gs}, avg_g={avg_g}, t={qt}, comp={comp_str}, ans={answer}")

print(f"  Matches: {g_exact} / {len(gravity)}")
for ex in g_fail_ex:
    print(ex)
