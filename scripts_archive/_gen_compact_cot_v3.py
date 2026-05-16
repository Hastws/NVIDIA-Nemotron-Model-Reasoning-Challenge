#!/usr/bin/env python3
"""Generate ultra-compact CoT v3. Always uses gold answer.

Rules:
- gravity: "g=XX.XX" (7 chars)
- unit_conv: "f=X.XXXX" (8 chars)
- numeral: "A->R" or "R->A" (3-4 chars)
- cipher: 26-char decryption key (a->?, b->?, ...) with '.' for unknown
- bit_ops: per-bit function (15-77 chars)
- symbol: empty (answer-only)
"""
import polars as pl
import re
import statistics
import csv


def classify_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or 'bitwise' in p or 'bit shift' in p:
        return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p or 'celestial' in p:
        return 'gravity'
    if 'unit conversion' in p or 'convert the following measurement' in p or 'secret unit' in p:
        return 'unit_conv'
    if 'encryption' in p or 'cipher' in p or 'encrypt' in p or 'decrypt' in p:
        return 'cipher'
    if 'numeral system' in p or 'roman numeral' in p or 'ancient numeral' in p:
        return 'numeral'
    if 'symbol' in p or 'equation' in p or 'transformation rule' in p:
        return 'symbol'
    return 'unknown'


# ===== GRAVITY =====
def rule_gravity(prompt):
    obs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    if not obs:
        return ''
    gs = []
    for t_str, d_str in obs:
        t, d = float(t_str), float(d_str)
        if t > 0:
            gs.append(2 * d / (t * t))
    if not gs:
        return ''
    g = statistics.median(gs)
    return f"g={g:.2f}"


# ===== UNIT_CONV =====
def rule_unit_conv(prompt):
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        return ''
    factors = []
    for inp, out in pairs:
        i, o = float(inp), float(out)
        if i > 0:
            factors.append(o / i)
    if not factors:
        return ''
    f = statistics.median(factors)
    return f"f={f:.4f}"


# ===== CIPHER: 26-char decryption key =====
def rule_cipher(prompt):
    lines = re.findall(r'(.+?)\s*->\s*(.+?)(?:\n|$)', prompt)
    if not lines:
        return ''

    mapping = {}
    for cipher_text, plain_text in lines:
        ct = cipher_text.strip().split()
        pt = plain_text.strip().split()
        if len(ct) != len(pt):
            continue
        for cw, pw in zip(ct, pt):
            if len(cw) != len(pw):
                continue
            for c, p in zip(cw, pw):
                if c.isalpha() and p.isalpha():
                    mapping[c.lower()] = p.lower()

    if not mapping:
        return ''

    # Build 26-char key: position i = decryption of chr(ord('a')+i)
    key = []
    for i in range(26):
        c = chr(ord('a') + i)
        key.append(mapping.get(c, '.'))
    return ''.join(key)


# ===== NUMERAL =====
def rule_numeral(prompt):
    examples = re.findall(r'(\S+)\s*->\s*(\S+)', prompt)
    if not examples:
        return ''
    left_numeric = all(re.match(r'^\d+$', ex[0]) for ex in examples)
    if left_numeric:
        return "A->R"
    return "R->A"


# ===== BIT_OPS =====
def rule_bit_ops(prompt):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    if not pairs:
        return ''

    inputs = [list(map(int, p[0])) for p in pairs]
    outputs = [list(map(int, p[1])) for p in pairs]
    n = len(inputs)

    rules = []
    for obit in range(8):
        out_vals = [outputs[j][obit] for j in range(n)]
        found = False

        # Constant
        if all(v == 0 for v in out_vals):
            rules.append("0")
            continue
        if all(v == 1 for v in out_vals):
            rules.append("1")
            continue

        # Direct copy or NOT
        for ibit in range(8):
            in_vals = [inputs[j][ibit] for j in range(n)]
            if in_vals == out_vals:
                rules.append(f"i{ibit}")
                found = True
                break
            if all(iv ^ 1 == ov for iv, ov in zip(in_vals, out_vals)):
                rules.append(f"~i{ibit}")
                found = True
                break
        if found:
            continue

        # XOR of two bits
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                xor_vals = [inputs[j][i1] ^ inputs[j][i2] for j in range(n)]
                if xor_vals == out_vals:
                    rules.append(f"i{i1}^i{i2}")
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # AND of two bits
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                and_vals = [inputs[j][i1] & inputs[j][i2] for j in range(n)]
                if and_vals == out_vals:
                    rules.append(f"i{i1}&i{i2}")
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # OR of two bits
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                or_vals = [inputs[j][i1] | inputs[j][i2] for j in range(n)]
                if or_vals == out_vals:
                    rules.append(f"i{i1}|i{i2}")
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # XNOR (NOT XOR)
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                xnor_vals = [(inputs[j][i1] ^ inputs[j][i2]) ^ 1 for j in range(n)]
                if xnor_vals == out_vals:
                    rules.append(f"~(i{i1}^i{i2})")
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # NAND
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                nand_vals = [(inputs[j][i1] & inputs[j][i2]) ^ 1 for j in range(n)]
                if nand_vals == out_vals:
                    rules.append(f"~(i{i1}&i{i2})")
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # NOR
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                nor_vals = [(inputs[j][i1] | inputs[j][i2]) ^ 1 for j in range(n)]
                if nor_vals == out_vals:
                    rules.append(f"~(i{i1}|i{i2})")
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # Majority of 3 bits
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                for i3 in range(i2 + 1, 8):
                    maj_vals = [int((inputs[j][i1] + inputs[j][i2] + inputs[j][i3]) >= 2) for j in range(n)]
                    if maj_vals == out_vals:
                        rules.append(f"maj({i1},{i2},{i3})")
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue

        # 3-bit XOR
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                for i3 in range(i2 + 1, 8):
                    xor3 = [inputs[j][i1] ^ inputs[j][i2] ^ inputs[j][i3] for j in range(n)]
                    if xor3 == out_vals:
                        rules.append(f"i{i1}^i{i2}^i{i3}")
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue

        # NOT of a single 2-bit op result
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                # NOT(AND), NOT(OR) already covered above
                pass

        # (A AND B) OR C, (A OR B) AND C, etc.
        for i1 in range(8):
            for i2 in range(i1 + 1, 8):
                for i3 in range(8):
                    if i3 == i1 or i3 == i2:
                        continue
                    # (i1 & i2) | i3
                    v = [(inputs[j][i1] & inputs[j][i2]) | inputs[j][i3] for j in range(n)]
                    if v == out_vals:
                        rules.append(f"(i{i1}&i{i2})|i{i3}")
                        found = True
                        break
                    # (i1 | i2) & i3
                    v = [(inputs[j][i1] | inputs[j][i2]) & inputs[j][i3] for j in range(n)]
                    if v == out_vals:
                        rules.append(f"(i{i1}|i{i2})&i{i3}")
                        found = True
                        break
                    # (i1 ^ i2) & i3
                    v = [(inputs[j][i1] ^ inputs[j][i2]) & inputs[j][i3] for j in range(n)]
                    if v == out_vals:
                        rules.append(f"(i{i1}^i{i2})&i{i3}")
                        found = True
                        break
                    # (i1 ^ i2) | i3
                    v = [(inputs[j][i1] ^ inputs[j][i2]) | inputs[j][i3] for j in range(n)]
                    if v == out_vals:
                        rules.append(f"(i{i1}^i{i2})|i{i3}")
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue

        rules.append("?")

    return ";".join(rules)


# ===== MAIN =====
train = pl.read_csv('competition_data/train.csv')

results = []
type_stats = {}

for row in train.iter_rows(named=True):
    prompt = row['prompt']
    gold = str(row['answer']).strip()
    t = classify_type(prompt)

    if t not in type_stats:
        type_stats[t] = {'total': 0, 'with_rule': 0, 'lens': []}
    type_stats[t]['total'] += 1

    rule = ''
    if t == 'gravity':
        rule = rule_gravity(prompt)
    elif t == 'unit_conv':
        rule = rule_unit_conv(prompt)
    elif t == 'cipher':
        rule = rule_cipher(prompt)
    elif t == 'numeral':
        rule = rule_numeral(prompt)
    elif t == 'bit_ops':
        rule = rule_bit_ops(prompt)

    if rule:
        type_stats[t]['with_rule'] += 1
        type_stats[t]['lens'].append(len(rule))

    results.append({
        'id': row['id'],
        'prompt': prompt,
        'answer': gold,  # ALWAYS gold
        'thinking': rule,
        'type': t,
    })

# Report
print("=" * 60)
print("ULTRA-COMPACT COT v3 REPORT (always gold answer)")
print("=" * 60)
total = len(results)
with_rule = sum(1 for r in results if r['thinking'])
print(f"Total: {total}, With rule: {with_rule} ({with_rule / total * 100:.1f}%)")

for t in sorted(type_stats.keys()):
    ts = type_stats[t]
    print(f"\n  {t}: {ts['total']} total, {ts['with_rule']} rules")
    if ts['lens']:
        print(f"    Rule len: min={min(ts['lens'])}, med={statistics.median(ts['lens']):.0f}, max={max(ts['lens'])}")
    # Show examples
    exs = [r for r in results if r['type'] == t][:3]
    for ex in exs:
        print(f"    rule='{ex['thinking'][:60]}' answer={ex['answer'][:30]}")

# Count '?' in bit_ops
bo_rules = [r['thinking'] for r in results if r['type'] == 'bit_ops' and r['thinking']]
has_q = sum(1 for r in bo_rules if '?' in r)
fully_solved = sum(1 for r in bo_rules if '?' not in r)
print(f"\n  bit_ops '?' analysis: {fully_solved} fully solved, {has_q} partial ({has_q/(has_q+fully_solved)*100:.1f}% have ?)")

# Save CSV - only rows with rules (for Stage 2) plus all rows (for mixed training)
# Version A: only rows WITH rules
out_a = 'data/sft_compact_rules.csv'
with open(out_a, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking', 'type'])
    writer.writeheader()
    for r in results:
        if r['thinking']:
            writer.writerow(r)
n_a = sum(1 for r in results if r['thinking'])
print(f"\nSaved {n_a} rows (with rules) to {out_a}")

# Version B: all 9500 rows (empty thinking for symbol/unknown)
out_b = 'data/sft_compact_all.csv'
with open(out_b, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking', 'type'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)
print(f"Saved {len(results)} rows (all) to {out_b}")
