#!/usr/bin/env python3
"""Debug solver failures for gravity, unit_conv, cipher."""
import polars as pl
import re
import statistics

train = pl.read_csv('competition_data/train.csv')

# === GRAVITY ROUNDING ===
print("=== GRAVITY ROUNDING ERRORS ===")
grav_err = 0
grav_total = 0
for row in train.iter_rows(named=True):
    p = row['prompt'].lower()
    if 'gravitational' not in p and 'gravity' not in p:
        continue
    prompt = row['prompt']
    gold = str(row['answer']).strip()
    grav_total += 1

    obs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    if not obs:
        continue
    gs = []
    for t_str, d_str in obs:
        t, d = float(t_str), float(d_str)
        if t > 0:
            gs.append(2 * d / (t * t))
    if not gs:
        continue
    g = statistics.median(gs)
    m = re.search(r'falling distance for t\s*=\s*([\d.]+)\s*s', prompt)
    if not m:
        continue
    t_target = float(m.group(1))
    d_answer = 0.5 * g * t_target * t_target
    computed = f"{d_answer:.2f}"
    if computed != gold:
        diff = abs(float(computed) - float(gold))
        grav_err += 1
        if grav_err <= 5:
            print(f"  g={g:.6f} t={t_target} computed={computed} gold={gold} diff={diff:.4f}")

print(f"Gravity: {grav_total} total, {grav_err} errors ({grav_err/grav_total*100:.1f}%)")

# === UNIT CONV ROUNDING ===
print("\n=== UNIT_CONV ROUNDING ERRORS ===")
uc_err = 0
uc_total = 0
for row in train.iter_rows(named=True):
    p = row['prompt'].lower()
    if 'unit conversion' not in p and 'convert the following measurement' not in p and 'secret unit' not in p:
        continue
    prompt = row['prompt']
    gold = str(row['answer']).strip()
    uc_total += 1

    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        continue
    factors = []
    for inp, out in pairs:
        i, o = float(inp), float(out)
        if i > 0:
            factors.append(o / i)
    if not factors:
        continue
    f = statistics.median(factors)
    m_match = re.search(r'convert.*?:\s*([\d.]+)\s*m', prompt)
    if not m_match:
        m_match = re.search(r'convert.*?([\d.]+)\s*m', prompt)
    if not m_match:
        continue
    target = float(m_match.group(1))
    answer = f * target
    computed = f"{answer:.2f}"
    if computed != gold:
        diff = abs(float(computed) - float(gold))
        uc_err += 1
        if uc_err <= 5:
            print(f"  f={f:.8f} target={target} computed={computed} gold={gold} diff={diff:.4f}")

print(f"Unit_conv: {uc_total} total, {uc_err} errors ({uc_err/uc_total*100:.1f}%)")

# === CIPHER FAILURE TYPES ===
print("\n=== CIPHER FAILURE ANALYSIS ===")
ci_total = 0
ci_correct = 0
ci_missing = 0
ci_wrong = 0
for row in train.iter_rows(named=True):
    p = row['prompt'].lower()
    if 'cipher' not in p and 'encrypt' not in p and 'decrypt' not in p:
        continue
    prompt = row['prompt']
    gold = str(row['answer']).strip()
    ci_total += 1

    lines = re.findall(r'(.+?)\s*->\s*(.+?)(?:\n|$)', prompt)
    mapping = {}
    for ct, pt in lines:
        cwords = ct.strip().split()
        pwords = pt.strip().split()
        if len(cwords) != len(pwords):
            continue
        for cw, pw in zip(cwords, pwords):
            if len(cw) != len(pw):
                continue
            for c, pp in zip(cw, pw):
                if c.isalpha() and pp.isalpha():
                    mapping[c.lower()] = pp.lower()

    m = re.search(r'(?:decrypt|decipher|translate).*?:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
    if not m:
        continue
    target = m.group(1).strip()

    decrypted = []
    has_missing = False
    for ch in target:
        if ch.lower() in mapping:
            mapped = mapping[ch.lower()]
            decrypted.append(mapped.upper() if ch.isupper() else mapped)
        else:
            decrypted.append(ch)
            if ch.isalpha():
                has_missing = True
    computed = ''.join(decrypted)

    if computed == gold:
        ci_correct += 1
    elif has_missing:
        ci_missing += 1
        if ci_missing <= 3:
            missing_chars = set(ch.lower() for ch in target if ch.isalpha() and ch.lower() not in mapping)
            print(f"  MISSING: chars={missing_chars}")
            print(f"    computed='{computed[:50]}' gold='{gold[:50]}'")
            print(f"    mapping has {len(mapping)} chars")
    else:
        ci_wrong += 1
        if ci_wrong <= 3:
            # Find which chars differ
            diffs = []
            for i, (c, g) in enumerate(zip(computed, gold)):
                if c != g:
                    diffs.append((i, c, g))
            print(f"  WRONG MAP: {diffs[:5]}")
            print(f"    computed='{computed[:50]}' gold='{gold[:50]}'")

print(f"\nCipher: {ci_total} total, {ci_correct} correct, {ci_missing} missing_char, {ci_wrong} wrong_map")

# === BIT_OPS: how many have unsolved '?' ===
print("\n=== BIT_OPS '?' ANALYSIS ===")
bo_total = 0
bo_solved = 0
bo_partial = 0
for row in train.iter_rows(named=True):
    p = row['prompt'].lower()
    if 'bit manipulation' not in p and 'bitwise' not in p and 'bit shift' not in p:
        continue
    bo_total += 1
    # Quick check: just count if we have '?' in the rule
    # (reusing logic from the solver would be complex, just looking at CSV output)

print(f"Bit_ops total: {bo_total}")
