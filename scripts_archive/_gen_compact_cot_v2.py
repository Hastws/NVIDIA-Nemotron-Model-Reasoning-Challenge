#!/usr/bin/env python3
"""Generate ultra-compact CoT v2 with correct answer computation.

Target: rule ≤ 50 chars, answer 100% correct where possible.
Types with no computable rule: answer-only (empty thinking).
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


# ===== GRAVITY: "g=XX.XX" =====
def solve_gravity(prompt, gold):
    obs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    if not obs:
        return '', gold
    gs = []
    for t_str, d_str in obs:
        t, d = float(t_str), float(d_str)
        if t > 0:
            gs.append(2 * d / (t * t))
    if not gs:
        return '', gold
    g = statistics.median(gs)
    m = re.search(r'falling distance for t\s*=\s*([\d.]+)\s*s', prompt)
    if not m:
        return '', gold
    t_target = float(m.group(1))
    d_answer = 0.5 * g * t_target * t_target
    return f"g={g:.2f}", f"{d_answer:.2f}"


# ===== UNIT_CONV: "f=X.XXXX" =====
def solve_unit_conv(prompt, gold):
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        pairs = re.findall(r'([\d.]+)\s*m\s*(?:->|→)\s*([\d.]+)', prompt)
    if not pairs:
        return '', gold
    factors = []
    for inp, out in pairs:
        i, o = float(inp), float(out)
        if i > 0:
            factors.append(o / i)
    if not factors:
        return '', gold
    f = statistics.median(factors)
    m = re.search(r'convert.*?:\s*([\d.]+)\s*m', prompt)
    if not m:
        m = re.search(r'convert.*?([\d.]+)\s*m', prompt)
    if not m:
        return '', gold
    target = float(m.group(1))
    answer = f * target
    return f"f={f:.4f}", f"{answer:.2f}"


# ===== CIPHER: build mapping, decrypt =====
def solve_cipher(prompt, gold):
    lines = re.findall(r'(.+?)\s*->\s*(.+?)(?:\n|$)', prompt)
    if not lines:
        return '', gold
    
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
        return '', gold
    
    # Check if Caesar shift
    shifts = set()
    for c, p in mapping.items():
        shifts.add((ord(p) - ord(c)) % 26)
    
    if len(shifts) == 1:
        s = shifts.pop()
        rule = f"shift={s}" if s <= 13 else f"shift=-{26-s}"
    else:
        # Compact: just "sub" (substitution)
        rule = "sub"
    
    # Decrypt the target text
    m = re.search(r'(?:decrypt|decipher|translate).*?:\s*(.+?)(?:\n|$)', prompt, re.IGNORECASE)
    if not m:
        return rule, gold
    target = m.group(1).strip()
    
    decrypted = []
    for ch in target:
        if ch.lower() in mapping:
            mapped = mapping[ch.lower()]
            decrypted.append(mapped.upper() if ch.isupper() else mapped)
        else:
            decrypted.append(ch)
    computed = ''.join(decrypted)
    
    return rule, computed


# ===== NUMERAL: determine system and convert =====
ROMAN_VALS = [
    (1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),
    (100,'C'),(90,'XC'),(50,'L'),(40,'XL'),
    (10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')
]
def int_to_roman(n):
    result = []
    for val, sym in ROMAN_VALS:
        while n >= val:
            result.append(sym)
            n -= val
    return ''.join(result)

def roman_to_int(s):
    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        if v < prev:
            total -= v
        else:
            total += v
        prev = v
    return total

def solve_numeral(prompt, gold):
    examples = re.findall(r'(\S+)\s*->\s*(\S+)', prompt)
    if not examples:
        return '', gold
    
    # Determine direction
    left_numeric = all(re.match(r'^\d+$', ex[0]) for ex in examples)
    right_numeric = all(re.match(r'^\d+$', ex[1]) for ex in examples)
    
    # Find target
    m = re.search(r'(?:write|convert|determine).*?(?:number|numeral)\s+(\S+)', prompt, re.IGNORECASE)
    target_str = m.group(1).strip('.,;:') if m else None
    
    if left_numeric and not right_numeric:
        rule = "A->R"  # Arabic to Roman
        if target_str and target_str.isdigit():
            return rule, int_to_roman(int(target_str))
    elif right_numeric and not left_numeric:
        rule = "R->A"  # Roman to Arabic
        if target_str:
            return rule, str(roman_to_int(target_str))
    
    return "numeral", gold


# ===== BIT_OPS: per-bit analysis, ultra-compact rule, compute answer =====
def solve_bit_ops(prompt, gold):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    if not pairs:
        return '', gold
    
    inputs = [list(map(int, p[0])) for p in pairs]
    outputs = [list(map(int, p[1])) for p in pairs]
    n = len(inputs)
    
    m = re.search(r'determine the output for:\s*([01]{8})', prompt)
    if not m:
        return '', gold
    target = list(map(int, m.group(1)))
    
    answer_bits = []
    rules = []
    all_solved = True
    
    for obit in range(8):
        out_vals = [outputs[j][obit] for j in range(n)]
        found = False
        
        # Direct copy
        for ibit in range(8):
            in_vals = [inputs[j][ibit] for j in range(n)]
            if in_vals == out_vals:
                rules.append(f"i{ibit}")
                answer_bits.append(target[ibit])
                found = True
                break
            if all(iv ^ 1 == ov for iv, ov in zip(in_vals, out_vals)):
                rules.append(f"~i{ibit}")
                answer_bits.append(target[ibit] ^ 1)
                found = True
                break
        if found:
            continue
        
        # XOR of two bits
        for i1 in range(8):
            for i2 in range(i1+1, 8):
                xor_vals = [inputs[j][i1] ^ inputs[j][i2] for j in range(n)]
                if xor_vals == out_vals:
                    rules.append(f"i{i1}^i{i2}")
                    answer_bits.append(target[i1] ^ target[i2])
                    found = True
                    break
            if found:
                break
        if found:
            continue
        
        # AND of two bits
        for i1 in range(8):
            for i2 in range(i1+1, 8):
                and_vals = [inputs[j][i1] & inputs[j][i2] for j in range(n)]
                if and_vals == out_vals:
                    rules.append(f"i{i1}&i{i2}")
                    answer_bits.append(target[i1] & target[i2])
                    found = True
                    break
            if found:
                break
        if found:
            continue
        
        # OR of two bits
        for i1 in range(8):
            for i2 in range(i1+1, 8):
                or_vals = [inputs[j][i1] | inputs[j][i2] for j in range(n)]
                if or_vals == out_vals:
                    rules.append(f"i{i1}|i{i2}")
                    answer_bits.append(target[i1] | target[i2])
                    found = True
                    break
            if found:
                break
        if found:
            continue
        
        # Constant
        if all(v == 0 for v in out_vals):
            rules.append("0")
            answer_bits.append(0)
            continue
        if all(v == 1 for v in out_vals):
            rules.append("1")
            answer_bits.append(1)
            continue
        
        # XOR with NOT
        for i1 in range(8):
            for i2 in range(8):
                if i1 == i2:
                    continue
                xnor_vals = [(inputs[j][i1] ^ inputs[j][i2]) ^ 1 for j in range(n)]
                if xnor_vals == out_vals:
                    rules.append(f"~(i{i1}^i{i2})")
                    answer_bits.append((target[i1] ^ target[i2]) ^ 1)
                    found = True
                    break
            if found:
                break
        if found:
            continue
        
        # Majority of 3 bits
        for i1 in range(8):
            for i2 in range(i1+1, 8):
                for i3 in range(i2+1, 8):
                    maj_vals = [(inputs[j][i1] + inputs[j][i2] + inputs[j][i3]) >= 2 for j in range(n)]
                    maj_vals = [int(v) for v in maj_vals]
                    if maj_vals == out_vals:
                        rules.append(f"maj({i1},{i2},{i3})")
                        answer_bits.append(int((target[i1] + target[i2] + target[i3]) >= 2))
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue
        
        rules.append("?")
        all_solved = False
    
    rule_str = ";".join(rules)
    if all_solved:
        computed = ''.join(map(str, answer_bits))
        return rule_str, computed
    else:
        return rule_str, gold


# ===== MAIN: generate dataset =====
train = pl.read_csv('competition_data/train.csv')

results = []
type_stats = {}

for row in train.iter_rows(named=True):
    prompt = row['prompt']
    gold = str(row['answer']).strip()
    t = classify_type(prompt)
    
    if t not in type_stats:
        type_stats[t] = {'total': 0, 'with_rule': 0, 'correct': 0, 'lens': []}
    type_stats[t]['total'] += 1
    
    rule, computed = '', gold
    if t == 'gravity':
        rule, computed = solve_gravity(prompt, gold)
    elif t == 'unit_conv':
        rule, computed = solve_unit_conv(prompt, gold)
    elif t == 'cipher':
        rule, computed = solve_cipher(prompt, gold)
    elif t == 'numeral':
        rule, computed = solve_numeral(prompt, gold)
    elif t == 'bit_ops':
        rule, computed = solve_bit_ops(prompt, gold)
    
    if rule:
        type_stats[t]['with_rule'] += 1
        type_stats[t]['lens'].append(len(rule))
    
    correct = (computed == gold)
    if correct:
        type_stats[t]['correct'] += 1
    
    results.append({
        'id': row['id'],
        'prompt': prompt,
        'answer': gold,
        'thinking': rule,
        'computed': computed,
        'correct': correct,
        'type': t,
    })

# Report
print("="*60)
print("ULTRA-COMPACT COT v2 REPORT")
print("="*60)
total = len(results)
with_rule = sum(1 for r in results if r['thinking'])
correct = sum(1 for r in results if r['correct'])
print(f"Total: {total}, With rule: {with_rule} ({with_rule/total*100:.1f}%), Correct: {correct} ({correct/total*100:.1f}%)")

for t in sorted(type_stats.keys()):
    ts = type_stats[t]
    pct = ts['correct']/ts['total']*100 if ts['total'] > 0 else 0
    print(f"\n  {t}: {ts['total']} total, {ts['with_rule']} rules, {ts['correct']} correct ({pct:.1f}%)")
    if ts['lens']:
        print(f"    Rule len: min={min(ts['lens'])}, med={statistics.median(ts['lens']):.0f}, max={max(ts['lens'])}")
    # Show 2 examples
    exs = [r for r in results if r['type'] == t][:2]
    for ex in exs:
        mark = "✓" if ex['correct'] else "✗"
        print(f"    [{mark}] rule='{ex['thinking'][:60]}' computed={ex['computed'][:20]} gold={ex['answer'][:20]}")

# Save CSV
out_path = 'data/sft_compact_cot.csv'
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking'])
    writer.writeheader()
    for r in results:
        # Use computed answer if correct, else gold
        answer = r['computed'] if r['correct'] else r['answer']
        writer.writerow({
            'id': r['id'],
            'prompt': r['prompt'],
            'answer': answer,
            'thinking': r['thinking'],
        })
print(f"\nSaved {len(results)} rows to {out_path}")
