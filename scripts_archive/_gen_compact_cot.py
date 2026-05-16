#!/usr/bin/env python3
"""Generate ultra-compact CoT: 20-50 chars, only the extracted rule.

Format per type:
  gravity:  "g=17.35"                         (~7 chars)
  unit_conv: "f=1.2380"                       (~8 chars)
  numeral:  "Arabic->Roman" or "Roman->Arabic" (~14 chars)
  cipher:   "shift=-7" or compact mapping      (~10-40 chars)
  bit_ops:  "b0=i3^i5;b1=i0^i4;..."           (~30-50 chars)
  symbol:   "" (answer-only, too complex)
"""
import polars as pl
import re
import statistics

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
def solve_gravity(prompt):
    """Extract g from examples, compute answer. Rule: "g=XX.XX" """
    # Parse observations: t=X.XX, d=X.XX
    obs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    if not obs:
        return None, None
    gs = []
    for t_str, d_str in obs:
        t, d = float(t_str), float(d_str)
        if t > 0:
            gs.append(2 * d / (t * t))
    if not gs:
        return None, None
    g = statistics.median(gs)
    
    # Parse target t
    m = re.search(r'falling distance for t\s*=\s*([\d.]+)\s*s', prompt)
    if not m:
        return None, None
    t_target = float(m.group(1))
    d_answer = 0.5 * g * t_target * t_target
    
    rule = f"g={g:.2f}"
    answer = f"{d_answer:.2f}"
    return rule, answer

# ===== UNIT CONVERSION =====
def solve_unit_conv(prompt):
    """Extract conversion factor. Rule: "f=X.XXXX" """
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        # Try "X.XX m -> Y.YY" format
        pairs = re.findall(r'([\d.]+)\s*m\s*(?:->|→)\s*([\d.]+)', prompt)
    if not pairs:
        return None, None
    
    factors = []
    for inp, out in pairs:
        i, o = float(inp), float(out)
        if i > 0:
            factors.append(o / i)
    if not factors:
        return None, None
    f = statistics.median(factors)
    
    # Parse target
    m = re.search(r'convert.*?:\s*([\d.]+)\s*m', prompt)
    if not m:
        m = re.search(r'convert.*?([\d.]+)\s*m', prompt)
    if not m:
        return None, None
    target = float(m.group(1))
    answer = f * target
    
    rule = f"f={f:.4f}"
    answer_str = f"{answer:.2f}"
    return rule, answer_str

# ===== NUMERAL =====
def solve_numeral(prompt):
    """Determine numeral conversion type. Rule: "Arabic->Roman" etc."""
    # Check direction from examples
    examples = re.findall(r'(\S+)\s*->\s*(\S+)', prompt)
    if not examples:
        return None, None
    
    # Check if left side is Arabic numbers
    left_numeric = all(re.match(r'^\d+$', ex[0]) for ex in examples[:3])
    right_numeric = all(re.match(r'^\d+$', ex[1]) for ex in examples[:3])
    
    if left_numeric and not right_numeric:
        rule = "Arabic->Roman"
    elif right_numeric and not left_numeric:
        rule = "Roman->Arabic"
    else:
        rule = "numeral"
    
    return rule, None  # Answer computed by model

# ===== CIPHER =====
def solve_cipher(prompt):
    """Build compact shift or substitution mapping. Rule: "shift=N" or "a>h,b>s,..." """
    # Extract plaintext -> ciphertext pairs
    lines = re.findall(r'(.+?)\s*->\s*(.+?)(?:\n|$)', prompt)
    if not lines:
        return None, None
    
    # Build char mapping from examples
    mapping = {}
    for cipher_text, plain_text in lines:
        cipher_words = cipher_text.strip().split()
        plain_words = plain_text.strip().split()
        if len(cipher_words) != len(plain_words):
            continue
        for cw, pw in zip(cipher_words, plain_words):
            if len(cw) != len(pw):
                continue
            for c, p in zip(cw, pw):
                if c.isalpha() and p.isalpha():
                    mapping[c.lower()] = p.lower()
    
    if not mapping:
        return None, None
    
    # Check if it's a simple Caesar shift
    shifts = []
    for c, p in mapping.items():
        shift = (ord(p) - ord(c)) % 26
        shifts.append(shift)
    
    if len(set(shifts)) == 1:
        s = shifts[0]
        rule = f"shift={s}" if s <= 13 else f"shift=-{26-s}"
    else:
        # Compact mapping: only mapped chars, sorted
        pairs = sorted(mapping.items())
        # Ultra compact: "acbdef..." -> "hrsomg..."
        keys = ''.join(k for k, v in pairs)
        vals = ''.join(v for k, v in pairs)
        rule = f"{keys}>{vals}"
    
    return rule, None

# ===== BIT_OPS =====
def solve_bit_ops(prompt):
    """Analyze per-bit function. Rule: compact bit mapping."""
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    if not pairs:
        return None, None
    
    inputs = [list(map(int, p[0])) for p in pairs]
    outputs = [list(map(int, p[1])) for p in pairs]
    
    # Try to find simple per-bit rules
    # For each output bit, check if it matches a simple function of input bits
    n = len(inputs)
    rules = []
    for obit in range(8):
        out_vals = [outputs[j][obit] for j in range(n)]
        found = False
        
        # Check: output_bit = input_bit[i]
        for ibit in range(8):
            in_vals = [inputs[j][ibit] for j in range(n)]
            if in_vals == out_vals:
                rules.append(f"i{ibit}")
                found = True
                break
            # Check NOT
            if all(iv ^ 1 == ov for iv, ov in zip(in_vals, out_vals)):
                rules.append(f"~i{ibit}")
                found = True
                break
        if found:
            continue
        
        # Check XOR of two bits
        for i1 in range(8):
            for i2 in range(i1+1, 8):
                xor_vals = [inputs[j][i1] ^ inputs[j][i2] for j in range(n)]
                if xor_vals == out_vals:
                    rules.append(f"i{i1}^i{i2}")
                    found = True
                    break
            if found:
                break
        if found:
            continue
        
        rules.append("?")
    
    rule = ";".join(f"o{i}={r}" for i, r in enumerate(rules))
    
    # Also compute answer if possible
    m = re.search(r'determine the output for:\s*([01]{8})', prompt)
    if m:
        target = list(map(int, m.group(1)))
        # Try to compute answer from rules
        answer_bits = []
        computable = True
        for i, r in enumerate(rules):
            if r == "?":
                computable = False
                break
            elif r.startswith("~i"):
                idx = int(r[2:])
                answer_bits.append(str(target[idx] ^ 1))
            elif r.startswith("i") and "^" in r:
                parts = r.split("^")
                i1, i2 = int(parts[0][1:]), int(parts[1][1:])
                answer_bits.append(str(target[i1] ^ target[i2]))
            elif r.startswith("i"):
                idx = int(r[1:])
                answer_bits.append(str(target[idx]))
            else:
                computable = False
                break
        if computable and len(answer_bits) == 8:
            return rule, ''.join(answer_bits)
    
    return rule, None


# ===== MAIN =====
train = pl.read_csv('competition_data/train.csv')
train = train.with_columns(
    pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('type')
)

results = []
stats = {'total': 0, 'with_rule': 0, 'correct': 0, 'rule_lens': []}
type_stats = {}

for row in train.iter_rows(named=True):
    prompt = row['prompt']
    gold_answer = str(row['answer']).strip()
    t = classify_type(prompt)
    
    if t not in type_stats:
        type_stats[t] = {'total': 0, 'with_rule': 0, 'correct': 0, 'lens': []}
    type_stats[t]['total'] += 1
    stats['total'] += 1
    
    rule, computed_answer = None, None
    if t == 'gravity':
        rule, computed_answer = solve_gravity(prompt)
    elif t == 'unit_conv':
        rule, computed_answer = solve_unit_conv(prompt)
    elif t == 'numeral':
        rule, computed_answer = solve_numeral(prompt)
    elif t == 'cipher':
        rule, computed_answer = solve_cipher(prompt)
    elif t == 'bit_ops':
        rule, computed_answer = solve_bit_ops(prompt)
    # symbol: no rule
    
    if rule:
        stats['with_rule'] += 1
        type_stats[t]['with_rule'] += 1
        stats['rule_lens'].append(len(rule))
        type_stats[t]['lens'].append(len(rule))
        
        # Check accuracy
        if computed_answer and computed_answer == gold_answer:
            stats['correct'] += 1
            type_stats[t]['correct'] += 1
    
    results.append({
        'id': row['id'],
        'prompt': prompt,
        'answer': gold_answer,
        'type': t,
        'thinking': rule or '',
        'computed': computed_answer or '',
    })

# Report
print("="*60)
print("ULTRA-COMPACT RULE GENERATION REPORT")
print("="*60)
print(f"\nTotal: {stats['total']}")
print(f"With rule: {stats['with_rule']} ({stats['with_rule']/stats['total']*100:.1f}%)")
if stats['rule_lens']:
    print(f"Rule length: min={min(stats['rule_lens'])}, med={statistics.median(stats['rule_lens']):.0f}, max={max(stats['rule_lens'])}")
print(f"Exactly correct: {stats['correct']}")

for t in sorted(type_stats.keys()):
    ts = type_stats[t]
    print(f"\n  {t}: {ts['total']} total, {ts['with_rule']} with rule, {ts['correct']} correct")
    if ts['lens']:
        print(f"    Rule len: min={min(ts['lens'])}, med={statistics.median(ts['lens']):.0f}, max={max(ts['lens'])}")

# Show examples
print("\n" + "="*60)
print("EXAMPLES PER TYPE")
print("="*60)
for t in sorted(type_stats.keys()):
    examples = [r for r in results if r['type'] == t and r['thinking']][:3]
    print(f"\n--- {t} ---")
    for ex in examples:
        correct = "✓" if ex['computed'] == ex['answer'] else "✗"
        print(f"  Rule ({len(ex['thinking'])} chars): {ex['thinking'][:80]}")
        print(f"  Computed: {ex['computed'][:30]}  Gold: {ex['answer'][:30]}  {correct}")
    if not examples:
        not_ex = [r for r in results if r['type'] == t][:1]
        if not_ex:
            print(f"  (no rule) Answer: {not_ex[0]['answer'][:50]}")
