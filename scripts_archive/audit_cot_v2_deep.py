#!/usr/bin/env python3
"""Deep manual verification: re-compute answers from thinking for select records."""

import json, random, re
from pathlib import Path
from collections import defaultdict

random.seed(123)
BASE = Path(__file__).resolve().parent.parent
COT_PATH = BASE / "data" / "cot_v2.jsonl"

records = []
with open(COT_PATH) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

by_type = defaultdict(list)
for r in records:
    by_type[r["type"]].append(r)

print("=" * 80)
print("DEEP MANUAL VERIFICATION — Re-computing select samples")
print("=" * 80)

# ── 1. GRAVITY: verify g computation ────────────────────────────────────────
print("\n--- GRAVITY deep check (5 samples) ---")
gravity_samples = random.sample(by_type["gravity"], 5)
for rec in gravity_samples:
    prompt = rec["prompt"]
    thinking = rec["thinking"]
    answer = rec["answer"]
    
    # Extract observations from prompt: number -> number pattern  
    obs_pattern = r'([\d.]+)\s*->\s*([\d.]+)'
    obs = re.findall(obs_pattern, prompt)
    
    # Extract query value
    query_match = re.search(r'determine the output for[:\s]*([\d.]+)', prompt, re.IGNORECASE)
    if not query_match:
        query_match = re.search(r'for the input[:\s]*([\d.]+)', prompt, re.IGNORECASE)
    
    if obs and query_match:
        query_val = float(query_match.group(1))
        # Compute g values from d = 0.5*g*t² => g = 2d/t²
        g_values = []
        for t_str, d_str in obs:
            t, d = float(t_str), float(d_str)
            if t > 0:
                g = 2 * d / (t * t)
                g_values.append(g)
        
        if g_values:
            avg_g = sum(g_values) / len(g_values)
            computed_d = 0.5 * avg_g * query_val * query_val
            computed_answer = f"{computed_d:.2f}"
            match = "✓" if computed_answer == answer else "✗"
            print(f"  [{match}] ID {rec['id']}: g_avg={avg_g:.4f}, query_t={query_val}, "
                  f"computed={computed_answer}, answer={answer}")
            if match == "✗":
                print(f"       g values: {[f'{g:.4f}' for g in g_values]}")

# ── 2. UNIT_CONV: verify factor computation ─────────────────────────────────
print("\n--- UNIT_CONV deep check (5 samples) ---")
uc_samples = random.sample(by_type["unit_conv"], 5)
for rec in uc_samples:
    prompt = rec["prompt"]
    thinking = rec["thinking"]
    answer = rec["answer"]
    
    obs = re.findall(r'([\d.]+)\s*->\s*([\d.]+)', prompt)
    query_match = re.search(r'determine the output for[:\s]*([\d.]+)', prompt, re.IGNORECASE)
    if not query_match:
        query_match = re.search(r'for the input[:\s]*([\d.]+)', prompt, re.IGNORECASE)
    
    if obs and query_match:
        query_val = float(query_match.group(1))
        # Linear conversion: out = factor * in
        factors = []
        for in_str, out_str in obs:
            inv, outv = float(in_str), float(out_str)
            if inv > 0:
                factors.append(outv / inv)
        
        if factors:
            avg_factor = sum(factors) / len(factors)
            computed = avg_factor * query_val
            computed_answer = f"{computed:.2f}"
            match = "✓" if computed_answer == answer else "✗"
            print(f"  [{match}] ID {rec['id']}: factor={avg_factor:.6f}, query={query_val}, "
                  f"computed={computed_answer}, answer={answer}")

# ── 3. NUMERAL: verify Roman numeral conversion ─────────────────────────────
print("\n--- NUMERAL deep check (5 samples) ---")
def to_roman(num):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = ''
    for v, s in vals:
        while num >= v:
            result += s
            num -= v
    return result

numeral_samples = random.sample(by_type["numeral"], 5)
for rec in numeral_samples:
    prompt = rec["prompt"]
    answer = rec["answer"]
    
    query_match = re.search(r'number\s+(\d+)\s+in', prompt)
    if query_match:
        num = int(query_match.group(1))
        computed = to_roman(num)
        match = "✓" if computed == answer else "✗"
        print(f"  [{match}] ID {rec['id']}: {num} -> computed={computed}, answer={answer}")

# ── 4. CIPHER: verify substitution mapping ──────────────────────────────────
print("\n--- CIPHER deep check (5 samples) ---")
cipher_samples = random.sample(by_type["cipher"], 5)
for rec in cipher_samples:
    prompt = rec["prompt"]
    thinking = rec["thinking"]
    answer = rec["answer"]
    
    # Extract mapping from thinking
    mapping_match = re.search(r'Mapping \(relevant chars\):\s*(.*)', thinking)
    if mapping_match:
        map_str = mapping_match.group(1).strip()
        # Parse a→b, c→d, ...
        char_map = {}
        for pair in map_str.split(', '):
            if '→' in pair:
                parts = pair.split('→')
                if len(parts) == 2 and len(parts[0].strip()) == 1 and len(parts[1].strip()) == 1:
                    char_map[parts[0].strip()] = parts[1].strip()
        
        # Extract encrypted text from prompt
        decrypt_match = re.search(r'decrypt the following text:\s*(.*)', prompt, re.IGNORECASE)
        if decrypt_match:
            encrypted = decrypt_match.group(1).strip()
            # Apply mapping
            decrypted = ''
            unmapped = set()
            for ch in encrypted:
                if ch in char_map:
                    decrypted += char_map[ch]
                elif ch == ' ':
                    decrypted += ' '
                else:
                    decrypted += ch
                    if ch.isalpha():
                        unmapped.add(ch)
            
            match = "✓" if decrypted == answer else "✗"
            print(f"  [{match}] ID {rec['id']}: decrypted='{decrypted}', answer='{answer}'")
            if unmapped:
                print(f"       Unmapped chars: {unmapped}")

# ── 5. BIT_OPS: verify bit operations ──────────────────────────────────────
print("\n--- BIT_OPS deep check (5 samples) ---")
bit_samples = random.sample(by_type["bit_ops"], 5)
for rec in bit_samples:
    thinking = rec["thinking"]
    answer = rec["answer"]
    prompt = rec["prompt"]
    
    # Extract input from prompt
    query_match = re.search(r'determine the output for[:\s]*(\d{8})', prompt)
    if not query_match:
        continue
    input_bits = query_match.group(1)
    inp = [int(b) for b in input_bits]
    
    # Parse rules from thinking
    rules = re.findall(r'bit (\d): (.+)', thinking)
    # Extract application results
    applied = re.findall(r'bit \d: .+ → (\d)', thinking)
    
    if len(applied) == 8:
        computed = ''.join(applied)
        match = "✓" if computed == answer else "✗"
        print(f"  [{match}] ID {rec['id']}: input={input_bits}, computed={computed}, answer={answer}")
        if match == "✗":
            print(f"       Rules: {rules[:4]}...")
    else:
        print(f"  [?] ID {rec['id']}: Could not extract 8 applied bits (found {len(applied)})")

# ── 6. SYMBOL: verify ───────────────────────────────────────────────────────
print("\n--- SYMBOL deep check (5 samples) ---")
symbol_samples = random.sample(by_type["symbol"], min(5, len(by_type["symbol"])))
for rec in symbol_samples:
    thinking = rec["thinking"]
    answer = rec["answer"]
    print(f"  ID {rec['id']}: answer='{answer}'")
    print(f"    Thinking: {thinking.strip()}")

# ── 7. Check thinking consistency: does "Result:" line match answer? ────────
print("\n--- RESULT LINE CONSISTENCY (all records) ---")
result_mismatches = []
for rec in records:
    thinking = rec["thinking"]
    answer = rec["answer"]
    # Find "Result: xxx" in thinking
    rm = re.search(r'Result:\s*(.+)', thinking)
    if rm:
        result_val = rm.group(1).strip()
        if result_val != answer:
            result_mismatches.append((rec["id"], rec["type"], result_val, answer))

print(f"Records with 'Result:' line that differs from answer: {len(result_mismatches)}")
for rid, rtype, rv, ans in result_mismatches[:10]:
    print(f"  [{rtype}] ID {rid}: Result='{rv}' vs answer='{ans}'")

# For types without explicit "Result:" line (numeral, symbol), check if answer appears
no_result_line = []
for rec in records:
    thinking = rec["thinking"]
    if "Result:" not in thinking and rec["answer"] not in thinking:
        no_result_line.append((rec["id"], rec["type"]))

print(f"\nRecords without 'Result:' line AND answer not in thinking: {len(no_result_line)}")
