#!/usr/bin/env python3
"""
Part 2: 验证 numeral (含罗马数字) + gravity/unit_conv 精确计算验证。
"""
import csv
import json
import re
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# Roman numeral utils
# ═══════════════════════════════════════════════════════════════════════════════
ROMAN_VALUES = {
    'I': 1, 'V': 5, 'X': 10, 'L': 50,
    'C': 100, 'D': 500, 'M': 1000
}

def roman_to_int(s):
    """Convert Roman numeral to integer."""
    s = s.strip().upper()
    total = 0
    prev = 0
    for ch in reversed(s):
        val = ROMAN_VALUES.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total

def int_to_roman(num):
    """Convert integer to Roman numeral."""
    vals = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    result = ''
    for v, s in vals:
        while num >= v:
            result += s
            num -= v
    return result

def detect_type(prompt):
    p = prompt.lower()
    if "bit manipulation" in p or "bit shift" in p:
        return "bit_ops"
    elif "gravitational" in p or "gravity" in p:
        return "gravity"
    elif "unit conversion" in p or "conversion factor" in p:
        return "unit_conv"
    elif "cipher" in p or "encrypt" in p or "decrypt" in p:
        return "cipher"
    elif "numeral" in p or ("base" in p and "convert" in p):
        return "numeral"
    elif "symbol" in p or "equation" in p:
        return "symbol"
    return "unknown"

# Load data
rows = []
with open("competition_data/train.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

numeral_rows = [r for r in rows if detect_type(r["prompt"]) == "numeral"]

# ═══════════════════════════════════════════════════════════════════════════════
# Analyze numeral subtypes
# ═══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("NUMERAL: Subtype analysis")
print("="*80)

subtypes = defaultdict(int)
for r in numeral_rows:
    prompt = r["prompt"]
    answer = r["answer"]
    
    # Check if answer or examples contain Roman numerals
    has_roman = bool(re.search(r'\b[IVXLCDM]{2,}\b', answer)) or bool(re.search(r'\bI\b', answer) and len(answer) <= 4)
    
    # Check examples
    arrow_lines = [l.strip() for l in prompt.split("\n") if "->" in l and "?" not in l]
    if arrow_lines:
        sample_out = arrow_lines[0].split("->")[1].strip() if "->" in arrow_lines[0] else ""
        sample_in = arrow_lines[0].split("->")[0].strip() if "->" in arrow_lines[0] else ""
    else:
        sample_out = ""
        sample_in = ""
    
    is_roman_out = bool(re.search(r'^[IVXLCDM]+$', sample_out.upper())) and len(sample_out) >= 1
    is_roman_in = bool(re.search(r'^[IVXLCDM]+$', sample_in.upper())) and len(sample_in) >= 1
    
    # Try to detect base
    is_hex_in = bool(re.search(r'[a-fA-F]', sample_in))
    is_hex_out = bool(re.search(r'[a-fA-F]', sample_out))
    
    if is_roman_out:
        subtypes["arabic->roman"] += 1
    elif is_roman_in:
        subtypes["roman->arabic"] += 1
    elif is_hex_in or is_hex_out:
        subtypes["hex_conversion"] += 1
    else:
        # check digits to guess bases
        try:
            max_digit_in = max(int(c) for c in sample_in if c.isdigit()) if sample_in else -1
            max_digit_out = max(int(c) for c in sample_out if c.isdigit()) if sample_out else -1
            subtypes["base_{}->base_{}(guess)".format(
                "?" if max_digit_in < 0 else ">=" + str(max_digit_in+1),
                "?" if max_digit_out < 0 else ">=" + str(max_digit_out+1)
            )] += 1
        except:
            subtypes["unknown"] += 1

for st, cnt in sorted(subtypes.items(), key=lambda x: -x[1]):
    print("  {}: {}".format(st, cnt))

# ═══════════════════════════════════════════════════════════════════════════════
# Verify Arabic -> Roman
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("NUMERAL: Verifying Arabic -> Roman conversion")
print("="*80)

correct = 0
wrong = 0
skipped = 0
wrong_list = []

for r in numeral_rows:
    prompt = r["prompt"]
    answer = r["answer"].strip()
    
    arrow_lines = [l.strip() for l in prompt.split("\n") if "->" in l]
    examples = []
    question = None
    
    for l in arrow_lines:
        parts = l.split("->")
        if len(parts) != 2:
            continue
        inp = parts[0].strip()
        out = parts[1].strip()
        if "?" in out:
            question = inp
        else:
            examples.append((inp, out))
    
    if not examples or question is None:
        skipped += 1
        continue
    
    # Detect: arabic->roman
    sample_out = examples[0][1]
    if re.match(r'^[IVXLCDM]+$', sample_out.upper()):
        # Arabic -> Roman
        try:
            q_val = int(question)
            expected = int_to_roman(q_val)
            # Check examples too
            all_ok = True
            for inp, out in examples:
                try:
                    ev = int(inp)
                    er = int_to_roman(ev)
                    if er != out:
                        all_ok = False
                        break
                except:
                    all_ok = False
                    break
            
            if not all_ok:
                skipped += 1
                continue
            
            if expected == answer:
                correct += 1
            else:
                wrong += 1
                wrong_list.append({
                    "id": r["id"], "gold": answer, "expected": expected,
                    "question": question
                })
        except:
            skipped += 1
    elif re.match(r'^[IVXLCDM]+$', examples[0][0].upper()):
        # Roman -> Arabic
        try:
            q_val = roman_to_int(question)
            expected = str(q_val)
            
            # Verify examples
            all_ok = True
            for inp, out in examples:
                ev = roman_to_int(inp)
                if str(ev) != out:
                    all_ok = False
                    break
            
            if not all_ok:
                skipped += 1
                continue
            
            if expected == answer:
                correct += 1
            else:
                wrong += 1
                wrong_list.append({
                    "id": r["id"], "gold": answer, "expected": expected,
                    "question": question
                })
        except:
            skipped += 1
    else:
        # Try general base conversion
        found = False
        for fb in range(2, 37):
            for tb in range(2, 37):
                if fb == tb:
                    continue
                ok = True
                for inp, out in examples:
                    try:
                        val = int(inp, fb)
                        if val == 0:
                            conv = "0"
                        else:
                            digits = []
                            v = val
                            while v > 0:
                                digits.append("0123456789abcdefghijklmnopqrstuvwxyz"[v % tb])
                                v //= tb
                            conv = "".join(reversed(digits))
                        if conv.lower() != out.lower():
                            ok = False
                            break
                    except:
                        ok = False
                        break
                
                if ok:
                    try:
                        val = int(question, fb)
                        if val == 0:
                            expected = "0"
                        else:
                            digits = []
                            v = val
                            while v > 0:
                                digits.append("0123456789abcdefghijklmnopqrstuvwxyz"[v % tb])
                                v //= tb
                            expected = "".join(reversed(digits))
                        
                        if expected.lower() == answer.lower():
                            correct += 1
                        else:
                            wrong += 1
                            wrong_list.append({
                                "id": r["id"], "gold": answer, "expected": expected,
                                "question": question, "rule": "base{}->base{}".format(fb, tb)
                            })
                        found = True
                        break
                    except:
                        pass
            if found:
                break
        
        if not found:
            skipped += 1

print("Correct: {}, Wrong: {}, Skipped: {}".format(correct, wrong, skipped))
if wrong_list:
    print("\nWRONG numeral answers (first 30):")
    for w in wrong_list[:30]:
        print("  id={}: gold=[{}] expected=[{}] question=[{}] {}".format(
            w["id"], w["gold"], w["expected"], w["question"], w.get("rule", "roman")))

# ═══════════════════════════════════════════════════════════════════════════════
# Verify specific gravity example by hand
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("GRAVITY: Manual verification of a specific case")
print("="*80)

# Pick a suspicious gravity case and print full prompt
for r in rows:
    if r["id"] == "088d07a3":
        print("\nFull prompt for id=088d07a3 (gold=97.85, model=97.83):")
        print(r["prompt"])
        print("\nGold answer: {}".format(r["answer"]))
        break

# Also print one unit_conv suspicious case
print("\n" + "="*80)
print("UNIT_CONV: Manual verification of a specific case")
print("="*80)
for r in rows:
    if r["id"] == "010055e2":
        print("\nFull prompt for id=010055e2 (gold=28.29, model=28.30):")
        print(r["prompt"])
        print("\nGold answer: {}".format(r["answer"]))
        break
