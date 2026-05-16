#!/usr/bin/env python3
"""
全量验证各题型 gold 答案正确性 (v3 — 修复解析)。
"""
import csv
import json
import re
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# Roman numeral utils
# ═══════════════════════════════════════════════════════════════════════════════
ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def roman_to_int(s):
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

by_type = defaultdict(list)
for r in rows:
    t = detect_type(r["prompt"])
    r["_type"] = t
    by_type[t].append(r)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. NUMERAL: 全类型验证 (罗马 + 通用进制)
# ═══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("NUMERAL VERIFICATION")
print("="*80)

correct = 0
wrong = 0
skipped = 0
wrong_list = []

for r in by_type.get("numeral", []):
    prompt = r["prompt"]
    answer = r["answer"].strip()
    
    # 提取 examples (箭头行)
    arrow_lines = [l.strip() for l in prompt.split("\n") if "->" in l]
    examples = []
    for l in arrow_lines:
        parts = l.split("->")
        if len(parts) == 2:
            inp = parts[0].strip()
            out = parts[1].strip()
            if inp and out:
                examples.append((inp, out))
    
    # 提取 question: "write the number X" or "convert X"
    question = None
    m = re.search(r'(?:write the number|convert|convert the number|determine)\s+(\S+)', prompt, re.IGNORECASE)
    if m:
        question = m.group(1).strip().rstrip('.')
    
    # Fallback: 找最后一个数字 before "in the Wonderland"
    if question is None:
        m2 = re.search(r'(\S+)\s+in the (?:Wonderland|new)', prompt, re.IGNORECASE)
        if m2:
            question = m2.group(1).strip().rstrip('.')
    
    if not examples or question is None:
        skipped += 1
        continue
    
    # Detect conversion type from examples
    sample_in = examples[0][0]
    sample_out = examples[0][1]
    
    # Check if Roman numerals involved
    is_roman_out = bool(re.match(r'^[IVXLCDM]+$', sample_out.upper()))
    is_roman_in = bool(re.match(r'^[IVXLCDM]+$', sample_in.upper()))
    
    verified = False
    
    if is_roman_out and not is_roman_in:
        # Arabic -> Roman
        try:
            all_ok = True
            for inp, out in examples:
                if int_to_roman(int(inp)) != out:
                    all_ok = False
                    break
            if all_ok:
                expected = int_to_roman(int(question))
                if expected == answer:
                    correct += 1
                else:
                    wrong += 1
                    wrong_list.append({"id": r["id"], "gold": answer, "expected": expected, "q": question, "rule": "arabic->roman"})
                verified = True
        except:
            pass
    
    elif is_roman_in and not is_roman_out:
        # Roman -> Arabic
        try:
            all_ok = True
            for inp, out in examples:
                if str(roman_to_int(inp)) != out:
                    all_ok = False
                    break
            if all_ok:
                expected = str(roman_to_int(question))
                if expected == answer:
                    correct += 1
                else:
                    wrong += 1
                    wrong_list.append({"id": r["id"], "gold": answer, "expected": expected, "q": question, "rule": "roman->arabic"})
                verified = True
        except:
            pass
    
    if not verified:
        # General base conversion (brute force)
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
                            wrong_list.append({"id": r["id"], "gold": answer, "expected": expected, "q": question, "rule": "b{}->b{}".format(fb, tb)})
                        found = True
                        verified = True
                        break
                    except:
                        pass
            if found:
                break
    
    if not verified:
        skipped += 1

print("Correct: {}, Wrong: {}, Skipped: {}".format(correct, wrong, skipped))
print("Verification rate: {:.1f}%".format((correct + wrong) / max(correct + wrong + skipped, 1) * 100))

if wrong_list:
    print("\nWRONG numeral answers:")
    for w in wrong_list[:30]:
        print("  id={}: gold=[{}] expected=[{}] q=[{}] rule={}".format(
            w["id"], w["gold"], w["expected"], w["q"], w["rule"]))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. GRAVITY: 手动验算 g 推断
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("GRAVITY: Programmatic verification (d = 0.5 * g * t^2)")
print("="*80)

g_correct = 0
g_wrong = 0
g_skipped = 0
g_wrong_list = []

for r in by_type.get("gravity", []):
    prompt = r["prompt"]
    answer = r["answer"].strip()
    
    # Extract (t, d) pairs
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    
    # Extract query t
    query_m = re.search(r't\s*=\s*([\d.]+)\s*s\s*given', prompt)
    if not query_m:
        query_m = re.search(r'for\s+t\s*=\s*([\d.]+)\s*s', prompt.split("determine")[-1] if "determine" in prompt else "")
    
    if not pairs or not query_m:
        g_skipped += 1
        continue
    
    query_t = float(query_m.group(1))
    
    # Infer g from each pair: g = 2*d / t^2
    g_values = []
    for t_str, d_str in pairs:
        t = float(t_str)
        d = float(d_str)
        if t > 0:
            g = 2 * d / (t * t)
            g_values.append(g)
    
    if not g_values:
        g_skipped += 1
        continue
    
    # Average g
    g_avg = sum(g_values) / len(g_values)
    
    # Compute expected distance
    expected_d = 0.5 * g_avg * query_t * query_t
    expected_str = "{:.2f}".format(expected_d)
    
    try:
        gold_val = float(answer)
        exp_val = float(expected_str)
        diff = abs(gold_val - exp_val)
        
        if diff <= 0.01:
            g_correct += 1
        else:
            g_wrong += 1
            g_wrong_list.append({
                "id": r["id"], "gold": answer, "computed": expected_str,
                "diff": diff, "g_avg": g_avg, "g_values": g_values, "query_t": query_t
            })
    except:
        g_skipped += 1

print("Correct (diff<=0.01): {}, Wrong: {}, Skipped: {}".format(g_correct, g_wrong, g_skipped))

if g_wrong_list:
    # Analyze error patterns
    small_diff = [w for w in g_wrong_list if w["diff"] < 0.1]
    mid_diff = [w for w in g_wrong_list if 0.1 <= w["diff"] < 1.0]
    big_diff = [w for w in g_wrong_list if w["diff"] >= 1.0]
    
    print("  diff < 0.1: {} (rounding issue)".format(len(small_diff)))
    print("  0.1 <= diff < 1.0: {} (methodology issue?)".format(len(mid_diff)))
    print("  diff >= 1.0: {} (fundamentally wrong?)".format(len(big_diff)))
    
    # Check: does gold match if we use round(d, 2) differently?
    # Maybe gold uses individual g from each pair and averages the distances?
    print("\n  Rounding examples (diff < 0.1):")
    for w in small_diff[:10]:
        # Also try: use the most consistent g (mode)
        g_round = [round(g, 2) for g in w["g_values"]]
        print("    id={}: gold={} computed={} diff={:.4f} g_values={} query_t={}".format(
            w["id"], w["gold"], w["computed"], w["diff"],
            [round(g, 4) for g in w["g_values"]], w["query_t"]))
    
    print("\n  Big diff examples:")
    for w in big_diff[:10]:
        print("    id={}: gold={} computed={} diff={:.2f} g_values={} query_t={}".format(
            w["id"], w["gold"], w["computed"], w["diff"],
            [round(g, 4) for g in w["g_values"]], w["query_t"]))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. UNIT_CONV: 验算
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("UNIT_CONV: Programmatic verification")
print("="*80)

u_correct = 0
u_wrong = 0
u_skipped = 0
u_wrong_list = []

for r in by_type.get("unit_conv", []):
    prompt = r["prompt"]
    answer = r["answer"].strip()
    
    # Extract conversion pairs: "X m becomes Y" or "X m -> Y"
    pairs = re.findall(r'([\d.]+)\s*\w+\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        pairs = re.findall(r'([\d.]+)\s*->\s*([\d.]+)', prompt)
    
    # Extract query value
    query_m = re.search(r'convert.*?:\s*([\d.]+)', prompt, re.IGNORECASE)
    if not query_m:
        query_m = re.search(r'convert.*?([\d.]+)\s*\w+', prompt.split("Now")[-1] if "Now" in prompt else "", re.IGNORECASE)
    
    if not pairs or not query_m:
        u_skipped += 1
        continue
    
    query_val = float(query_m.group(1))
    
    # Infer conversion factor: out = factor * in
    factors = []
    for in_str, out_str in pairs:
        in_val = float(in_str)
        out_val = float(out_str)
        if in_val > 0:
            factors.append(out_val / in_val)
    
    if not factors:
        u_skipped += 1
        continue
    
    # Average factor
    factor_avg = sum(factors) / len(factors)
    
    expected = factor_avg * query_val
    expected_str = "{:.2f}".format(expected)
    
    try:
        gold_val = float(answer)
        exp_val = float(expected_str)
        diff = abs(gold_val - exp_val)
        
        if diff <= 0.01:
            u_correct += 1
        else:
            u_wrong += 1
            u_wrong_list.append({
                "id": r["id"], "gold": answer, "computed": expected_str,
                "diff": diff, "factor_avg": factor_avg, "factors": factors,
                "query": query_val
            })
    except:
        u_skipped += 1

print("Correct (diff<=0.01): {}, Wrong: {}, Skipped: {}".format(u_correct, u_wrong, u_skipped))

if u_wrong_list:
    small_diff = [w for w in u_wrong_list if w["diff"] < 0.1]
    big_diff = [w for w in u_wrong_list if w["diff"] >= 0.1]
    
    print("  diff < 0.1: {} (rounding)".format(len(small_diff)))
    print("  diff >= 0.1: {} (methodology)".format(len(big_diff)))
    
    print("\n  Examples:")
    for w in u_wrong_list[:10]:
        print("    id={}: gold={} computed={} diff={:.4f} factor={:.6f} factors={}".format(
            w["id"], w["gold"], w["computed"], w["diff"], w["factor_avg"],
            [round(f, 6) for f in w["factors"]]))

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("FINAL SUMMARY — Gold Answer Errors")
print("="*80)
print("numeral: {} correct, {} wrong, {} skipped".format(correct, wrong, skipped))
print("gravity: {} correct, {} wrong, {} skipped".format(g_correct, g_wrong, g_skipped))
print("unit_conv: {} correct, {} wrong, {} skipped".format(u_correct, u_wrong, u_skipped))
print("(bit_ops, cipher, symbol: need specialized verifiers)")
