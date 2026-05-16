#!/usr/bin/env python3
"""
深入验证 gold 答案 — 特别关注 gravity/unit_conv 的四舍五入问题。
"""
import csv
import json
import re
from collections import defaultdict

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

# Load cot_t0 data
cot_data = {}
with open("data/cot_t0.jsonl") as f:
    for line in f:
        r = json.loads(line)
        cot_data[r["id"]] = r

# Load train.csv
rows = {}
with open("competition_data/train.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows[r["id"]] = r

# ═══════════════════════════════════════════════════════════════════════════════
# 1. gravity/unit_conv 精度分析
# ═══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("GRAVITY + UNIT_CONV: Rounding analysis")
print("="*80)

for target_type in ["gravity", "unit_conv"]:
    print("\n--- {} ---".format(target_type))
    
    # 找到 3/3 一致但与 gold 不同的
    rounding_errors = []
    exact_diff = []
    
    for sid, cot in cot_data.items():
        if cot.get("type") != target_type:
            continue
        
        samples = cot.get("samples", [])
        preds = [str(s.get("predicted", "")).strip() for s in samples if s.get("predicted") is not None]
        
        if len(preds) < 3:
            continue
        if len(set(preds)) != 1:
            continue  # not unanimous
        
        model_ans = preds[0]
        gold = str(cot.get("gold", "")).strip()
        
        if model_ans == gold:
            continue
        
        # 尝试数值比较
        try:
            m = float(model_ans)
            g = float(gold)
            diff = abs(m - g)
            if diff <= 0.01:
                rounding_errors.append({
                    "id": sid, "gold": gold, "model": model_ans, 
                    "diff": diff, "within_tol": True
                })
            elif diff <= 0.1:
                rounding_errors.append({
                    "id": sid, "gold": gold, "model": model_ans,
                    "diff": diff, "within_tol": False
                })
            else:
                exact_diff.append({
                    "id": sid, "gold": gold, "model": model_ans, "diff": diff
                })
        except ValueError:
            exact_diff.append({
                "id": sid, "gold": gold, "model": model_ans, "diff": -1
            })
    
    within_tol = [e for e in rounding_errors if e["within_tol"]]
    outside_tol = [e for e in rounding_errors if not e["within_tol"]]
    
    print("  Within tolerance (|diff| <= 0.01): {}".format(len(within_tol)))
    print("  Close but outside (0.01 < |diff| <= 0.1): {}".format(len(outside_tol)))
    print("  Large diff or non-numeric: {}".format(len(exact_diff)))
    
    print("\n  Close-but-outside examples (model 3/3 agree, diff 0.01~0.1):")
    for e in outside_tol[:15]:
        print("    id={}: gold={} model={} diff={:.4f}".format(
            e["id"], e["gold"], e["model"], e["diff"]))
    
    print("\n  Large-diff examples:")
    for e in exact_diff[:10]:
        print("    id={}: gold={} model={} diff={}".format(
            e["id"], e["gold"], e["model"], e["diff"]))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. numeral: 解析 prompt 并验证
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("NUMERAL: Detailed parsing and verification")
print("="*80)

# 看几个 numeral prompt 的结构
numeral_rows = [r for r in rows.values() if detect_type(r["prompt"]) == "numeral"]
print("\nSample numeral prompt structure:")
for r in numeral_rows[:3]:
    prompt = r["prompt"]
    # 找 -> 行
    arrow_lines = [l.strip() for l in prompt.split("\n") if "->" in l]
    print("\n  ID: {}".format(r["id"]))
    print("  Answer: {}".format(r["answer"]))
    print("  Arrow lines:")
    for al in arrow_lines:
        print("    {}".format(al))

# 尝试更灵活的 numeral 验证
correct = 0
wrong = 0
unknown = 0
wrong_list = []

for r in numeral_rows:
    prompt = r["prompt"]
    answer = r["answer"]
    
    lines = [l.strip() for l in prompt.split("\n") if "->" in l]
    examples = []
    question = None
    
    for l in lines:
        if "-> ?" in l or "->?" in l:
            question = l.split("->")[0].strip()
        elif "->" in l:
            parts = l.split("->")
            if len(parts) == 2:
                inp = parts[0].strip()
                out = parts[1].strip()
                if inp and out and out != "?":
                    examples.append((inp, out))
    
    if not examples or question is None:
        unknown += 1
        continue
    
    # 暴力搜索 base pair
    found = False
    for fb in range(2, 37):
        for tb in range(2, 37):
            if fb == tb:
                continue
            ok = True
            for inp, out in examples:
                try:
                    val = int(inp, fb)
                    # convert to tb
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
                # Try to compute answer
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
                            "rule": "base{}->base{}".format(fb, tb),
                            "question": question
                        })
                    found = True
                    break
                except:
                    pass
        if found:
            break
    
    if not found:
        unknown += 1

print("\nNumeral verification:")
print("  Correct: {}, Wrong: {}, Unknown: {}".format(correct, wrong, unknown))
if wrong_list:
    print("\n  WRONG numeral answers (first 20):")
    for w in wrong_list[:20]:
        print("    id={}: gold=[{}] expected=[{}] rule={} question={}".format(
            w["id"], w["gold"], w["expected"], w["rule"], w["question"]))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 总结
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("numeral: {} verified correct, {} wrong, {} unknown".format(correct, wrong, unknown))
print("gravity: model 3/3 agree but != gold — check rounding policy")
print("unit_conv: model 3/3 agree but != gold — check rounding policy")
print("symbol: 226 cases model returns empty (can't solve, not gold errors)")
print("cipher: 7 cases backslash escaping (model formatting issue)")
