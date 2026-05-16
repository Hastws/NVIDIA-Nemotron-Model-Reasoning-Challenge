#!/usr/bin/env python3
"""
验证官方 train.csv 数据集答案正确性。
对每种题型，提取规则并程序化验证答案。
"""
import csv
import re
import sys
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

def verify_numeral(prompt, answer):
    """验证进制转换题：从 prompt 中提取 examples，推断规则，验证答案。"""
    # 找到 examples
    lines = prompt.strip().split("\n")
    examples = []
    question_val = None
    question_line = None
    
    for line in lines:
        line = line.strip()
        # Pattern: "value_in_baseX -> value_in_baseY"  or similar
        if " -> " in line and "?" not in line:
            parts = line.split(" -> ")
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        elif " -> ?" in line:
            question_val = line.split(" -> ")[0].strip()
            question_line = line
    
    if not examples or question_val is None:
        return None, "Could not parse"
    
    # 尝试推断：从 base X 到 base Y
    # 检查所有常见进制对 (2-36)
    for from_base in range(2, 37):
        for to_base in range(2, 37):
            if from_base == to_base:
                continue
            all_match = True
            for inp, out in examples:
                try:
                    # 解析输入为 from_base 整数
                    val = int(inp, from_base)
                    # 转换为 to_base
                    if to_base == 10:
                        converted = str(val)
                    elif to_base == 16:
                        converted = hex(val)[2:]
                    elif to_base == 8:
                        converted = oct(val)[2:]
                    elif to_base == 2:
                        converted = bin(val)[2:]
                    else:
                        # 通用进制转换
                        if val == 0:
                            converted = "0"
                        else:
                            digits = []
                            v = val
                            while v > 0:
                                digits.append("0123456789abcdefghijklmnopqrstuvwxyz"[v % to_base])
                                v //= to_base
                            converted = "".join(reversed(digits))
                    
                    if converted.lower() != out.lower():
                        all_match = False
                        break
                except (ValueError, IndexError):
                    all_match = False
                    break
            
            if all_match:
                # 验证答案
                try:
                    val = int(question_val, from_base)
                    if to_base == 10:
                        expected = str(val)
                    elif to_base == 16:
                        expected = hex(val)[2:]
                    elif to_base == 8:
                        expected = oct(val)[2:]
                    elif to_base == 2:
                        expected = bin(val)[2:]
                    else:
                        if val == 0:
                            expected = "0"
                        else:
                            digits = []
                            v = val
                            while v > 0:
                                digits.append("0123456789abcdefghijklmnopqrstuvwxyz"[v % to_base])
                                v //= to_base
                            expected = "".join(reversed(digits))
                    
                    match = expected.lower() == answer.lower()
                    return match, "base{}->base{}: expected={}, got={}".format(from_base, to_base, expected, answer)
                except:
                    pass
    
    return None, "Could not determine base conversion rule"


def verify_gravity(prompt, answer):
    """验证重力常数推断题：检查是否符合 F = G * m1 * m2 / r^2。"""
    # 这类题通常给出几组 (m1, m2, r, F)，推断 G，然后算新的 F。
    # 比较复杂，先检查答案格式
    try:
        float(answer)
        return None, "numeric answer, needs manual check"
    except:
        return None, "non-numeric answer"


def verify_unit_conv(prompt, answer):
    """验证单位换算题。"""
    try:
        float(answer)
        return None, "numeric answer, needs manual check"
    except:
        return None, "non-numeric answer"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
rows = []
with open("competition_data/train.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print("Total rows: {}".format(len(rows)))

by_type = defaultdict(list)
for r in rows:
    t = detect_type(r["prompt"])
    r["_type"] = t
    by_type[t].append(r)

for t in sorted(by_type):
    print("{}: {} rows".format(t, len(by_type[t])))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 验证 numeral (进制转换 — 最容易程序化验证)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("VERIFYING NUMERAL (base conversion)")
print("="*80)

correct = 0
wrong = 0
unknown = 0
wrong_examples = []

for r in by_type.get("numeral", []):
    result, detail = verify_numeral(r["prompt"], r["answer"])
    if result is True:
        correct += 1
    elif result is False:
        wrong += 1
        wrong_examples.append((r["id"], r["answer"], detail))
    else:
        unknown += 1

print("Correct: {}, Wrong: {}, Unknown: {}".format(correct, wrong, unknown))
if wrong_examples:
    print("\nWrong examples (first 20):")
    for eid, ans, detail in wrong_examples[:20]:
        print("  id={}: answer=[{}] {}".format(eid, ans, detail))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 检查答案格式异常
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("CHECKING ANSWER FORMAT ANOMALIES")
print("="*80)

for t in sorted(by_type):
    answers = [r["answer"] for r in by_type[t]]
    # 检查空答案
    empty = [a for a in answers if not a.strip()]
    # 检查超长答案
    long_ans = [a for a in answers if len(a) > 100]
    # 检查含空格或特殊字符
    weird = [a for a in answers if "\n" in a or "\t" in a]
    
    lens = [len(a) for a in answers]
    avg_len = sum(lens) / len(lens) if lens else 0
    max_len = max(lens) if lens else 0
    min_len = min(lens) if lens else 0
    
    print("{:12s}: count={}, len_avg={:.1f}, len_range=[{},{}], empty={}, long={}, weird={}".format(
        t, len(answers), avg_len, min_len, max_len, len(empty), len(long_ans), len(weird)
    ))
    
    # 显示最长的几个答案
    if long_ans:
        print("  Long answers: {}".format(long_ans[:3]))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 用 cot_t0.jsonl 交叉验证：模型答对但答案与 gold 不同的 => 可能是 gold 错
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("CROSS-VALIDATION: model answers vs gold (from cot_t0.jsonl)")
print("="*80)

import json

# 找出模型多次给出一致答案但与 gold 不同的情况
suspicious = []
total_checked = 0

try:
    with open("data/cot_t0.jsonl") as f:
        for line in f:
            r = json.loads(line)
            total_checked += 1
            samples = r.get("samples", [])
            predicted_answers = []
            for s in samples:
                pred = s.get("predicted")
                if pred is not None:
                    predicted_answers.append(str(pred).strip())
            
            if len(predicted_answers) < 2:
                continue
            
            # 模型答案完全一致 (3/3 agree) 但与 gold 不同
            gold = str(r.get("gold", "")).strip()
            if len(set(predicted_answers)) == 1 and predicted_answers[0] != gold:
                # 检查是否是数值近似
                try:
                    model_val = float(predicted_answers[0])
                    gold_val = float(gold)
                    if abs(model_val - gold_val) < 0.01:
                        continue  # 数值近似，不算
                except:
                    pass
                
                suspicious.append({
                    "id": r["id"],
                    "type": r.get("type", "?"),
                    "gold": gold,
                    "model_answer": predicted_answers[0],
                    "agreement": len(predicted_answers),
                })
    
    print("Total checked: {}".format(total_checked))
    print("Suspicious (model 3/3 agree but != gold): {}".format(len(suspicious)))
    
    # 按题型统计
    sus_by_type = defaultdict(int)
    for s in suspicious:
        sus_by_type[s["type"]] += 1
    for t in sorted(sus_by_type):
        print("  {}: {}".format(t, sus_by_type[t]))
    
    # 显示 numeral 的可疑案例 (最容易验证)
    print("\nSuspicious numeral cases (first 10):")
    num_sus = [s for s in suspicious if s["type"] == "numeral"]
    for s in num_sus[:10]:
        print("  id={}: gold=[{}] model=[{}]".format(s["id"], s["gold"], s["model_answer"]))
    
    print("\nSuspicious other cases (first 10 per type):")
    for t in sorted(sus_by_type):
        if t == "numeral":
            continue
        cases = [s for s in suspicious if s["type"] == t]
        print("  --- {} ---".format(t))
        for s in cases[:5]:
            print("    id={}: gold=[{}] model=[{}]".format(s["id"], s["gold"], s["model_answer"]))

except FileNotFoundError:
    print("cot_t0.jsonl not found, skipping cross-validation")

PYEOF
