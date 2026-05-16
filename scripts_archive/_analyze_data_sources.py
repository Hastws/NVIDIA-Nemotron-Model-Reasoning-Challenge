#!/usr/bin/env python3
"""Analyze data sources for difficulty-aware dataset construction."""
import csv, json

# 1. DSL rules structure
print("=== DSL Rules Structure ===")
with open("data/train_dsl_rules.csv") as f:
    reader = csv.DictReader(f)
    by_type = {}
    for row in reader:
        t = row["type"]
        rules = row.get("dsl_rules", "")
        if t not in by_type:
            by_type[t] = {"total": 0, "has_rules": 0, "rule_lengths": [], "examples": []}
        by_type[t]["total"] += 1
        if rules:
            by_type[t]["has_rules"] += 1
            # Count operations (heuristic: count brackets or commas)
            ops = rules.count("[") if "[" in rules else rules.count(",") + 1
            by_type[t]["rule_lengths"].append(ops)
            if len(by_type[t]["examples"]) < 3:
                by_type[t]["examples"].append(rules[:150])

    for t, info in sorted(by_type.items()):
        rl = info["rule_lengths"]
        avg_ops = sum(rl) / len(rl) if rl else 0
        print(f"\n{t}: {info['has_rules']}/{info['total']} have rules, avg_ops={avg_ops:.1f}")
        if rl:
            from collections import Counter
            c = Counter(rl)
            print(f"  Op counts: {dict(sorted(c.items()))}")
        for ex in info["examples"]:
            print(f"  Rule: {ex}")

# 2. Check compact rules structure
print("\n\n=== Compact Rules (thinking field) ===")
with open("data/sft_compact_rules.csv") as f:
    reader = csv.DictReader(f)
    types_compact = {}
    for row in reader:
        t = row["type"]
        thinking = row.get("thinking", "")
        if t not in types_compact:
            types_compact[t] = {"lens": [], "examples": []}
        types_compact[t]["lens"].append(len(thinking))
        if len(types_compact[t]["examples"]) < 2:
            types_compact[t]["examples"].append(thinking[:120])

    for t, info in sorted(types_compact.items()):
        avg = sum(info["lens"]) / len(info["lens"])
        mn = min(info["lens"])
        mx = max(info["lens"])
        print(f"\n{t}: n={len(info['lens'])}, thinking_len avg={avg:.0f} min={mn} max={mx}")
        for ex in info["examples"]:
            print(f"  Example: {ex}")

# 3. Check cot_v2 thinking lengths
print("\n\n=== CoT V2 ===")
with open("data/sft_cot_v2.csv") as f:
    reader = csv.DictReader(f)
    types_v2 = {}
    for row in reader:
        t = row["type"]
        thinking = row.get("thinking", "")
        if t not in types_v2:
            types_v2[t] = {"lens": [], "examples": []}
        types_v2[t]["lens"].append(len(thinking))
        if len(types_v2[t]["examples"]) < 1:
            types_v2[t]["examples"].append(thinking[:200])

    for t, info in sorted(types_v2.items()):
        avg = sum(info["lens"]) / len(info["lens"])
        print(f"\n{t}: n={len(info['lens'])}, avg_thinking_len={avg:.0f}")
        for ex in info["examples"]:
            print(f"  Example: {ex[:120]}...")

# 4. symbol_solved stats
print("\n\n=== Symbol Solved ===")
data = [json.loads(l) for l in open("data/symbol_solved.jsonl")]
solved = [d for d in data if d["solved"]]
failed = [d for d in data if not d["solved"]]
c_lens = [len(d["content"]) for d in solved]
print(f"Solved: {len(solved)}, Failed: {len(failed)}")
print(f"Content avg: {sum(c_lens)/len(c_lens):.0f}, min: {min(c_lens)}, max: {max(c_lens)}")

# 5. Check what columns each CSV has
print("\n\n=== CSV Columns ===")
for fn in ["sft_compact_rules.csv", "sft_cot_v2.csv", "sft_ao_7741.csv", "sft_full_cot.csv"]:
    with open(f"data/{fn}") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        print(f"{fn}: columns={list(row.keys())}")

# 6. Full CoT coverage by type
print("\n\n=== Full CoT Coverage ===")
with open("data/sft_full_cot.csv") as f:
    reader = csv.DictReader(f)
    types_full = {}
    for row in reader:
        t = row["type"]
        thinking = row.get("thinking", "")
        if t not in types_full:
            types_full[t] = {"count": 0, "lens": []}
        types_full[t]["count"] += 1
        types_full[t]["lens"].append(len(thinking))
    
    for t, info in sorted(types_full.items()):
        avg = sum(info["lens"]) / len(info["lens"]) if info["lens"] else 0
        print(f"{t}: n={info['count']}, avg_thinking_len={avg:.0f}")
