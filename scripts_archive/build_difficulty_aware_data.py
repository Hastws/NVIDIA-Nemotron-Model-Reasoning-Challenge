#!/usr/bin/env python3
"""
难度感知训练数据生成器 — Difficulty-Aware Training Data Generator

核心思想：让模型学会"何时思考"的隐式策略选择（Emergent Routing）
- 简单题 → 直接答（answer_only）
- 中等题 → 用紧凑规则（compact）
- 难题   → 必要时展开推理（full_cot）

每个问题只分配一种模式（不重复），通过概率分布控制各桶的模式比例。

Usage:
  python3 scripts/build_difficulty_aware_data.py                     # 全量 9500
  python3 scripts/build_difficulty_aware_data.py --seed 42           # 可复现
  python3 scripts/build_difficulty_aware_data.py --max-cot-len 1500  # 限制 CoT 长度
  python3 scripts/build_difficulty_aware_data.py --truncate-pct 0.1  # 10% 难题 CoT 截断
"""
import csv
import json
import random
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ═══════════════════════════════════════════════════════════════════════════════
#  难度模型参数
# ═══════════════════════════════════════════════════════════════════════════════

# 类型级别基础难度（1 - 基座模型准确率）
TYPE_DIFFICULTY = {
    "numeral":   0.00,  # 100% base acc → trivial
    "gravity":   0.28,  # 72%
    "unit_conv": 0.44,  # 56%
    "cipher":    0.66,  # 34%
    "bit_ops":   0.90,  # 10%
    "symbol":    0.92,  # 8%
}

# 全局最大操作数（用于归一化规则复杂度）
GLOBAL_MAX_OPS = 12  # bit_ops 最大

# 分桶阈值
EASY_THRESHOLD = 0.30
HARD_THRESHOLD = 0.65

# 每个难度桶的输出模式概率分布
MODE_PROBS = {
    "easy":   {"answer_only": 0.70, "compact": 0.30, "full_cot": 0.00},
    "medium": {"answer_only": 0.40, "compact": 0.40, "full_cot": 0.20},
    "hard":   {"answer_only": 0.20, "compact": 0.50, "full_cot": 0.30},
}

# Loss 权重建议（写入 CSV，供训练脚本使用）
THINK_LOSS_WEIGHT = {
    "answer_only": 0.0,   # 无 thinking
    "compact":     0.10,  # thinking loss 权重 = 0.1
    "full_cot":    0.05,  # thinking loss 权重 = 0.05
}


# ═══════════════════════════════════════════════════════════════════════════════
#  难度计算
# ═══════════════════════════════════════════════════════════════════════════════

def count_ops(dsl_rules):
    """Count operations in a DSL rule string (number of [...] blocks)."""
    if not dsl_rules:
        return 0
    return dsl_rules.count("[")


def compute_difficulty(ptype, dsl_rules):
    """
    difficulty = 0.7 * type_base_difficulty + 0.3 * rule_complexity_normalized

    type_base_difficulty: from base model accuracy (higher = harder)
    rule_complexity_normalized: ops / GLOBAL_MAX_OPS (no rules → 1.0 = hardest)
    """
    base = TYPE_DIFFICULTY[ptype]
    ops = count_ops(dsl_rules)

    if ops == 0:
        rule_norm = 1.0  # no rules = assume hardest
    else:
        rule_norm = min(ops / GLOBAL_MAX_OPS, 1.0)

    return 0.7 * base + 0.3 * rule_norm


def assign_bucket(difficulty):
    if difficulty < EASY_THRESHOLD:
        return "easy"
    elif difficulty < HARD_THRESHOLD:
        return "medium"
    else:
        return "hard"


# ═══════════════════════════════════════════════════════════════════════════════
#  模式分配
# ═══════════════════════════════════════════════════════════════════════════════

def assign_mode(bucket, available_modes, rng):
    """
    Based on bucket probabilities, assign one output mode.
    If desired mode not available, redistribute probability proportionally.
    """
    probs = MODE_PROBS[bucket].copy()

    # Redistribute unavailable modes' probability
    unavailable_total = sum(probs[m] for m in probs if m not in available_modes)
    if unavailable_total > 0:
        for m in list(probs):
            if m not in available_modes:
                probs[m] = 0.0
        remaining_total = sum(probs.values())
        if remaining_total > 0:
            for m in probs:
                probs[m] += unavailable_total * (probs[m] / remaining_total)

    # Sample
    r = rng.random()
    cumulative = 0.0
    for mode, p in probs.items():
        cumulative += p
        if r < cumulative:
            return mode
    return list(probs.keys())[-1]


# ═══════════════════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load all data sources and build lookup tables."""

    # 1. All problems with types and DSL rules
    problems = {}
    with open(DATA_DIR / "train_dsl_rules.csv", "r") as f:
        for row in csv.DictReader(f):
            problems[row["id"]] = {
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "type": row["type"],
                "dsl_rules": row.get("dsl_rules", ""),
            }

    # 2. Compact thinking (5 types, ~7945 rows)
    compact = {}
    with open(DATA_DIR / "sft_compact_rules.csv", "r") as f:
        for row in csv.DictReader(f):
            compact[row["id"]] = row["thinking"]

    # 3. Full CoT from cot_v2 (6 types, ~8032 rows)
    full_cot = {}
    with open(DATA_DIR / "sft_cot_v2.csv", "r") as f:
        for row in csv.DictReader(f):
            full_cot[row["id"]] = row["thinking"]

    # 4. Symbol solved (1275 LLM-generated CoT)
    with open(DATA_DIR / "symbol_solved.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            if d["solved"] and d["id"] not in full_cot:
                full_cot[d["id"]] = d["content"]

    # 5. Create compact for symbol from DSL rules (229 problems)
    for pid, p in problems.items():
        if p["type"] == "symbol" and p["dsl_rules"] and pid not in compact:
            # Simplify DSL → compact format: [OP:*→CONCAT] → *→CONCAT
            rules = p["dsl_rules"]
            compact_text = rules.replace("[", "").replace("]", "").replace("OP:", "")
            # Multi-op: join with semicolons
            parts = [x.strip() for x in compact_text.split("\n") if x.strip()]
            compact[pid] = ";".join(parts)

    print(f"Loaded: {len(problems)} problems, {len(compact)} compact, {len(full_cot)} full_cot")
    return problems, compact, full_cot


# ═══════════════════════════════════════════════════════════════════════════════
#  CoT 截断（进阶技巧）
# ═══════════════════════════════════════════════════════════════════════════════

def truncate_cot(thinking, max_len, rng, do_partial_truncate=False, trunc_pct=0.1):
    """
    1. Hard cap at max_len chars
    2. Optionally: partial truncation for some hard problems (simulate incomplete reasoning)
    """
    if not thinking:
        return thinking

    # Hard cap
    if len(thinking) > max_len:
        # Truncate at last complete sentence/step before max_len
        truncated = thinking[:max_len]
        # Try to find a good break point
        for sep in ["\n\n", "\n", ". ", ", "]:
            idx = truncated.rfind(sep)
            if idx > max_len * 0.5:
                truncated = truncated[:idx]
                break
        thinking = truncated.rstrip()

    # Partial truncation for difficulty (optional)
    if do_partial_truncate and rng.random() < trunc_pct:
        cut_point = rng.randint(int(len(thinking) * 0.4), int(len(thinking) * 0.8))
        thinking = thinking[:cut_point].rstrip()

    return thinking


# ═══════════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Difficulty-Aware Training Data Generator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="sft_difficulty_aware.csv",
                        help="Output filename (in data/)")
    parser.add_argument("--max-cot-len", type=int, default=1500,
                        help="Max thinking length in chars for full_cot")
    parser.add_argument("--truncate-pct", type=float, default=0.0,
                        help="Fraction of hard full_cot to partially truncate (0-1)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    problems, compact, full_cot = load_all_data()

    # ─── 计算难度 & 分桶 ─────────────────────────────────────────────────
    buckets = defaultdict(list)
    for pid, p in problems.items():
        diff = compute_difficulty(p["type"], p["dsl_rules"])
        bucket = assign_bucket(diff)
        p["difficulty"] = diff
        p["bucket"] = bucket
        buckets[bucket].append(pid)

    print(f"\nBucket distribution:")
    for b in ["easy", "medium", "hard"]:
        types_in = Counter(problems[pid]["type"] for pid in buckets[b])
        print(f"  {b}: {len(buckets[b])} problems | {dict(types_in)}")

    # ─── 分配模式 ──────────────────────────────────────────────────────
    output_rows = []
    stats = defaultdict(lambda: defaultdict(Counter))  # type → bucket → mode → count

    for pid, p in problems.items():
        bucket = p["bucket"]

        # Available modes
        available = {"answer_only"}
        if pid in compact:
            available.add("compact")
        if pid in full_cot:
            available.add("full_cot")

        # Assign ONE mode
        mode = assign_mode(bucket, available, rng)

        # Get thinking content
        thinking = ""
        if mode == "compact":
            thinking = compact.get(pid, "")
        elif mode == "full_cot":
            thinking = full_cot.get(pid, "")

        # Fallback: if thinking empty for a thinking mode → answer_only
        if mode != "answer_only" and not thinking.strip():
            mode = "answer_only"
            thinking = ""

        # Truncate long CoT
        if mode == "full_cot" and thinking:
            do_partial = (bucket == "hard" and args.truncate_pct > 0)
            thinking = truncate_cot(
                thinking, args.max_cot_len, rng,
                do_partial_truncate=do_partial,
                trunc_pct=args.truncate_pct,
            )

        # Build output row
        output_rows.append({
            "id": pid,
            "prompt": p["prompt"],
            "answer": p["answer"],
            "thinking": thinking,
            "type": p["type"],
            "difficulty": bucket,
            "mode": mode,
            "think_loss_weight": THINK_LOSS_WEIGHT[mode],
        })

        stats[p["type"]][bucket][mode] += 1

    # ─── 保存 ──────────────────────────────────────────────────────────
    output_path = DATA_DIR / args.output
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "prompt", "answer", "thinking", "type",
            "difficulty", "mode", "think_loss_weight",
        ])
        writer.writeheader()
        writer.writerows(output_rows)

    # ─── 报告 ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Output: {output_path} ({len(output_rows)} rows)")

    # Mode distribution
    mode_counts = Counter(r["mode"] for r in output_rows)
    total = len(output_rows)
    print(f"\nOverall mode distribution:")
    for mode in ["answer_only", "compact", "full_cot"]:
        n = mode_counts[mode]
        print(f"  {mode:12s}: {n:5d} ({n/total*100:5.1f}%)")

    # Per-bucket distribution
    print(f"\nPer-bucket × mode distribution:")
    print(f"{'Bucket':>8s} | {'answer_only':>12s} | {'compact':>12s} | {'full_cot':>12s} | {'Total':>8s}")
    print("-" * 65)
    for bucket in ["easy", "medium", "hard"]:
        counts = Counter(r["mode"] for r in output_rows if r["difficulty"] == bucket)
        tot = sum(counts.values())
        parts = []
        for mode in ["answer_only", "compact", "full_cot"]:
            n = counts[mode]
            parts.append(f"{n:5d} ({n/tot*100:4.1f}%)")
        print(f"{bucket:>8s} | {'  |  '.join(parts)}  | {tot:8d}")

    # Per-type breakdown
    print(f"\nPer-type × mode distribution:")
    print(f"{'Type':>10s} | {'Bucket':>6s} | {'answer_only':>12s} | {'compact':>12s} | {'full_cot':>12s} | {'Total':>6s}")
    print("-" * 75)
    for ptype in ["numeral", "gravity", "unit_conv", "cipher", "bit_ops", "symbol"]:
        type_rows = [r for r in output_rows if r["type"] == ptype]
        type_bucket = type_rows[0]["difficulty"] if type_rows else "?"
        counts = Counter(r["mode"] for r in type_rows)
        tot = len(type_rows)
        parts = []
        for mode in ["answer_only", "compact", "full_cot"]:
            n = counts[mode]
            parts.append(f"{n:5d} ({n/tot*100:4.1f}%)")
        print(f"{ptype:>10s} | {type_bucket:>6s} | {'  |  '.join(parts)}  | {tot:6d}")

    # Thinking length stats
    print(f"\nThinking length stats by mode:")
    for mode in ["compact", "full_cot"]:
        lens = [len(r["thinking"]) for r in output_rows if r["mode"] == mode and r["thinking"]]
        if lens:
            avg = sum(lens) / len(lens)
            print(f"  {mode:12s}: n={len(lens)}, avg={avg:.0f}, min={min(lens)}, max={max(lens)}")

    # Stage split (for 2-stage training)
    stage1 = [r for r in output_rows if r["mode"] in ("compact", "full_cot")]
    stage2 = [r for r in output_rows if r["mode"] == "answer_only"]
    print(f"\n2-Stage split:")
    print(f"  Stage 1 (thinking): {len(stage1)} rows")
    print(f"  Stage 2 (answer):   {len(stage2)} rows")

    print(f"\nDone! ✅")


if __name__ == "__main__":
    main()
