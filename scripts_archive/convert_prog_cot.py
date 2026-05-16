#!/usr/bin/env python3
"""
将程序化 CoT 数据 (programmatic_cot.jsonl) 转换为 SFT 训练 CSV。

输出两种格式供对比实验:
1. answer-only (与 E1 对齐): prompt + \boxed{answer}
2. cot-thinking (新策略): prompt + <think>CoT</think>\boxed{answer}

训练 notebook 的 build_training_text() 会用 apply_chat_template(enable_thinking=True)
把 assistant content 包装成:
  <|im_start|>assistant\n<think>\n</think>\n{content}<|im_end|>

所以:
- answer-only: content = \boxed{answer} → 模型自己思考
- cot-thinking: content = <think>CoT</think>\boxed{answer} → 显式教推理过程

设计决策:
- 由于 enable_thinking=True 的 chat template 已经加了 <think></think>，
  如果我们的 content 也包含 <think>...</think>，会形成嵌套。
  所以 cot 版本需要在 build_training_text 中特殊处理。
- 最简单: cot 版本直接输出完整 text 字段 (跳过 build_training_text)。
- 或者: 输出 CSV 多加一个 thinking 列，让 build_training_text 支持它。

本脚本输出两种 CSV:
- sft_prog_answer_only.csv: (id, prompt, answer) — 与 train.csv 格式一致
- sft_prog_with_cot.csv: (id, prompt, answer, thinking) — 多了 thinking 列
"""
import os
import csv
import json
import random
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT_PATH = os.path.join(DATA_DIR, "programmatic_cot.jsonl")
OUTPUT_ANSWER_ONLY = os.path.join(DATA_DIR, "sft_prog_answer_only.csv")
OUTPUT_WITH_COT = os.path.join(DATA_DIR, "sft_prog_with_cot.csv")


def load_data():
    records = []
    with open(INPUT_PATH) as f:
        for line in f:
            r = json.loads(line.strip())
            records.append(r)
    return records


def main():
    records = load_data()
    print("Loaded {} programmatic CoT records".format(len(records)))

    # 按题型统计
    by_type = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)

    for t in sorted(by_type):
        print("  {}: {}".format(t, len(by_type[t])))

    # ═══════════════════════════════════════════════════════════════════════════
    # 输出 1: answer-only CSV (与 train.csv 完全兼容)
    # ═══════════════════════════════════════════════════════════════════════════
    with open(OUTPUT_ANSWER_ONLY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r["id"],
                "prompt": r["prompt"],
                "answer": r["gold"],
            })
    print("\nAnswer-only: {} -> {}".format(len(records), OUTPUT_ANSWER_ONLY))

    # ═══════════════════════════════════════════════════════════════════════════
    # 输出 2: with-CoT CSV (多一个 thinking 列)
    # ═══════════════════════════════════════════════════════════════════════════
    with open(OUTPUT_WITH_COT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "thinking"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r["id"],
                "prompt": r["prompt"],
                "answer": r["gold"],
                "thinking": r["thinking"],
            })
    print("With-CoT: {} -> {}".format(len(records), OUTPUT_WITH_COT))

    # ═══════════════════════════════════════════════════════════════════════════
    # 输出 3: 按题型平衡采样版 (每类最多 N 条)
    # ═══════════════════════════════════════════════════════════════════════════
    for max_per in [100, 200]:
        sampled = []
        random.seed(42)
        for t in sorted(by_type):
            pool = by_type[t]
            n = min(len(pool), max_per)
            sampled.extend(random.sample(pool, n))
        
        random.shuffle(sampled)
        
        # answer-only
        out_ao = os.path.join(DATA_DIR, "sft_prog_ao_{}.csv".format(max_per))
        with open(out_ao, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
            writer.writeheader()
            for r in sampled:
                writer.writerow({"id": r["id"], "prompt": r["prompt"], "answer": r["gold"]})
        
        # with-cot
        out_cot = os.path.join(DATA_DIR, "sft_prog_cot_{}.csv".format(max_per))
        with open(out_cot, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer", "thinking"])
            writer.writeheader()
            for r in sampled:
                writer.writerow({
                    "id": r["id"], "prompt": r["prompt"],
                    "answer": r["gold"], "thinking": r["thinking"],
                })
        
        print("Balanced {}/type: {} records -> {} + {}".format(max_per, len(sampled), out_ao, out_cot))

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary token estimates
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n--- Token estimates (chars/4) ---")
    for r in records[:3]:
        # answer-only: prompt + suffix + \boxed{answer}
        ao_len = len(r["prompt"]) + 50 + len(r["gold"]) + 10
        # with-cot: prompt + suffix + <think>thinking</think>\boxed{answer}
        cot_len = len(r["prompt"]) + 50 + len(r["thinking"]) + 20 + len(r["gold"]) + 10
        print("  id={} type={}: ao~{} tok, cot~{} tok".format(
            r["id"][:8], r["type"], ao_len // 4, cot_len // 4))


if __name__ == "__main__":
    main()
