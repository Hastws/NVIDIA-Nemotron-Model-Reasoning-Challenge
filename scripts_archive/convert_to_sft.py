#!/usr/bin/env python3
"""
将 multi_model_cot.py 输出的 JSONL 转换为 SFT 训练用的 CSV。

输出格式与 train.csv 完全一致 (id, prompt, answer)，
可直接被 Kaggle notebook 的 build_training_text() 处理：
  user_msg = prompt + METRIC_SUFFIX
  assistant_msg = f"\\boxed{{{answer}}}"

用法示例:
  # 基础: 转换全部 gold 数据
  python scripts/convert_to_sft.py

  # 指定输入/输出
  python scripts/convert_to_sft.py --input data/sft_cot_data.jsonl --output data/sft_train.csv

  # 只要 gold 级别 (推荐第一轮)
  python scripts/convert_to_sft.py --quality gold

  # 限制总量 600 条
  python scripts/convert_to_sft.py --total 600

  # 每类最多 100 条
  python scripts/convert_to_sft.py --max-per-type 100

设计原则:
  - 输出纯 \\boxed{answer}，不含 CoT (历史: CoT=0.63 < answer-only=0.68)
  - 与 E1 (0.68 最佳) 的数据流完全对齐
  - gold 优先: 基座稳定掌握的题 > teacher 教的题
"""
import os
import csv
import json
import random
import argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

DEFAULT_INPUT = os.path.join(DATA_DIR, "sft_cot_data.jsonl")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "sft_train.csv")


def load_jsonl(path):
    """加载 JSONL 文件，跳过损坏行。"""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def convert(args):
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records from {input_path}")

    # 按质量筛选
    if args.quality:
        allowed = set(args.quality.split(","))
        before = len(records)
        records = [r for r in records if r.get("quality") in allowed]
        print(f"Quality filter ({args.quality}): {before} → {len(records)}")

    # 按类型统计
    type_counts = defaultdict(list)
    for r in records:
        type_counts[r.get("type", "unknown")].append(r)

    # 每类限量
    if args.max_per_type > 0:
        limited = []
        for ptype, recs in type_counts.items():
            # gold 优先排序: gold > silver > silver+
            quality_order = {"gold": 0, "silver": 1, "silver+": 2}
            recs.sort(key=lambda x: quality_order.get(x.get("quality", ""), 9))
            take = recs[:args.max_per_type]
            limited.extend(take)
            if len(recs) > args.max_per_type:
                print(f"  {ptype}: {len(recs)} → {len(take)} (capped)")
        records = limited

    # 总量限制
    if args.total > 0 and len(records) > args.total:
        # 按类型均匀采样
        random.seed(args.seed)
        per_type = max(1, args.total // len(type_counts))
        selected = []
        type_lists = defaultdict(list)
        for r in records:
            type_lists[r.get("type", "unknown")].append(r)

        # 先每类取 per_type 条
        remaining_budget = args.total
        for ptype in sorted(type_lists):
            take = min(per_type, len(type_lists[ptype]), remaining_budget)
            selected.extend(type_lists[ptype][:take])
            remaining_budget -= take

        # 有剩余配额: 从各类未选中的里补
        if remaining_budget > 0:
            already = {r["id"] for r in selected}
            pool = [r for r in records if r["id"] not in already]
            random.shuffle(pool)
            selected.extend(pool[:remaining_budget])

        records = selected
        print(f"Total limit: → {len(records)} (target={args.total})")

    # 写出 CSV (与 train.csv 格式一致)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "id": r["id"],
                "prompt": r["prompt"],
                "answer": r["gold"],
            })

    # 统计报告
    final_types = defaultdict(lambda: defaultdict(int))
    for r in records:
        final_types[r.get("type", "unknown")][r.get("quality", "?")] += 1

    print(f"\n{'='*60}")
    print(f"📦 SFT Training CSV: {len(records)} samples")
    print(f"   Output: {output_path}")
    print(f"\n   By type × quality:")
    for ptype in sorted(final_types):
        quals = final_types[ptype]
        parts = [f"{q}={c}" for q, c in sorted(quals.items())]
        total = sum(quals.values())
        print(f"   {ptype:12s}: {total:4d}  ({', '.join(parts)})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert multi_model_cot JSONL → SFT training CSV")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="输入 JSONL 文件 (default: data/sft_cot_data.jsonl)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="输出 CSV 文件 (default: data/sft_train.csv)")
    parser.add_argument("--quality", default=None,
                        help="质量过滤, 逗号分隔 (如 'gold' 或 'gold,silver')")
    parser.add_argument("--max-per-type", type=int, default=0,
                        help="每类题最多 N 条 (0=不限)")
    parser.add_argument("--total", type=int, default=0,
                        help="总量限制 (0=不限)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (用于总量限制时的采样)")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
