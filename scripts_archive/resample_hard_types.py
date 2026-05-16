"""
硬题型重采样脚本 — 用修正后的 max_tokens=7680 重新采样

═══ 背景 ═══
2026-03-24 确认官方 Eval Page 参数:
  max_tokens=7680, max_model_len=8192, temperature=0.0

之前 T0 采样用 max_tokens=3584，导致 bit_ops/cipher/symbol/gravity 截断率 80-99%。
本脚本只重采这些硬题型，利用已有的 multi_model_cot.py 框架。

═══ 策略 ═══
Round 1 (默认): T0 基座模型 重采 unit_conv/gravity/symbol/bit_ops
  - max_tokens=7680 (修正后)
  - 输出: data/cot_t0_v2.jsonl

Round 2 (--round 2): 强模型采 cipher + T0v2 仍失败的 bit_ops/symbol  
  - 模型: ultra-253b 或 super-120b
  - 输出: data/cot_teacher.jsonl

═══ 用法 ═══
  # Round 1: T0 重采硬题型 (先测试10条)
  python scripts/resample_hard_types.py --round 1 --limit 10

  # Round 1: T0 全量重采 (断点续跑)
  python scripts/resample_hard_types.py --round 1 --resume

  # Round 2: 强模型采 cipher + 残余难题
  python scripts/resample_hard_types.py --round 2 --resume

  # 合并: 把 v2 数据合入并重新分级
  python scripts/resample_hard_types.py --merge
"""
import os
import sys
import json
import argparse

# 复用 multi_model_cot.py 的核心框架
sys.path.insert(0, os.path.dirname(__file__))
from multi_model_cot import (
    MODEL_TIERS, DATA_DIR, TRAIN_CSV, METRIC_SUFFIX,
    detect_type, load_rows, run_sampling_phase, run_merge,
    get_zero_correct_ids, load_done_ids,
)
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"

# 需要重采的题型 (Round 1)
HARD_TYPES_R1 = {"unit_conv", "gravity", "symbol", "bit_ops"}

# Round 2: 基座完全不会的题型 + R1 仍失败的
HARD_TYPES_R2 = {"cipher", "bit_ops", "symbol"}

# 强模型配置 (Round 2)
TEACHER_MODEL = {
    "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "desc": "Ultra-253B teacher (最强推理)",
    "samples_per_question": 2,
    "temperatures": [0.0, 0.4],   # greedy + 低温 (对齐评测 temp=0.0)
    "max_tokens": 7680,
    "enable_thinking": True,
}


def filter_rows_by_type(rows, types):
    """只保留指定题型的行。"""
    filtered = []
    for row in rows:
        ptype = detect_type(row["prompt"])
        if ptype in types:
            filtered.append(row)
    return filtered


def get_r1_failed_ids():
    """获取 Round 1 重采后仍然 correct_count=0 的 ID。"""
    path = os.path.join(DATA_DIR, "cot_t0_v2.jsonl")
    if not os.path.exists(path):
        return set()
    failed = set()
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec["correct_count"] == 0:
                    failed.add(rec["id"])
            except:
                pass
    return failed


def run_round1(args, client, rows):
    """Round 1: T0 基座模型，用修正后的 max_tokens=7680 重采硬题型。"""
    hard_rows = filter_rows_by_type(rows, HARD_TYPES_R1)
    print(f"\n📋 Round 1: 重采 {len(hard_rows)} 道硬题 ({', '.join(sorted(HARD_TYPES_R1))})")
    
    # 临时覆盖 T0 输出路径为 v2
    original_dir = DATA_DIR
    
    # 创建临时 tier 配置
    import copy
    tier = copy.deepcopy(MODEL_TIERS["t0"])
    # max_tokens 已在 multi_model_cot.py 中更新为 7680
    
    # 直接用 t0 配置，但输出到 cot_t0_v2.jsonl
    # 我们通过修改 run_sampling_phase 的输出文件名实现
    # 最简单的方式：临时修改 tier key
    
    # 方案：直接注册一个虚拟 tier
    MODEL_TIERS["t0_v2"] = tier
    MODEL_TIERS["t0_v2"]["desc"] = "T0 基座重采 (max_tokens=7680)"
    
    run_sampling_phase("t0_v2", args, client, hard_rows)


def run_round2(args, client, rows):
    """Round 2: 强模型采 cipher 全量 + R1 仍失败的 bit_ops/symbol。"""
    # cipher 全量
    cipher_rows = filter_rows_by_type(rows, {"cipher"})
    
    # R1 仍失败的 bit_ops/symbol
    r1_failed = get_r1_failed_ids()
    other_hard = filter_rows_by_type(rows, {"bit_ops", "symbol"})
    other_hard = [r for r in other_hard if r["id"] in r1_failed]
    
    target_rows = cipher_rows + other_hard
    print(f"\n📋 Round 2: 强模型采样 {len(target_rows)} 道")
    print(f"   cipher: {len(cipher_rows)}, bit_ops/symbol 残余: {len(other_hard)}")
    
    # 注册 teacher tier
    MODEL_TIERS["teacher"] = TEACHER_MODEL
    
    run_sampling_phase("teacher", args, client, target_rows)


def run_merge_v2():
    """合并原始 T0 + T0v2 + teacher 数据并重新分级。
    
    策略:
    - numeral: 直接用原始 T0 (已经有 1456 gold)
    - unit_conv/gravity/symbol/bit_ops: 用 T0v2 替代原始 T0
    - cipher + 残余: 用 teacher 数据
    """
    # 加载原始 T0 (只保留 numeral)
    t0_path = os.path.join(DATA_DIR, "cot_t0.jsonl")
    t0_v2_path = os.path.join(DATA_DIR, "cot_t0_v2.jsonl")
    teacher_path = os.path.join(DATA_DIR, "cot_teacher.jsonl")
    
    merged_t0_path = os.path.join(DATA_DIR, "cot_t0_merged.jsonl")
    
    records = {}
    
    # 1. 原始 T0 的 numeral
    if os.path.exists(t0_path):
        with open(t0_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") == "numeral":
                    records[rec["id"]] = rec
        print(f"Loaded numeral from original T0: {len(records)}")
    
    # 2. T0v2 (覆盖硬题型)
    if os.path.exists(t0_v2_path):
        count = 0
        with open(t0_v2_path) as f:
            for line in f:
                rec = json.loads(line)
                # 修正 tier 标记为 t0 以兼容 merge 逻辑
                rec["tier"] = "t0"
                records[rec["id"]] = rec
                count += 1
        print(f"Loaded T0v2 (hard types): {count}")
    
    # 写出合并的 T0
    with open(merged_t0_path, "w") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Written merged T0: {len(records)} → {merged_t0_path}")
    
    # 3. 如果有 teacher 数据，拷贝为 cot_t1.jsonl (merge 逻辑读 t1)
    if os.path.exists(teacher_path):
        import shutil
        t1_path = os.path.join(DATA_DIR, "cot_t1.jsonl")
        shutil.copy2(teacher_path, t1_path)
        with open(teacher_path) as f:
            teacher_count = sum(1 for _ in f)
        print(f"Linked teacher → t1: {teacher_count} records")
    
    # 4. 临时把 merged_t0 当作 cot_t0.jsonl
    import shutil
    backup_path = os.path.join(DATA_DIR, "cot_t0_original_backup.jsonl")
    if os.path.exists(t0_path) and not os.path.exists(backup_path):
        shutil.copy2(t0_path, backup_path)
        print(f"Backed up original T0 → {backup_path}")
    shutil.copy2(merged_t0_path, t0_path)
    print(f"Replaced cot_t0.jsonl with merged version")
    
    # 5. 运行标准 merge
    run_merge(gold_only=False)
    
    # 6. 恢复原始 T0
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, t0_path)
        print(f"Restored original cot_t0.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Resample hard question types with corrected max_tokens")
    parser.add_argument("--round", type=int, choices=[1, 2], default=1,
                        help="1=T0重采硬题, 2=强模型采cipher+残余")
    parser.add_argument("--merge", action="store_true",
                        help="合并所有数据并重新分级")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gold-only", action="store_true")
    args = parser.parse_args()
    
    if args.merge:
        run_merge_v2()
        return
    
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        return
    
    client = OpenAI(base_url=API_BASE, api_key=api_key)
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from train.csv")
    
    if args.round == 1:
        run_round1(args, client, rows)
    elif args.round == 2:
        run_round2(args, client, rows)


if __name__ == "__main__":
    main()
