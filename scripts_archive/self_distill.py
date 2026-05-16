"""
Self-Distillation Sampling — 用基座模型自己的推理作为训练数据

核心理念:
  - 纯自蒸馏: 只用 nano-30b，不引入其他模型的分布
  - 多次采样 + rejection sampling: 每题 N 次，选正确且最优的 CoT
  - 分题型 prompt: 针对各题型特点引导更短/更准的推理
  - max_tokens=7680: 对齐官方评测参数

输出: data/self_distill.jsonl
每行: {id, type, prompt, gold, samples: [...], best_thinking, best_response, correct_count}
"""
import os
import re
import csv
import json
import math
import time
import argparse
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════
class RateLimiter:
    """Token bucket rate limiter — max N requests per second."""
    def __init__(self, rps=5):
        self.rps = rps
        self.lock = threading.Lock()
        self.tokens = rps
        self.last = time.monotonic()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                self.tokens = min(self.rps, self.tokens + elapsed * self.rps)
                self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.05)

rate_limiter = RateLimiter(rps=5)  # 5 requests/sec to avoid 429

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"

# 官方评测后缀
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# 采样配置: 每题 N 次，不同温度增加多样性
SAMPLES_PER_QUESTION = 8
TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 0.7, 0.9, 1.0, 1.2]  # 8 temps, 从 greedy 到探索
MAX_TOKENS = 7680  # 对齐官方 Eval Page

# 分题型 system prompt — 引导更短更准的推理
# 设计原则: 不改变问题本身，只通过 system prompt 引导推理风格
TYPE_SYSTEM_PROMPTS = {
    "gravity": (
        "You are solving a physics problem about gravitational constants. "
        "Strategy: Use the formula d = 0.5*g*t^2 to find g from any example pair, "
        "then apply g to the target time. Be precise to 2 decimal places. "
        "Think step by step but be concise."
    ),
    "unit_conv": (
        "You are solving a unit conversion problem. "
        "Strategy: Find the conversion factor by dividing output by input from any example pair, "
        "then multiply the target by this factor. Round to 2 decimal places. "
        "Think step by step but be concise."
    ),
    "numeral": (
        "You are solving a numeral system conversion problem. "
        "Strategy: Identify the target numeral system (e.g., Roman numerals) from examples, "
        "then convert the given number. Be precise and concise."
    ),
    "cipher": (
        "You are decrypting text using a substitution cipher. "
        "Strategy: Build a letter mapping from the example pairs (encrypted -> decrypted), "
        "then apply the mapping to decrypt the target text. "
        "Work letter by letter. Be systematic and concise."
    ),
    "bit_ops": (
        "You are solving a bit manipulation problem on 8-bit binary numbers. "
        "Strategy: Compare input and output bits position by position across examples "
        "to find the per-bit transformation rule (e.g., XOR, AND, OR, NOT, shift, rotate). "
        "Then apply the rule to the target input. Be systematic and concise."
    ),
    "symbol": (
        "You are solving a symbol transformation problem on equations. "
        "Strategy: Analyze how each symbol maps to another in the examples. "
        "Build a character-level mapping table and apply it to the target. "
        "Be systematic and concise."
    ),
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TRAIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "competition_data", "train.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def detect_type(prompt):
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p):
        return "bit_ops"
    elif "encrypt" in p or "decrypt" in p or "cipher" in p:
        return "cipher"
    elif "gravit" in p:
        return "gravity"
    elif "numeral" in p or "wonderland numbers" in p:
        return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p):
        return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p):
        return "symbol"
    return "unknown"


def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return None


def answers_match(pred, gold):
    if pred is None:
        return False
    pred, gold = pred.strip(), gold.strip()
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, OverflowError):
        return pred.lower() == gold.lower()


def score_cot(sample):
    """评分: 完整性 > 推理深度 > 长度适中 > 格式。偏好短且正确的 CoT。"""
    score = 0
    thinking = sample.get("thinking") or ""
    response = sample.get("response") or ""
    think_len = len(thinking)
    total_len = think_len + len(response)

    # 完整输出 (没被截断)
    if sample.get("finish_reason") == "stop":
        score += 5
    elif sample.get("finish_reason") == "length":
        score -= 5  # 截断的不要

    # 有实质推理
    if think_len > 100:
        score += 2
    elif think_len > 30:
        score += 1

    # 偏好更短的正确推理 (短=高效=更好的训练信号)
    if 100 <= total_len <= 2000:
        score += 3   # 甜区: 短且有内容
    elif 2000 < total_len <= 4000:
        score += 1   # 可以接受
    elif total_len > 5000:
        score -= 2   # 太长

    # boxed 在 content 中
    if extract_boxed(response):
        score += 2

    return score


def call_api(client, prompt, temperature, ptype, retries=6):
    """调用 API，分题型添加 system prompt。带 rate limiter + 指数退避。"""
    user_content = prompt + METRIC_SUFFIX
    
    messages = []
    sys_prompt = TYPE_SYSTEM_PROMPTS.get(ptype)
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_content})

    extra = {"chat_template_kwargs": {"enable_thinking": True}}

    for attempt in range(retries):
        rate_limiter.acquire()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                top_p=1.0,
                max_tokens=MAX_TOKENS,
                extra_body=extra,
                timeout=300,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            thinking = getattr(choice.message, 'reasoning_content', None) or ""
            finish = choice.finish_reason
            return thinking, content, finish
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                wait = min(2 ** (attempt + 1), 60)  # 2,4,8,16,32,60
                if attempt < 3:  # Only print first few
                    print(f"    ⚠️ Rate limited, wait {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            elif attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    ⚠️ Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                return "", f"[ERROR] {e}", "error"


def call_api_no_system(client, prompt, temperature, retries=6):
    """调用 API，不加 system prompt（对照组/fallback）。"""
    user_content = prompt + METRIC_SUFFIX
    messages = [{"role": "user", "content": user_content}]
    extra = {"chat_template_kwargs": {"enable_thinking": True}}

    for attempt in range(retries):
        rate_limiter.acquire()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                top_p=1.0,
                max_tokens=MAX_TOKENS,
                extra_body=extra,
                timeout=300,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            thinking = getattr(choice.message, 'reasoning_content', None) or ""
            finish = choice.finish_reason
            return thinking, content, finish
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                wait = min(2 ** (attempt + 1), 60)
                if attempt < 3:
                    print(f"    ⚠️ Rate limited, wait {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            elif attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    ⚠️ Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                return "", f"[ERROR] {e}", "error"


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLING — 单题多次采样
# ═══════════════════════════════════════════════════════════════════════════════
def sample_one(client, prompt, gold, temp, ptype, use_system=True):
    """单次采样。"""
    t0 = time.time()
    if use_system:
        thinking, content, finish = call_api(client, prompt, temp, ptype)
    else:
        thinking, content, finish = call_api_no_system(client, prompt, temp)
    elapsed = time.time() - t0

    pred = extract_boxed(content)
    correct = answers_match(pred, gold)

    return {
        "temperature": temp,
        "thinking": thinking,
        "response": content,
        "predicted": pred,
        "correct": correct,
        "finish_reason": finish,
        "elapsed": round(elapsed, 2),
        "think_len": len(thinking),
        "use_system": use_system,
    }


def process_question(client, row, use_system=True):
    """处理一道题: 8次并发采样 → 统计 → 选最佳 CoT。"""
    sid = row["id"]
    prompt = row["prompt"]
    gold = row["answer"]
    ptype = detect_type(prompt)

    # 并发采样 (限制内层并发防止 429)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for i in range(SAMPLES_PER_QUESTION):
            temp = TEMPERATURES[i] if i < len(TEMPERATURES) else TEMPERATURES[-1]
            futures.append(pool.submit(sample_one, client, prompt, gold, temp, ptype, use_system))
        samples = [f.result() for f in futures]

    correct_count = sum(1 for s in samples if s["correct"])
    truncated_count = sum(1 for s in samples if s["finish_reason"] == "length")

    # 选最佳正确 CoT
    best = None
    best_score = -999
    for s in samples:
        if not s["correct"]:
            continue
        sc = score_cot(s)
        if sc > best_score:
            best_score = sc
            best = s

    return {
        "id": sid,
        "type": ptype,
        "prompt": prompt,
        "gold": gold,
        "n_samples": SAMPLES_PER_QUESTION,
        "correct_count": correct_count,
        "truncated_count": truncated_count,
        "samples": samples,
        "best_thinking": best["thinking"] if best else "",
        "best_response": best["response"] if best else "",
        "best_predicted": best["predicted"] if best else None,
        "best_score": best_score if best else None,
        "best_think_len": best["think_len"] if best else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Self-distillation sampling for Nemotron")
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--resume", action="store_true", help="Skip already done IDs")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent questions")
    parser.add_argument("--no-system", action="store_true", help="Skip system prompts (baseline)")
    parser.add_argument("--output", type=str, default="self_distill.jsonl", help="Output filename")
    parser.add_argument("--type-filter", type=str, default="", help="Only sample this type (e.g. bit_ops)")
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        return

    client = OpenAI(base_url=API_BASE, api_key=api_key)

    # Load data
    train_path = os.path.abspath(TRAIN_CSV)
    with open(train_path) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from train.csv")

    # Filter by type if requested
    if args.type_filter:
        rows = [r for r in rows if detect_type(r["prompt"]) == args.type_filter]
        print(f"Filtered to {len(rows)} rows of type '{args.type_filter}'")

    rows = rows[args.start:]
    if args.limit > 0:
        rows = rows[:args.limit]

    # Resume
    output_path = os.path.join(DATA_DIR, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    done_ids = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except:
                    pass
        print(f"Resume: {len(done_ids)} already done")
    rows = [r for r in rows if r["id"] not in done_ids]

    total = len(rows)
    use_system = not args.no_system
    print(f"\n{'='*70}")
    print(f"Self-Distillation Sampling")
    print(f"  Model:       {MODEL}")
    print(f"  Samples:     {SAMPLES_PER_QUESTION}/question, temps={TEMPERATURES}")
    print(f"  Max tokens:  {MAX_TOKENS}")
    print(f"  System:      {'per-type prompts' if use_system else 'none (baseline)'}")
    print(f"  Workers:     {args.workers}")
    print(f"  Questions:   {total}")
    print(f"  Output:      {output_path}")
    print(f"{'='*70}\n")

    # Stats
    processed = 0
    type_stats = defaultdict(lambda: {"total": 0, "any_correct": 0, "all_correct": 0, "truncated": 0})
    write_lock = threading.Lock()

    open_mode = "a" if args.resume else "w"
    with open(output_path, open_mode) as out_f:
        def handle_row(row):
            return process_question(client, row, use_system=use_system)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(handle_row, r): r for r in rows}

            for future in as_completed(future_map):
                try:
                    record = future.result()
                except Exception as exc:
                    row = future_map[future]
                    print(f"❌ ERROR {row['id']}: {exc}")
                    continue

                with write_lock:
                    processed += 1
                    ptype = record["type"]
                    type_stats[ptype]["total"] += 1
                    if record["correct_count"] > 0:
                        type_stats[ptype]["any_correct"] += 1
                    if record["correct_count"] == SAMPLES_PER_QUESTION:
                        type_stats[ptype]["all_correct"] += 1
                    type_stats[ptype]["truncated"] += record["truncated_count"]

                    # Write (strip full samples to save space, keep only best)
                    slim = {k: v for k, v in record.items() if k != "samples"}
                    # Keep sample summaries (correct, think_len, finish_reason) for analysis
                    slim["sample_summary"] = [
                        {"t": s["temperature"], "ok": s["correct"], "fin": s["finish_reason"],
                         "tlen": s["think_len"]}
                        for s in record["samples"]
                    ]
                    out_f.write(json.dumps(slim, ensure_ascii=False) + "\n")
                    out_f.flush()

                    # Progress
                    any_total = sum(s["any_correct"] for s in type_stats.values())
                    all_total = sum(s["all_correct"] for s in type_stats.values())
                    icon = "🟢" if record["correct_count"] == SAMPLES_PER_QUESTION else (
                        "🟡" if record["correct_count"] > 0 else "🔴")
                    think_info = f"think={record['best_think_len']:5d}" if record["best_thinking"] else "think=    0"
                    print(f"{icon} [{processed:4d}/{total}] {ptype:10s} "
                          f"{record['correct_count']}/{SAMPLES_PER_QUESTION} correct "
                          f"trunc={record['truncated_count']} "
                          f"{think_info} "
                          f"| any={any_total/processed:.3f} all={all_total/processed:.3f}")

                    # Checkpoint every 200
                    if processed % 200 == 0:
                        print(f"\n{'='*70}")
                        print(f"📊 CHECKPOINT {processed}/{total}")
                        for t in sorted(type_stats):
                            s = type_stats[t]
                            n = s["total"]
                            if n == 0:
                                continue
                            a_any = s["any_correct"] / n
                            a_all = s["all_correct"] / n
                            trunc_pct = s["truncated"] / (n * SAMPLES_PER_QUESTION)
                            print(f"  {t:12s}: any={a_any:.3f} all={a_all:.3f} trunc={trunc_pct:.1%} ({n})")
                        print(f"{'='*70}\n")

    # Final report
    print(f"\n{'='*70}")
    print(f"🏁 FINAL REPORT — {processed} questions")
    print(f"{'Type':14s} {'N':>5s} {'Any%':>7s} {'All%':>7s} {'Trunc%':>7s}")
    for t in sorted(type_stats):
        s = type_stats[t]
        n = s["total"]
        if n == 0:
            continue
        print(f"{t:14s} {n:5d} {s['any_correct']/n*100:6.1f}% {s['all_correct']/n*100:6.1f}% "
              f"{s['truncated']/(n*SAMPLES_PER_QUESTION)*100:6.1f}%")
    n_total = sum(s["total"] for s in type_stats.values())
    n_any = sum(s["any_correct"] for s in type_stats.values())
    n_all = sum(s["all_correct"] for s in type_stats.values())
    n_trunc = sum(s["truncated"] for s in type_stats.values())
    print(f"{'TOTAL':14s} {n_total:5d} {n_any/n_total*100:6.1f}% {n_all/n_total*100:6.1f}% "
          f"{n_trunc/(n_total*SAMPLES_PER_QUESTION)*100:6.1f}%")
    print(f"\nOutput: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
