#!/usr/bin/env python3
"""
Evaluate Nemotron-3-Nano-30B base model via NVIDIA API.
Mirrors the OFFICIAL Kaggle evaluation pipeline exactly.

Usage:
  python scripts/eval_base_model_api.py                          # 50/type = 300 samples
  python scripts/eval_base_model_api.py --samples-per-type 100   # 100/type = 600 samples
  python scripts/eval_base_model_api.py --all                    # all 9500 samples
  python scripts/eval_base_model_api.py --resume                 # resume from last run
"""
import os
import re
import csv
import json
import math
import time
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict

from openai import OpenAI

# ============================================================
#  OFFICIAL EVALUATION PARAMETERS (from Kaggle Eval Page)
# ============================================================
API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
TEMPERATURE = 0.0       # Official: greedy
TOP_P = 1.0
MAX_TOKENS = 7680       # Official eval page override


# ============================================================
#  OFFICIAL answer extraction (verbatim from metric script)
# ============================================================
def extract_final_answer(text):
    r"""Extracts the final answer from the model response.
    Prioritizes extracting answers inside `\boxed{}`.
    Verbatim from official metric.
    """
    if text is None:
        return 'NOT_FOUND'

    # Match all instances of \boxed{...} or unclosed \boxed{ at the end
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # Extract the last valid number
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer, predicted):
    """Official verify function — verbatim from metric script."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


# ============================================================
#  Type classifier (same as E1)
# ============================================================
def classify_type(prompt_text):
    p = prompt_text.lower()
    if 'bit manipulation' in p or '8-bit binary' in p:
        return 'bit_ops'
    elif 'encrypt' in p or 'decrypt' in p:
        return 'cipher'
    elif 'gravitational' in p or 'falling distance' in p:
        return 'gravity'
    elif 'numeral system' in p:
        return 'numeral'
    elif 'transformation rules' in p:
        return 'symbol'
    elif 'unit conversion' in p or 'convert the following measurement' in p:
        return 'unit_conv'
    return 'unknown'


# ============================================================
#  API call (matches official eval: enable_thinking + greedy)
# ============================================================
def call_api(client, prompt, enable_thinking=True, max_retries=3):
    """Call NVIDIA API with official eval-matching parameters."""
    messages = [{"role": "user", "content": prompt + METRIC_SUFFIX}]
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                timeout=300,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            thinking = getattr(choice.message, 'reasoning_content', None) or ""
            finish = choice.finish_reason
            return {
                "content": content,
                "thinking": thinking,
                "finish_reason": finish,
                "error": None,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                time.sleep(wait)
            else:
                return {
                    "content": None,
                    "thinking": None,
                    "finish_reason": None,
                    "error": str(e),
                }


def process_one(client, item, enable_thinking=True):
    """Process one sample, return full result dict."""
    prompt = item["prompt"]
    gold = str(item["answer"])
    qtype = item.get("type", "unknown")
    
    result = call_api(client, prompt, enable_thinking)
    
    if result["error"]:
        return {
            "id": item["id"],
            "type": qtype,
            "correct": False,
            "predicted": "ERROR",
            "gold": gold,
            "finish_reason": None,
            "error": result["error"],
            "content_len": 0,
            "thinking_len": 0,
        }
    
    predicted = extract_final_answer(result["content"])
    correct = verify(gold, predicted)
    
    return {
        "id": item["id"],
        "type": qtype,
        "correct": correct,
        "predicted": predicted,
        "gold": gold,
        "finish_reason": result["finish_reason"],
        "error": None,
        "content_len": len(result["content"]),
        "thinking_len": len(result["thinking"]) if result["thinking"] else 0,
    }


def load_train_data(csv_path):
    """Load train.csv and add type classification."""
    data = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['type'] = classify_type(row['prompt'])
            data.append(row)
    return data


def load_existing_results(output_path):
    """Load already-completed results for resume."""
    done_ids = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["id"])
                results.append(r)
    return done_ids, results


def print_summary(results):
    """Print per-type and overall accuracy summary."""
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0, "truncated": 0})
    
    for r in results:
        t = r["type"]
        type_stats[t]["total"] += 1
        if r["correct"]:
            type_stats[t]["correct"] += 1
        if r.get("error"):
            type_stats[t]["errors"] += 1
        if r.get("finish_reason") == "length":
            type_stats[t]["truncated"] += 1
    
    total_correct = sum(s["correct"] for s in type_stats.values())
    total_count = sum(s["total"] for s in type_stats.values())
    
    print()
    print("=" * 65)
    print("  NEMOTRON BASE MODEL EVALUATION (via NVIDIA API)")
    print(f"  Official params: temp={TEMPERATURE}, max_tokens={MAX_TOKENS}, enable_thinking=True")
    print("=" * 65)
    print(f"  {'Type':<12} {'Correct':>8} {'Total':>6} {'Acc%':>7} {'Trunc':>6} {'Err':>5}")
    print("  " + "-" * 53)
    
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {t:<12} {s['correct']:>8} {s['total']:>6} {acc:>6.1f}% {s['truncated']:>6} {s['errors']:>5}")
    
    overall = total_correct / total_count * 100 if total_count > 0 else 0
    print("  " + "-" * 53)
    print(f"  {'OVERALL':<12} {total_correct:>8} {total_count:>6} {overall:>6.1f}%")
    print("=" * 65)
    
    # Show avg thinking/content length
    thinking_lens = [r["thinking_len"] for r in results if r.get("thinking_len", 0) > 0]
    content_lens = [r["content_len"] for r in results if r.get("content_len", 0) > 0]
    if thinking_lens:
        print(f"  Avg thinking length: {sum(thinking_lens)/len(thinking_lens):.0f} chars")
    if content_lens:
        print(f"  Avg content length:  {sum(content_lens)/len(content_lens):.0f} chars")
    
    # Show failure examples per type
    print("\n  Sample failures:")
    for t in sorted(type_stats.keys()):
        failures = [r for r in results if r["type"] == t and not r["correct"] and not r.get("error")]
        if failures:
            f = failures[0]
            print(f"    {t}: predicted='{f['predicted']}' gold='{f['gold']}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nemotron base model via NVIDIA API")
    parser.add_argument("--samples-per-type", type=int, default=50,
                        help="Samples per type (default: 50, total ~300)")
    parser.add_argument("--all", action="store_true",
                        help="Eval all 9500 samples")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    parser.add_argument("--output", default="competition_data/base_model_eval.jsonl")
    parser.add_argument("--data", default="competition_data/train.csv")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        return
    
    client = OpenAI(base_url=API_BASE, api_key=api_key)

    # Load data
    print(f"Loading data from {args.data}...")
    all_data = load_train_data(args.data)
    print(f"Total samples: {len(all_data)}")
    
    # Type distribution
    by_type = defaultdict(list)
    for item in all_data:
        by_type[item["type"]].append(item)
    
    for t in sorted(by_type.keys()):
        print(f"  {t}: {len(by_type[t])}")

    # Sample or use all
    if args.all:
        sampled = all_data
        print(f"\nUsing ALL {len(sampled)} samples")
    else:
        random.seed(args.seed)
        sampled = []
        for t in sorted(by_type.keys()):
            items = by_type[t]
            n = min(args.samples_per_type, len(items))
            chosen = random.sample(items, n)
            sampled.extend(chosen)
            print(f"  Sampled {t}: {n}/{len(items)}")
        print(f"\nTotal sampled: {len(sampled)}")

    # Resume support
    done_ids, existing_results = set(), []
    if args.resume:
        done_ids, existing_results = load_existing_results(args.output)
        print(f"Resuming: {len(done_ids)} already done")
    
    todo = [item for item in sampled if item["id"] not in done_ids]
    print(f"Remaining: {len(todo)} API calls")
    
    if not todo:
        print("All done! Showing results from previous run.")
        print_summary(existing_results)
        return

    # Execute
    results = list(existing_results)
    lock = Lock()
    done_count = [len(existing_results)]
    total = len(existing_results) + len(todo)
    correct_count = [sum(1 for r in existing_results if r["correct"])]
    start_time = time.time()
    
    # Open output file for incremental saving
    output_mode = "a" if args.resume else "w"
    outf = open(args.output, output_mode)
    
    def run_task(item):
        result = process_one(client, item, enable_thinking=True)
        with lock:
            done_count[0] += 1
            if result["correct"]:
                correct_count[0] += 1
            results.append(result)
            outf.write(json.dumps(result, ensure_ascii=False) + "\n")
            outf.flush()
            
            # Progress
            n = done_count[0]
            elapsed = time.time() - start_time
            rate = (n - len(existing_results)) / elapsed if elapsed > 0 else 0
            eta = (total - n) / rate / 60 if rate > 0 else 0
            running_acc = correct_count[0] / n * 100
            status = "✓" if result["correct"] else "✗"
            
            if n % 5 == 0 or n == total:
                print(f"  [{n}/{total}] {status} {result['type']:<10} "
                      f"pred='{result['predicted'][:30]}' gold='{result['gold'][:20]}' "
                      f"| running={running_acc:.1f}% | ETA={eta:.1f}min")
        return result

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_task, item): item for item in todo}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  THREAD ERROR: {e}")
    finally:
        outf.close()
    
    print(f"\nSaved {len(results)} results to {args.output}")
    print_summary(results)


if __name__ == "__main__":
    main()
