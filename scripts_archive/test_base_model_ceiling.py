"""
Test Nemotron base model ceiling: original vs stripped prompts.
Compares accuracy per type to understand the model's raw math ability.

Usage:
  python scripts/test_base_model_ceiling.py [--samples-per-type 30] [--mode both|original|stripped]
"""
import os
import re
import json
import math
import time
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

# --- Config ---
API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
MAX_TOKENS = 7680  # Match eval setting


# --- Answer extraction & comparison (mirrors official metric) ---
def extract_boxed(text):
    """Extract last non-empty \\boxed{...} match."""
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    return None


def answers_match(pred, gold):
    if pred is None:
        return False
    pred, gold = pred.strip(), gold.strip()
    if pred.lower() == gold.lower():
        return True
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, OverflowError):
        return False


# --- API call ---
def call_api(client, prompt, enable_thinking=True):
    """Call Nemotron API with evaluation-matching params."""
    messages = [{"role": "user", "content": prompt + METRIC_SUFFIX}]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,  # Match eval: greedy
            top_p=1.0,
            max_tokens=MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            timeout=180,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish = choice.finish_reason
        return content, finish
    except Exception as e:
        return None, str(e)


def process_one(client, item, prompt_key, enable_thinking=True):
    """Process one sample, return result dict."""
    prompt = item[prompt_key]
    gold = item["answer"]
    
    content, finish = call_api(client, prompt, enable_thinking)
    if content is None:
        return {
            "id": item["id"], "type": item["type"],
            "correct": False, "error": finish,
            "predicted": None, "gold": gold,
            "finish_reason": None, "prompt_key": prompt_key,
        }
    
    predicted = extract_boxed(content)
    correct = answers_match(predicted, gold)
    
    return {
        "id": item["id"], "type": item["type"],
        "correct": correct, "predicted": predicted, "gold": gold,
        "finish_reason": finish, "prompt_key": prompt_key,
        "content_len": len(content),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-type", type=int, default=30)
    parser.add_argument("--mode", choices=["both", "original", "stripped"], default="both")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--output", default="competition_data/base_model_ceiling.jsonl")
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        return
    client = OpenAI(base_url=API_BASE, api_key=api_key)

    # Load stripped prompts
    data = []
    with open("competition_data/stripped_prompts.jsonl") as f:
        for line in f:
            data.append(json.loads(line))
    
    # Sample per type
    random.seed(args.seed)
    by_type = {}
    for item in data:
        by_type.setdefault(item["type"], []).append(item)
    
    sampled = []
    for t, items in sorted(by_type.items()):
        chosen = random.sample(items, min(args.samples_per_type, len(items)))
        sampled.extend(chosen)
        print(f"  {t}: sampled {len(chosen)}/{len(items)}")
    
    print(f"\nTotal samples: {len(sampled)}")
    print(f"Mode: {args.mode}, enable_thinking: {args.enable_thinking}")
    print(f"Workers: {args.workers}")
    print()

    # Build task list
    tasks = []
    if args.mode in ("both", "original"):
        for item in sampled:
            tasks.append((item, "original_prompt"))
    if args.mode in ("both", "stripped"):
        for item in sampled:
            tasks.append((item, "stripped_prompt"))
    
    print(f"Total API calls: {len(tasks)}")

    # Execute with thread pool
    results = []
    lock = Lock()
    done_count = [0]
    
    def run_task(task):
        item, prompt_key = task
        result = process_one(client, item, prompt_key, args.enable_thinking)
        with lock:
            done_count[0] += 1
            status = "✓" if result["correct"] else "✗"
            if done_count[0] % 10 == 0 or done_count[0] == len(tasks):
                print(f"  [{done_count[0]}/{len(tasks)}] {result['type']} ({prompt_key.split('_')[0]}) {status}")
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  ERROR: {e}")

    # Save raw results
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} results to {args.output}")

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("BASE MODEL CEILING ANALYSIS")
    print("=" * 70)

    for prompt_key in (["original_prompt", "stripped_prompt"] if args.mode == "both" 
                       else [("original_prompt" if args.mode == "original" else "stripped_prompt")]):
        subset = [r for r in results if r["prompt_key"] == prompt_key]
        if not subset:
            continue
        
        label = "ORIGINAL (with narrative)" if "original" in prompt_key else "STRIPPED (clean math)"
        print(f"\n--- {label} ---")
        
        # Per-type accuracy
        type_stats = {}
        for r in subset:
            t = r["type"]
            type_stats.setdefault(t, {"correct": 0, "total": 0, "errors": 0, "truncated": 0})
            type_stats[t]["total"] += 1
            if r["correct"]:
                type_stats[t]["correct"] += 1
            if r.get("error"):
                type_stats[t]["errors"] += 1
            if r.get("finish_reason") == "length":
                type_stats[t]["truncated"] += 1
        
        total_correct = sum(s["correct"] for s in type_stats.values())
        total_count = sum(s["total"] for s in type_stats.values())
        
        print(f"{'Type':<12} {'Correct':>8} {'Total':>6} {'Acc%':>7} {'Trunc':>6} {'Err':>5}")
        print("-" * 50)
        for t in sorted(type_stats.keys()):
            s = type_stats[t]
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"{t:<12} {s['correct']:>8} {s['total']:>6} {acc:>6.1f}% {s['truncated']:>6} {s['errors']:>5}")
        
        overall = total_correct / total_count * 100 if total_count > 0 else 0
        print("-" * 50)
        print(f"{'OVERALL':<12} {total_correct:>8} {total_count:>6} {overall:>6.1f}%")

    # Comparison if both modes
    if args.mode == "both":
        print(f"\n--- COMPARISON: Original vs Stripped ---")
        orig = {r["id"]: r["correct"] for r in results if r["prompt_key"] == "original_prompt"}
        strip = {r["id"]: r["correct"] for r in results if r["prompt_key"] == "stripped_prompt"}
        
        common_ids = set(orig.keys()) & set(strip.keys())
        both_correct = sum(1 for i in common_ids if orig[i] and strip[i])
        only_orig = sum(1 for i in common_ids if orig[i] and not strip[i])
        only_strip = sum(1 for i in common_ids if not orig[i] and strip[i])
        both_wrong = sum(1 for i in common_ids if not orig[i] and not strip[i])
        
        print(f"  Both correct:    {both_correct:>4} ({both_correct/len(common_ids)*100:.1f}%)")
        print(f"  Only original:   {only_orig:>4} ({only_orig/len(common_ids)*100:.1f}%)")
        print(f"  Only stripped:   {only_strip:>4} ({only_strip/len(common_ids)*100:.1f}%)")
        print(f"  Both wrong:      {both_wrong:>4} ({both_wrong/len(common_ids)*100:.1f}%)")

        # Per-type comparison
        print(f"\n  Per-type delta (stripped - original):")
        for t in sorted(set(r["type"] for r in results)):
            t_orig = [r for r in results if r["type"] == t and r["prompt_key"] == "original_prompt"]
            t_strip = [r for r in results if r["type"] == t and r["prompt_key"] == "stripped_prompt"]
            acc_o = sum(r["correct"] for r in t_orig) / len(t_orig) * 100 if t_orig else 0
            acc_s = sum(r["correct"] for r in t_strip) / len(t_strip) * 100 if t_strip else 0
            delta = acc_s - acc_o
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"    {t:<12} {acc_o:>5.1f}% → {acc_s:>5.1f}% ({arrow}{abs(delta):.1f}%)")


if __name__ == "__main__":
    main()
