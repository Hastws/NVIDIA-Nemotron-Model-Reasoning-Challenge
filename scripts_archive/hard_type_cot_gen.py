"""
Hard-Type CoT Generator — Aggressive multi-temperature rejection sampling
Targets: bit_ops (need ~130), cipher (need ~140), symbol (need ~80)
Uses NVIDIA Nemotron API with enable_thinking=True
"""
import os
import re
import csv
import json
import math
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

# --- Config ---
API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# Temperature ladder: start low (deterministic), escalate for diversity
TEMP_LADDER = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SAMPLES_PER_TEMP = 3  # try 3 times at each temperature
MAX_TOKENS = 4096

# CoT prefix for enable_thinking=False mode
COT_PREFIX = "Think step by step, then give your final answer.\n\n"

# Hard types that need more data
HARD_TYPES = {"bit_ops", "cipher", "symbol"}

# --- Puzzle type detection ---
def detect_type(prompt):
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p):
        return "bit_ops"
    elif "encrypt" in p or "cipher" in p:
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

# --- Answer extraction & comparison ---
def extract_boxed(text):
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
def call_api(client, prompt, temperature, max_tokens=MAX_TOKENS):
    """Call API with enable_thinking=False + CoT prefix.
    
    enable_thinking=True causes reasoning to consume all tokens on hard puzzles
    (finish_reason=length, no answer produced). Using False + prompt-based CoT
    keeps reasoning in content where we can extract it.
    """
    messages = [{"role": "user", "content": COT_PREFIX + prompt + METRIC_SUFFIX}]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=120,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish = choice.finish_reason
        
        if not content.strip():
            return None, "empty_content"
        
        # Wrap content in <think> tags for training format
        # Extract the reasoning (everything before \boxed) and the answer part
        boxed_match = re.search(r'(\\boxed\{[^}]*\})', content)
        if boxed_match:
            # Keep full content as reasoning, answer is in boxed
            completion = f"<think>\n{content}\n</think>\n\n{boxed_match.group(1)}"
        else:
            # No boxed answer found
            completion = f"<think>\n{content}\n</think>\n\n"
        
        return completion, finish
    except Exception as e:
        return None, str(e)

# --- Process one sample across all temperatures ---
def process_sample(client, row, existing_ids, stats, lock):
    sample_id = row["id"]
    prompt = row["prompt"]
    gold = row["answer"]
    ptype = detect_type(prompt)
    
    if sample_id in existing_ids:
        return None
    
    for temp in TEMP_LADDER:
        for attempt in range(SAMPLES_PER_TEMP):
            completion, finish = call_api(client, prompt, temp)
            
            if completion is None:
                if "Connection error" in str(finish) or "timeout" in str(finish).lower():
                    time.sleep(5)  # Longer delay on connection errors
                else:
                    time.sleep(1)
                continue
            
            predicted = extract_boxed(completion)
            
            if answers_match(predicted, gold):
                # Extract reasoning length
                think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
                reasoning_len = len(think_match.group(1).strip()) if think_match else 0
                
                result = {
                    "id": sample_id,
                    "prompt": prompt,
                    "gold_answer": gold,
                    "completion": completion,
                    "source": "hard_type_sampling",
                    "puzzle_type": ptype,
                    "reasoning_len": reasoning_len,
                    "temperature_used": temp,
                    "attempt": attempt + 1,
                    "temp_stage": TEMP_LADDER.index(temp) + 1,
                    "api_correct": True,
                }
                
                with lock:
                    stats["solved"] += 1
                    stats["by_type"][ptype] = stats["by_type"].get(ptype, 0) + 1
                    stats["total_attempts"] += (TEMP_LADDER.index(temp) * SAMPLES_PER_TEMP + attempt + 1)
                
                return result
            
            # Small delay between attempts
            time.sleep(0.3)
    
    # All attempts failed
    with lock:
        stats["failed"] += 1
        stats["failed_ids"].append(sample_id)
        stats["total_attempts"] += len(TEMP_LADDER) * SAMPLES_PER_TEMP
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Hard-type CoT generator")
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--output", default="data/hard_type_cot.jsonl")
    parser.add_argument("--progress", default="data/hard_type_progress.json")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Max samples per type (0=all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--types", nargs="+", default=["bit_ops", "cipher", "symbol"],
                        help="Which types to sample")
    args = parser.parse_args()
    
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY or NVIDIA_NIM_API_KEY environment variable")
        return
    
    client = OpenAI(base_url=API_BASE, api_key=api_key)
    
    # Load existing results for resume
    existing_ids = set()
    existing_results = []
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                existing_ids.add(d["id"])
                existing_results.append(d)
        print(f"Resuming: {len(existing_ids)} samples already done")
    
    # Also exclude IDs already in hybrid_cot (api_correct) and multi_round
    for fpath in ["data/hybrid_cot_data.jsonl", "data/multi_round_correct.jsonl"]:
        if Path(fpath).exists():
            with open(fpath) as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("api_correct"):
                        existing_ids.add(d["id"])
    print(f"Excluding {len(existing_ids)} already-solved samples")
    
    # Load train data, filter to hard types only
    target_types = set(args.types) & HARD_TYPES
    rows = []
    with open(args.train_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ptype = detect_type(row["prompt"])
            if ptype in target_types and row["id"] not in existing_ids:
                row["_type"] = ptype
                rows.append(row)
    
    # Optionally limit per type
    if args.limit > 0:
        limited = []
        type_counts = {}
        for r in rows:
            t = r["_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
            if type_counts[t] <= args.limit:
                limited.append(r)
        rows = limited
    
    # Show plan
    from collections import Counter
    type_dist = Counter(r["_type"] for r in rows)
    print(f"\n=== Sampling Plan ===")
    print(f"Target types: {target_types}")
    print(f"Samples to process: {len(rows)}")
    for t in sorted(type_dist):
        print(f"  {t}: {type_dist[t]} unsolved")
    print(f"Temp ladder: {TEMP_LADDER}")
    print(f"Max attempts per sample: {len(TEMP_LADDER) * SAMPLES_PER_TEMP}")
    print(f"Workers: {args.workers}")
    print()
    
    # Process
    stats = {
        "solved": 0, "failed": 0, "total_attempts": 0,
        "by_type": {}, "failed_ids": [],
        "start_time": time.time(),
    }
    lock = Lock()
    results = list(existing_results)
    
    save_every = 5
    processed = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_sample, client, row, existing_ids, stats, lock): row
            for row in rows
        }
        
        for future in as_completed(futures):
            processed += 1
            result = future.result()
            
            if result:
                results.append(result)
                ptype = result["puzzle_type"]
                temp = result["temperature_used"]
                rlen = result["reasoning_len"]
                print(f"  ✓ [{processed}/{len(rows)}] {ptype} solved @ temp={temp}, reasoning={rlen} chars")
            
            # Periodic save
            if processed % save_every == 0:
                elapsed = time.time() - stats["start_time"]
                print(f"\n--- Progress: {processed}/{len(rows)} | "
                      f"Solved: {stats['solved']} | Failed: {stats['failed']} | "
                      f"Time: {elapsed:.0f}s ---")
                for t in sorted(stats["by_type"]):
                    print(f"  {t}: +{stats['by_type'][t]} new")
                print()
                
                # Save results
                with open(args.output, "w") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
                # Save progress
                progress = {**stats, "processed": processed, "total": len(rows),
                            "elapsed_s": elapsed}
                with open(args.progress, "w") as f:
                    json.dump(progress, f, indent=2, ensure_ascii=False)
    
    # Final save
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    elapsed = time.time() - stats["start_time"]
    print(f"\n=== DONE ===")
    print(f"Processed: {processed}")
    print(f"Solved: {stats['solved']}")
    print(f"Failed: {stats['failed']}")
    print(f"Time: {elapsed:.0f}s")
    print(f"Results saved to: {args.output}")
    print(f"\nBy type:")
    for t in sorted(stats["by_type"]):
        print(f"  {t}: +{stats['by_type'][t]} new correct CoT samples")


if __name__ == "__main__":
    main()
