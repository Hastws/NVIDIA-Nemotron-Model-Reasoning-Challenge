#!/usr/bin/env python3
"""
固定格式蒸馏验证 — 用强模型按固定模板解题，验证覆盖率和答案质量

用法:
  export NVIDIA_API_KEY=nvapi-xxx
  python3 scripts/distill_fixed_format.py --n 100
  python3 scripts/distill_fixed_format.py --n 9500 --output data/distill_fixed.jsonl
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

rate_limiter = RateLimiter(rps=0.5)  # 1 request per 2 seconds

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
API_BASE = "https://integrate.api.nvidia.com/v1"

# 强模型候选 — 按优先级
STRONG_MODELS = [
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "google/gemma-3-27b-it",
    "meta/llama-3.1-405b-instruct",
]

METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TRAIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "competition_data", "train.csv")

# ═══════════════════════════════════════════════════════════════════════════════
#  FIXED-FORMAT SYSTEM PROMPTS — 核心：让强模型按极简模板输出
# ═══════════════════════════════════════════════════════════════════════════════
FIXED_FORMAT_PROMPTS = {
    "gravity": """You are solving a physics puzzle. The secret gravitational constant g is hidden.

STRATEGY:
1. Pick any example: d = 0.5 * g * t²  →  g = 2*d / t²
2. Compute g from that example
3. Apply: d = 0.5 * g * target_t²

OUTPUT FORMAT (use EXACTLY this format in your thinking):
g = 2 * {d} / {t}^2 = {g_value}
d = 0.5 * {g_value} * {target_t}^2 = {answer}

Keep your reasoning to 3 lines max. Then put the final numeric answer in \\boxed{}.""",

    "unit_conv": """You are solving a unit conversion puzzle. A secret conversion factor f is used.

STRATEGY:
1. Pick any example: output = f * input  →  f = output / input
2. Compute f
3. Apply: answer = f * target_input

OUTPUT FORMAT (use EXACTLY this format in your thinking):
f = {output} / {input} = {f_value}
answer = {f_value} * {target} = {answer}

Keep your reasoning to 3 lines max. Then put the final numeric answer in \\boxed{}.""",

    "numeral": """You are solving a numeral system conversion puzzle.

STRATEGY:
1. Look at the example input→output pairs to identify the target numeral system (Roman, binary, hex, custom base, etc.)
2. Convert the target number to that system

OUTPUT FORMAT (use EXACTLY this format in your thinking):
System: {identified_system}
{number} in {system} = {result}

Keep your reasoning to 3 lines max. Then put the final answer in \\boxed{}.""",

    "cipher": """You are decrypting text using a substitution cipher.

STRATEGY:
1. Align characters from ciphertext→plaintext in the examples to build a letter mapping
2. Apply the mapping to decrypt the target ciphertext
3. Work letter by letter, space by space

OUTPUT FORMAT (use EXACTLY this format in your thinking):
Mapping: {cipher_letter}→{plain_letter} for each letter found in examples
Apply mapping to target: {result}

Be systematic. Build the FULL 26-letter mapping if possible. Then put the decrypted text in \\boxed{}.""",

    "bit_ops": """You are solving a bit manipulation puzzle on 8-bit binary numbers.

STRATEGY:
1. For each bit position (0-7), examine how that bit changes across ALL examples
2. Determine the rule for each output bit as a function of input bits (XOR, AND, OR, NOT, shift, rotate, etc.)
3. Apply the rules to the target input

OUTPUT FORMAT (use EXACTLY this format in your thinking):
For each output bit position, state: out[i] = function(in[...])
Then apply to target: {result}

Be extremely systematic. Check your rule against ALL examples before applying. Then put the 8-bit result in \\boxed{}.""",

    "symbol": """You are solving a symbol/equation transformation puzzle.

STRATEGY:
1. Analyze examples to find the transformation rule (character substitution, position swap, mathematical operation on char codes, etc.)
2. Try character-by-character mapping first
3. If that doesn't work, try position-based rules or other patterns

OUTPUT FORMAT (use EXACTLY this format in your thinking):
Rule: {describe the transformation}
Apply to target: {step by step} = {result}

Be systematic. Verify your rule against ALL examples. Then put the final answer in \\boxed{}.""",

    "unknown": """You are solving a pattern-finding puzzle. Analyze the examples carefully to find the hidden rule, then apply it to the target.

Keep your reasoning concise and systematic. Put the final answer in \\boxed{}.""",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def detect_type(prompt):
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p):
        return "bit_ops"
    elif "encrypt" in p or "decrypt" in p or "cipher" in p or "secret language" in p:
        return "cipher"
    elif "gravit" in p or "free fall" in p or "free-fall" in p:
        return "gravity"
    elif "numeral" in p or "wonderland number" in p:
        return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and ("measurement" in p or "meter" in p or " m " in p)):
        return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p or "symbol" in p):
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


# ═══════════════════════════════════════════════════════════════════════════════
#  API CALL
# ═══════════════════════════════════════════════════════════════════════════════
def call_model(client, model, system_prompt, user_prompt, temperature=0.3, max_tokens=2048):
    """Call a model with rate limiting and retry."""
    for attempt in range(5):
        time.sleep(5)  # Hard delay between calls
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + METRIC_SUFFIX},
                ],
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
                timeout=120,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            return {
                "content": content,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                },
            }
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if "timeout" in err.lower() or "timed out" in err.lower():
                print(f"    Timeout, retrying...")
                time.sleep(2)
                continue
            if attempt == 2:
                print(f"    ERROR after 3 attempts: {err[:100]}")
                return {"content": f"ERROR: {err}", "finish_reason": "error", "usage": {}}
            time.sleep(1)
    return {"content": "ERROR: max retries", "finish_reason": "error", "usage": {}}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def process_question(client, model, qid, prompt, gold, qtype):
    """Process one question: call model, extract answer, check correctness."""
    system_prompt = FIXED_FORMAT_PROMPTS.get(qtype, FIXED_FORMAT_PROMPTS["unknown"])
    
    result = call_model(client, model, system_prompt, prompt, temperature=0.3, max_tokens=2048)
    content = result["content"]
    
    # Extract answer
    pred = extract_boxed(content)
    correct = answers_match(pred, gold) if pred else False
    
    return {
        "id": qid,
        "type": qtype,
        "gt_answer": gold,
        "predicted": pred,
        "correct": correct,
        "content": content,
        "finish_reason": result["finish_reason"],
        "usage": result["usage"],
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of questions to test")
    parser.add_argument("--model", type=str, default=None, help="Model to use (default: auto-select)")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        return

    client = OpenAI(base_url=API_BASE, api_key=api_key)

    # Select model
    model = args.model
    if not model:
        # Test each strong model
        for m in STRONG_MODELS:
            print(f"Testing model: {m}...")
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": "Say hello"}],
                    max_tokens=10,
                )
                print(f"  ✓ {m} available")
                model = m
                break
            except Exception as e:
                print(f"  ✗ {m}: {e}")
        if not model:
            print("ERROR: No strong model available")
            return

    print(f"\nUsing model: {model}")

    # Load questions
    questions = []
    with open(TRAIN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qtype = detect_type(row["prompt"])
            questions.append({
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "type": qtype,
            })

    print(f"Loaded {len(questions)} questions")
    print(f"Type distribution: {Counter(q['type'] for q in questions)}")

    # Sample N questions (balanced across types)
    if args.n < len(questions):
        types = sorted(set(q["type"] for q in questions))
        per_type = max(1, args.n // len(types))
        sampled = []
        by_type = defaultdict(list)
        for q in questions:
            by_type[q["type"]].append(q)
        for t in types:
            avail = by_type[t]
            # deterministic sample
            step = max(1, len(avail) // per_type)
            sampled.extend(avail[::step][:per_type])
        questions = sampled[:args.n]
        print(f"Sampled {len(questions)} questions: {Counter(q['type'] for q in questions)}")

    # Process
    results = []
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)
    content_lens = []
    
    lock = threading.Lock()
    done = [0]

    def do_one(q):
        r = process_question(client, model, q["id"], q["prompt"], q["answer"], q["type"])
        with lock:
            done[0] += 1
            total_by_type[r["type"]] = total_by_type.get(r["type"], 0) + 1
            if r["correct"]:
                correct_by_type[r["type"]] = correct_by_type.get(r["type"], 0) + 1
            c = sum(correct_by_type.values())
            t = done[0]
            mark = "✓" if r["correct"] else "✗"
            print(f"  [{done[0]:3d}/{len(questions)}] {mark} {r['type']:12s} pred={repr(r['predicted'])[:30]:30s} gold={repr(r['gt_answer'])[:20]} acc={c}/{t} ({c/t*100:.1f}%)", flush=True)
        return r

    print(f"\nProcessing {len(questions)} questions with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(do_one, q): q for q in questions}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            content_lens.append(len(r["content"]))

    # Results
    print("\n" + "=" * 60)
    print(f"MODEL: {model}")
    print(f"TOTAL: {sum(correct_by_type.values())}/{len(results)} ({sum(correct_by_type.values())/len(results)*100:.1f}%)")
    print("=" * 60)
    print(f"{'Type':12s} {'Total':>6s} {'Correct':>8s} {'Acc%':>7s} {'AvgLen':>8s}")
    for t in sorted(total_by_type.keys()):
        tot = total_by_type[t]
        corr = correct_by_type[t]
        type_lens = [len(r["content"]) for r in results if r["type"] == t]
        avg_len = sum(type_lens) / len(type_lens) if type_lens else 0
        print(f"{t:12s} {tot:6d} {corr:8d} {corr/tot*100:6.1f}% {avg_len:8.0f}")

    # Show some failures
    print("\n--- Sample failures ---")
    for t in sorted(total_by_type.keys()):
        failures = [r for r in results if r["type"] == t and not r["correct"]]
        if failures:
            f = failures[0]
            print(f"\n[{t}] pred={repr(f['predicted'])} gold={repr(f['gt_answer'])}")
            print(f"  Content (first 300): {f['content'][:300]}")

    # Save
    if args.output:
        out_path = os.path.join(DATA_DIR, args.output) if not args.output.startswith("/") else args.output
        with open(out_path, "w") as fp:
            for r in results:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} results to {out_path}")

    # Also save as CSV for quick review
    csv_path = os.path.join(DATA_DIR, "distill_fixed_test.csv")
    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "type", "gt_answer", "predicted", "correct", "content_len"])
        for r in results:
            writer.writerow([r["id"], r["type"], r["gt_answer"], r["predicted"], r["correct"], len(r["content"])])
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
