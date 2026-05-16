#!/usr/bin/env python3
"""
Rewrite solution processes using NVIDIA API.
Reads train_annotated.csv, uses LLM to rewrite compact solution_process
into natural step-by-step reasoning, adds as 'rewritten_solution' column.

Usage:
  python3 scripts/rewrite_solutions.py [--n N] [--model MODEL] [--workers W]
"""
import os
import csv
import json
import time
import argparse
import threading
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

# Rate limiter: 40 requests/min = 0.667 rps
class RateLimiter:
    def __init__(self, rpm=38):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.last = 0

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = self.last + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()

rate_limiter = RateLimiter(rpm=38)

# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPTS — per-type rewrite instructions
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are rewriting a compact machine-generated solution into a clear, natural chain-of-thought reasoning trace. This will be used as training data for a language model.

CRITICAL RULES:
1. The rewritten solution MUST arrive at EXACTLY the same final answer — do not change it.
2. Write as if you are thinking through the problem step-by-step, naturally and concisely.
3. Use math notation where helpful (e.g., formulas, equations).
4. Keep it concise — aim for 3-8 sentences. No fluff, no meta-commentary.
5. End with the answer in \\boxed{} format: \\boxed{ANSWER}
6. Output ONLY the reasoning trace + boxed answer. Nothing else — no preamble like "Here is..." or "Sure..."."""

TYPE_HINTS = {
    "gravity": "This is a physics puzzle where a hidden gravitational constant g governs the formula d = 0.5 * g * t². The solver computed g from examples and applied it.",
    "unit_conv": "This is a unit conversion puzzle with a hidden linear conversion factor f where output = f * input. The solver computed f from examples and applied it.",
    "numeral": "This is a numeral system conversion puzzle (e.g., Arabic to Roman, base conversion). The solver identified the target system and converted.",
    "cipher": "This is a substitution cipher decryption puzzle. The solver built a character mapping from examples and decoded the target text.",
    "bit_ops": "This is a bitwise transformation puzzle on 8-bit binary numbers. The solver identified per-bit rules (XOR, NOT, AND, etc.) from input→output examples.",
    "symbol": "This is a symbolic equation puzzle where a custom operator is defined by examples. The solver identified the operation (concatenation, addition, charwise operation, etc.) and applied it.",
}

def build_user_message(row):
    """Build the user message for a single row."""
    ptype = row['type']
    hint = TYPE_HINTS.get(ptype, "")
    
    # Truncate prompt to key info (first 500 chars + last 100)
    prompt = row['prompt']
    if len(prompt) > 700:
        prompt_short = prompt[:500] + "\n...\n" + prompt[-200:]
    else:
        prompt_short = prompt
    
    return f"""Problem type: {ptype}
{hint}

Problem (abbreviated):
{prompt_short}

Machine-generated solution: {row['solution_process']}
Correct answer: {row['answer']}

Rewrite the solution into a natural step-by-step reasoning trace that arrives at \\boxed{{{row['answer']}}}."""


def call_api(api_key, model, row, max_retries=3):
    """Call NVIDIA API to rewrite a single solution."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    user_msg = build_user_message(row)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.4,
        "max_tokens": 512,
    }
    
    for attempt in range(max_retries):
        rate_limiter.acquire()
        try:
            resp = requests.post(API_BASE, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content']
                tokens = data.get('usage', {}).get('completion_tokens', 0)
                return content, tokens
            elif resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  Timeout, retry {attempt+1}/{max_retries}")
            time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}, retry {attempt+1}/{max_retries}")
            time.sleep(2)
    
    return None, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='Number of rows to process (0=all matched)')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--workers', type=int, default=1, help='Concurrent workers')
    parser.add_argument('--output', default='train_annotated_rewritten.csv')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output')
    args = parser.parse_args()
    
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        return
    
    # Load data
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'train_annotated.csv')
    with open(input_path) as f:
        all_rows = list(csv.DictReader(f))
    
    print(f"Loaded {len(all_rows)} rows from {input_path}")
    
    # Filter to matched rows that have solution_process
    to_rewrite = [r for r in all_rows if r['match'] == 'True' and r.get('solution_process')]
    print(f"Rows with match=True and solution_process: {len(to_rewrite)}")
    
    # Load existing progress if resuming
    done_ids = set()
    done_results = {}
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', args.output)
    
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            for row in csv.DictReader(f):
                if row.get('rewritten_solution'):
                    done_ids.add(row['id'])
                    done_results[row['id']] = row['rewritten_solution']
        print(f"Resuming: {len(done_ids)} already done")
    
    # Filter out already done
    pending = [r for r in to_rewrite if r['id'] not in done_ids]
    
    if args.n > 0:
        pending = pending[:args.n]
    
    print(f"To process: {len(pending)} rows")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_path}")
    print()
    
    # Process
    results = dict(done_results)  # id -> rewritten_solution
    success = len(done_ids)
    fail = 0
    total_tokens = 0
    type_counts = Counter()
    t_start = time.time()
    
    def process_one(row):
        content, tokens = call_api(api_key, args.model, row)
        return row['id'], row['type'], content, tokens
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, row): row for row in pending}
        
        for i, future in enumerate(as_completed(futures)):
            rid, rtype, content, tokens = future.result()
            
            if content:
                results[rid] = content
                success += 1
                total_tokens += tokens
                type_counts[rtype] += 1
            else:
                fail += 1
                results[rid] = ""  # empty on failure
            
            # Progress
            elapsed = time.time() - t_start
            done_now = i + 1
            rpm = done_now / elapsed * 60 if elapsed > 0 else 0
            if done_now % 10 == 0 or done_now == len(pending):
                print(f"  [{done_now}/{len(pending)}] success={success} fail={fail} "
                      f"rpm={rpm:.1f} tokens={total_tokens} "
                      f"types={dict(type_counts)}")
            
            # Save checkpoint every 100
            if done_now % 100 == 0:
                save_output(all_rows, results, output_path)
                print(f"  ** Checkpoint saved ({success} rows)")
    
    # Final save
    save_output(all_rows, results, output_path)
    
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"Success: {success}, Failed: {fail}")
    print(f"Total tokens: {total_tokens}")
    print(f"Output: {output_path}")
    print(f"Types: {dict(type_counts)}")


def save_output(all_rows, results, output_path):
    """Save all rows with rewritten_solution column."""
    fieldnames = ['id', 'prompt', 'answer', 'type', 'solvable', 'solver_answer', 
                  'solution_process', 'match', 'fail_reason', 'rewritten_solution']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            out = dict(row)
            out['rewritten_solution'] = results.get(row['id'], '')
            writer.writerow(out)


if __name__ == '__main__':
    main()
