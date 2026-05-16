#!/usr/bin/env python3
"""
Symbol Rule Solver — 由果推因，用 LLM 为 1326 条无规则 symbol 题生成推理过程。

策略:
1. 先用 NVIDIA Nemotron API (enable_thinking) 尝试，给出答案做验证
2. 失败的用 DeepSeek-V3 重试
3. 输出结果到 data/symbol_solved.jsonl（增量保存，可断点续传）

Usage:
  python3 scripts/solve_symbol_rules.py                # 全量 1326 条
  python3 scripts/solve_symbol_rules.py --n 20         # 测试 20 条
  python3 scripts/solve_symbol_rules.py --resume       # 断点续传
  python3 scripts/solve_symbol_rules.py --deepseek-only # 只用 DeepSeek
"""
import os
import re
import csv
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
OUTPUT_FILE = DATA_DIR / 'symbol_solved.jsonl'
CHECKPOINT_INTERVAL = 50

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

MAX_TOKENS = 4096

# ═══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, rps=8):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.last = 0.0
    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = self.last + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()

nvidia_limiter = RateLimiter(rps=8)
deepseek_limiter = RateLimiter(rps=20)

# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expert puzzle solver. Given a symbol transformation puzzle with examples and the correct answer, your task is to deduce the transformation rules and explain your reasoning.

RULES FOR YOUR RESPONSE:
1. Analyze the examples carefully to find the pattern/rule.
2. Show your step-by-step reasoning of how you deduced the rule.
3. Apply the rule to the query to verify it produces the correct answer.
4. Be concise: focus on the key insight and verification.
5. Use plain ASCII only. No LaTeX.
6. Do NOT wrap anything in \\boxed{}.
7. Output ONLY the reasoning process. Nothing else."""

def build_user_prompt(prompt, answer):
    return f"""Here is a symbol transformation puzzle. The correct answer is given. Deduce the transformation rules from the examples and explain your reasoning.

PUZZLE:
{prompt}

CORRECT ANSWER: {answer}

Please explain step-by-step how the transformation rules work and verify they produce the correct answer."""

# ═══════════════════════════════════════════════════════════════════════════════
#  ANSWER MATCHING
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_answer(s):
    """Normalize answer string for comparison."""
    return str(s).strip()

def check_answer_in_response(response, gold):
    """Check if the response mentions/produces the correct answer."""
    gold_norm = normalize_answer(gold)
    if gold_norm in response:
        return True
    # Also check with different whitespace
    if gold_norm.replace(' ', '') in response.replace(' ', ''):
        return True
    return False

# ═══════════════════════════════════════════════════════════════════════════════
#  API CALLERS
# ═══════════════════════════════════════════════════════════════════════════════
def call_nvidia(client, prompt, answer, temperature=0.3):
    """Call NVIDIA Nemotron API. Returns (thinking, content, success)."""
    nvidia_limiter.acquire()
    try:
        resp = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(prompt, answer)},
            ],
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        thinking = getattr(choice.message, 'reasoning_content', '') or ""
        
        # Combine thinking + content as the full reasoning
        full = (thinking + "\n" + content).strip() if thinking else content.strip()
        
        # Check if response references the correct answer
        ok = check_answer_in_response(full, answer)
        return thinking.strip(), content.strip(), ok
    except Exception as e:
        print(f"  NVIDIA error: {e}")
        return "", f"ERROR: {e}", False

def call_deepseek(client, prompt, answer, temperature=0.3):
    """Call DeepSeek API. Returns (thinking, content, success)."""
    deepseek_limiter.acquire()
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(prompt, answer)},
            ],
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )
        content = resp.choices[0].message.content or ""
        ok = check_answer_in_response(content, answer)
        return "", content.strip(), ok
    except Exception as e:
        print(f"  DeepSeek error: {e}")
        return "", f"ERROR: {e}", False

# ═══════════════════════════════════════════════════════════════════════════════
#  SOLVE ONE PROBLEM
# ═══════════════════════════════════════════════════════════════════════════════
def solve_one(row, nvidia_client, deepseek_client, deepseek_only=False):
    """Try to solve a symbol problem. Returns result dict."""
    pid = row['id']
    prompt = row['prompt']
    answer = str(row['answer'])
    
    result = {
        'id': pid,
        'type': 'symbol',
        'prompt': prompt,
        'answer': answer,
        'thinking': '',
        'content': '',
        'source': '',
        'solved': False,
    }
    
    # --- Step 1: Try NVIDIA Nemotron first (fastest) ---
    if not deepseek_only:
        for temp in [0.3, 0.7]:
            thinking, content, ok = call_nvidia(nvidia_client, prompt, answer, temp)
            # NVIDIA often returns reasoning in thinking field with empty content
            full_len = len(thinking) + len(content)
            if ok and full_len > 20:
                result['thinking'] = thinking
                result['content'] = content if content else thinking
                result['source'] = f'nvidia_t{temp}'
                result['solved'] = True
                return result
    
    # --- Step 2: Fallback to DeepSeek ---
    for temp in [0.3, 0.7]:
        thinking, content, ok = call_deepseek(deepseek_client, prompt, answer, temp)
        full_len = len(thinking) + len(content)
        if ok and full_len > 20:
            result['thinking'] = thinking
            result['content'] = content
            result['source'] = f'deepseek_t{temp}'
            result['solved'] = True
            return result
    
    # --- Failed ---
    result['source'] = 'failed'
    return result

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None, help='Limit to N problems')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--deepseek-only', action='store_true', help='Skip NVIDIA, use DeepSeek only')
    parser.add_argument('--workers', type=int, default=30, help='Parallel workers')
    args = parser.parse_args()
    
    # Load data
    dsl = []
    with open(DATA_DIR / 'train_dsl_rules.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == 'symbol' and (not row['dsl_rules'] or row['dsl_rules'] == ''):
                dsl.append(row)
    
    print(f"Loaded {len(dsl)} unsolved symbol problems")
    
    # Resume: skip already solved IDs
    done_ids = set()
    if args.resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                d = json.loads(line)
                done_ids.add(d['id'])
        print(f"Resuming: {len(done_ids)} already done")
    
    remaining = [r for r in dsl if r['id'] not in done_ids]
    if args.n:
        remaining = remaining[:args.n]
    
    print(f"Processing {len(remaining)} problems ({args.workers} workers)")
    
    # Init clients with generous timeouts
    nvidia_client = OpenAI(
        base_url=NVIDIA_API_BASE,
        api_key=os.environ.get('NVIDIA_API_KEY', ''),
        timeout=120.0,
    )
    deepseek_client = OpenAI(
        base_url=DEEPSEEK_API_BASE,
        api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
        timeout=30.0,  # DeepSeek is slow; short timeout, skip if too slow
    )
    
    # Process
    results = []
    solved = 0
    failed = 0
    source_counts = {}
    lock = threading.Lock()
    
    def process(row):
        return solve_one(row, nvidia_client, deepseek_client, args.deepseek_only)
    
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, r): r for r in remaining}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            
            with lock:
                results.append(result)
                
                if result['solved']:
                    solved += 1
                else:
                    failed += 1
                
                src = result['source']
                source_counts[src] = source_counts.get(src, 0) + 1
                
                # Progress
                total_done = solved + failed
                elapsed = time.time() - t0
                rate = total_done / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - total_done) / rate / 60 if rate > 0 else 0
                
                if total_done % 5 == 0 or total_done <= 10 or total_done == len(remaining):
                    print(f"[{total_done}/{len(remaining)}] solved={solved} failed={failed} "
                          f"rate={rate:.1f}/s ETA={eta:.1f}min | {result['source']}")
                
                # Checkpoint
                if total_done % CHECKPOINT_INTERVAL == 0:
                    _save(results, done_ids)
                    print(f"  💾 Checkpoint saved ({len(results)} new results)")
    
    # Final save
    _save(results, done_ids)
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"Total: {len(remaining)}, Solved: {solved} ({solved/len(remaining)*100:.1f}%), Failed: {failed}")
    print(f"Sources: {source_counts}")
    print(f"Output: {OUTPUT_FILE}")

def _save(results, done_ids):
    """Append new results to output file."""
    mode = 'a' if done_ids else 'w'
    # If not resuming, write all; if resuming, append only new
    with open(OUTPUT_FILE, 'a') as f:
        for r in results:
            if r['id'] not in done_ids:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                done_ids.add(r['id'])
    results.clear()

if __name__ == '__main__':
    main()
