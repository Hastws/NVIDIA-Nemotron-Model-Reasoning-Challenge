#!/usr/bin/env python3
"""
Generate DSL rules for all solvable training examples using DeepSeek API.
- 4 generations per row
- Evaluate and pick best
- Output final dataset

Usage:
  python3 scripts/generate_dsl_rules.py [--n N] [--workers W] [--resume]
"""
import os
import re
import csv
import json
import time
import argparse
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ═══════════════════════════════════════════════════════════════════════════════
API_URL = 'https://api.deepseek.com/v1/chat/completions'
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
NUM_GENS = 4
CHECKPOINT_INTERVAL = 50

SYSTEM = """You are a rule compiler.

Convert the given problem and its machine-generated solution into a compact symbolic program.

Rules:
- Use ONLY symbolic tokens in format: [TYPE:ARGS]
- No natural language descriptions
- No execution steps (no "apply", "compute", "convert")
- No reference to specific inputs or outputs
- Minimize token count — express the rule in fewest tokens possible
- One token per line

Available token types:
[SCALE:value] — linear scaling: output = value * input (ONE token, no separate CONST needed)
[CONST:name=value] — define a named constant (only when used by FORMULA)
[FORMULA:expr] — a formula using named constants (e.g. d=0.5*g*t^2)
[CIPHER_MAP:a>b,c>d,...] — full substitution cipher, all pairs in ONE token
[Bn:OP(args)] — bit n output rule (B0-B7), args: IN0-IN7, ops: NOT,AND,OR,XOR,XNOR,CONST(val)
[OP:sym→FUNC] — custom operator (e.g. [OP:*→CONCAT]). Only list NON-TRIVIAL ops. Unlisted symbols = identity/passthrough by default.
[ROMAN_GREEDY] — standard Roman numeral encoding
[BASE:from→to] — base conversion
[UNSOLVABLE] — no consistent rule found

Key principles:
- For scaling/unit_conv: use ONE [SCALE:value] token. Do NOT add separate CONST or FORMULA.
- For bit_ops: use [Bn:CONST(1)] directly, NOT [CONST:ONE=1] + [B7:CONST(ONE)]
- For cipher: ONE [CIPHER_MAP:...] token with all pairs
- For symbol: ONLY list operators that DO something (CONCAT, ADD, XOR, etc). Do NOT list KEEP/IDENTITY ops — unlisted = KEEP by default.
- For gravity/physics: [CONST:name=value] + [FORMULA:expr] is fine (2 tokens)"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Rate limiter (simple token-bucket)
# ═══════════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, rps=3):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.last = 0
    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = self.last + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()

rate_limiter = RateLimiter(rps=200)  # No rate limit on DeepSeek

# ═══════════════════════════════════════════════════════════════════════════════
#  API call
# ═══════════════════════════════════════════════════════════════════════════════
def build_user_msg(row):
    prompt = row['prompt']
    if len(prompt) > 600:
        prompt = prompt[:450] + "\n...\n" + prompt[-150:]
    return (
        f"Problem:\n{prompt}\n\n"
        f"Machine solution: {row['solution_process']}\n"
        f"Answer: {row['answer']}"
    )

def call_api(user_msg, temperature=0.5, max_retries=3):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user_msg},
        ],
        'temperature': temperature,
        'max_tokens': 300,
    }
    for attempt in range(max_retries):
        rate_limiter.acquire()
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content'].strip()
                tokens = data.get('usage', {}).get('completion_tokens', 0)
                return content, tokens
            elif resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2)
    return None, 0

# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation: score a DSL output
# ═══════════════════════════════════════════════════════════════════════════════
TOKEN_PATTERN = re.compile(r'^\[.+\]$')
VALID_TYPES = {'SCALE', 'CONST', 'FORMULA', 'CIPHER_MAP', 'OP', 'ROMAN_GREEDY',
               'BASE', 'UNSOLVABLE',
               'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'}
NL_WORDS = {'apply', 'compute', 'convert', 'using', 'the', 'from', 'each', 
            'step', 'determine', 'identify', 'recognize', 'observe'}

def score_dsl(text):
    """Score a DSL output. Higher = better. Returns (score, details)."""
    if not text:
        return -100, "empty"
    
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return -100, "empty"
    
    score = 0
    details = []
    
    # 1. Format compliance: all lines match [TYPE:ARGS]
    format_ok = 0
    for line in lines:
        if TOKEN_PATTERN.match(line):
            format_ok += 1
        else:
            # Penalty for non-token lines
            score -= 10
            details.append(f"non-token: {line[:50]}")
    
    compliance = format_ok / len(lines) if lines else 0
    score += int(compliance * 50)  # up to 50 points for format
    
    # 2. Check for NL words (penalty)
    text_lower = text.lower()
    for w in NL_WORDS:
        if w in text_lower:
            score -= 5
            details.append(f"nl-word:{w}")
    
    # 3. Compactness bonus (fewer tokens = better)
    n_tokens = len(lines)
    if n_tokens <= 2:
        score += 20
    elif n_tokens <= 4:
        score += 15
    elif n_tokens <= 8:
        score += 10
    elif n_tokens <= 10:
        score += 5
    else:
        score -= 5  # too many tokens
    
    # 4. Valid type names bonus
    for line in lines:
        m = re.match(r'\[([A-Z0-9_]+)[:\]]', line)
        if m and m.group(1) in VALID_TYPES:
            score += 3
    
    # 5. Short text bonus
    if len(text) < 50:
        score += 5
    elif len(text) < 100:
        score += 3
    
    return score, "; ".join(details) if details else "clean"

def pick_best(generations):
    """Pick the best DSL from multiple generations."""
    scored = []
    for i, gen in enumerate(generations):
        if gen is None:
            continue
        s, detail = score_dsl(gen)
        scored.append((s, i, gen, detail))
    
    if not scored:
        return "", -1, "all_failed"
    
    # Sort by score descending
    scored.sort(key=lambda x: -x[0])
    
    # Check consensus: if multiple gens produce same output, prefer that
    texts = [g for g in generations if g]
    if texts:
        counter = Counter(texts)
        most_common, count = counter.most_common(1)[0]
        if count >= 2:
            # Consensus bonus — prefer the most common
            s, _ = score_dsl(most_common)
            return most_common, s, f"consensus={count}"
    
    # Otherwise return highest scored
    best = scored[0]
    return best[2], best[0], best[3]

# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════
import requests  # import here to avoid issues

def process_row(row):
    """Generate NUM_GENS DSL rules for one row, return best."""
    user_msg = build_user_msg(row)
    gens = []
    total_tokens = 0
    
    # Use slightly different temperatures for diversity
    temps = [0.3, 0.5, 0.5, 0.7]
    
    for i in range(NUM_GENS):
        content, tok = call_api(user_msg, temperature=temps[i])
        gens.append(content)
        total_tokens += tok
    
    best_text, best_score, reason = pick_best(gens)
    return best_text, best_score, reason, total_tokens, gens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='Rows to process (0=all)')
    parser.add_argument('--workers', type=int, default=50, help='Concurrent workers')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', default='train_dsl_rules.jsonl')
    args = parser.parse_args()
    
    if not API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY")
        return
    
    # Load data
    base = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base, '..', 'data', 'train_annotated.csv')
    with open(input_path) as f:
        all_rows = list(csv.DictReader(f))
    
    # Filter to solvable rows
    solvable = [r for r in all_rows if r['match'] == 'True' and r.get('solution_process')]
    unsolvable = [r for r in all_rows if r['match'] != 'True' or not r.get('solution_process')]
    print(f"Total: {len(all_rows)}, Solvable: {len(solvable)}, Unsolvable: {len(unsolvable)}")
    
    # Resume support: only count rows with actual DSL (score > 0) as done
    output_path = os.path.join(base, '..', 'data', args.output)
    done = {}
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get('score', -1) > 0 and obj.get('dsl'):
                    done[obj['id']] = obj
        print(f"Resuming: {len(done)} already done (with valid DSL)")
    
    pending = [r for r in solvable if r['id'] not in done]
    if args.n > 0:
        pending = pending[:args.n]
    
    print(f"To process: {len(pending)} rows × {NUM_GENS} gens = {len(pending)*NUM_GENS} API calls")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_path}")
    print()
    
    # Process
    results = dict(done)
    success = len(done)
    fail = 0
    total_tokens = 0
    type_counts = Counter()
    score_sum = 0
    t_start = time.time()
    
    def do_one(row):
        best_text, best_score, reason, tok, gens = process_row(row)
        return row['id'], row['type'], best_text, best_score, reason, tok, gens
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(do_one, r): r for r in pending}
        
        for i, future in enumerate(as_completed(futures)):
            rid, rtype, best_text, best_score, reason, tok, gens = future.result()
            
            result = {
                'id': rid,
                'type': rtype,
                'dsl': best_text,
                'score': best_score,
                'reason': reason,
                'all_gens': gens,
            }
            results[rid] = result
            total_tokens += tok
            
            if best_text:
                success += 1
                type_counts[rtype] += 1
                score_sum += best_score
            else:
                fail += 1
            
            done_now = i + 1
            elapsed = time.time() - t_start
            rps = done_now / elapsed if elapsed > 0 else 0
            
            if done_now % 20 == 0 or done_now == len(pending):
                avg_score = score_sum / max(success - len(done), 1)
                print(f"  [{done_now}/{len(pending)}] ok={success} fail={fail} "
                      f"rps={rps:.2f} tok={total_tokens} avg_score={avg_score:.1f} "
                      f"types={dict(type_counts)}")
            
            # Checkpoint
            if done_now % CHECKPOINT_INTERVAL == 0:
                save_results(results, unsolvable, all_rows, output_path)
                print(f"  ** Checkpoint saved ({len(results)} rows)")
    
    # Final save
    save_results(results, unsolvable, all_rows, output_path)
    
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Success: {success}, Failed: {fail}, Total tokens: {total_tokens}")
    print(f"Avg score: {score_sum/max(success,1):.1f}")
    print(f"Output: {output_path}")
    
    # Also save a clean CSV version
    csv_path = output_path.replace('.jsonl', '.csv')
    save_csv(results, unsolvable, all_rows, csv_path)
    print(f"CSV: {csv_path}")

def save_results(results, unsolvable, all_rows, output_path):
    """Save JSONL with all results."""
    with open(output_path, 'w') as f:
        for row in all_rows:
            rid = row['id']
            if rid in results:
                obj = results[rid]
                if isinstance(obj, dict):
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                else:
                    f.write(json.dumps({'id': rid, 'type': row['type'], 'dsl': '', 'score': -1}, ensure_ascii=False) + '\n')
            else:
                # unsolvable
                f.write(json.dumps({'id': rid, 'type': row['type'], 'dsl': '', 'score': -1, 'reason': 'unsolvable'}, ensure_ascii=False) + '\n')

def save_csv(results, unsolvable, all_rows, csv_path):
    """Save clean CSV: id, prompt, answer, type, dsl_rules."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'type', 'dsl_rules'])
        writer.writeheader()
        for row in all_rows:
            rid = row['id']
            dsl = ''
            if rid in results:
                obj = results[rid]
                if isinstance(obj, dict):
                    dsl = obj.get('dsl', '')
            writer.writerow({
                'id': rid,
                'prompt': row['prompt'],
                'answer': row['answer'],
                'type': row['type'],
                'dsl_rules': dsl,
            })

if __name__ == '__main__':
    main()
