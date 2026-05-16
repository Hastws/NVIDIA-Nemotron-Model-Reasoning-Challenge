#!/usr/bin/env python3
"""
Generate Chain-of-Thought reasoning for all training rows with valid DSL rules.
Uses DeepSeek API with few-shot prompting to produce concise, natural CoT.

Usage:
  python3 scripts/generate_cot.py [--n N] [--workers W] [--resume]
"""
import os, re, csv, json, time, argparse, threading, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = 'https://api.deepseek.com/v1/chat/completions'
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
CHECKPOINT_INTERVAL = 100

# ─── System Prompt (v3 — structured & concise) ────────────────────────────────
SYSTEM = """You are writing the internal reasoning a strong model would produce when solving a pattern-matching puzzle.

You receive: a puzzle (with worked examples), the underlying rule (DSL notation), and the correct answer.

YOUR TASK: Write a short, structured chain-of-thought that derives the rule from the examples and applies it.

FORMAT RULES:
- Use short labeled sections: "Pattern:", "Check:", "Apply:", "Map:", "Bits:" etc.
- NO filler phrases ("Looking at", "From the examples", "Let me", "I notice")
- Ground every claim in concrete numbers
- End with the final answer value on its own line

ABSOLUTE RULES:
1. NEVER mention "DSL", "rule provided", "symbolic rule", "given rule" — write as if discovering the pattern yourself.
2. For bit-ops: bits 0 (LSB) to 7 (MSB). One line per output bit rule.
3. For cipher: show 3-4 letter derivations, then full map. Do NOT trace every decryption letter.
4. For operator-symbol: test arithmetic per operator, verify once, conclude.

LENGTH: 30-120 words. Shorter is better."""

# ─── Few-shot golden examples ─────────────────────────────────────────────────
EXAMPLES = {
    'numeral': {
        'prompt_snippet': '57 -> LVII, 45 -> XLV, 78 -> LXXVIII ... convert 43',
        'answer': 'XLIII',
        'golden_cot': 'Pattern: standard Roman numerals (57=LVII, 45=XLV, 78=LXXVIII).\nApply: 43 = 40(XL) + 3(III) → XLIII'
    },
    'gravity': {
        'prompt_snippet': 't=1.72s d=24.41m, t=3.23s d=86.08m, t=2.48s d=50.74m ... find d for t=2.17s given d=0.5*g*t²',
        'answer': '38.85',
        'golden_cot': 'Formula: d = 0.5*g*t². Solve for g:\n- (1.72, 24.41): g = 2×24.41/1.72² = 48.82/2.958 ≈ 16.501\n- (3.23, 86.08): g = 172.16/10.433 ≈ 16.502\nConsistent g ≈ 16.5009.\nApply: t=2.17 → d = 0.5 × 16.5009 × 4.7089 ≈ 38.85'
    },
    'unit_conv': {
        'prompt_snippet': '6.24 m becomes 3.77, 18.52 m becomes 11.19, 26.37 m becomes 15.93 ... convert 7.7 m',
        'answer': '4.65',
        'golden_cot': 'Ratio: 3.77/6.24 = 0.6042, 11.19/18.52 = 0.6042, 15.93/26.37 = 0.6041\nConstant factor ≈ 0.6041\nApply: 7.7 × 0.6041 ≈ 4.65'
    },
    'cipher': {
        'prompt_snippet': '"ysu" → "the", "fjtcujy" → "ancient" ... decrypt "qvcjtuxx"',
        'answer': 'princess',
        'golden_cot': 'Substitution cipher.\nDerive: "ysu"→"the": y→t, s→h, u→e. "fjtcujy"→"ancient": f→a, j→n, t→c, c→i.\nFull map: a→u, c→i, f→a, j→n, q→p, t→c, u→e, v→r, x→s, y→t\nApply "qvcjtuxx": p-r-i-n-c-e-s-s → princess'
    },
    'bit_ops': {
        'prompt_snippet': '11011101->11010001, 00010111->01110000, 00010000->00000000 ... find output for 11101101',
        'answer': '11010001',
        'golden_cot': 'Bits (0=LSB, 7=MSB):\nB0-B2: always 0 → CONST(0)\nB3: copies IN7\nB4: copies IN0\nB5: copies IN1, B6: copies IN2, B7: copies IN3\nApply 11101101 (IN7=1,IN6=1,IN5=1,IN4=0,IN3=1,IN2=1,IN1=0,IN0=1):\nB7..B0 = 1,1,0,1,0,0,0,1 → 11010001'
    },
    'symbol': {
        'prompt_snippet': '24/63=87, 22/92=114, 96{75=21, 50{12=38 ... find 41/85',
        'answer': '126',
        'golden_cot': 'Test /: 24+63=87 ✓, 22+92=114 ✓ → ADD\nTest {: 96−75=21 ✓, 50−12=38 ✓ → SUB\nApply: 41/85 = 41+85 = 126'
    }
}

TYPES_ORDER = ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']

# ─── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rps=200):
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

rate_limiter = RateLimiter(rps=200)

# ─── Build few-shot block (exclude same type) ─────────────────────────────────
def build_few_shot(exclude_type):
    shots = []
    for t in TYPES_ORDER:
        if t == exclude_type:
            continue
        ex = EXAMPLES[t]
        shots.append(
            f"Problem snippet: {ex['prompt_snippet']}\n"
            f"Answer: {ex['answer']}\n"
            f"Reasoning:\n{ex['golden_cot']}"
        )
    return "\n\n---\n\n".join(shots)

# Pre-build few-shot strings (one per type)
FEW_SHOT_CACHE = {t: build_few_shot(t) for t in TYPES_ORDER}

# ─── API call ──────────────────────────────────────────────────────────────────
def call_api(system, user_msg, temperature=0.3, max_retries=3):
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_msg},
        ],
        'temperature': temperature,
        'max_tokens': 500,
    }
    for attempt in range(max_retries):
        rate_limiter.acquire()
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content'].strip()
                tokens = data.get('usage', {}).get('total_tokens', 0)
                return content, tokens
            elif resp.status_code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(2)
    return None, 0
# ─── Post-processing: normalize special chars to ASCII ───────────────────────
def sanitize_cot(text):
    """Replace Unicode math symbols with ASCII equivalents."""
    text = text.replace('\u00d7', '*')    # × -> *
    text = text.replace('\u2192', '->')   # \u2192 -> ->
    text = text.replace('\u2248', '~')    # \u2248 -> ~
    text = text.replace('\u00b2', '^2')   # \u00b2 -> ^2
    text = text.replace('\u2713', 'ok')   # \u2713 -> ok
    text = text.replace('\u2212', '-')    # \u2212 -> -
    text = text.replace('\u00f7', '/')    # \u00f7 -> /
    return text
# ─── Quality checks ───────────────────────────────────────────────────────────
DSL_LEAK_WORDS = ['dsl', 'symbolic rule', 'rule provided', 'rule says', 'rule tells',
                  'given rule', 'the rule is provided']

def check_cot(cot, answer):
    """Return (pass: bool, issues: list)."""
    if not cot:
        return False, ['empty']
    issues = []
    words = len(cot.split())
    if words < 15:
        issues.append(f'too_short({words}w)')
    cot_lower = cot.lower()
    for w in DSL_LEAK_WORDS:
        if w in cot_lower:
            issues.append(f'dsl_leak:{w}')
    if answer.strip() not in cot:
        issues.append('answer_missing')
    return len(issues) == 0, issues

# ─── Process one row ──────────────────────────────────────────────────────────
def process_row(row_id, prompt, answer, ptype, dsl):
    few_shot = FEW_SHOT_CACHE.get(ptype, FEW_SHOT_CACHE['numeral'])
    user_msg = (
        f"Here are examples of good reasoning for similar puzzles:\n\n{few_shot}\n\n"
        f"---\n\n"
        f"Now write the reasoning for this puzzle:\n\n"
        f"Problem:\n{prompt}\n\n"
        f"Symbolic rule: {dsl}\n"
        f"Correct answer: {answer}\n\n"
        f"Write your reasoning (30-120 words, structured with labels):"
    )
    
    # Try up to 2 generations, pick the passing one (or best)
    best_cot = None
    best_tokens = 0
    for temp in [0.3, 0.5]:
        cot, tokens = call_api(SYSTEM, user_msg, temperature=temp)
        best_tokens += tokens
        if cot is None:
            continue
        cot = sanitize_cot(cot)
        passed, issues = check_cot(cot, answer)
        if passed:
            return cot, best_tokens, True, []
        if best_cot is None:
            best_cot = cot
    
    # Return best even if imperfect
    _, issues = check_cot(best_cot, answer) if best_cot else (False, ['all_failed'])
    return best_cot or '', best_tokens, False, issues

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='Rows to process (0=all)')
    parser.add_argument('--workers', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', default='train_cot.jsonl')
    args = parser.parse_args()
    
    if not API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY"); return
    
    output_path = f'data/{args.output}'
    
    # Load DSL data
    dsl_data = {}
    with open('data/train_dsl_rules.jsonl') as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('dsl') and obj.get('score', 0) > 0:
                dsl_data[obj['id']] = obj
    
    # Load annotated data for prompts
    ann = {}
    with open('data/train_annotated.csv') as f:
        for r in csv.DictReader(f):
            ann[r['id']] = r
    
    # Resume: load already-done IDs
    done_ids = set()
    results = []
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get('cot') and obj.get('passed'):
                    done_ids.add(obj['id'])
                    results.append(obj)
        print(f"Resuming: {len(done_ids)} already done")
    
    # Build work queue
    work = []
    for rid, d in dsl_data.items():
        if rid in done_ids:
            continue
        if rid not in ann:
            continue
        work.append((rid, ann[rid]['prompt'], ann[rid]['answer'], d['type'], d['dsl']))
    
    if args.n > 0:
        work = work[:args.n]
    
    total = len(work)
    print(f"Processing {total} rows ({len(done_ids)} resumed) | workers={args.workers}")
    
    # Counters
    lock = threading.Lock()
    counters = {'done': 0, 'passed': 0, 'failed': 0, 'tokens': 0}
    t_start = time.time()
    
    def worker(item):
        rid, prompt, answer, ptype, dsl = item
        cot, tokens, passed, issues = process_row(rid, prompt, answer, ptype, dsl)
        
        result = {
            'id': rid,
            'type': ptype,
            'cot': cot,
            'answer': answer,
            'passed': passed,
            'issues': issues,
            'words': len(cot.split()) if cot else 0,
        }
        
        with lock:
            counters['done'] += 1
            counters['tokens'] += tokens
            if passed:
                counters['passed'] += 1
            else:
                counters['failed'] += 1
            results.append(result)
            
            n = counters['done']
            if n % 50 == 0 or n == total:
                elapsed = time.time() - t_start
                rps = n / elapsed if elapsed > 0 else 0
                pct = counters['passed'] / n * 100
                print(f"  [{n}/{total}] {rps:.1f} rows/s | pass={counters['passed']} ({pct:.0f}%) | fail={counters['failed']} | tokens={counters['tokens']//1000}K")
            
            # Checkpoint
            if n % CHECKPOINT_INTERVAL == 0:
                _save(output_path, results)
        
        return result
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, item): item for item in work}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Final save
    _save(output_path, results)
    
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"Total: {counters['done']} | Passed: {counters['passed']} | Failed: {counters['failed']}")
    print(f"Tokens: {counters['tokens']//1000}K")
    print(f"Output: {output_path}")
    
    # Also save as CSV for easy viewing
    csv_path = output_path.replace('.jsonl', '.csv')
    _save_csv(csv_path, results)
    print(f"CSV: {csv_path}")

def _save(path, results):
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def _save_csv(path, results):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'type', 'cot', 'answer', 'passed', 'words'])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})

if __name__ == '__main__':
    main()
