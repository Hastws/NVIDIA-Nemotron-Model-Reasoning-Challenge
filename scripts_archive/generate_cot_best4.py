#!/usr/bin/env python3
"""
Production CoT generation: best-of-4 sampling.
For each row with a valid solution_process (rule), generates 4 CoT candidates
and picks the best one (correct answer + shortest).

Usage:
  python3 scripts/generate_cot_best4.py                     # all 8121 rows
  python3 scripts/generate_cot_best4.py --n 60 --workers 10 # test 60
  python3 scripts/generate_cot_best4.py --resume             # resume

Output: data/cot_best4.jsonl
"""
import os, re, csv, json, math, time, argparse, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_URL = 'https://api.deepseek.com/v1/chat/completions'
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CHECKPOINT_INTERVAL = 200

# ═══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, rps=50):
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

rate_limiter = RateLimiter(rps=50)

# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM = (
    "You are rewriting a compact machine-generated solution into a clean, "
    "natural chain-of-thought reasoning trace.\n\n"
    "RULES:\n"
    "1. Write as if you are thinking through the problem step-by-step.\n"
    "2. Be concise: 2-6 sentences. No fluff, no preamble like 'Here is' or 'Sure'.\n"
    "3. Include key numbers/values (e.g. g=15.90, f=0.6636, mapping).\n"
    "4. Use plain ASCII only. No LaTeX (no \\( \\) or $ $). No special Unicode.\n"
    "5. The final answer should appear naturally at the end of your reasoning (e.g. '= 154.62' or 'gives: cat imagines book').\n"
    "6. Do NOT wrap the answer in \\boxed{}. Do NOT write \\boxed{} at all.\n"
    "7. Output ONLY the reasoning process. Nothing else."
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PER-TYPE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
TYPE_EXAMPLES = {
    'gravity': (
        "EXAMPLE:\n"
        "Machine solution: d=0.5*g*t^2. g=2*14.92/1.37^2=15.90; g=2*144.96/4.27^2=15.90. g_avg=15.90. d=0.5*15.90*4.41^2=154.62\n"
        "Rewrite:\n"
        "Using d = 0.5*g*t^2, compute g from the first example: g = 2*14.92/1.37^2 = 15.90. "
        "The second example confirms g = 2*144.96/4.27^2 = 15.90. "
        "For t = 4.41: d = 0.5 * 15.90 * 4.41^2 = 154.62."
    ),
    'unit_conv': (
        "EXAMPLE:\n"
        "Machine solution: Linear conversion. 10.08->6.69(f=0.6637); 17.83->11.83(f=0.6635). avg_f=0.6636. 25.09*0.6636=16.65\n"
        "Rewrite:\n"
        "Each example gives a constant ratio: 6.69/10.08 = 0.6637, 11.83/17.83 = 0.6635. "
        "Average factor f = 0.6636. Apply: 25.09 * 0.6636 = 16.65."
    ),
    'numeral': (
        "EXAMPLE:\n"
        "Machine solution: Arabic->Roman. 38 = 10*3=XXX, 5*1=V, 1*3=III -> XXXVIII\n"
        "Rewrite:\n"
        "The examples show standard Arabic to Roman numeral conversion. "
        "38 = 30(XXX) + 5(V) + 3(III) = XXXVIII."
    ),
    'cipher': (
        "EXAMPLE:\n"
        "Machine solution: Substitution cipher. Mapping: b->t, f->o, g->s, h->b, k->k, o->e. Result: cat imagines book\n"
        "Rewrite:\n"
        "Build the substitution map from examples: b->t, f->o, g->s, h->b, k->k, o->e, r->a, s->g, t->c, v->n, w->i, z->m. "
        "Applying letter by letter to the ciphertext gives: cat imagines book."
    ),
    'bit_ops': (
        "EXAMPLE:\n"
        "Machine solution: Per-bit rules: b0=XNOR(in[1],in[7]); b1=NOT in[2]; b2=NOT in[3]; b3=NOT in[4]; b4=NOT in[5]; b5=NOT in[6]; b6=NOT in[7]; b7=1\n"
        "Rewrite:\n"
        "Analyzing each bit position across examples: b7 is always 1, b6 through b1 are NOT of in[7] through in[2] respectively, "
        "and b0 = XNOR(in[1], in[7]). For input 00110100: b7=1, b6=NOT(0)=1, b5=NOT(0)=1, b4=NOT(1)=0, "
        "b3=NOT(1)=0, b2=NOT(1)=0, b1=NOT(0)=1, b0=XNOR(0,0)=1. Result: 10010111."
    ),
    'symbol': (
        "EXAMPLE:\n"
        "Machine solution: Symbol op '*' = concat. \\( * [# = \\([#\n"
        "Rewrite:\n"
        "Testing the * operator on examples: it concatenates the two operands. "
        "Apply: \\( * [# = \\([#."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ═══════════════════════════════════════════════════════════════════════════════
def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None

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

def sanitize(text):
    for old, new in [('\u00d7','*'), ('\u2192','->'), ('\u2248','~'),
                     ('\u00b2','^2'), ('\u2713','ok'), ('\u2212','-'),
                     ('\u00f7','/'), ('\u2019',"'"), ('\u201c','"'), ('\u201d','"')]:
        text = text.replace(old, new)
    text = re.sub(r'\\\(|\\\)', '', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    return text

DSL_LEAK_WORDS = ['dsl', 'machine solution', 'machine-generated', 'machine generated',
                  'rewrite', 'rewriting']

def answer_in_text(text, answer):
    """Check if the correct answer appears in the generated text."""
    if not text or not answer:
        return False
    # Normalize for comparison
    text_norm = text.strip().lower()
    ans_norm = answer.strip().lower()
    if ans_norm in text_norm:
        return True
    # Try numeric match
    try:
        ans_float = float(answer)
        # Find all numbers in text
        for m in re.finditer(r'[\d]+\.?[\d]*', text):
            try:
                if math.isclose(float(m.group()), ans_float, rel_tol=1e-2, abs_tol=1e-5):
                    return True
            except ValueError:
                pass
    except ValueError:
        pass
    return False

def score_cot(cot, answer):
    """Score a CoT: higher is better. Returns (score, issues)."""
    if not cot:
        return -100, ['empty']
    issues = []

    # Check if answer appears in reasoning
    has_answer = answer_in_text(cot, answer)
    score = 100 if has_answer else 0

    # Penalize if boxed still appears (shouldn't)
    if '\\boxed' in cot or '\boxed' in cot:
        issues.append('has_boxed')
        score -= 30

    # Penalize DSL leaks
    cot_lower = cot.lower()
    for w in DSL_LEAK_WORDS:
        if w in cot_lower:
            issues.append(f'leak:{w}')
            score -= 50

    # Prefer shorter (fewer words)
    words = len(cot.split())
    if words < 10:
        issues.append('too_short')
        score -= 30
    elif words > 200:
        issues.append('too_long')
        score -= 10

    # Prefer moderate length
    score -= words * 0.1  # slight penalty per word

    if not has_answer:
        issues.append('answer_not_found')

    return score, issues

# ═══════════════════════════════════════════════════════════════════════════════
#  API CALL
# ═══════════════════════════════════════════════════════════════════════════════
def call_api(user_msg, temperature=0.3, timeout_s=60):
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user_msg},
        ],
        'temperature': temperature,
        'max_tokens': 500,
    }
    rate_limiter.acquire()
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout_s)
        if resp.status_code == 200:
            data = resp.json()
            content = data['choices'][0]['message']['content'].strip()
            tokens = data.get('usage', {}).get('total_tokens', 0)
            return sanitize(content), tokens
        elif resp.status_code == 429:
            time.sleep(5)
            return None, 0
        return None, 0
    except Exception:
        return None, 0

# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESS ONE ROW — best of 4
# ═══════════════════════════════════════════════════════════════════════════════
def process_row(row):
    rid = row['id']
    ptype = row['type']
    answer = row['answer']
    sol = row['solution_process']

    example = TYPE_EXAMPLES.get(ptype, '')
    prompt = row['prompt']
    if len(prompt) > 600:
        prompt = prompt[:400] + "\n...\n" + prompt[-150:]

    user_msg = (
        f"Problem type: {ptype}\n\n"
        f"Problem:\n{prompt}\n\n"
        f"Machine solution: {sol}\n"
        f"Correct answer: {answer}\n\n"
        f"{example}\n\n"
        f"Rewrite the machine solution into natural step-by-step reasoning. "
        f"The answer should appear naturally in your reasoning but do NOT use \\boxed{{}}."
    )

    # Generate 4 candidates at different temperatures
    candidates = []
    total_tokens = 0
    temps = [0.2, 0.4, 0.6, 0.8]

    for temp in temps:
        content, tokens = call_api(user_msg, temperature=temp)
        total_tokens += tokens
        if content:
            sc, issues = score_cot(content, answer)
            candidates.append({
                'content': content,
                'score': sc,
                'issues': issues,
                'temp': temp,
                'words': len(content.split()),
            })

    if not candidates:
        return {
            'id': rid, 'type': ptype, 'answer': answer,
            'thinking': '', 'content': '', 'correct': False,
            'tokens': total_tokens, 'n_candidates': 0,
            'best_score': -100,
        }

    # Pick best candidate
    best = max(candidates, key=lambda c: c['score'])
    content = best['content']

    # Strip any boxed if model still adds it
    clean = re.sub(r'\\boxed\{[^}]*\}', '', content).strip()
    has_answer = answer_in_text(content, answer)

    return {
        'id': rid, 'type': ptype, 'answer': answer,
        'thinking': clean, 'content': clean,
        'correct': has_answer,
        'tokens': total_tokens,
        'n_candidates': len(candidates),
        'best_score': best['score'],
        'best_temp': best['temp'],
        'best_words': best['words'],
        'all_scores': [c['score'] for c in candidates],
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='Rows to process (0=all)')
    parser.add_argument('--workers', type=int, default=20, help='Parallel workers')
    parser.add_argument('--output', type=str, default='cot_best4.jsonl')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY"); return

    out_path = os.path.join(DATA_DIR, args.output)

    # Load source: train_annotated.csv (rows with match=True and solution_process)
    src = os.path.join(DATA_DIR, 'train_annotated.csv')
    with open(src) as f:
        all_rows = [r for r in csv.DictReader(f)
                    if r['match'] == 'True' and r.get('solution_process', '').strip()]

    print(f"Source: {len(all_rows)} matched rows with solution_process")
    print(f"Types: {Counter(r['type'] for r in all_rows)}")

    # Resume
    done_ids = set()
    existing = []
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done_ids.add(r['id'])
                    existing.append(r)
        print(f"Resume: {len(done_ids)} already done")

    rows = [r for r in all_rows if r['id'] not in done_ids]
    if args.n > 0:
        rows = rows[:args.n]

    print(f"Processing {len(rows)} rows with {args.workers} workers (4 samples each)...")

    results = list(existing)
    lock = threading.Lock()
    done_count = [len(existing)]
    correct_count = [sum(1 for r in existing if r.get('correct'))]
    type_stats = defaultdict(lambda: [0, 0])
    for r in existing:
        type_stats[r['type']][0] += 1
        if r.get('correct'):
            type_stats[r['type']][1] += 1
    total_tokens = [0]
    t0 = time.time()

    def do_one(row):
        r = process_row(row)
        with lock:
            results.append(r)
            done_count[0] += 1
            total_tokens[0] += r.get('tokens', 0)
            type_stats[r['type']][0] += 1
            if r['correct']:
                correct_count[0] += 1
                type_stats[r['type']][1] += 1
            n = done_count[0]
            c = correct_count[0]
            elapsed = time.time() - t0
            new_done = n - len(existing)
            rate = new_done / max(elapsed, 1) * 60
            mark = 'ok' if r['correct'] else 'FAIL'
            print(f"  [{n:5d}/{len(rows)+len(existing)}] {mark:4s} {r['type']:12s} "
                  f"w={r.get('best_words',0):3d} sc={r.get('best_score',0):5.1f} "
                  f"acc={c}/{n} ({c/n*100:.1f}%) "
                  f"rate={rate:.0f}/min", flush=True)

            if new_done > 0 and new_done % CHECKPOINT_INTERVAL == 0:
                _save_jsonl(results, out_path)
                print(f"  [CHECKPOINT] saved {len(results)} rows", flush=True)
        return r

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(do_one, r): r for r in rows}
            for f in as_completed(futures):
                f.result()
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
    finally:
        _save_jsonl(results, out_path)

    elapsed = time.time() - t0
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s | {total} rows | {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Tokens: {total_tokens[0]:,}")
    print(f"{'='*60}")
    print(f"{'Type':12s} {'Total':>6s} {'Correct':>8s} {'Acc':>7s}")
    for t in sorted(type_stats):
        tot, cor = type_stats[t]
        print(f"{t:12s} {tot:6d} {cor:8d} {cor/tot*100:6.1f}%")
    print(f"\nSaved: {out_path}")


def _save_jsonl(results, path):
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
