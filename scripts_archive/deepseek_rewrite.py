#!/usr/bin/env python3
"""
Production DeepSeek rewrite pipeline.
Rewrites machine-generated solution_process into clean step-by-step reasoning.

Usage:
  python3 scripts/deepseek_rewrite.py                        # all 8121 rows
  python3 scripts/deepseek_rewrite.py --n 30 --workers 5     # test 30 rows
  python3 scripts/deepseek_rewrite.py --resume               # resume from checkpoint

Output: data/rewrite_v1.jsonl + data/sft_rewrite_v1.csv
"""
import os, re, csv, json, math, time, argparse, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_URL = 'https://api.deepseek.com/v1/chat/completions'
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CHECKPOINT_INTERVAL = 50

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
    "5. End with the answer in \\boxed{} format.\n"
    "6. The answer MUST be exactly: \\boxed{ANSWER} — do not change the given answer.\n"
    "7. Output ONLY the reasoning + boxed answer. Nothing else."
)

# ═══════════════════════════════════════════════════════════════════════════════
#  TYPE-SPECIFIC EXAMPLES (few-shot)
# ═══════════════════════════════════════════════════════════════════════════════
TYPE_EXAMPLES = {
    'gravity': (
        "EXAMPLE:\n"
        "Machine solution: d=0.5*g*t^2. g=2*14.92/1.37^2=15.90; g=2*144.96/4.27^2=15.90. g_avg=15.90. d=0.5*15.90*4.41^2=154.62\n"
        "Rewrite:\n"
        "Using d = 0.5*g*t^2, compute g from the first example: g = 2*14.92/1.37^2 = 15.90. "
        "The second example confirms g = 2*144.96/4.27^2 = 15.90. "
        "For t = 4.41: d = 0.5 * 15.90 * 4.41^2 = 154.62.\n"
        "\\boxed{154.62}"
    ),
    'unit_conv': (
        "EXAMPLE:\n"
        "Machine solution: Linear conversion. 10.08->6.69(f=0.6637); 17.83->11.83(f=0.6635). avg_f=0.6636. 25.09*0.6636=16.65\n"
        "Rewrite:\n"
        "Each example gives a constant ratio: 6.69/10.08 = 0.6637, 11.83/17.83 = 0.6635. "
        "Average factor f = 0.6636. Apply: 25.09 * 0.6636 = 16.65.\n"
        "\\boxed{16.65}"
    ),
    'numeral': (
        "EXAMPLE:\n"
        "Machine solution: Arabic->Roman. 38 = 10*3=XXX, 5*1=V, 1*3=III -> XXXVIII\n"
        "Rewrite:\n"
        "The examples show standard Arabic to Roman numeral conversion. "
        "38 = 30(XXX) + 5(V) + 3(III) = XXXVIII.\n"
        "\\boxed{XXXVIII}"
    ),
    'cipher': (
        "EXAMPLE:\n"
        "Machine solution: Substitution cipher. Mapping: b->t, f->o, g->s, h->b, k->k, o->e. Result: cat imagines book\n"
        "Rewrite:\n"
        "Build the substitution map from examples: b->t, f->o, g->s, h->b, k->k, o->e, r->a, s->g, t->c, v->n, w->i, z->m. "
        "Applying letter by letter to the ciphertext gives: cat imagines book.\n"
        "\\boxed{cat imagines book}"
    ),
    'bit_ops': (
        "EXAMPLE:\n"
        "Machine solution: Per-bit rules: b0=XNOR(in[1],in[7]); b1=NOT in[2]; b2=NOT in[3]; b3=NOT in[4]; b4=NOT in[5]; b5=NOT in[6]; b6=NOT in[7]; b7=1\n"
        "Rewrite:\n"
        "Analyzing each bit position across examples: b7 is always 1, b6 through b1 are NOT of in[7] through in[2] respectively, "
        "and b0 = XNOR(in[1], in[7]). For input 00110100: b7=1, b6=NOT(0)=1, b5=NOT(0)=1, b4=NOT(1)=0, "
        "b3=NOT(1)=0, b2=NOT(1)=0, b1=NOT(0)=1, b0=XNOR(0,0)=1. Result: 10010111.\n"
        "\\boxed{10010111}"
    ),
    'symbol': (
        "EXAMPLE:\n"
        "Machine solution: Symbol op '*' = concat. \\( * [# = \\([#\n"
        "Rewrite:\n"
        "Testing the * operator on examples: 24/63 = 87 (24+63=87, addition works). "
        "Confirming: 22/92 = 114 (22+92=114). So / means addition. "
        "Apply: 41/85 = 41+85 = 126.\n"
        "\\boxed{126}"
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  ANSWER MATCHING
# ═══════════════════════════════════════════════════════════════════════════════
def extract_boxed(text):
    """Extract last \\boxed{...} content."""
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
    """Replace Unicode math symbols with ASCII."""
    for old, new in [('\u00d7','*'), ('\u2192','->'), ('\u2248','~'),
                     ('\u00b2','^2'), ('\u2713','ok'), ('\u2212','-'),
                     ('\u00f7','/'), ('\u2019',"'"), ('\u201c','"'), ('\u201d','"')]:
        text = text.replace(old, new)
    # Strip any remaining LaTeX delimiters
    text = re.sub(r'\\\(|\\\)', '', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    return text

# ═══════════════════════════════════════════════════════════════════════════════
#  API CALL
# ═══════════════════════════════════════════════════════════════════════════════
def call_api(system, user_msg, temperature=0.3, timeout_s=60):
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_msg},
        ],
        'temperature': temperature,
        'max_tokens': 400,
    }
    rate_limiter.acquire()
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout_s)
        if resp.status_code == 200:
            data = resp.json()
            content = data['choices'][0]['message']['content'].strip()
            tokens = data.get('usage', {}).get('total_tokens', 0)
            return content, tokens
        return None, 0
    except Exception:
        return None, 0

# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESS ONE ROW
# ═══════════════════════════════════════════════════════════════════════════════
def process_row(row):
    """Rewrite solution_process for one row. Returns result dict."""
    rid = row['id']
    ptype = row['type']
    answer = row['answer']
    sol = row['solution_process']

    # Build user message
    example = TYPE_EXAMPLES.get(ptype, '')
    # Truncate prompt if too long
    prompt = row['prompt']
    if len(prompt) > 600:
        prompt = prompt[:400] + "\n...\n" + prompt[-150:]

    user_msg = (
        f"Problem type: {ptype}\n\n"
        f"Problem:\n{prompt}\n\n"
        f"Machine solution: {sol}\n"
        f"Correct answer: {answer}\n\n"
        f"{example}\n\n"
        f"Rewrite the machine solution into natural reasoning, ending with \\boxed{{{answer}}}."
    )

    best_content = None
    best_ok = False
    total_tokens = 0

    for temp in [0.3, 0.5]:
        content, tokens = call_api(SYSTEM, user_msg, temperature=temp)
        total_tokens += tokens
        if content is None:
            continue

        content = sanitize(content)
        pred = extract_boxed(content)
        ok = answers_match(pred, answer)

        if ok:
            # Extract thinking (everything before \boxed{})
            thinking = re.sub(r'\\boxed\{[^}]*\}', '', content).strip()
            return {
                'id': rid, 'type': ptype, 'answer': answer,
                'thinking': thinking, 'predicted': pred,
                'correct': True, 'tokens': total_tokens,
                'content': content,
            }
        if best_content is None:
            best_content = content

    # Return best attempt even if wrong answer
    thinking = ''
    predicted = None
    if best_content:
        thinking = re.sub(r'\\boxed\{[^}]*\}', '', best_content).strip()
        predicted = extract_boxed(best_content)

    return {
        'id': rid, 'type': ptype, 'answer': answer,
        'thinking': thinking, 'predicted': predicted,
        'correct': False, 'tokens': total_tokens,
        'content': best_content or '',
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='Rows to process (0=all)')
    parser.add_argument('--workers', type=int, default=20, help='Parallel workers')
    parser.add_argument('--output', type=str, default='rewrite_v1.jsonl', help='Output filename')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable")
        return

    out_path = os.path.join(DATA_DIR, args.output)

    # Load source data
    src = os.path.join(DATA_DIR, 'train_annotated.csv')
    with open(src) as f:
        all_rows = [r for r in csv.DictReader(f)
                    if r['match'] == 'True' and r.get('solution_process', '').strip()]

    print(f"Source: {len(all_rows)} matched rows with solution_process")
    print(f"Types: {Counter(r['type'] for r in all_rows)}")

    # Resume: load existing results
    done_ids = set()
    existing = []
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r['id'])
                existing.append(r)
        print(f"Resume: {len(done_ids)} already done")

    # Filter remaining
    rows = [r for r in all_rows if r['id'] not in done_ids]
    if args.n > 0:
        rows = rows[:args.n]

    print(f"Processing {len(rows)} rows with {args.workers} workers...")

    # Process
    results = list(existing)
    lock = threading.Lock()
    done_count = [len(existing)]
    correct_count = [sum(1 for r in existing if r.get('correct'))]
    type_stats = defaultdict(lambda: [0, 0])  # [total, correct]
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
            rate = (n - len(existing)) / max(elapsed, 1) * 60
            mark = 'ok' if r['correct'] else 'FAIL'
            print(f"  [{n:5d}/{len(rows)+len(existing)}] {mark:4s} {r['type']:12s} "
                  f"pred={repr(r.get('predicted',''))[:25]:25s} "
                  f"acc={c}/{n} ({c/n*100:.1f}%) "
                  f"rate={rate:.0f}/min", flush=True)

            # Checkpoint
            if (n - len(existing)) % CHECKPOINT_INTERVAL == 0:
                save_jsonl(results, out_path)
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(do_one, r): r for r in rows}
        for f in as_completed(futures):
            f.result()  # propagate exceptions

    # Final save
    save_jsonl(results, out_path)
    save_csv(results, out_path)

    # Summary
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


def save_jsonl(results, path):
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def save_csv(results, jsonl_path):
    """Save as SFT-ready CSV with prompt/thinking/answer columns."""
    csv_path = jsonl_path.replace('.jsonl', '.csv')

    # Load original prompts
    src = os.path.join(DATA_DIR, 'train_annotated.csv')
    prompts = {}
    with open(src) as f:
        for r in csv.DictReader(f):
            prompts[r['id']] = r['prompt']

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'prompt', 'answer', 'type', 'thinking', 'correct'])
        for r in results:
            if r.get('correct'):
                w.writerow([
                    r['id'],
                    prompts.get(r['id'], ''),
                    r['answer'],
                    r['type'],
                    r.get('thinking', ''),
                    r['correct'],
                ])

    correct = sum(1 for r in results if r.get('correct'))
    print(f"CSV: {correct} correct rows -> {csv_path}")


if __name__ == '__main__':
    main()
