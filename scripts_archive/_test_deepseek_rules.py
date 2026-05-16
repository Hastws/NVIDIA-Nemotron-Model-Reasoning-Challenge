#!/usr/bin/env python3
"""Test DeepSeek rule-inference-engine prompt on 6 samples (1 per type)."""
import csv, time, requests, os

API_KEY = os.environ['DEEPSEEK_API_KEY']
API_URL = 'https://api.deepseek.com/v1/chat/completions'

with open('data/train_annotated.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['match'] == 'True' and r.get('solution_process')]

seen = {}
samples = []
for r in rows:
    t = r['type']
    if t not in seen:
        seen[t] = True
        samples.append(r)
    if len(samples) == 6:
        break

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

for s in samples:
    prompt = s['prompt']
    if len(prompt) > 600:
        prompt = prompt[:450] + "\n...\n" + prompt[-150:]

    user_msg = (
        f"Problem:\n{prompt}\n\n"
        f"Machine solution: {s['solution_process']}\n"
        f"Answer: {s['answer']}"
    )

    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user_msg},
        ],
        'temperature': 0.3,
        'max_tokens': 300,
    }

    t0 = time.time()
    resp = requests.post(API_URL, headers={
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }, json=payload, timeout=30)
    dt = time.time() - t0

    if resp.status_code == 200:
        data = resp.json()
        content = data['choices'][0]['message']['content']
        tokens = data.get('usage', {}).get('completion_tokens', 0)
        print(f"=== {s['type']} ({dt:.1f}s, {tokens}tok) ===")
        print(f"ORIG: {s['solution_process'][:200]}")
        print(f"NEW:\n{content}")
        print()
    else:
        print(f"ERROR {resp.status_code}: {resp.text[:300]}")

    time.sleep(1)
