#!/usr/bin/env python3
"""Test DeepSeek rewrite on 6 samples (1 per type)."""
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

SYSTEM = (
    "You are rewriting a compact machine-generated solution process into a clean, "
    "standardized step-by-step reasoning procedure.\n\n"
    "FORMAT:\n"
    "Step 1: <action>\n"
    "Step 2: <action>\n"
    "...\n\n"
    "RULES:\n"
    "1. Each step is ONE short sentence describing a concrete action or computation.\n"
    "2. Use 2-5 steps. No fluff, no meta-commentary, no preamble.\n"
    "3. Include key values/numbers in the steps (e.g. 'g=15.90', 'factor=0.6636').\n"
    "4. The final step should state the computed result.\n"
    "5. Output ONLY the steps. Nothing else — no answer box, no summary."
)

TYPE_EXAMPLES = {
    'bit_ops': (
        "EXAMPLE OUTPUT:\n"
        "Step 1: Identify per-bit rules: b0=XNOR(in[1],in[7]), b1-b6=NOT of in[2]-in[7], b7=1\n"
        "Step 2: Apply rules to input 00110100\n"
        "Step 3: Combine bits to get 10010111"
    ),
    'gravity': (
        "EXAMPLE OUTPUT:\n"
        "Step 1: Compute g from each example using g=2d/t^2\n"
        "Step 2: Average to get g=15.90\n"
        "Step 3: Compute d=0.5*15.90*4.41^2=154.62"
    ),
    'unit_conv': (
        "EXAMPLE OUTPUT:\n"
        "Step 1: Compute conversion factor f=output/input for each example\n"
        "Step 2: Average to get f=0.6636\n"
        "Step 3: Compute 25.09*0.6636=16.65"
    ),
    'cipher': (
        "EXAMPLE OUTPUT:\n"
        "Step 1: Build substitution mapping from examples: b->t, f->o, g->s, ...\n"
        "Step 2: Apply mapping to encrypted text\n"
        "Step 3: Decoded result is cat imagines book"
    ),
    'numeral': (
        "EXAMPLE OUTPUT:\n"
        "Step 1: Identify target system as Roman numerals\n"
        "Step 2: Decompose 38 = 30+5+3 = XXX+V+III = XXXVIII"
    ),
    'symbol': (
        "EXAMPLE OUTPUT:\n"
        "Step 1: Identify operator * as concatenation from examples\n"
        "Step 2: Apply concatenation to get the result"
    ),
}

for s in samples:
    example = TYPE_EXAMPLES.get(s['type'], '')
    prompt = s['prompt']
    if len(prompt) > 500:
        prompt = prompt[:400] + "\n...\n" + prompt[-100:]

    user_msg = (
        f"Problem type: {s['type']}\n\n"
        f"Problem (abbreviated):\n{prompt}\n\n"
        f"Machine solution: {s['solution_process']}\n"
        f"Correct answer: {s['answer']}\n\n"
        f"{example}\n\n"
        f"Now rewrite the solution process into Step 1/2/3... format. Only output the steps."
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
        print(f"NEW:  {content}")
        print()
    else:
        print(f"ERROR {resp.status_code}: {resp.text[:300]}")

    time.sleep(1)
