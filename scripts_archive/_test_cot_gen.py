#!/usr/bin/env python3
"""
Test DeepSeek CoT generation — one sample per type.
Goal: generate concise, correct, natural CoT from DSL rules.
"""
import os, json, csv, requests, time

API_URL = 'https://api.deepseek.com/v1/chat/completions'
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')

# ─── The System Prompt ─────────────────────────────────────────────────────
SYSTEM = """You are writing the internal reasoning a strong model would produce when solving a pattern-matching puzzle.

You receive: a puzzle (with worked examples), the underlying rule (DSL notation), and the correct answer.

YOUR TASK: Write a short chain-of-thought that derives the rule from the examples and applies it.

ABSOLUTE RULES:
1. NEVER mention "DSL", "rule provided", "symbolic rule", "given rule", or anything suggesting you were told the answer — write as if discovering the pattern yourself.
2. Ground every claim in concrete numbers from the examples.
3. For bit-operation puzzles: index bits 0 (rightmost/LSB) to 7 (leftmost/MSB). State the rule for each output bit in ONE line each (e.g., "Bit 3 copies input bit 7"). Do NOT exhaustively verify — check 1 example per bit max.
4. For cipher puzzles: show 3-4 letter derivations from examples, then state "Building full map: a→x, b→y, …". Do NOT trace every letter of the decryption.
5. For operator-symbol puzzles: test standard arithmetic against each operator, verify once, conclude.
6. End with the final answer value.

LENGTH: 40-120 words. HARD LIMIT: 150 words. Shorter is better."""

# ─── Hand-crafted golden examples per type ──────────────────────────────────
EXAMPLES = {
    'numeral': {
        'dsl': '[ROMAN_GREEDY]',
        'prompt_snippet': '57 -> LVII, 45 -> XLV, 78 -> LXXVIII ... convert 43',
        'answer': 'XLIII',
        'golden_cot': 'The examples are Arabic-to-Roman: 57=LVII, 45=XLV, 78=LXXVIII — standard Roman numeral notation. For 43: 40=XL, 3=III → XLIII.'
    },
    'gravity': {
        'dsl': '[CONST:g=16.5009]\n[FORMULA:d=0.5*g*t^2]',
        'prompt_snippet': 't=1.72s d=24.41m, t=3.23s d=86.08m, t=2.48s d=50.74m ... find d for t=2.17s given d=0.5*g*t²',
        'answer': '38.85',
        'golden_cot': 'The formula d = 0.5gt² is given, so I need g. From (1.72, 24.41): g = 2×24.41/1.72² = 48.82/2.958 ≈ 16.501. From (3.23, 86.08): g = 172.16/10.433 ≈ 16.502. Consistent g ≈ 16.5009. For t = 2.17: d = 0.5 × 16.5009 × 4.7089 ≈ 38.85.'
    },
    'unit_conv': {
        'dsl': '[SCALE:0.604080]',
        'prompt_snippet': '6.24 m becomes 3.77, 18.52 m becomes 11.19, 26.37 m becomes 15.93 ... convert 7.7 m',
        'answer': '4.65',
        'golden_cot': 'Computing output/input: 3.77/6.24 = 0.6042, 11.19/18.52 = 0.6042, 15.93/26.37 = 0.6041. Constant factor ≈ 0.6041. For 7.7: 7.7 × 0.6041 ≈ 4.65.'
    },
    'cipher': {
        'dsl': '[CIPHER_MAP:a>u,c>i,f>a,j>n,t>c,u>e,v>r,x>s,y>t]',
        'prompt_snippet': '"ysu" → "the", "fjtcujy" → "ancient" ... decrypt "qvcjtuxx"',
        'answer': 'princess',
        'golden_cot': 'Substitution cipher. From "ysu"→"the": y→t, s→h, u→e. From "fjtcujy"→"ancient": f→a, j→n, t→c, c→i. Building full map: a→u, c→i, f→a, j→n, q→p, t→c, u→e, v→r, x→s, y→t. Decrypting "qvcjtuxx": p-r-i-n-c-e-s-s → "princess".'
    },
    'bit_ops': {
        'dsl': '[B0:CONST(0)]\n[B1:CONST(0)]\n[B2:CONST(0)]\n[B3:IN7]\n[B4:IN0]\n[B5:IN1]\n[B6:IN2]\n[B7:IN3]',
        'prompt_snippet': '11011101->11010001, 00010111->01110000, 00010000->00000000 ... find output for 11101101',
        'answer': '11010001',
        'golden_cot': 'Indexing bits 0 (LSB) to 7 (MSB). Bits 0-2: always 0 across all examples → CONST(0). Bit 3 copies input bit 7. Bit 4 copies input bit 0. Bit 5 copies IN1, bit 6 copies IN2, bit 7 copies IN3. For 11101101 (bits: IN7=1,IN6=1,IN5=1,IN4=0,IN3=1,IN2=1,IN1=0,IN0=1): B7-B0 = IN3,IN2,IN1,IN0,IN7,0,0,0 = 1,1,0,1,0,0,0,1 → 11010001.'
    },
    'symbol': {
        'dsl': '[OP:/→ADD]\n[OP:{→SUB]',
        'prompt_snippet': '24/63=87, 22/92=114, 96{75=21, 50{12=38 ... find 41/85',
        'answer': '126',
        'golden_cot': 'Testing / with addition: 24+63=87 ✓, 22+92=114 ✓. Testing { with subtraction: 96−75=21 ✓, 50−12=38 ✓. So / means add, { means subtract. For 41/85: 41+85 = 126.'
    }
}

def call_api(system, user_msg, temperature=0.3):
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
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    data = resp.json()
    return data['choices'][0]['message']['content'].strip()

# Load real examples
with open('data/train_annotated.csv') as f:
    ann = {r['id']: r for r in csv.DictReader(f)}

dsl_data = {}
with open('data/train_dsl_rules.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        dsl_data[obj['id']] = obj

# Pick 2 real examples per type for more robust testing
import random
random.seed(42)
types_order = ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']
test_ids = {}
for t in types_order:
    candidates = [k for k, v in dsl_data.items() if v['type'] == t and v.get('dsl') and v['score'] > 0]
    random.shuffle(candidates)
    test_ids[t] = candidates[:2]

# Build few-shot examples string
def build_few_shot(exclude_type):
    shots = []
    for t in types_order:
        if t == exclude_type:
            continue
        ex = EXAMPLES[t]
        shots.append(
            f"Problem snippet: {ex['prompt_snippet']}\n"
            f"Answer: {ex['answer']}\n"
            f"Reasoning:\n{ex['golden_cot']}"
        )
    return "\n\n---\n\n".join(shots)

print("=" * 80)
print("Testing CoT generation (v2) — 2 samples per type")
print("=" * 80)

stats = {}
for t in types_order:
    stats[t] = []
    for idx, rid in enumerate(test_ids[t]):
        row = ann[rid]
        d = dsl_data[rid]
        
        few_shot = build_few_shot(t)
        user_msg = (
            f"Here are examples of good reasoning for similar puzzles:\n\n{few_shot}\n\n"
            f"---\n\n"
            f"Now write the reasoning for this puzzle:\n\n"
            f"Problem:\n{row['prompt']}\n\n"
            f"Symbolic rule: {d['dsl']}\n"
            f"Correct answer: {row['answer']}\n\n"
            f"Write your reasoning (40-120 words, flowing prose):"
        )
        
        print(f"\n{'='*60}")
        print(f"TYPE: {t} [{idx+1}/2] | ID: {rid}")
        print(f"DSL: {d['dsl']}")
        print(f"ANSWER: {row['answer']}")
        print(f"{'='*60}")
        
        t0 = time.time()
        cot = call_api(SYSTEM, user_msg, temperature=0.3)
        elapsed = time.time() - t0
        
        word_count = len(cot.split())
        char_count = len(cot)
        has_dsl_leak = any(w in cot.lower() for w in ['dsl', 'symbolic rule', 'rule provided', 'rule says', 'rule tells'])
        has_answer = row['answer'].strip() in cot
        
        status = '✅' if word_count <= 150 and not has_dsl_leak and has_answer else '⚠️'
        if word_count > 150:
            status = '❌ TOO LONG'
        if has_dsl_leak:
            status = '❌ DSL LEAK'
        
        stats[t].append({'words': word_count, 'chars': char_count, 'status': status})
        
        print(f"{status} | {word_count} words, {char_count} chars, {elapsed:.1f}s")
        print(cot)
        print()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for t in types_order:
    avg_words = sum(s['words'] for s in stats[t]) / len(stats[t])
    statuses = ', '.join(s['status'] for s in stats[t])
    print(f"  {t:12s}: avg {avg_words:.0f} words | {statuses}")
