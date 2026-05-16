#!/usr/bin/env python3
"""Test deepseek-reasoner (thinking mode) vs deepseek-chat on hard types."""
import csv, time, requests, os, json

API_KEY = os.environ['DEEPSEEK_API_KEY']
API_URL = 'https://api.deepseek.com/v1/chat/completions'

with open('data/train_annotated.csv') as f:
    all_rows = list(csv.DictReader(f))

# Pick harder samples: symbol (unsolvable), bit_ops (complex), cipher
# Also pick one where match=False to see if reasoner can handle
samples = []
for r in all_rows:
    if r['match'] == 'True' and r.get('solution_process'):
        if r['type'] == 'symbol' and len(samples) < 1:
            samples.append(r)
        elif r['type'] == 'bit_ops' and len([s for s in samples if s['type']=='bit_ops']) < 1:
            samples.append(r)
        elif r['type'] == 'cipher' and len([s for s in samples if s['type']=='cipher']) < 1:
            samples.append(r)
        elif r['type'] == 'gravity' and len([s for s in samples if s['type']=='gravity']) < 1:
            samples.append(r)
    if len(samples) >= 4:
        break

# Also grab an unsolvable symbol
for r in all_rows:
    if r['type'] == 'symbol' and r['match'] == 'False':
        samples.append(r)
        break

SYSTEM = """You are a rule compiler.

Convert the given problem and its solution into a compact symbolic program using [TYPE:ARGS] tokens.

Token types:
[CONST:name=value] — a constant derived from examples
[FORMULA:expr] — a formula using constants  
[SCALE:factor] — linear scaling: output = factor * input
[CIPHER_MAP:a>b,c>d,...] — substitution cipher (all pairs in one token)
[Bn:OP(args)] — bit n rule, ops: NOT,AND,OR,XOR,XNOR,CONST, args: IN0-IN7
[OP:sym→FUNC] — custom operator definition
[ROMAN_GREEDY] — Roman numeral conversion
[UNSOLVABLE] — no consistent rule found

Rules:
- ONLY [TYPE:ARGS] tokens, one per line
- No natural language, no execution steps
- Max 10 tokens
- For cipher: ONE [CIPHER_MAP:...] token
- For operators: ONE [OP:...] token"""

MODELS = ['deepseek-chat', 'deepseek-reasoner']

for s in samples:
    prompt = s['prompt']
    if len(prompt) > 600:
        prompt = prompt[:450] + "\n...\n" + prompt[-150:]
    
    sol = s.get('solution_process', '(none)')
    user_msg = (
        f"Problem:\n{prompt}\n\n"
        f"Machine solution: {sol}\n"
        f"Answer: {s['answer']}"
    )
    
    print(f"{'='*70}")
    print(f"TYPE: {s['type']} | match={s['match']} | id={s['id']}")
    print(f"ORIG: {sol[:150]}")
    print()
    
    for model in MODELS:
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': user_msg},
            ],
            'max_tokens': 4096,
        }
        # reasoner doesn't support temperature
        if model == 'deepseek-chat':
            payload['temperature'] = 0.3
        
        t0 = time.time()
        try:
            resp = requests.post(API_URL, headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
            }, json=payload, timeout=60)
            dt = time.time() - t0
            
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content']
                reasoning = data['choices'][0]['message'].get('reasoning_content', '')
                tokens = data.get('usage', {})
                comp_tok = tokens.get('completion_tokens', 0)
                reason_tok = tokens.get('completion_tokens_details', {}).get('reasoning_tokens', 0) if 'completion_tokens_details' in tokens else 0
                
                print(f"  [{model}] ({dt:.1f}s, {comp_tok}tok, reason={reason_tok}tok)")
                if reasoning:
                    # Show first 200 chars of reasoning
                    print(f"  THINK: {reasoning[:300]}...")
                print(f"  OUT:   {content}")
                print()
            else:
                print(f"  [{model}] ERROR {resp.status_code}: {resp.text[:200]}")
                print()
        except Exception as e:
            print(f"  [{model}] EXCEPTION: {e}")
            print()
        
        time.sleep(1)
