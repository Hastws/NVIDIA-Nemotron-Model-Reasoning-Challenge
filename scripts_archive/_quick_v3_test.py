#!/usr/bin/env python3
"""Quick test: generate 6 CoT rows with v3 prompt, verify ASCII output."""
import json, csv, os, requests, time, sys

API_URL = 'https://api.deepseek.com/v1/chat/completions'
API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')

# Quick API test first
print("Testing API...", flush=True)
t0 = time.time()
r = requests.post(API_URL,
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    json={'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': 'say ok'}], 'max_tokens': 5},
    timeout=30)
print(f"  API status={r.status_code} in {time.time()-t0:.1f}s", flush=True)
if r.status_code != 200:
    print(f"  ERROR: {r.text[:200]}")
    sys.exit(1)

# Now run generate_cot
print("\nRunning generate_cot.py --n 6 --workers 2 ...", flush=True)
os.system("python3 scripts/generate_cot.py --n 6 --workers 2 --output train_cot_v3.jsonl 2>&1")

# Check result
try:
    data = [json.loads(l) for l in open('data/train_cot_v3.jsonl')]
    print(f"\nResult: {len(data)} rows")
    for d in data:
        cot = d['cot']
        non_ascii = [c for c in cot if ord(c) > 127]
        status = 'ASCII-clean' if not non_ascii else f'HAS UNICODE: {non_ascii}'
        print(f"  [{d['type']:10s}] {d['words']:3d}w | {status} | {d['cot'][:80]}...")
except FileNotFoundError:
    print("ERROR: output file not found!")
