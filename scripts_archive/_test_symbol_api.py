#!/usr/bin/env python3
"""Quick test: single NVIDIA API call with a real symbol problem."""
import csv, os, time
from openai import OpenAI

# Load one symbol problem
with open('data/train_dsl_rules.csv') as f:
    for row in csv.DictReader(f):
        if row['type'] == 'symbol' and not row['dsl_rules']:
            pid = row['id']
            prompt = row['prompt']
            answer = row['answer']
            break

print(f"ID: {pid}")
print(f"Answer: {answer}")
print(f"Prompt length: {len(prompt)} chars")
print()

# Test NVIDIA API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
    timeout=60.0,
)

t0 = time.time()
print("Calling NVIDIA API...")
try:
    resp = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=[
            {"role": "system", "content": "You are a puzzle solver. Explain the transformation rule briefly."},
            {"role": "user", "content": f"PUZZLE:\n{prompt}\n\nCORRECT ANSWER: {answer}\n\nExplain the rule."},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content or ""
    thinking = getattr(resp.choices[0].message, "reasoning_content", "") or ""
    print(f"NVIDIA OK in {elapsed:.1f}s")
    print(f"Thinking length: {len(thinking)}")
    print(f"Content length: {len(content)}")
    print(f"Content preview: {content[:500]}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"NVIDIA FAILED in {elapsed:.1f}s: {e}")

# Also test DeepSeek
print("\n" + "="*40)
client2 = OpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    timeout=60.0,
)

t0 = time.time()
print("Calling DeepSeek API...")
try:
    resp2 = client2.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a puzzle solver. Explain the transformation rule briefly."},
            {"role": "user", "content": f"PUZZLE:\n{prompt}\n\nCORRECT ANSWER: {answer}\n\nExplain the rule."},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    elapsed = time.time() - t0
    content2 = resp2.choices[0].message.content or ""
    print(f"DeepSeek OK in {elapsed:.1f}s")
    print(f"Content length: {len(content2)}")
    print(f"Content preview: {content2[:500]}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"DeepSeek FAILED in {elapsed:.1f}s: {e}")
