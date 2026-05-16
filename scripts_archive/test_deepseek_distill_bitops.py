"""
Test: DeepSeek distillation for bit_ops.

Strategy:
  Give DeepSeek our programmatic CoT (rules + derivation) for a bit_ops problem,
  ask it to "think through" the problem naturally, then output \boxed{answer}.
  
  Goal: Generate natural-sounding training data for Nemotron SFT.

Usage:
  python scripts/test_deepseek_distill_bitops.py --n 5
"""
import os
import re
import csv
import sys
import math
import time
import json
import argparse
import random

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-reasoner"  # R1 model with thinking

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
_BINARY_RE = re.compile(r'^[01]+$')

def extract_answer(text):
    if not text:
        return "NOT_FOUND"
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    return "NOT_FOUND"

def verify(gold, predicted):
    gold = str(gold).strip()
    predicted = str(predicted).strip()
    if predicted.lower() == gold.lower():
        return True
    if len(gold) > 1 and _BINARY_RE.match(gold):
        return False
    try:
        return math.isclose(float(gold), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except:
        return False


def call_deepseek_stream(client, messages, temperature=0.0, max_tokens=8192):
    """Call DeepSeek R1 with streaming to capture reasoning_content (thinking)."""
    thinking_parts = []
    content_parts = []
    
    stream = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # DeepSeek R1 streams reasoning_content in delta
        rc = getattr(delta, 'reasoning_content', None)
        if rc:
            thinking_parts.append(rc)
        if delta.content:
            content_parts.append(delta.content)
    
    thinking = ''.join(thinking_parts)
    content = ''.join(content_parts)
    return thinking, content


def call_nemotron_stream(client, messages, temperature=0.0, max_tokens=3584):
    """Call Nemotron with streaming."""
    thinking_parts = []
    content_parts = []
    
    stream = client.chat.completions.create(
        model=NEMOTRON_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        extra_body={"thinking": {"type": "enabled"}},
    )
    
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        rc = getattr(delta, 'reasoning_content', None)
        if rc:
            thinking_parts.append(rc)
        if delta.content:
            content_parts.append(delta.content)
    
    thinking = ''.join(thinking_parts)
    content = ''.join(content_parts)
    return thinking, content


# ═══════════════════════════════════════════════════════════════════════════════
#  DISTILLATION PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

DISTILL_SYSTEM = """You are solving a bit manipulation puzzle. You will be given:
1. The puzzle (input/output examples + a query)
2. The solution process that explains the rules

Your task: Read and understand the solution, then solve the puzzle yourself. Show your reasoning step by step, and put your final answer in \\boxed{}.

Important:
- Follow the rules given in the solution process
- Apply them to the query input to get the output
- Be concise but show your work
- Output the 8-bit binary result in \\boxed{}"""

def build_distill_prompt(prompt, thinking, answer):
    """Build the distillation prompt: problem + our CoT."""
    return f"""{prompt}
{PROMPT_SUFFIX}

--- Solution Process (for reference) ---
{thinking}"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5, help='Number of samples to test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify-nemotron', action='store_true', 
                        help='Also test if Nemotron can use the distilled thinking')
    args = parser.parse_args()

    random.seed(args.seed)

    # Load bit_ops data with CoT
    with open('data/sft_thinking.csv', 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    bit_rows = [r for r in rows if r['type'] == 'bit_ops' and r.get('thinking', '').strip()]
    random.shuffle(bit_rows)
    samples = bit_rows[:args.n]

    print(f"=== DeepSeek Distillation Test: bit_ops ===")
    print(f"Samples: {args.n}, Model: {DEEPSEEK_MODEL}")
    print()

    ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
    
    nm_client = None
    if args.verify_nemotron:
        nm_client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_API_BASE)

    results = []
    for i, row in enumerate(samples):
        prompt = row['prompt']
        gold = row['answer']
        our_cot = row['thinking']

        print(f"{'='*70}")
        print(f"[{i+1}/{args.n}] id={row['id']}, gold={gold}")
        print(f"{'='*70}")
        
        # --- Our programmatic CoT ---
        print(f"\n📋 Our CoT ({len(our_cot)}c):")
        print(our_cot)
        
        # --- Step 1: DeepSeek distillation ---
        print(f"\n🤖 Calling DeepSeek R1...")
        t0 = time.time()
        
        user_msg = build_distill_prompt(prompt, our_cot, gold)
        messages = [
            {"role": "system", "content": DISTILL_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        
        try:
            ds_thinking, ds_content = call_deepseek_stream(ds_client, messages)
            ds_time = time.time() - t0
            ds_answer = extract_answer(ds_content)
            ds_correct = verify(gold, ds_answer)
            
            print(f"\n💭 DeepSeek Thinking ({len(ds_thinking)}c, {ds_time:.1f}s):")
            print(ds_thinking[:2000] if len(ds_thinking) > 2000 else ds_thinking)
            if len(ds_thinking) > 2000:
                print(f"  ... [truncated, {len(ds_thinking)}c total]")
            
            print(f"\n📝 DeepSeek Content ({len(ds_content)}c):")
            print(ds_content[:1000] if len(ds_content) > 1000 else ds_content)
            
            print(f"\n{'✅' if ds_correct else '❌'} DeepSeek: predicted={ds_answer}, gold={gold}")
            
        except Exception as e:
            print(f"❌ DeepSeek error: {e}")
            ds_thinking, ds_content, ds_answer, ds_correct, ds_time = "", "", "ERROR", False, 0
        
        result = {
            'id': row['id'],
            'gold': gold,
            'our_cot': our_cot,
            'our_cot_len': len(our_cot),
            'ds_thinking': ds_thinking,
            'ds_thinking_len': len(ds_thinking),
            'ds_content': ds_content,
            'ds_content_len': len(ds_content),
            'ds_answer': ds_answer,
            'ds_correct': ds_correct,
            'ds_time': ds_time,
        }
        
        # --- Step 2 (optional): Test if Nemotron can use the distilled thinking ---
        if args.verify_nemotron and ds_correct:
            print(f"\n🧪 Testing Nemotron with DeepSeek's content as thinking...")
            # Use DeepSeek's content (not thinking) as the CoT for Nemotron
            # Format: Give Nemotron the problem + DeepSeek's reasoning as system context
            nm_messages = [
                {"role": "user", "content": prompt + PROMPT_SUFFIX + 
                 f"\n\n--- Reasoning guidance ---\n{ds_content}"},
            ]
            try:
                nm_thinking, nm_content = call_nemotron_stream(nm_client, nm_messages)
                nm_answer = extract_answer(nm_content)
                nm_correct = verify(gold, nm_answer)
                print(f"  Nemotron thinking: {len(nm_thinking)}c")
                print(f"  Nemotron content: {nm_content[:300]}")
                print(f"  {'✅' if nm_correct else '❌'} Nemotron: predicted={nm_answer}")
                result['nm_answer'] = nm_answer
                result['nm_correct'] = nm_correct
            except Exception as e:
                print(f"  ❌ Nemotron error: {e}")
        
        results.append(result)
        print()

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY: DeepSeek Distillation for bit_ops")
    print(f"{'='*70}")
    
    ds_correct_count = sum(1 for r in results if r['ds_correct'])
    print(f"DeepSeek accuracy: {ds_correct_count}/{len(results)} = {100*ds_correct_count/len(results):.0f}%")
    
    thinking_lens = [r['ds_thinking_len'] for r in results if r['ds_thinking_len'] > 0]
    content_lens = [r['ds_content_len'] for r in results if r['ds_content_len'] > 0]
    if thinking_lens:
        print(f"DeepSeek thinking length: min={min(thinking_lens)}, avg={sum(thinking_lens)/len(thinking_lens):.0f}, max={max(thinking_lens)}")
    if content_lens:
        print(f"DeepSeek content length: min={min(content_lens)}, avg={sum(content_lens)/len(content_lens):.0f}, max={max(content_lens)}")
    
    our_lens = [r['our_cot_len'] for r in results]
    print(f"Our CoT length: min={min(our_lens)}, avg={sum(our_lens)/len(our_lens):.0f}, max={max(our_lens)}")
    
    if args.verify_nemotron:
        nm_correct_count = sum(1 for r in results if r.get('nm_correct', False))
        nm_total = sum(1 for r in results if 'nm_correct' in r)
        if nm_total:
            print(f"Nemotron (with distilled thinking): {nm_correct_count}/{nm_total}")


if __name__ == "__main__":
    main()
