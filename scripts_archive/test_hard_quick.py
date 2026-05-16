"""
Quick focused test: 3 strategies × 2 samples × 2 types = 12 API calls
Key question: can ANY prompt strategy prevent thinking overflow on bit_ops/symbol?
"""
import os, re, math, time
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
client = OpenAI(base_url=API_BASE, api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY"))

def call_api(prompt, enable_thinking=True, system_msg=None):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=0.0, top_p=1.0, max_tokens=7680,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        timeout=300,
    )
    c = resp.choices[0]
    content = c.message.content or ""
    return content, c.finish_reason

def extract_answer(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    if '</think>' in text:
        text = text.split('</think>')[-1]
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else None

def answers_match(pred, gold):
    if pred is None: return False
    pred, gold = pred.strip(), gold.strip()
    if pred.lower() == gold.lower(): return True
    try: return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except: pass
    if gold.lower() in pred.lower(): return True
    return False

def strip_wl(p):
    return re.sub(r"^In Alice's Wonderland, ", "", p.strip())

# ============================================================================
# 3 STRATEGIES (most differentiated)
# ============================================================================
STRATEGIES = {
    # S1: thinking=True, original prompt, no guidance (baseline)
    "A_baseline_think": {
        "thinking": True,
        "system": None,
        "prompt": lambda p: p,
    },
    # S2: thinking=True, structured algorithm prompt
    "B_algo_guide": {
        "thinking": True,
        "system": None,
        "prompt": lambda p: strip_wl(p) + "\nCheck these transforms in order, stop at first match that works for ALL examples:\n1. XOR with constant (input[0] XOR output[0])\n2. Bit rotation left/right by 1-7\n3. NOT, reverse bits, swap nibbles\n4. Two-step combo of above\nVerify, then give ONLY the 8-bit binary answer.",
    },
    # S3: thinking=False, let model answer directly without internal reasoning
    "C_nothink_direct": {
        "thinking": False,
        "system": None,
        "prompt": lambda p: strip_wl(p) + "\nAnalyze the pattern and give only the answer.",
    },
}

# Symbol-specific variants
SYMBOL_STRATEGIES = {
    "A_baseline_think": {
        "thinking": True,
        "system": None,
        "prompt": lambda p: p,
    },
    "B_algo_guide": {
        "thinking": True,
        "system": None,
        "prompt": lambda p: strip_wl(p) + "\nAnalysis steps:\n1. Input is always 5 chars. Compare input/output lengths.\n2. Check if output = input with some positions removed.\n3. If digits present: each operator maps to an arithmetic op.\n4. If pure symbols: look for character substitution or deletion.\nVerify on all examples. Give ONLY the final answer.",
    },
    "C_nothink_direct": {
        "thinking": False,
        "system": None,
        "prompt": lambda p: strip_wl(p) + "\nAnalyze the pattern and give only the answer.",
    },
}

# ============================================================================
# LOAD DATA
# ============================================================================
import pandas as pd
df = pd.read_csv('competition_data/train.csv')

bit_df = df[df['prompt'].str.contains('bit manipulation', na=False)].reset_index(drop=True)
sym_df = df[df['prompt'].str.contains('transformation rules', na=False)].reset_index(drop=True)

N = 2  # samples per type

# ============================================================================
# TEST BIT_OPS
# ============================================================================
print("=" * 90)
print("BIT_OPS: 3 strategies × 2 samples")
print("=" * 90)

bit_scores = {s: 0 for s in STRATEGIES}
for si in range(N):
    row = bit_df.iloc[si * 100]
    gold = row['answer']
    prompt = row['prompt']
    print(f"\n--- Sample {si}: gold={gold} ---")
    
    for sname, cfg in STRATEGIES.items():
        p = cfg["prompt"](prompt)
        t0 = time.time()
        r, f = call_api(p, cfg["thinking"], cfg["system"])
        elapsed = time.time() - t0
        a = extract_answer(r)
        ok = answers_match(a, gold)
        
        think_part = r.split('</think>')[0] if '</think>' in r else ""
        after = r.split('</think>')[-1].strip() if '</think>' in r else r.strip()
        out_len = len(after)
        
        tag = 'OK' if ok else ('TRUNC' if f == 'length' else 'WRONG')
        print(f"  {sname:22s}: {tag:5s} finish={f:6s} think={len(think_part):5d}ch out={out_len:4d}ch ans={str(a)[:30]:30s} {elapsed:.1f}s")
        
        if ok:
            bit_scores[sname] += 1

# ============================================================================
# TEST SYMBOL
# ============================================================================
print("\n\n" + "=" * 90)
print("SYMBOL: 3 strategies × 2 samples")
print("=" * 90)

sym_scores = {s: 0 for s in SYMBOL_STRATEGIES}
for si in range(N):
    row = sym_df.iloc[si * 100]
    gold = row['answer']
    prompt = row['prompt']
    print(f"\n--- Sample {si}: gold={gold!r} ---")
    
    for sname, cfg in SYMBOL_STRATEGIES.items():
        p = cfg["prompt"](prompt)
        t0 = time.time()
        r, f = call_api(p, cfg["thinking"], cfg["system"])
        elapsed = time.time() - t0
        a = extract_answer(r)
        ok = answers_match(a, gold)
        
        think_part = r.split('</think>')[0] if '</think>' in r else ""
        after = r.split('</think>')[-1].strip() if '</think>' in r else r.strip()
        out_len = len(after)
        
        tag = 'OK' if ok else ('TRUNC' if f == 'length' else 'WRONG')
        print(f"  {sname:22s}: {tag:5s} finish={f:6s} think={len(think_part):5d}ch out={out_len:4d}ch ans={str(a)[:30]:30s} {elapsed:.1f}s")
        
        if ok:
            sym_scores[sname] += 1

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 90)
print("FINAL SUMMARY")
print("=" * 90)
print(f"\nBIT_OPS ({N} samples):")
for s, sc in bit_scores.items():
    print(f"  {s:22s}: {sc}/{N}")
print(f"\nSYMBOL ({N} samples):")
for s, sc in sym_scores.items():
    print(f"  {s:22s}: {sc}/{N}")
