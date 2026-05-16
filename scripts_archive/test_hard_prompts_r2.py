"""
Round 2: More aggressive prompt strategies for hard types.
Focus on: system message, ultra-compact prompts, answer-first framing.
"""
import os, re, math, time
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
client = OpenAI(base_url=API_BASE, api_key=os.environ.get("NVIDIA_API_KEY"))

def call_api(prompt, enable_thinking=True, system=None, max_tokens=7680):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=0.0, top_p=1.0, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        timeout=180,
    )
    c = resp.choices[0]
    content = c.message.content or ""
    return content, c.finish_reason

def extract_answer(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    for m in reversed(matches):
        if m.strip(): return m.strip()
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

def strip_w(p):
    return re.sub(r"^In Alice's Wonderland, ", "", p.strip())

# ============================================================================
import polars as pl
df = pl.read_csv('competition_data/train.csv')
bit_df = df.filter(pl.col('prompt').str.contains('bit manipulation'))
sym_df = df.filter(pl.col('prompt').str.contains('transformation rules'))

# Strategies to test (both types share same meta-strategies)
STRATEGIES = {
    # R1 winner: ultra-compact algorithmic hint
    "cot_compact": {
        "system": None,
        "bit_suffix": "\nApproach: For each example, compute input XOR output. If XOR is constant, that's the rule. Otherwise try bit rotations (1-7), NOT, reverse. Test against all examples. Output only the 8-bit binary result.",
        "sym_suffix": "\nApproach: Compare input (5 chars) to output. Identify which chars survive and their positions. If numbers+operators, each operator maps to an arithmetic op. Keep reasoning brief. Output only the answer.",
    },
    
    # System message: force brevity
    "sys_brief": {
        "system": "You are a pattern matching expert. Think step by step but be extremely concise. Limit your reasoning to under 200 words. Always output a final answer.",
        "bit_suffix": "\nOutput only the 8-bit binary result.",
        "sym_suffix": "\nOutput only the result.",
    },
    
    # System message: no thinking needed
    "sys_direct": {
        "system": "You are an expert at pattern recognition. Analyze the examples, find the rule, and immediately give the answer. Do not overthink. Be direct.",
        "bit_suffix": "\nThe 8-bit binary answer is:",
        "sym_suffix": "\nThe answer is:",
    },
    
    # Answer-first framing: force model to commit immediately
    "answer_first": {
        "system": None,
        "bit_suffix": "\nAnswer (8-bit binary only, nothing else):",
        "sym_suffix": "\nAnswer (only the result, nothing else):",
    },
    
    # XOR-first for bit_ops, position analysis for symbol
    "algorithmic": {
        "system": None,
        "bit_suffix": "\nStep 1: XOR first input with first output = constant C.\nStep 2: Check if input XOR C = output for all examples.\nStep 3: If yes, answer = query XOR C. If no, try rotate left by 1,2,...7.\nAnswer (8 bits):",
        "sym_suffix": "\nStep 1: Compare each input-output pair character by character.\nStep 2: Note which positions are kept vs removed.\nStep 3: Apply the pattern to the query.\nAnswer:",
    },
    
    # Minimal: just say "answer directly"
    "minimal": {
        "system": None,
        "bit_suffix": "\nDon't think, directly output the 8-bit binary answer.",
        "sym_suffix": "\nDon't think, directly output the answer.",
    },
}

N = 5  # samples per type

# ============================================================================
print("=" * 90)
print("BIT_OPS - ROUND 2")
print("=" * 90)

bit_scores = {s: 0 for s in STRATEGIES}
bit_trunc = {s: 0 for s in STRATEGIES}
for si in range(N):
    row = bit_df.row(si * 30, named=True)
    gold = row['answer']
    prompt_raw = strip_w(row['prompt'])
    
    print(f"\n{'─'*70} Sample {si}: gold={gold}")
    
    for sname, cfg in STRATEGIES.items():
        p = prompt_raw + cfg["bit_suffix"]
        r, f = call_api(p, True, system=cfg.get("system"), max_tokens=7680)
        a = extract_answer(r)
        ok = answers_match(a, gold)
        
        think_part = r.split('</think>')[0] if '</think>' in r else r
        after = r.split('</think>')[-1].strip() if '</think>' in r else r.strip()
        
        if ok: bit_scores[sname] += 1
        if f == 'length': bit_trunc[sname] += 1
        
        tag = '✓' if ok else ('TRUNC' if f == 'length' else '✗')
        # Show first 80 chars of output for non-truncated
        out_preview = after[:80].replace('\n', ' ') if after else '(empty)'
        print(f"  {sname:15s}: {tag:5s} think={len(think_part):5d}  out={out_preview}")
        time.sleep(0.3)

print(f"\n{'─'*70}")
print("BIT_OPS SUMMARY:")
for s in STRATEGIES:
    print(f"  {s:15s}: {bit_scores[s]}/{N} correct, {bit_trunc[s]}/{N} truncated")

# ============================================================================
print("\n\n" + "=" * 90)
print("SYMBOL - ROUND 2")
print("=" * 90)

sym_scores = {s: 0 for s in STRATEGIES}
sym_trunc = {s: 0 for s in STRATEGIES}
for si in range(N):
    row = sym_df.row(si * 30, named=True)
    gold = row['answer']
    prompt_raw = strip_w(row['prompt'])
    
    print(f"\n{'─'*70} Sample {si}: gold={gold}")
    
    for sname, cfg in STRATEGIES.items():
        p = prompt_raw + cfg["sym_suffix"]
        r, f = call_api(p, True, system=cfg.get("system"), max_tokens=7680)
        a = extract_answer(r)
        ok = answers_match(a, gold)
        
        think_part = r.split('</think>')[0] if '</think>' in r else r
        after = r.split('</think>')[-1].strip() if '</think>' in r else r.strip()
        
        if ok: sym_scores[sname] += 1
        if f == 'length': sym_trunc[sname] += 1
        
        tag = '✓' if ok else ('TRUNC' if f == 'length' else '✗')
        out_preview = after[:80].replace('\n', ' ') if after else '(empty)'
        print(f"  {sname:15s}: {tag:5s} think={len(think_part):5d}  out={out_preview}")
        time.sleep(0.3)

print(f"\n{'─'*70}")
print("SYMBOL SUMMARY:")
for s in STRATEGIES:
    print(f"  {s:15s}: {sym_scores[s]}/{N} correct, {sym_trunc[s]}/{N} truncated")

print("\n\n=== OVERALL BEST ===")
for s in STRATEGIES:
    total = bit_scores[s] + sym_scores[s]
    total_trunc = bit_trunc[s] + sym_trunc[s]
    print(f"  {s:15s}: {total}/{2*N} correct, {total_trunc}/{2*N} truncated")
