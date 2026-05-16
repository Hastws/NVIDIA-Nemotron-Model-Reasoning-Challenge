"""
Test different prompt strategies for bit_ops and symbol to find what works best.
Key problem: model's thinking exhausts all 7680 tokens → finish_reason=length, 0 output.
Goal: prompt that guides efficient, structured reasoning.
"""
import os, re, math, time
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
client = OpenAI(base_url=API_BASE, api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY"))

def call_api(prompt, enable_thinking=True):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, top_p=1.0, max_tokens=7680,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        timeout=180,
    )
    c = resp.choices[0]
    content = c.message.content or ""
    return content, c.finish_reason

def extract_answer(text):
    # boxed
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    # after </think>
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

def strip_wonderland(prompt):
    return re.sub(r"^In Alice's Wonderland, ", "", prompt.strip())

# ============================================================================
# PROMPT STRATEGIES
# ============================================================================

# --- BIT_OPS strategies ---
BIT_STRATEGIES = {
    "baseline": lambda p: p + "\nThink briefly, then output the answer.",
    
    "recipe": lambda p: strip_wonderland(p) + """
Solve step by step:
1. Try simple operations first: XOR with a constant, bit rotation (left/right by 1-7), NOT, reverse bits, swap nibbles.
2. If none fit, try two-step combos: rotation+XOR, NOT+rotation, etc.
3. Verify your hypothesis on ALL examples before answering.
4. Output ONLY the 8-bit binary result.""",

    "concise_recipe": lambda p: strip_wonderland(p) + """
Check these transforms in order, verify against ALL examples:
- XOR with constant (first_input XOR first_output)
- Rotate left/right by 1-7 bits  
- NOT (flip all bits)
- Reverse bit order
- Two-step combo of above
Output only the 8-bit binary answer.""",

    "cot_compact": lambda p: strip_wonderland(p) + """
Approach: For each example, compute input XOR output. If XOR is constant → that's the rule. Otherwise try bit rotations (1-7), NOT, reverse. Test hypothesis against all examples. Give only the 8-bit binary result.""",

    "no_overthink": lambda p: strip_wonderland(p) + """
Important: Keep your reasoning under 500 words. Try common bit operations (XOR, rotation, NOT, reverse, swap) and their 2-step combinations. Verify on all examples. Output only the 8-bit binary answer.""",

    "direct_systematic": lambda p: strip_wonderland(p) + """
Systematically test:
1. Compute XOR of first pair. Check if same XOR works for all → answer = query XOR constant.
2. For rotation N=1..7: check if rotating all inputs left by N gives outputs.
3. Try NOT, reverse bits, nibble swap.
4. Try pairs: rotate then XOR, XOR then rotate, NOT then rotate.
Be concise. Output only the 8-bit result.""",
}

# --- SYMBOL strategies ---
SYMBOL_STRATEGIES = {
    "baseline": lambda p: p + "\nThink briefly, then output the answer.",
    
    "recipe": lambda p: strip_wonderland(p) + """
Solve step by step:
1. Input is always 5 characters. Check if output is formed by REMOVING certain positions from the input.
2. If inputs contain digits and operators: each operator symbol maps to a different arithmetic operation (add, subtract, multiply, concatenate, etc.). Find the mapping.
3. If pure symbols: check if it's a character substitution cipher, or if certain characters are deleted.
4. Verify your rule on all examples, then apply to the query.
Output only the final answer.""",

    "concise_recipe": lambda p: strip_wonderland(p) + """
Patterns to check (input is always 5 chars):
- Position removal: which positions are kept in output? Consistent across examples?
- Character removal: which chars are always deleted?
- If digits+operators: each operator maps to an arithmetic op (+-×÷, concat, etc.)
- Character substitution: each input char → fixed output char
Verify on all examples. Output only the answer.""",

    "cot_compact": lambda p: strip_wonderland(p) + """
Approach: Compare input (5 chars) to output. Identify which input characters survive to output and in what order. If numbers are involved, figure out what arithmetic each operator symbol performs. Keep reasoning brief. Output only the final answer.""",

    "no_overthink": lambda p: strip_wonderland(p) + """
Important: Keep reasoning under 500 words. The input is always 5 characters. Look at which characters or positions from input appear in the output. Check for character deletion, substitution, or operator-based arithmetic. Verify on examples. Output only the answer.""",

    "structural": lambda p: strip_wonderland(p) + """
Analysis method:
1. Note input length (5) vs output length for each example.
2. For each output char, find which input position it came from.
3. Determine the rule: which positions are kept? Are chars transformed?
4. Apply rule to query.
Be concise. Output only the result.""",
}

# ============================================================================
# LOAD DATA
# ============================================================================
import polars as pl
df = pl.read_csv('competition_data/train.csv')

bit_df = df.filter(pl.col('prompt').str.contains('bit manipulation'))
sym_df = df.filter(pl.col('prompt').str.contains('transformation rules'))

# Sample 3 per type for testing
N_SAMPLES = 3

print("=" * 90)
print("BIT_OPS PROMPT STRATEGY COMPARISON")
print("=" * 90)

bit_results = {s: [] for s in BIT_STRATEGIES}
for si in range(N_SAMPLES):
    row = bit_df.row(si * 50, named=True)  # spread out samples
    gold = row['answer']
    prompt = row['prompt']
    
    print(f"\n{'─'*90}")
    print(f"Sample {si}: gold={gold}")
    print(f"{'─'*90}")
    
    for sname, sfunc in BIT_STRATEGIES.items():
        p = sfunc(prompt)
        r, f = call_api(p, True)
        a = extract_answer(r)
        ok = answers_match(a, gold)
        
        think_part = r.split('</think>')[0] if '</think>' in r else ""
        after = r.split('</think>')[-1].strip() if '</think>' in r else r.strip()
        
        status = '✓' if ok else ('TRUNC' if f == 'length' else '✗')
        print(f"  {sname:20s}: finish={f:6s} think={len(think_part):5d} ans={str(a):20s} {status}")
        if ok:
            bit_results[sname].append(1)
        else:
            bit_results[sname].append(0)
        
        time.sleep(0.5)  # Rate limit

print(f"\n{'─'*90}")
print("BIT_OPS SUMMARY:")
for sname in BIT_STRATEGIES:
    score = sum(bit_results[sname])
    print(f"  {sname:20s}: {score}/{N_SAMPLES}")

print("\n\n" + "=" * 90)
print("SYMBOL PROMPT STRATEGY COMPARISON")
print("=" * 90)

sym_results = {s: [] for s in SYMBOL_STRATEGIES}
for si in range(N_SAMPLES):
    row = sym_df.row(si * 50, named=True)
    gold = row['answer']
    prompt = row['prompt']
    
    print(f"\n{'─'*90}")
    print(f"Sample {si}: gold={gold}")
    print(f"{'─'*90}")

    for sname, sfunc in SYMBOL_STRATEGIES.items():
        p = sfunc(prompt)
        r, f = call_api(p, True)
        a = extract_answer(r)
        ok = answers_match(a, gold)
        
        think_part = r.split('</think>')[0] if '</think>' in r else ""
        after = r.split('</think>')[-1].strip() if '</think>' in r else r.strip()
        
        status = '✓' if ok else ('TRUNC' if f == 'length' else '✗')
        print(f"  {sname:20s}: finish={f:6s} think={len(think_part):5d} ans={str(a):20s} {status}")
        if ok:
            sym_results[sname].append(1)
        else:
            sym_results[sname].append(0)
        
        time.sleep(0.5)

print(f"\n{'─'*90}")
print("SYMBOL SUMMARY:")
for sname in SYMBOL_STRATEGIES:
    score = sum(sym_results[sname])
    print(f"  {sname:20s}: {score}/{N_SAMPLES}")
