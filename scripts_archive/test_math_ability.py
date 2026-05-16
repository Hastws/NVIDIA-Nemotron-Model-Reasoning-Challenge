"""
Test base model math ability:
- Remove "In Alice's Wonderland, " prefix only
- No \boxed{} suffix
- 4 easy types: "Don't think, directly output the answer"
- 2 hard types (bit_ops, symbol): "Think briefly, then output the answer"
- Let model figure out the type itself
"""
import os, json, re, math
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"

api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
client = OpenAI(base_url=API_BASE, api_key=api_key)

# Type detection (same as training pipeline)
def detect_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p: return 'bit_ops'
    elif 'gravitational' in p: return 'gravity'
    elif 'unit conversion' in p: return 'unit_conv'
    elif 'encryption' in p: return 'cipher'
    elif 'numeral system' in p: return 'numeral'
    elif 'transformation rules' in p: return 'symbol'
    return 'unknown'

HARD_TYPES = {'bit_ops', 'symbol'}

def strip_wonderland(prompt):
    """Only remove 'In Alice's Wonderland, ' prefix. Keep everything else."""
    # The prefix is always "In Alice's Wonderland, " followed by the actual problem
    stripped = re.sub(r"^In Alice's Wonderland, ", "", prompt.strip())
    return stripped

def build_prompt(original_prompt, ptype):
    """Strip wonderland + add type-appropriate suffix."""
    core = strip_wonderland(original_prompt)
    if ptype in HARD_TYPES:
        return core + "\nThink briefly, then output the answer."
    else:
        return core + "\nDon't think, directly output the answer."

def call_api(prompt, enable_thinking):
    messages = [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=0.0, top_p=1.0, max_tokens=7680,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        timeout=180,
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    return content, choice.finish_reason

def extract_answer(text):
    """Try to extract answer - check \boxed first, then last line."""
    # Try boxed
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    # Fallback: last non-empty line after </think>
    if '</think>' in text:
        text = text.split('</think>')[-1]
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else None

def answers_match(pred, gold):
    if pred is None: return False
    pred, gold = pred.strip(), gold.strip()
    if pred.lower() == gold.lower(): return True
    # Try numeric comparison
    try: return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except: pass
    # Check if pred contains gold
    if gold.lower() in pred.lower(): return True
    return False

# Load data
import polars as pl
df = pl.read_csv('competition_data/train.csv')

# Sample 1 per type
for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    rows = df.filter(pl.col('prompt').str.contains(
        {'numeral':'numeral system','gravity':'gravitational','unit_conv':'unit conversion',
         'cipher':'encryption','bit_ops':'bit manipulation','symbol':'transformation rules'}[t]
    ))
    row = rows.row(0, named=True)
    
    original = row['prompt']
    gold = row['answer']
    ptype = t
    
    new_prompt = build_prompt(original, ptype)
    
    print(f"\n{'='*80}")
    print(f"TYPE: {t}  |  GOLD: {gold}")
    print(f"{'='*80}")
    print(f"\n[NEW PROMPT]")
    print(new_prompt)
    
    # Test with original prompt (no boxed, no stripping, as baseline)
    orig_suffix = '\nDon\'t think, directly output the answer.' if ptype not in HARD_TYPES else '\nThink briefly, then output the answer.'
    print(f"\n--- ORIGINAL (with Wonderland, thinking=True) ---")
    r1, f1 = call_api(original + orig_suffix, True)
    a1 = extract_answer(r1)
    ok1 = answers_match(a1, gold)
    think1 = r1.split('</think>')[0] if '</think>' in r1 else ""
    after1 = r1.split('</think>')[-1].strip() if '</think>' in r1 else r1.strip()
    print(f"  finish={f1}  len={len(r1)}  think_len={len(think1)}  answer={a1}  {'✓' if ok1 else '✗'}")
    if after1: print(f"  output: {after1[:300]}")

    # Test with stripped prompt (always thinking=True)
    print(f"\n--- STRIPPED (no Wonderland, thinking=True) ---")
    r2, f2 = call_api(new_prompt, True)
    a2 = extract_answer(r2)
    ok2 = answers_match(a2, gold)
    think2 = r2.split('</think>')[0] if '</think>' in r2 else ""
    after2 = r2.split('</think>')[-1].strip() if '</think>' in r2 else r2.strip()
    print(f"  finish={f2}  len={len(r2)}  think_len={len(think2)}  answer={a2}  {'✓' if ok2 else '✗'}")
    if after2: print(f"  output: {after2[:300]}")
    
    print()
