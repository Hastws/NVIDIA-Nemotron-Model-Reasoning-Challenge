"""
Round 3: Efficient test with shorter max_tokens for screening.
If model truncates at 2048, it'll truncate at 7680 too.
Only promote non-truncated prompts to full test.
"""
import os, re, math, time
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
client = OpenAI(base_url=API_BASE, api_key=os.environ.get("NVIDIA_API_KEY"))

def call_api(prompt, enable_thinking=True, system=None, max_tokens=2048):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=0.0, top_p=1.0, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        timeout=120,
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

import polars as pl
df = pl.read_csv('competition_data/train.csv')
bit_df = df.filter(pl.col('prompt').str.contains('bit manipulation'))
sym_df = df.filter(pl.col('prompt').str.contains('transformation rules'))

# --- Key strategies to test ---
STRATS = {
    # Baseline: just the raw prompt, no hint
    "raw": {
        "sys": None,
        "suf": "",
    },
    # Compact algorithm hint (R1 winner for bit_ops)
    "algo": {
        "sys": None,
        "suf": "\nApproach: Compute input XOR output for first pair. If constant across all pairs, answer=query XOR constant. Otherwise try rotate left/right 1-7, NOT, reverse. Verify all examples. Output only the answer.",
    },
    # System: pattern expert, be brief
    "sys_expert": {
        "sys": "You are a pattern matching expert. Be extremely concise in your reasoning. Limit thinking to under 300 words. Always output a final answer.",
        "suf": "\nOutput only the answer.",
    },
    # Direct answer prompt
    "direct": {
        "sys": None,
        "suf": "\nDon't think, directly output the answer.",
    },
    # Forced format
    "forced": {
        "sys": None,
        "suf": "\nAnswer:",
    },
}

N = 3

for type_name, type_df, label in [("BIT_OPS", bit_df, "bit"), ("SYMBOL", sym_df, "sym")]:
    print(f"\n{'='*80}")
    print(f"{type_name} (max_tokens=2048 screening)")
    print(f"{'='*80}")
    
    scores = {s: 0 for s in STRATS}
    truncs = {s: 0 for s in STRATS}
    
    for si in range(N):
        row = type_df.row(si * 40, named=True)
        gold = row['answer']
        base = strip_w(row['prompt'])
        
        print(f"\n  Sample {si}: gold={gold}")
        
        for sname, cfg in STRATS.items():
            p = base + cfg["suf"]
            t0 = time.time()
            r, f = call_api(p, True, system=cfg.get("sys"), max_tokens=2048)
            dt = time.time() - t0
            a = extract_answer(r)
            ok = answers_match(a, gold)
            
            if ok: scores[sname] += 1
            if f == 'length': truncs[sname] += 1
            
            tag = '✓' if ok else ('T' if f == 'length' else '✗')
            print(f"    {sname:12s}: {tag} ans={str(a)[:30]:30s} ({dt:.1f}s)")
    
    print(f"\n  {'─'*60}")
    print(f"  {type_name} Summary (max_tokens=2048):")
    for s in STRATS:
        print(f"    {s:12s}: {scores[s]}/{N} correct, {truncs[s]}/{N} truncated")

# Now test the most promising non-truncating strategies with full 7680
print(f"\n\n{'='*80}")
print("FULL TEST (max_tokens=7680) - 'direct' and 'forced' on more samples")
print("Best anti-truncation strategies only")
print(f"{'='*80}")

for type_name, type_df in [("BIT_OPS", bit_df), ("SYMBOL", sym_df)]:
    print(f"\n  {type_name}:")
    for sname in ["direct", "forced", "algo"]:
        correct = 0
        trunc = 0
        for si in range(5):
            row = type_df.row(si * 20, named=True)
            gold = row['answer']
            base = strip_w(row['prompt'])
            p = base + STRATS[sname]["suf"]
            r, f = call_api(p, True, system=STRATS[sname].get("sys"), max_tokens=7680)
            a = extract_answer(r)
            ok = answers_match(a, gold)
            if ok: correct += 1
            if f == 'length': trunc += 1
            tag = '✓' if ok else ('T' if f == 'length' else '✗')
            print(f"    {sname:12s} #{si}: {tag} ans={str(a)[:30]:30s} gold={gold}")
        print(f"    {sname:12s} TOTAL: {correct}/5 correct, {trunc}/5 truncated")
