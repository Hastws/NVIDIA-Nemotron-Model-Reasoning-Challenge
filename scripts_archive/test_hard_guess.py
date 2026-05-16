"""
Test base model's raw guessing ability for bit_ops and symbol.
Key: enable_thinking=False forces direct output (no thinking trap).
This tells us the model's "intuition" ceiling for these types.
Then we know what to put in LoRA training data.
"""
import os, re, math, time, csv
from openai import OpenAI

API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-nano-30b-a3b"
client = OpenAI(base_url=API_BASE, api_key=os.environ.get("NVIDIA_API_KEY"))

def call_api(prompt, enable_thinking=False, max_tokens=512):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, top_p=1.0, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        timeout=120,
    )
    c = resp.choices[0]
    return c.message.content or "", c.finish_reason

def extract_answer(text):
    # Try boxed
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    for m in reversed(matches):
        if m.strip(): return m.strip()
    # After </think>
    if '</think>' in text:
        text = text.split('</think>')[-1]
    # Last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return None
    ans = lines[-1]
    # Clean up markdown/formatting
    ans = re.sub(r'^\*\*(.+?)\*\*\.?$', r'\1', ans)
    ans = re.sub(r'^`+(.+?)`+$', r'\1', ans)
    ans = re.sub(r'^Answer:\s*', '', ans, flags=re.IGNORECASE)
    return ans.strip() if ans.strip() else None

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

# Load data
with open('competition_data/train.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

bit_rows = [r for r in rows if 'bit manipulation' in r['prompt']]
sym_rows = [r for r in rows if 'transformation rules' in r['prompt']]

SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# ============================================================================
# Test 10 samples each, enable_thinking=False, with eval suffix (as in real eval)
# This is the "raw guess" mode
# ============================================================================
N = 10

print("=" * 80)
print("BIT_OPS: enable_thinking=FALSE, with \\boxed suffix (eval mode)")
print("10 samples, testing raw guessing ability")
print("=" * 80)

bit_correct = 0
for i in range(N):
    row = bit_rows[i * 20]
    gold = row['answer']
    prompt = row['prompt'] + SUFFIX
    
    r, f = call_api(prompt, enable_thinking=False, max_tokens=512)
    a = extract_answer(r)
    ok = answers_match(a, gold)
    if ok: bit_correct += 1
    
    tag = '✓' if ok else '✗'
    print(f"  #{i}: {tag} gold={gold} pred={str(a)[:40]} finish={f} len={len(r)}")

print(f"\n  BIT_OPS no-think: {bit_correct}/{N} ({100*bit_correct/N:.0f}%)")

print(f"\n{'='*80}")
print("SYMBOL: enable_thinking=FALSE, with \\boxed suffix (eval mode)")
print("10 samples, testing raw guessing ability")
print(f"{'='*80}")

sym_correct = 0
for i in range(N):
    row = sym_rows[i * 20]
    gold = row['answer']
    prompt = row['prompt'] + SUFFIX
    
    r, f = call_api(prompt, enable_thinking=False, max_tokens=512)
    a = extract_answer(r)
    ok = answers_match(a, gold)
    if ok: sym_correct += 1
    
    tag = '✓' if ok else '✗'
    print(f"  #{i}: {tag} gold={gold!r} pred={str(a)[:40]!r} finish={f} len={len(r)}")

print(f"\n  SYMBOL no-think: {sym_correct}/{N} ({100*sym_correct/N:.0f}%)")

# ============================================================================
# Now test: enable_thinking=TRUE but with "guess immediately" prompt
# This simulates what LoRA-trained model should do
# ============================================================================
print(f"\n\n{'='*80}")
print("BIT_OPS: enable_thinking=TRUE + 'guess quickly' prompt")
print("Testing if prompt can prevent thinking overflow")
print(f"{'='*80}")

GUESS_SUFFIX = "\nThis is a pattern matching puzzle. Make your best guess quickly - do not exhaustively analyze every possibility. Just try the most likely transformation and output your answer."

bit2_correct = 0
bit2_trunc = 0
for i in range(N):
    row = bit_rows[i * 20]
    gold = row['answer']
    prompt = strip_w(row['prompt']) + GUESS_SUFFIX + SUFFIX
    
    r, f = call_api(prompt, enable_thinking=True, max_tokens=7680)
    a = extract_answer(r)
    ok = answers_match(a, gold)
    if ok: bit2_correct += 1
    if f == 'length': bit2_trunc += 1
    
    think_len = len(r.split('</think>')[0]) if '</think>' in r else len(r)
    tag = '✓' if ok else ('T' if f == 'length' else '✗')
    print(f"  #{i}: {tag} gold={gold} pred={str(a)[:40]} think={think_len} finish={f}")

print(f"\n  BIT_OPS guess+think: {bit2_correct}/{N}, {bit2_trunc}/{N} truncated")

print(f"\n{'='*80}")
print("SYMBOL: enable_thinking=TRUE + 'guess quickly' prompt")
print(f"{'='*80}")

sym2_correct = 0
sym2_trunc = 0
for i in range(N):
    row = sym_rows[i * 20]
    gold = row['answer']
    prompt = strip_w(row['prompt']) + GUESS_SUFFIX + SUFFIX
    
    r, f = call_api(prompt, enable_thinking=True, max_tokens=7680)
    a = extract_answer(r)
    ok = answers_match(a, gold)
    if ok: sym2_correct += 1
    if f == 'length': sym2_trunc += 1
    
    think_len = len(r.split('</think>')[0]) if '</think>' in r else len(r)
    tag = '✓' if ok else ('T' if f == 'length' else '✗')
    print(f"  #{i}: {tag} gold={gold!r} pred={str(a)[:40]!r} think={think_len} finish={f}")

print(f"\n  SYMBOL guess+think: {sym2_correct}/{N}, {sym2_trunc}/{N} truncated")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")
print(f"  BIT_OPS no-think (raw guess):  {bit_correct}/{N}")
print(f"  BIT_OPS think+guess prompt:    {bit2_correct}/{N} ({bit2_trunc} truncated)")
print(f"  SYMBOL  no-think (raw guess):  {sym_correct}/{N}")
print(f"  SYMBOL  think+guess prompt:    {sym2_correct}/{N} ({sym2_trunc} truncated)")
print(f"\n  → no-think total: {bit_correct+sym_correct}/{2*N}")
print(f"  → think+guess total: {bit2_correct+sym2_correct}/{2*N}")
