import json
from collections import defaultdict
import statistics

def detect_type(prompt):
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p): return "bit_ops"
    elif "encrypt" in p or "decrypt" in p or "cipher" in p: return "cipher"
    elif "gravit" in p: return "gravity"
    elif "numeral" in p or "wonderland numbers" in p: return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p): return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p): return "symbol"
    return "unknown"

# Load all correct samples
correct = defaultdict(list)
with open('data/distill_output.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if not d.get('correct'):
            continue
        t = detect_type(d['prompt'])
        comp_tokens = d.get('usage', {}).get('completion_tokens', 0)
        prompt_tokens = d.get('usage', {}).get('prompt_tokens', 0)
        think_len = len(d.get('think') or '')
        correct[t].append({
            'id': d['id'],
            'think_len': think_len,
            'comp_tokens': comp_tokens,
            'prompt_tokens': prompt_tokens,
            'total_tokens': prompt_tokens + comp_tokens,
        })

print("=== ALL CORRECT SAMPLES: Token Budget Analysis ===\n")
print(f"{'Type':12s} {'N':>5s} {'CompTokAvg':>11s} {'CompTokMed':>11s} {'CompTokP25':>11s} {'TotalTokAvg':>12s}")
print("-" * 70)

grand_total_tokens = 0
grand_n = 0
for t in sorted(correct):
    items = correct[t]
    n = len(items)
    comp = [x['comp_tokens'] for x in items]
    total = [x['total_tokens'] for x in items]
    grand_total_tokens += sum(total)
    grand_n += n
    comp_s = sorted(comp)
    print(f"{t:12s} {n:5d} {statistics.mean(comp):11.0f} {statistics.median(comp):11.0f} "
          f"{comp_s[n//4]:11d} {statistics.mean(total):12.0f}")

print("-" * 70)
print(f"{'TOTAL':12s} {grand_n:5d}")
print(f"\nTotal training tokens (all correct): {grand_total_tokens:,}")
print(f"Average tokens per sample: {grand_total_tokens/grand_n:,.0f}")

# Compare with E1 baseline
e1_tokens = 600 * 80  # ~80 tokens per answer-only sample (prompt ~60 + answer ~20)
print(f"\nE1 baseline: 600 × ~80 = {e1_tokens:,} tokens")
print(f"All correct: {grand_n} × {grand_total_tokens//grand_n:,} = {grand_total_tokens:,} tokens")
print(f"Ratio: {grand_total_tokens/e1_tokens:.1f}x more training signal than E1")

# Scenario analysis
print("\n\n=== STRATEGY OPTIONS ===\n")

# Option A: All 3211
print(f"Option A: ALL correct ({grand_n} samples)")
print(f"  Total tokens: {grand_total_tokens:,}")
print(f"  vs E1: {grand_total_tokens/e1_tokens:.0f}x → HIGH risk of forgetting")

# Option B: Top 600 shortest
print(f"\nOption B: Shortest 600 (100/type, fill rest from large types)")
all_sorted = []
for t in correct:
    for item in correct[t]:
        item['type'] = t
        all_sorted.append(item)
all_sorted.sort(key=lambda x: x['comp_tokens'])

# Take shortest 600
selected_b = all_sorted[:600]
b_tokens = sum(x['total_tokens'] for x in selected_b)
b_types = defaultdict(int)
for x in selected_b:
    b_types[x['type']] += 1
print(f"  Total tokens: {b_tokens:,} ({b_tokens/e1_tokens:.1f}x E1)")
print(f"  Distribution: {dict(sorted(b_types.items()))}")

# Option C: Balanced per-type, prefer shorter
print(f"\nOption C: Balanced 600 (100/type, shortest first)")
selected_c = []
for t in sorted(correct):
    items = sorted(correct[t], key=lambda x: x['comp_tokens'])
    take = min(100, len(items))
    selected_c.extend(items[:take])
c_tokens = sum(x['total_tokens'] for x in selected_c)
c_types = defaultdict(int)
for x in selected_c:
    c_types[x['type']] += 1
print(f"  Total tokens: {c_tokens:,} ({c_tokens/e1_tokens:.1f}x E1)")
print(f"  Distribution: {dict(sorted(c_types.items()))}")
print(f"  Avg completion tokens: {statistics.mean([x['comp_tokens'] for x in selected_c]):.0f}")

# Option D: All correct but cap think length
print(f"\nOption D: ALL correct with think < 4000 chars")
selected_d = [x for t in correct for x in correct[t] if x['think_len'] < 4000]
d_tokens = sum(x['total_tokens'] for x in selected_d)
d_types = defaultdict(int)
for x in selected_d:
    d_types[x['type']] += 1
print(f"  Count: {len(selected_d)}")
print(f"  Total tokens: {d_tokens:,} ({d_tokens/e1_tokens:.1f}x E1)")
print(f"  Distribution: {dict(sorted(d_types.items()))}")

# Option E: All 3211, full data
print(f"\nOption E: ALL {grand_n} correct, full CoT (proposed)")
print(f"  Total tokens: {grand_total_tokens:,} ({grand_total_tokens/e1_tokens:.0f}x E1)")
for t in sorted(correct):
    items = correct[t]
    avg_comp = statistics.mean([x['comp_tokens'] for x in items])
    print(f"  {t:12s}: {len(items):5d} samples, avg {avg_comp:.0f} completion tokens")

# What LR adjustment compensates for Nx more data?
print(f"\n\n=== LR ADJUSTMENT for data volume ===")
print(f"E1: 600 samples × 1 epoch × LR=2e-4 = baseline")
print(f"If using {grand_n} samples with same total gradient magnitude:")
print(f"  LR = 2e-4 × (600/{grand_n}) = {2e-4 * 600/grand_n:.2e}")
print(f"  But token-weighted: LR = 2e-4 × ({e1_tokens}/{grand_total_tokens}) = {2e-4 * e1_tokens/grand_total_tokens:.2e}")
