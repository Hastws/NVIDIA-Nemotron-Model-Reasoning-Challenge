"""Deep analysis of CoT data quality by type."""
import csv
from collections import Counter

def detect_type(p):
    p = p[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p): return "bit_ops"
    elif "encrypt" in p or "cipher" in p: return "cipher"
    elif "gravit" in p: return "gravity"
    elif "numeral" in p or "wonderland numbers" in p: return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p): return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p): return "symbol"
    return "unknown"

with open("data/sft_full_cot.csv") as f:
    rows = list(csv.DictReader(f))

print(f"=== sft_full_cot.csv CoT Analysis ===")
print(f"Total: {len(rows)}, Columns: {list(rows[0].keys())}")
print()

# Group by type
by_type = {}
for r in rows:
    t = detect_type(r["prompt"])
    if t not in by_type: by_type[t] = []
    by_type[t].append(r)

for t in sorted(by_type):
    items = by_type[t]
    think_lens = [len(r.get("thinking", "")) for r in items]
    has_think = sum(1 for l in think_lens if l > 0)
    avg = sum(think_lens) / len(think_lens) if think_lens else 0
    med = sorted(think_lens)[len(think_lens)//2] if think_lens else 0
    mx = max(think_lens) if think_lens else 0
    mn = min(think_lens) if think_lens else 0
    print(f"{t}: {len(items)} samples")
    print(f"  Has thinking: {has_think}/{len(items)}")
    print(f"  Thinking len: min={mn}, med={med}, avg={avg:.0f}, max={mx}")
    
    # Token estimate (rough: 4 chars per token)
    think_tokens = [l // 4 for l in think_lens]
    ans_lens = [len(r.get("answer", "")) for r in items]
    # total = prompt + thinking + answer, estimate prompt tokens
    prompt_lens = [len(r["prompt"]) // 4 for r in items]
    total_tokens = [p + t + a for p, t, a in zip(prompt_lens, think_tokens, [l//4 for l in ans_lens])]
    over_1024 = sum(1 for t in total_tokens if t > 1024)
    over_2048 = sum(1 for t in total_tokens if t > 2048)
    print(f"  Est. total tokens: med={sorted(total_tokens)[len(total_tokens)//2]}, max={max(total_tokens)}")
    print(f"  Over 1024 tokens: {over_1024}/{len(items)} ({over_1024*100//len(items)}%)")
    print(f"  Over 2048 tokens: {over_2048}/{len(items)} ({over_2048*100//len(items)}%)")
    
    # Sample thinking content
    for i, item in enumerate(items[:2]):
        think = item.get("thinking", "")
        ans = item.get("answer", "")
        print(f"  Sample {i}: answer={repr(ans[:50])}, thinking[:150]={repr(think[:150])}")
    print()

# Also check: how many train.csv samples are NOT in sft_full_cot?
with open("competition_data/train.csv") as f:
    all_train = list(csv.DictReader(f))

cot_ids = set(r["id"] for r in rows)
missing = [r for r in all_train if r["id"] not in cot_ids]
missing_types = Counter(detect_type(r["prompt"]) for r in missing)
print(f"=== Samples NOT in sft_full_cot.csv ===")
print(f"Total missing: {len(missing)}/{len(all_train)}")
for t in sorted(missing_types):
    print(f"  {t}: {missing_types[t]}")
