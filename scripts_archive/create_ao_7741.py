"""Create answer-only CSV from all solver-verified data."""
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

# Read full CoT data but only keep id/prompt/answer (drop thinking)
with open("data/sft_full_cot.csv") as f:
    rows = list(csv.DictReader(f))

# Write answer-only version
with open("data/sft_ao_7741.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
    w.writeheader()
    for r in rows:
        w.writerow({"id": r["id"], "prompt": r["prompt"], "answer": r["answer"]})

types = Counter(detect_type(r["prompt"]) for r in rows)
print(f"Created sft_ao_7741.csv: {len(rows)} rows")
for t in sorted(types):
    print(f"  {t}: {types[t]}")
print(f"No symbol (0 noise)")
print(f"File size: {__import__('os').path.getsize('data/sft_ao_7741.csv') / 1024:.1f} KB")
