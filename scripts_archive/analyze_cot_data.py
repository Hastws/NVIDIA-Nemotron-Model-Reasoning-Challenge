import json
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

with open("data/hybrid_cot_data.jsonl") as f:
    data = [json.loads(l) for l in f]

sources = Counter(d["source"] for d in data)
correct = sum(1 for d in data if d.get("api_correct"))
print(f"Total: {len(data)}")
print(f"Correct: {correct}")
print(f"Sources: {dict(sources)}")

correct_data = [d for d in data if d.get("api_correct")]
type_counts = Counter(detect_type(d["prompt"]) for d in correct_data)
print(f"\nCorrect samples by type:")
for t in sorted(type_counts):
    lens = [d["reasoning_len"] for d in correct_data if detect_type(d["prompt"]) == t]
    avg_len = sum(lens) / len(lens)
    print(f"  {t}: {type_counts[t]} samples, avg reasoning len: {avg_len:.0f} chars")

with open("data/multi_round_correct.jsonl") as f:
    mr = [json.loads(l) for l in f]
mr_types = Counter(detect_type(d["prompt"]) for d in mr)
print(f"\nMulti-round correct by type:")
for t in sorted(mr_types):
    print(f"  {t}: {mr_types[t]}")
