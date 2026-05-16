"""
Convert distill_output.jsonl → training CSV for weighted-loss SFT.
Only keeps correct samples. Output: data/sft_distill.csv
Columns: id, prompt, answer, thinking, type
"""
import json
import csv
import os

def detect_type(prompt):
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p): return "bit_ops"
    elif "encrypt" in p or "decrypt" in p or "cipher" in p: return "cipher"
    elif "gravit" in p: return "gravity"
    elif "numeral" in p or "wonderland numbers" in p: return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p): return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p): return "symbol"
    return "unknown"

input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'distill_output.jsonl')
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sft_distill.csv')

rows = []
with open(input_path) as f:
    for line in f:
        d = json.loads(line)
        if not d.get('correct'):
            continue
        think = (d.get('think') or '').strip()
        answer = (d.get('answer') or '').strip()
        if not answer:
            continue
        rows.append({
            'id': d['id'],
            'prompt': d['prompt'],
            'answer': answer,
            'thinking': think,
            'type': detect_type(d['prompt']),
        })

with open(output_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer', 'thinking', 'type'])
    w.writeheader()
    w.writerows(rows)

from collections import Counter
types = Counter(r['type'] for r in rows)
print(f"Total: {len(rows)} correct samples → {output_path}")
for t in sorted(types):
    print(f"  {t:12s}: {types[t]}")
