#!/usr/bin/env python3
"""Analyze hard type puzzle patterns to determine if programmatic solving is feasible."""
import csv
import json
from collections import Counter

types = {"bit_ops": [], "cipher": [], "symbol": []}

with open("competition_data/train.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompt = row["prompt"]
        p = prompt[:300].lower()
        if "8-bit binary" in p or ("bit" in p and "binary" in p):
            types["bit_ops"].append(row)
        elif "encrypt" in p or "cipher" in p or "secret code" in p:
            types["cipher"].append(row)
        elif "transformation" in p and ("equation" in p or "rule" in p):
            types["symbol"].append(row)

for t, rows in types.items():
    print(f"\n{'='*60}")
    print(f"TYPE: {t} ({len(rows)} samples)")
    print(f"{'='*60}")
    
    # Show 3 examples
    for i, row in enumerate(rows[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt (first 600 chars):")
        print(row["prompt"][:600])
        print(f"\nAnswer: {row['answer']}")
    
    # Answer pattern analysis
    answers = [r["answer"] for r in rows]
    if t == "bit_ops":
        lens = Counter(len(a) for a in answers)
        print(f"\nAnswer length distribution: {dict(lens)}")
        binary_count = sum(1 for a in answers if all(c in '01' for c in a))
        print(f"Binary answers: {binary_count}/{len(answers)}")
    elif t == "cipher":
        avg_len = sum(len(a) for a in answers) / len(answers)
        print(f"\nAvg answer length: {avg_len:.1f}")
        print(f"Sample answers: {answers[:5]}")
    elif t == "symbol":
        print(f"\nSample answers: {answers[:10]}")
