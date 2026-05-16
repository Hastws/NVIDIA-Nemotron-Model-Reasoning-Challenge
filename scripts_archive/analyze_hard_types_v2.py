#!/usr/bin/env python3
"""Analyze existing T0 sampling data for hard types."""
import json
from collections import defaultdict

stats = defaultdict(lambda: {"total": 0, "correct": 0, "truncated": 0, "think_lens": []})

with open("data/cot_t0_v2.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        t = d.get("type", "unknown")
        stats[t]["total"] += 1
        
        if d.get("correct", False):
            stats[t]["correct"] += 1
        
        thinking = d.get("thinking", "")
        if thinking:
            stats[t]["think_lens"].append(len(thinking))
            if len(thinking) > 3000:
                stats[t]["truncated"] += 1

print("cot_t0_v2.jsonl analysis:")
print(f"{'Type':<12} {'Total':>6} {'Correct':>8} {'Truncated':>10} {'Avg Think Len':>14}")
for t in sorted(stats.keys()):
    s = stats[t]
    avg_len = sum(s["think_lens"]) / max(len(s["think_lens"]), 1)
    print(f"{t:<12} {s['total']:>6} {s['correct']:>8} {s['truncated']:>10} {avg_len:>14.0f}")
total = sum(s["total"] for s in stats.values())
total_correct = sum(s["correct"] for s in stats.values())
print(f"{'TOTAL':<12} {total:>6} {total_correct:>8}")

# Show sample entry
print("\nFields available:")
with open("data/cot_t0_v2.jsonl", "r") as f:
    sample = json.loads(f.readline())
print(list(sample.keys()))

# Show a correct hard type sample if available
print("\n--- Looking for correct hard type samples ---")
for hard_type in ["bit_ops", "cipher", "symbol"]:
    count = 0
    with open("data/cot_t0_v2.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == hard_type and d.get("correct", False):
                count += 1
                if count == 1:
                    print(f"\n{hard_type} sample (correct):")
                    thinking = d.get("thinking", "")
                    print(f"  thinking length: {len(thinking)}")
                    print(f"  answer: {d.get('predicted', d.get('answer', 'N/A'))[:100]}")
                    print(f"  gold: {d.get('gold', 'N/A')[:100]}")
    print(f"  Total correct {hard_type}: {count}")
