#!/usr/bin/env python3
"""Check programmatic CoT length distribution."""
import json
from collections import defaultdict

stats = defaultdict(lambda: {"count": 0, "think_lens": [], "ans_lens": []})
with open("data/programmatic_cot.jsonl") as f:
    for line in f:
        r = json.loads(line)
        t = r["type"]
        stats[t]["count"] += 1
        stats[t]["think_lens"].append(len(r["thinking"]))
        stats[t]["ans_lens"].append(len(r["computed_answer"]))

header = "{:12s} {:>6s} {:>10s} {:>10s} {:>10s}".format("Type", "Count", "Think_avg", "Think_min", "Think_max")
print(header)
print("-" * len(header))
for t in sorted(stats):
    s = stats[t]
    tl = s["think_lens"]
    print("{:12s} {:6d} {:10.0f} {:10d} {:10d}".format(
        t, s["count"], sum(tl)/len(tl), min(tl), max(tl)))

total = sum(s["count"] for s in stats.values())
all_lens = []
for s in stats.values():
    all_lens.extend(s["think_lens"])
print("-" * len(header))
print("{:12s} {:6d} {:10.0f} {:10d} {:10d}".format(
    "TOTAL", total, sum(all_lens)/len(all_lens), min(all_lens), max(all_lens)))

avg_tokens = sum(all_lens) / len(all_lens) / 4
max_tokens = max(all_lens) / 4
print()
print("Estimated avg tokens: {:.0f}".format(avg_tokens))
print("Estimated max tokens: {:.0f}".format(max_tokens))
print("All fit in 2048 seq_len: {}".format(max_tokens < 1800))
