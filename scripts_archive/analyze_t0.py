#!/usr/bin/env python3
"""Quick analysis of cot_t0.jsonl quality distribution."""
import json
from collections import Counter

gold, silver, discard, trunc = Counter(), Counter(), Counter(), Counter()
total_by_type = Counter()

with open("data/cot_t0.jsonl") as f:
    for line in f:
        r = json.loads(line)
        t = r.get("type", "?")
        total_by_type[t] += 1
        cc = r.get("correct_count", 0)
        consist = r.get("answers_consistent", False)
        agree = r.get("agreement_ratio", 0)
        samples = r.get("samples", [])
        is_trunc = any(s.get("finish_reason") == "length" for s in samples)
        if is_trunc:
            trunc[t] += 1
        if cc >= 2 and consist and agree >= 0.67:
            gold[t] += 1
        elif cc >= 1:
            silver[t] += 1
        else:
            discard[t] += 1

header = "{:12s} {:>6s} {:>6s} {:>6s} {:>7s} {:>7s}".format(
    "Type", "Total", "Gold", "Silver", "Discard", "Trunc%"
)
print(header)
print("-" * len(header))
for t in sorted(total_by_type):
    tot = total_by_type[t]
    g, s, d, tr = gold[t], silver[t], discard[t], trunc[t]
    pct = tr / tot * 100 if tot else 0
    print("{:12s} {:6d} {:6d} {:6d} {:7d} {:6.1f}%".format(t, tot, g, s, d, pct))

tot_all = sum(total_by_type.values())
print("-" * len(header))
print("{:12s} {:6d} {:6d} {:6d} {:7d}".format(
    "TOTAL", tot_all, sum(gold.values()), sum(silver.values()), sum(discard.values())
))
