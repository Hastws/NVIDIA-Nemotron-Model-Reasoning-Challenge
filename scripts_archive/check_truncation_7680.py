"""分析 cot_t0_v2.jsonl (max_tokens=7680) 的截断率"""
import json
from collections import defaultdict

with open('data_archive/cot_t0_v2.jsonl') as f:
    data = [json.loads(l) for l in f]

print(f"Total problems: {len(data)}")

# Per-type stats
stats = defaultdict(lambda: {"total": 0, "truncated": 0, "samples_total": 0, 
                              "samples_truncated": 0, "think_lens": []})

for d in data:
    t = d['type']
    stats[t]["total"] += 1
    
    samples = d.get('samples', [])
    any_truncated = False
    for s in samples:
        stats[t]["samples_total"] += 1
        fr = s.get('finish_reason', '')
        thinking = s.get('thinking', '')
        if fr == 'length':
            stats[t]["samples_truncated"] += 1
            any_truncated = True
        if thinking:
            stats[t]["think_lens"].append(len(thinking))
    
    if any_truncated:
        stats[t]["truncated"] += 1

print(f"\n{'Type':<12} {'Problems':>8} {'Truncated':>10} {'Trunc%':>8} | {'Samples':>8} {'S_Trunc':>8} {'S_Trunc%':>8} | {'Avg Think':>10} {'Max Think':>10}")
print("-" * 110)

for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    s = stats.get(t, {"total":0, "truncated":0, "samples_total":0, "samples_truncated":0, "think_lens":[]})
    total = s["total"]
    trunc = s["truncated"]
    trunc_pct = trunc/total*100 if total else 0
    s_total = s["samples_total"]
    s_trunc = s["samples_truncated"]
    s_trunc_pct = s_trunc/s_total*100 if s_total else 0
    avg_think = sum(s["think_lens"])/len(s["think_lens"]) if s["think_lens"] else 0
    max_think = max(s["think_lens"]) if s["think_lens"] else 0
    print(f"{t:<12} {total:>8} {trunc:>10} {trunc_pct:>7.1f}% | {s_total:>8} {s_trunc:>8} {s_trunc_pct:>7.1f}% | {avg_think:>10.0f} {max_think:>10}")

# Also check unknown types
for t in stats:
    if t not in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
        s = stats[t]
        print(f"{t:<12} {s['total']:>8} {s['truncated']:>10}")

# Summary
total_all = sum(s["total"] for s in stats.values())
trunc_all = sum(s["truncated"] for s in stats.values())
print(f"\n{'TOTAL':<12} {total_all:>8} {trunc_all:>10} {trunc_all/total_all*100:>7.1f}%")

# Also check how many samples per problem
sample_counts = [len(d.get('samples', [])) for d in data]
print(f"\nSamples per problem: min={min(sample_counts)}, max={max(sample_counts)}, avg={sum(sample_counts)/len(sample_counts):.1f}")

# Check data for cot_t0.jsonl too (the old one with max_tokens=3584)
print("\n\n=== cot_t0.jsonl (old, max_tokens=3584) ===")
try:
    with open('data_archive/cot_t0.jsonl') as f:
        data_old = [json.loads(l) for l in f]
    
    stats_old = defaultdict(lambda: {"total": 0, "truncated": 0, "samples_total": 0, "samples_truncated": 0})
    for d in data_old:
        t = d.get('type', 'unknown')
        stats_old[t]["total"] += 1
        samples = d.get('samples', [])
        any_trunc = False
        for s in samples:
            stats_old[t]["samples_total"] += 1
            if s.get('finish_reason', '') == 'length':
                stats_old[t]["samples_truncated"] += 1
                any_trunc = True
        if any_trunc:
            stats_old[t]["truncated"] += 1
    
    print(f"{'Type':<12} {'Problems':>8} {'Truncated':>10} {'Trunc%':>8} | {'Samples':>8} {'S_Trunc':>8} {'S_Trunc%':>8}")
    print("-" * 80)
    for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
        s = stats_old.get(t, {"total":0, "truncated":0, "samples_total":0, "samples_truncated":0})
        total = s["total"]
        trunc = s["truncated"]
        trunc_pct = trunc/total*100 if total else 0
        s_total = s["samples_total"]
        s_trunc = s["samples_truncated"]
        s_trunc_pct = s_trunc/s_total*100 if s_total else 0
        print(f"{t:<12} {total:>8} {trunc:>10} {trunc_pct:>7.1f}% | {s_total:>8} {s_trunc:>8} {s_trunc_pct:>7.1f}%")
    
    total_all = sum(s["total"] for s in stats_old.values())
    trunc_all = sum(s["truncated"] for s in stats_old.values())
    print(f"{'TOTAL':<12} {total_all:>8} {trunc_all:>10} {trunc_all/total_all*100:>7.1f}%")
except FileNotFoundError:
    print("cot_t0.jsonl not found")
