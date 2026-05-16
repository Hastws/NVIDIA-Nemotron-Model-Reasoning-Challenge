import json
from collections import defaultdict

stats = defaultdict(lambda: {'total':0, 'any_correct':0, 'all_correct':0, 'truncated':0})
with open('data_archive/cot_t0.jsonl') as f:
    for line in f:
        d = json.loads(line)
        t = d['type']
        stats[t]['total'] += 1
        if d['correct_count'] > 0:
            stats[t]['any_correct'] += 1
        if d['correct_count'] == d['n_samples']:
            stats[t]['all_correct'] += 1
        for s in d.get('samples', []):
            if s.get('finish_reason') == 'length':
                stats[t]['truncated'] += 1

print(f"{'Type':12s} {'Total':>6s} {'Any%':>7s} {'All%':>7s} {'Trunc%':>7s}")
for t in sorted(stats):
    s = stats[t]
    n = s['total']
    n_samp = n * 3
    print(f"{t:12s} {n:6d} {s['any_correct']/n*100:6.1f}% {s['all_correct']/n*100:6.1f}% {s['truncated']/n_samp*100:6.1f}%")

total = sum(s['total'] for s in stats.values())
any_c = sum(s['any_correct'] for s in stats.values())
all_c = sum(s['all_correct'] for s in stats.values())
trunc = sum(s['truncated'] for s in stats.values())
print(f"{'TOTAL':12s} {total:6d} {any_c/total*100:6.1f}% {all_c/total*100:6.1f}% {trunc/(total*3)*100:6.1f}%")
