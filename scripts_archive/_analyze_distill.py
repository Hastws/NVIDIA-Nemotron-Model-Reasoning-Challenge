import json
from collections import defaultdict

stats = defaultdict(lambda: {
    'total': 0, 'correct': 0, 'truncated': 0,
    'think_lens': [], 'answer_lens': [], 'prompt_tokens': [], 'completion_tokens': []
})

# Detect type from prompt
def detect_type(prompt):
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p):
        return "bit_ops"
    elif "encrypt" in p or "decrypt" in p or "cipher" in p:
        return "cipher"
    elif "gravit" in p:
        return "gravity"
    elif "numeral" in p or "wonderland numbers" in p:
        return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p):
        return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p):
        return "symbol"
    return "unknown"

with open('data/distill_output.jsonl') as f:
    for line in f:
        d = json.loads(line)
        t = detect_type(d['prompt'])
        s = stats[t]
        s['total'] += 1
        if d.get('correct'):
            s['correct'] += 1
        if d.get('finish_reason') == 'length':
            s['truncated'] += 1
        think = d.get('think') or ''
        answer = d.get('answer') or ''
        s['think_lens'].append(len(think))
        s['answer_lens'].append(len(answer))
        usage = d.get('usage', {})
        s['prompt_tokens'].append(usage.get('prompt_tokens', 0))
        s['completion_tokens'].append(usage.get('completion_tokens', 0))

import statistics

print(f"{'Type':12s} {'N':>5s} {'Correct':>8s} {'Acc%':>7s} {'Trunc':>6s} {'Trunc%':>7s} "
      f"{'ThinkAvg':>9s} {'ThinkMed':>9s} {'ThinkMax':>9s} {'AnsAvg':>7s} {'CompTok':>8s}")
print("-" * 110)

total_n = total_c = total_t = 0
all_think = []
all_comp = []

for t in sorted(stats):
    s = stats[t]
    n = s['total']
    c = s['correct']
    tr = s['truncated']
    total_n += n
    total_c += c
    total_t += tr
    
    tl = s['think_lens']
    al = s['answer_lens']
    ct = s['completion_tokens']
    all_think.extend(tl)
    all_comp.extend(ct)
    
    tavg = statistics.mean(tl) if tl else 0
    tmed = statistics.median(tl) if tl else 0
    tmax = max(tl) if tl else 0
    aavg = statistics.mean(al) if al else 0
    cavg = statistics.mean(ct) if ct else 0
    
    print(f"{t:12s} {n:5d} {c:8d} {c/n*100:6.1f}% {tr:6d} {tr/n*100:6.1f}% "
          f"{tavg:9.0f} {tmed:9.0f} {tmax:9d} {aavg:7.1f} {cavg:8.0f}")

print("-" * 110)
print(f"{'TOTAL':12s} {total_n:5d} {total_c:8d} {total_c/total_n*100:6.1f}% {total_t:6d} {total_t/total_n*100:6.1f}% "
      f"{statistics.mean(all_think):9.0f} {statistics.median(all_think):9.0f} {max(all_think):9d} "
      f"{'':>7s} {statistics.mean(all_comp):8.0f}")

# Correct-only stats
print(f"\n\n=== CORRECT SAMPLES ONLY ===")
correct_stats = defaultdict(lambda: {'count': 0, 'think_lens': [], 'completion_tokens': []})
with open('data/distill_output.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if not d.get('correct'):
            continue
        t = detect_type(d['prompt'])
        cs = correct_stats[t]
        cs['count'] += 1
        cs['think_lens'].append(len(d.get('think') or ''))
        cs['completion_tokens'].append(d.get('usage', {}).get('completion_tokens', 0))

print(f"{'Type':12s} {'N':>5s} {'ThinkAvg':>9s} {'ThinkMed':>9s} {'ThinkP25':>9s} {'ThinkP75':>9s} {'ThinkMax':>9s} {'CompTokAvg':>11s}")
print("-" * 80)
for t in sorted(correct_stats):
    cs = correct_stats[t]
    tl = cs['think_lens']
    ct = cs['completion_tokens']
    if not tl:
        continue
    tl_sorted = sorted(tl)
    p25 = tl_sorted[len(tl)//4]
    p75 = tl_sorted[3*len(tl)//4]
    print(f"{t:12s} {cs['count']:5d} {statistics.mean(tl):9.0f} {statistics.median(tl):9.0f} "
          f"{p25:9d} {p75:9d} {max(tl):9d} {statistics.mean(ct):11.0f}")
