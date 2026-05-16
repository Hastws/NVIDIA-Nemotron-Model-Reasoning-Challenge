import json
from collections import defaultdict

with open('data/test_dsl_30.jsonl') as f:
    rows = [json.loads(l) for l in f]

processed = [r for r in rows if r['score'] > 0]
print(f'Total rows: {len(rows)}, Processed with DSL: {len(processed)}')

scores = [r['score'] for r in processed]
print(f'Scores: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.1f}')

type_info = defaultdict(list)
for r in processed:
    type_info[r['type']].append(r)

for t in sorted(type_info):
    rr = type_info[t]
    ss = [r['score'] for r in rr]
    print(f'  {t}: n={len(rr)} avg_score={sum(ss)/len(ss):.1f}')
    for r in rr:
        dsl_preview = r['dsl'].replace('\n', ' | ')
        print(f'    score={r["score"]}: {dsl_preview[:120]}')

print()
print('=== ALL_GENS STRUCTURE ===')
r = processed[0]
print(f'all_gens count: {len(r["all_gens"])}')
print(f'all_gens[0] type: {type(r["all_gens"][0])}')
if isinstance(r['all_gens'][0], str):
    for i, g in enumerate(r['all_gens']):
        preview = g.replace('\n', ' | ')[:120]
        print(f'  gen{i}: {preview}')
elif isinstance(r['all_gens'][0], dict):
    for i, g in enumerate(r['all_gens']):
        preview = g['text'].replace('\n', ' | ')[:120]
        print(f'  gen{i} (score={g["score"]}): {preview}')
