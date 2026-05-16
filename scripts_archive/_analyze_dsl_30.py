import json
from collections import Counter, defaultdict

with open('data/test_dsl_30.jsonl') as f:
    rows = [json.loads(l) for l in f]

scores = [r['score'] for r in rows]
print(f'Score distribution: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.1f}')

score_buckets = Counter()
for s in scores:
    if s >= 90: score_buckets['90+'] += 1
    elif s >= 80: score_buckets['80-89'] += 1
    elif s >= 70: score_buckets['70-79'] += 1
    elif s >= 60: score_buckets['60-69'] += 1
    else: score_buckets['<60'] += 1
print(f'Score buckets: {dict(sorted(score_buckets.items()))}')

type_scores = defaultdict(list)
for r in rows:
    type_scores[r['type']].append(r['score'])
for t, ss in sorted(type_scores.items()):
    print(f'  {t}: avg={sum(ss)/len(ss):.1f} n={len(ss)} scores={sorted(ss)}')

print()
for r in rows[:6]:
    print(f'--- {r["type"]} (id={r["id"]}) score={r["score"]} ---')
    print(f'DSL: {r["dsl"]}')
    print()

consensus_count = sum(1 for r in rows if r.get('consensus'))
print(f'Consensus (2+ gens agree): {consensus_count}/{len(rows)} ({consensus_count/len(rows)*100:.0f}%)')

dsl_lens = [len(r['dsl']) for r in rows]
print(f'DSL length (chars): min={min(dsl_lens)}, max={max(dsl_lens)}, avg={sum(dsl_lens)/len(dsl_lens):.0f}')

# Show all 4 gens for a couple examples
print('\n=== ALL GENS COMPARISON ===')
for r in rows[:3]:
    print(f'\n--- {r["type"]} (id={r["id"]}) best_score={r["score"]} ---')
    for i, g in enumerate(r['all_gens']):
        print(f'  gen{i} (score={g["score"]}): {g["text"]}')
