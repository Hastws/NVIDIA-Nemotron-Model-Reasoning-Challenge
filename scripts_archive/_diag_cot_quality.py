#!/usr/bin/env python3
"""CoT v2 质量检查：每类抽3条打印完整 thinking，统计长度分布。"""
import json
import random
from collections import defaultdict

def main():
    random.seed(42)

    records = []
    with open('data/cot_v2.jsonl') as f:
        for line in f:
            records.append(json.loads(line))

    # Group by type
    by_type = defaultdict(list)
    for r in records:
        by_type[r['type']].append(r)

    print('=' * 70)
    print(f'CoT v2 Quality Check — {len(records)} total records')
    print('=' * 70)

    # 1. Overall stats
    print(f'\nType distribution:')
    for t in sorted(by_type.keys()):
        items = by_type[t]
        lengths = [len(r['thinking']) for r in items]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        print(f'  {t:<12}: {len(items):5d} records | thinking len: avg={avg_len:.0f}, min={min_len}, max={max_len}')

    # 2. Per-type samples
    for t in sorted(by_type.keys()):
        items = by_type[t]
        samples = random.sample(items, min(3, len(items)))
        print(f'\n{"="*70}')
        print(f'Type: {t} — {len(items)} records, showing {len(samples)} samples')
        print(f'{"="*70}')

        for i, s in enumerate(samples):
            print(f'\n--- [{t} sample {i+1}] ID={s["id"]} ---')
            print(f'Answer: {s["answer"]}')
            print(f'Thinking ({len(s["thinking"])} chars):')
            print(s['thinking'])
            print()

    # 3. Answer format check
    print(f'\n{"="*70}')
    print('Answer format consistency check:')
    print('='*70)
    for t in sorted(by_type.keys()):
        items = by_type[t]
        # Check if thinking ends with the answer
        ends_with_answer = sum(1 for r in items if r['thinking'].strip().endswith(r['answer']))
        contains_result = sum(1 for r in items if 'Result:' in r['thinking'] or 'result:' in r['thinking'].lower())
        print(f'  {t:<12}: ends_with_answer={ends_with_answer}/{len(items)}, contains "Result:"={contains_result}/{len(items)}')

    # 4. Thinking token estimate (rough: chars / 4)
    print(f'\n{"="*70}')
    print('Estimated thinking tokens (chars/4):')
    print('='*70)
    total_chars = sum(len(r['thinking']) for r in records)
    print(f'  Total thinking chars: {total_chars:,}')
    print(f'  Estimated tokens: ~{total_chars // 4:,}')
    for t in sorted(by_type.keys()):
        items = by_type[t]
        chars = sum(len(r['thinking']) for r in items)
        print(f'  {t:<12}: {chars:,} chars (~{chars // 4:,} tokens)')


if __name__ == '__main__':
    main()
