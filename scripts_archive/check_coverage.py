import json, csv

# Check cipher solver coverage
with open('data_archive/cipher_programmatic_cot.jsonl') as f:
    cipher_cot = [json.loads(l) for l in f]
print(f'cipher programmatic CoT: {len(cipher_cot)} samples')
print(f'  verified: {sum(1 for x in cipher_cot if x.get("verified", False))}')
lens = sorted(len(x["thinking"]) for x in cipher_cot)
print(f'  thinking len (chars): min={lens[0]}, median={lens[len(lens)//2]}, max={lens[-1]}')

# Check bit_ops
with open('data_archive/bit_ops_programmatic_cot.jsonl') as f:
    bit_ops_cot = [json.loads(l) for l in f]
print(f'\nbit_ops programmatic CoT: {len(bit_ops_cot)} samples')
print(f'  verified: {sum(1 for x in bit_ops_cot if x.get("verified", False))}')
lens2 = sorted(len(x["thinking"]) for x in bit_ops_cot)
print(f'  thinking len (chars): min={lens2[0]}, median={lens2[len(lens2)//2]}, max={lens2[-1]}')

# Check train.csv type distribution
with open('competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

type_counts = {}
for r in rows:
    p = r['prompt']
    if 'bit manipulation' in p:
        t = 'bit_ops'
    elif 'gravitational constant' in p or 'gravity' in p.lower():
        t = 'gravity'
    elif 'unit' in p.lower() and 'convert' in p.lower():
        t = 'unit_conv'
    elif 'encryption' in p or 'decrypt' in p:
        t = 'cipher'
    elif 'numeral system' in p or ('base' in p.lower() and 'numeral' in p.lower()):
        t = 'numeral'
    elif 'symbol' in p.lower() and 'transform' in p.lower():
        t = 'symbol'
    else:
        t = 'unknown'
    type_counts[t] = type_counts.get(t, 0) + 1

print(f'\ntrain.csv type distribution:')
for t, c in sorted(type_counts.items()):
    print(f'  {t}: {c}')
print(f'  total: {sum(type_counts.values())}')

print(f'\nSolver coverage rates:')
for t, total in sorted(type_counts.items()):
    if t == 'cipher':
        solved = len(cipher_cot)
        print(f'  {t}: {solved}/{total} = {solved/total*100:.1f}%')
    elif t == 'bit_ops':
        solved = len(bit_ops_cot)
        print(f'  {t}: {solved}/{total} = {solved/total*100:.1f}%')
    elif t == 'symbol':
        print(f'  {t}: 0/{total} = 0.0%')
    elif t in ('numeral', 'gravity', 'unit_conv'):
        print(f'  {t}: ~{total}/{total} = ~100% (programmatic)')

# Estimate token counts for thinking
print(f'\nEstimated token counts (chars/4):')
print(f'  cipher thinking: median ~{lens[len(lens)//2]//4} tokens, max ~{lens[-1]//4} tokens')
print(f'  bit_ops thinking: median ~{lens2[len(lens2)//2]//4} tokens, max ~{lens2[-1]//4} tokens')
