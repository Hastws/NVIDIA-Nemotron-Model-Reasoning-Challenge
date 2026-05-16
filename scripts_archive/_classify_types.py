"""Test keyword-based type classification on train.csv prompts."""
import polars as pl
from collections import Counter

df = pl.read_csv('competition_data/train.csv')
samples = df['prompt'].to_list()

keywords = {
    'bit_ops': ['bit manipulation', 'binary number', '8-bit', 'bit shift', 'XOR', 'AND, OR, NOT'],
    'gravity': ['gravitational', 'gravity', 'celestial', 'planet', 'gravitational constant'],
    'unit_conv': ['unit conversion', 'convert the following measurement', 'secret unit'],
    'cipher': ['encryption', 'cipher', 'encrypt', 'decrypt', 'encoded', 'secret code'],
    'numeral': ['numeral system', 'Roman numeral', 'ancient numeral', 'number system'],
    'symbol': ['symbol', 'symbolic', 'equation', 'transformation rule', 'symbol manipulation'],
}

def classify(prompt):
    p_lower = prompt.lower()
    for t, kws in keywords.items():
        for kw in kws:
            if kw.lower() in p_lower:
                return t
    return None

type_counts = Counter()
unclassified_examples = []
for p in samples:
    t = classify(p)
    if t:
        type_counts[t] += 1
    else:
        unclassified_examples.append(p[:200])

print('Type distribution:')
for t, c in sorted(type_counts.items()):
    print(f'  {t}: {c}')
print(f'  unclassified: {len(unclassified_examples)}')
print(f'  total: {sum(type_counts.values()) + len(unclassified_examples)}')

if unclassified_examples:
    print(f'\nFirst 5 unclassified:')
    for ex in unclassified_examples[:5]:
        print(f'  {ex}\n')
