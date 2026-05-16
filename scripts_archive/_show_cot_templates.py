import pandas as pd

df = pd.read_csv('data/sft_full_cot.csv')

# Proper type classification
def classify(p):
    pl = p.lower()
    if 'bit manipulation' in pl: return 'bit_ops'
    if 'gravit' in pl or 'planet' in pl: return 'gravity'
    if 'numeral system' in pl: return 'numeral'
    if 'encrypt' in pl or 'cipher' in pl or 'decrypt' in pl: return 'cipher'
    if 'unit' in pl and 'convert' in pl: return 'unit_conv'
    if 'secretly converted' in pl and ('measurement' in pl or 'weight' in pl or 'length' in pl or 'volume' in pl or 'temperature' in pl or 'distance' in pl):
        return 'unit_conv'
    return 'other'

df['type'] = df['prompt'].apply(classify)
print('=== sft_full_cot.csv type distribution ===')
print(df['type'].value_counts())
print()

# Show one example of each type with its CoT
for t in ['bit_ops', 'gravity', 'numeral', 'cipher', 'unit_conv', 'other']:
    subset = df[df['type'] == t]
    if len(subset) > 0:
        row = subset.iloc[0]
        print(f'=== {t} ({len(subset)} total) ===')
        print(f'Prompt: {row["prompt"][:150]}...')
        print(f'Think:\n{row["thinking"]}')
        print(f'Answer: {row["answer"]}')
        print()

# Also check CoT lengths per type
print('\n=== CoT character length by type ===')
for t in df['type'].unique():
    subset = df[df['type'] == t]
    lens = subset['thinking'].str.len()
    print(f'{t}: mean={lens.mean():.0f}, max={lens.max()}, min={lens.min()}')
