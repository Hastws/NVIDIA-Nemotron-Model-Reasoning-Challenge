import pandas as pd

train = pd.read_csv('competition_data/train.csv')

def infer_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or '8-bit binary' in p:
        return 'bit_ops'
    elif 'numeral system' in p:
        return 'numeral'
    elif 'encrypt' in p or 'decrypt' in p:
        return 'cipher'
    elif 'gravitational' in p or 'gravity' in p or 'free-fall' in p:
        return 'gravity'
    elif 'unit' in p and ('convert' in p or 'conversion' in p):
        return 'unit_conv'
    elif 'symbol' in p or 'transformation rule' in p:
        return 'symbol'
    else:
        return 'unknown'

train['type'] = train['prompt'].apply(infer_type)
print(train['type'].value_counts())
print(f"\nUnknown: {(train['type']=='unknown').sum()}")
if (train['type']=='unknown').sum() > 0:
    for _, r in train[train['type']=='unknown'].head(5).iterrows():
        print(f"  {r['prompt'][:200]}")
