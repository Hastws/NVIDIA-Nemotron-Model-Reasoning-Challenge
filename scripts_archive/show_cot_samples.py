import pandas as pd

df = pd.read_csv('data/sft_full_cot.csv')

types_seen = set()
for idx in range(len(df)):
    prompt = df.iloc[idx]['prompt']
    thinking = df.iloc[idx]['thinking']
    answer = str(df.iloc[idx]['answer'])
    
    if 'bit manipulation' in prompt and 'bit_ops' not in types_seen:
        types_seen.add('bit_ops')
    elif 'gravitational' in prompt and 'gravity' not in types_seen:
        types_seen.add('gravity')
    elif 'unit conversion' in prompt and 'unit_conv' not in types_seen:
        types_seen.add('unit_conv')
    elif 'encryption' in prompt and 'cipher' not in types_seen:
        types_seen.add('cipher')
    elif 'numeral' in prompt and 'numeral' not in types_seen:
        types_seen.add('numeral')
    else:
        continue
    
    t = list(types_seen)[-1]
    print(f"=== {t} (row {idx}) ===")
    print(f"Answer: {answer}")
    print(f"FULL Thinking:")
    print(thinking)
    print()
    
    if len(types_seen) >= 5:
        break
