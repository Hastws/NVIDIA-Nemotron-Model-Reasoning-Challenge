"""Quick analysis of data assets for evaluation analyst."""
import pandas as pd
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def classify(prompt):
    p = str(prompt).lower()
    if 'bit manipulation' in p or 'binary number' in p: return 'bit_ops'
    if 'encryption' in p or 'decrypt' in p: return 'cipher'
    if 'transformation rules' in p and ('equation' in p or '=' in str(prompt)[:200]): return 'symbol'
    if 'gravitational' in p or 'gravity' in p: return 'gravity'
    if 'unit' in p and 'convert' in p: return 'unit_conv'
    if 'numeral system' in p or 'roman' in p or 'number system' in p: return 'numeral'
    if 'ancient numeral' in p or 'numeral' in p: return 'numeral'
    return 'unknown'

# 1. E1 composition
e1 = pd.read_csv('data/sft_e1_replica.csv')
e1['type'] = e1['prompt'].apply(classify)
print("=== E1 (0.68 best) ===")
print(f"Total: {len(e1)}")
print(e1['type'].value_counts().to_string())

# 2. Full train
train = pd.read_csv('competition_data/train.csv')
train['type'] = train['prompt'].apply(classify)
print(f"\n=== Full train.csv ===")
print(f"Total: {len(train)}")
print(train['type'].value_counts().to_string())

# 3. Programmatic CoT data
print(f"\n=== Programmatic CoT Data ===")
cot_data = []
for fn in ['data/programmatic_cot.jsonl', 'data/cipher_programmatic_cot.jsonl', 'data/bit_ops_programmatic_cot.jsonl']:
    try:
        with open(fn) as f:
            for line in f:
                cot_data.append(json.loads(line))
        print(f"  Loaded: {fn}")
    except FileNotFoundError:
        print(f"  Not found: {fn}")

print(f"Total entries: {len(cot_data)}")
types = {}
for d in cot_data:
    t = d.get('type', 'unknown')
    types[t] = types.get(t, 0) + 1
for t, c in sorted(types.items()):
    print(f"  {t}: {c}")

thinking_lens = [len(d.get('thinking', '')) for d in cot_data]
if thinking_lens:
    print(f"\nThinking length (chars): mean={sum(thinking_lens)/len(thinking_lens):.0f}, max={max(thinking_lens)}, min={min(thinking_lens)}")

# 4. Token estimation
print(f"\n=== Token estimates (chars/4 rough) ===")
for d in cot_data[:3]:
    prompt_toks = len(d.get('prompt', '')) // 4
    think_toks = len(d.get('thinking', '')) // 4
    ans_toks = len(str(d.get('answer', ''))) // 4
    print(f"  type={d.get('type')}: prompt~{prompt_toks}tok, think~{think_toks}tok, ans~{ans_toks}tok")

# 5. Check how much budget the prompt uses
prompt_lens = [len(str(row['prompt'])) for _, row in train.iterrows()]
print(f"\n=== Prompt length stats (chars) ===")
print(f"  Mean: {sum(prompt_lens)/len(prompt_lens):.0f}")
print(f"  Max: {max(prompt_lens)}")
print(f"  Min: {min(prompt_lens)}")
for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    sub = train[train['type'] == t]
    if len(sub) > 0:
        avg = sub['prompt'].str.len().mean()
        mx = sub['prompt'].str.len().max()
        print(f"  {t}: mean={avg:.0f}, max={mx}")
