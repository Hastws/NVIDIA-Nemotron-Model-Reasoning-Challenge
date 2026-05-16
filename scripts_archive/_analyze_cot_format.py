#!/usr/bin/env python3
"""Analyze current CoT data to design ultra-short format."""
import polars as pl

df = pl.read_csv('data/sft_typed_cot_600.csv')
print(f"Columns: {df.columns}")
print(f"Total: {len(df)}")

def classify(prompt):
    p = prompt.lower()
    if 'bitwise' in p or 'bit manipulation' in p or 'bit shift' in p:
        return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p or 'celestial' in p:
        return 'gravity'
    if 'unit conversion' in p or 'convert the following measurement' in p or 'secret unit' in p:
        return 'unit_conv'
    if 'encryption' in p or 'cipher' in p or 'encrypt' in p or 'decrypt' in p:
        return 'cipher'
    if 'numeral system' in p or 'roman numeral' in p or 'ancient numeral' in p:
        return 'numeral'
    if 'symbol' in p or 'equation' in p or 'transformation rule' in p:
        return 'symbol'
    return 'unknown'

df = df.with_columns(pl.col('prompt').map_elements(classify, return_dtype=pl.Utf8).alias('type'))

for t in sorted(df['type'].unique().to_list()):
    sub = df.filter(pl.col('type') == t)
    has_think = sub.filter(pl.col('thinking').str.len_chars() > 0)
    print(f"\n{'='*60}")
    print(f"{t}: {len(sub)} rows, {len(has_think)} with CoT")
    if len(has_think) > 0:
        lens = has_think['thinking'].str.len_chars()
        print(f"  CoT len: min={lens.min()}, med={lens.median():.0f}, max={lens.max()}")
        print(f"  Example thinking:")
        print(f"    {has_think[0]['thinking'][0][:300]}")
    else:
        print(f"  (answer-only)")
        print(f"  Example answer: {sub[0]['answer'][0][:100]}")

# Also look at a few train.csv examples per type to understand structure
print(f"\n\n{'='*60}")
print("TRAIN.CSV EXAMPLES (for rule structure reference)")
print('='*60)
train = pl.read_csv('competition_data/train.csv')
train = train.with_columns(pl.col('prompt').map_elements(classify, return_dtype=pl.Utf8).alias('type'))
for t in sorted(train['type'].unique().to_list()):
    sub = train.filter(pl.col('type') == t)
    print(f"\n--- {t} ({len(sub)} total) ---")
    ex = sub[0]
    prompt = ex['prompt'][0]
    # Show just the key structure (first 400 chars)
    print(f"  Prompt (first 400 chars):\n    {prompt[:400]}")
    print(f"  Answer: {ex['answer'][0]}")
