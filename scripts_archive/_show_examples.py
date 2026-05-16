#!/usr/bin/env python3
"""Sample per-type examples and prepare NVIDIA API test prompts."""
import polars as pl

train = pl.read_csv('competition_data/train.csv')

def classify(p):
    p = p.lower()
    if 'bit manipulation' in p or 'bitwise' in p or 'bit shift' in p: return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p or 'celestial' in p: return 'gravity'
    if 'unit conversion' in p or 'convert the following measurement' in p or 'secret unit' in p: return 'unit_conv'
    if 'encryption' in p or 'cipher' in p or 'encrypt' in p or 'decrypt' in p: return 'cipher'
    if 'numeral system' in p or 'roman numeral' in p or 'ancient numeral' in p: return 'numeral'
    if 'symbol' in p or 'equation' in p or 'transformation rule' in p: return 'symbol'
    return 'unknown'

train = train.with_columns(pl.col('prompt').map_elements(classify, return_dtype=pl.Utf8).alias('type'))

for t in ['gravity', 'unit_conv', 'numeral', 'cipher', 'bit_ops', 'symbol']:
    row = train.filter(pl.col('type') == t).head(1).row(0, named=True)
    print(f'=== {t} (prompt {len(row["prompt"])} chars) ===')
    print(row['prompt'][:500])
    print(f'...')
    print(f'ANSWER: {row["answer"]}')
    print()
