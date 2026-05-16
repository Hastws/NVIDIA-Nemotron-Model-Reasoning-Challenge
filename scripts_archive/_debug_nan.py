import polars as pl
import math

df = pl.read_csv('data_upload/sft_merged_v1.csv')
print('Schema:', df.schema)
print('Null count:', df['thinking'].null_count())
print('Total rows:', len(df))

pdf = df.to_pandas()
t = pdf['thinking']
print(f'\nAfter to_pandas():')
print(f'  NaN count: {t.isna().sum()}')
print(f'  String "nan": {(t == "nan").sum()}')

filled = t.fillna('')
ao_mask = filled.str.strip().str.len() == 0
print(f'\nfillna + strip filter:')
print(f'  answer-only (empty): {ao_mask.sum()}')
print(f'  has thinking: {(~ao_mask).sum()}')

# Get first AO row and simulate build_stage1_text
ao_idx = ao_mask.idxmax()
row = pdf.iloc[ao_idx]
thinking_val = row['thinking']
print(f'\nFirst AO row:')
print(f'  thinking value: {repr(thinking_val)}')
print(f'  type: {type(thinking_val)}')

# Now simulate the HuggingFace dataset path
# When we do Dataset.from_pandas -> .map, NaN becomes None? Or stays NaN?
from datasets import Dataset
test_df = pdf.iloc[[ao_idx]]
hf = Dataset.from_pandas(test_df)
example = hf[0]
print(f'\nAfter HF Dataset conversion:')
print(f'  thinking value: {repr(example.get("thinking"))}')
print(f'  type: {type(example.get("thinking"))}')

# The actual check
thinking = example.get("thinking", None)
print(f'\n  thinking={repr(thinking)}')
print(f'  bool(thinking)={thinking is not None and thinking}')
if thinking:
    s = str(thinking).strip()
    print(f'  str(thinking).strip()={repr(s)}')
    print(f'  RESULT: enters thinking branch = {bool(s)}')
else:
    print(f'  RESULT: enters answer-only branch')

# Also check with a thinking row
think_mask = filled.str.strip().str.len() > 0
think_idx = think_mask.idxmax()
test_df2 = pdf.iloc[[think_idx]]
hf2 = Dataset.from_pandas(test_df2)
ex2 = hf2[0]
print(f'\nFirst thinking row after HF:')
print(f'  thinking={repr(str(ex2.get("thinking"))[:100])}')
