import pandas as pd

df2 = pd.read_csv("data/sft_difficulty_aware.csv")
print(f"difficulty_aware modes: {df2['mode'].value_counts().to_dict()}")

df1 = pd.read_csv("data/sft_merged_v1.csv")
# Match IDs to train.csv to get types
train = pd.read_csv("competition_data/train.csv")
merged = df1.merge(train[['id']], on='id', how='left', indicator=True)
print(f"\nmerged_v1: {len(df1)} rows, {(merged['_merge']=='both').sum()} match train.csv")

# Check if merged_v1 has symbol data by matching against difficulty_aware which has types
da_types = df2[['id','type']].drop_duplicates()
df1_typed = df1.merge(da_types, on='id', how='left')
print(f"\nmerged_v1 type distribution (via difficulty_aware matching):")
print(df1_typed['type'].value_counts(dropna=False))

# Check thinking status by type
h = df1_typed['thinking'].fillna('').str.strip().str.len() > 0
for t in sorted(df1_typed['type'].dropna().unique()):
    mask = df1_typed['type'] == t
    n_think = (mask & h).sum()
    n_ao = (mask & ~h).sum()
    print(f"  {t}: thinking={n_think}, answer-only={n_ao}")
