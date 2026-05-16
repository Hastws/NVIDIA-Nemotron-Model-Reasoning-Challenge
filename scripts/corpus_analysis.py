"""Analyze corpus.jsonl: schema, segment / category distribution, token-length stats,
inclusion rate, and per-category token statistics.
"""
import json
import numpy as np
import pandas as pd
from collections import Counter

PATH = "data/corpus.jsonl"

rows = []
with open(PATH, "r") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.DataFrame(rows)
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"dtypes:\n{df.dtypes}\n")

# Uniqueness
print(f"Unique problem_id: {df['problem_id'].nunique()}")
print(f"Unique segment   : {df['segment'].nunique()}  -> {df['segment'].unique().tolist()}")
print(f"Unique category  : {df['category'].nunique()}")

# included flag
print(f"\nincluded=True : {df['included'].sum()}  ({100*df['included'].mean():.2f}%)")
print(f"included=False: {(~df['included']).sum()}")

# Segment distribution
print("\n=== Segment distribution ===")
seg = df.groupby("segment").agg(
    n=("problem_id","size"),
    included=("included","sum"),
    incl_rate=("included","mean"),
    tok_mean=("token_count","mean"),
    tok_p50=("token_count", lambda s: int(np.percentile(s,50))),
    tok_p95=("token_count", lambda s: int(np.percentile(s,95))),
    tok_max=("token_count","max"),
).sort_values("n", ascending=False)
seg["incl_rate"] = (seg["incl_rate"]*100).round(1)
seg["tok_mean"]  = seg["tok_mean"].round(0).astype(int)
print(seg.to_string())

# Category distribution
print("\n=== Category distribution (top 30) ===")
cat = df.groupby("category").agg(
    n=("problem_id","size"),
    incl=("included","sum"),
    incl_rate=("included","mean"),
    tok_mean=("token_count","mean"),
    tok_p50=("token_count", lambda s: int(np.percentile(s,50))),
    tok_p95=("token_count", lambda s: int(np.percentile(s,95))),
    tok_max=("token_count","max"),
    masked_mean=("masked_token_count","mean"),
    unmasked_mean=("unmasked_token_count","mean"),
).sort_values("n", ascending=False)
cat["incl_rate"] = (cat["incl_rate"]*100).round(1)
for c in ["tok_mean","masked_mean","unmasked_mean"]:
    cat[c] = cat[c].round(0).astype(int)
print(cat.head(30).to_string())
print(f"\nTotal categories: {len(cat)}")

# Token-count distribution overall
print("\n=== Token-count distribution (all records) ===")
def stats(name, arr):
    pct = np.percentile(arr, [50,75,90,95,99,99.9,100])
    print(f"{name:<20} mean={arr.mean():8.1f}  std={arr.std():7.1f}  "
          f"min={int(arr.min()):5d}  p50={int(pct[0]):5d}  p75={int(pct[1]):5d}  "
          f"p90={int(pct[2]):5d}  p95={int(pct[3]):5d}  p99={int(pct[4]):5d}  "
          f"p99.9={int(pct[5]):6d}  max={int(pct[6]):6d}")

stats("token_count",          df["token_count"].values)
stats("masked_token_count",   df["masked_token_count"].values)
stats("unmasked_token_count", df["unmasked_token_count"].values)

# included only
incl = df[df["included"]]
print(f"\n=== Token distribution (included=True only, n={len(incl)}) ===")
stats("token_count",          incl["token_count"].values)
stats("unmasked_token_count", incl["unmasked_token_count"].values)

# Bucketize total token_count
print("\n=== token_count buckets (all) ===")
edges  = [0, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 10**9]
labels = ["<=256","<=512","<=1024","<=2048","<=3072","<=4096","<=6144","<=8192","<=12288","<=16384",">16384"]
counts,_ = np.histogram(df["token_count"], bins=edges)
N = len(df)
for lab,c in zip(labels, counts):
    print(f"  {lab:>10}: {c:6d}  ({100*c/N:5.2f}%)")

for limit in (4096, 6144, 8192, 16384):
    n = int((df["token_count"] > limit).sum())
    print(f"  超过 {limit:>5} tokens: {n} ({100*n/N:.2f}%)")

# masked / unmasked ratio
ratio = df["unmasked_token_count"] / df["token_count"].clip(lower=1)
print(f"\nunmasked / total ratio: mean={ratio.mean():.3f}  median={ratio.median():.3f}  "
      f"p10={ratio.quantile(0.1):.3f}  p90={ratio.quantile(0.9):.3f}")
print("(unmasked = the part that contributes to loss; high ratio = more learning signal)")

# Sum of tokens (effective training size)
print(f"\nTotal tokens (all)        : {int(df['token_count'].sum()):,}")
print(f"Total unmasked tokens (all): {int(df['unmasked_token_count'].sum()):,}")
print(f"Total tokens (included)   : {int(incl['token_count'].sum()):,}")
print(f"Total unmasked (included) : {int(incl['unmasked_token_count'].sum()):,}")

# Duplicate problem_id?
dup = df["problem_id"].value_counts()
print(f"\nProblem_id duplicates: {int((dup>1).sum())} ids appear >1 time (max repeats={int(dup.max())})")

# answer length
ans_lens = df["answer"].astype(str).str.len()
print(f"\nanswer string length: mean={ans_lens.mean():.1f}  median={int(ans_lens.median())}  "
      f"p95={int(ans_lens.quantile(0.95))}  max={int(ans_lens.max())}")
