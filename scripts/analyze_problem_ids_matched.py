"""Analyze competition_data/nemotron-cot-tong/problem_ids_matched.csv:
schema, dedup, type distribution, prompt/CoT length, token-length distribution
(Nemotron tokenizer), augmentation rate, overlap with corpus.jsonl & rule_found.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "competition_data/nemotron-cot-tong/problem_ids_matched.csv"
TOK = ROOT / "tonghuikang_repo/tokenizer.json"
CORPUS = ROOT / "data" / "corpus.jsonl"
PROBLEMS = ROOT / "tonghuikang_repo/problems.jsonl"
PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

print(f"Loading {CSV.name} ...")
df = pd.read_csv(CSV)
print(f"  rows={len(df)}  cols={list(df.columns)}")
print(f"  dtypes:\n{df.dtypes.to_string()}")
print(f"  unique ids: {df['id'].nunique()}  (duplicate id rows: {len(df) - df['id'].nunique()})")

# Empty answer / CoT?
df["ans_empty"] = df["answer"].fillna("").astype(str).str.strip().eq("")
df["cot_empty"] = df["generated_cot"].fillna("").astype(str).str.strip().eq("")
print(f"\n  empty answer rows: {int(df['ans_empty'].sum())}  ({100*df['ans_empty'].mean():.2f}%)")
print(f"  empty CoT rows   : {int(df['cot_empty'].sum())}  ({100*df['cot_empty'].mean():.2f}%)")

# Type distribution.
print("\n=== Type distribution ===")
print(df["type"].value_counts().to_string())

# Per-type empty-answer share (i.e. how augmentation-heavy each type is).
print("\n=== Per-type empty-answer share ===")
g = df.groupby("type").agg(
    n=("id", "size"),
    empty_ans=("ans_empty", "sum"),
    empty_cot=("cot_empty", "sum"),
)
g["empty_ans_pct"] = (g["empty_ans"] / g["n"] * 100).round(1)
print(g.sort_values("n", ascending=False).to_string())

# Duplicate id structure (which ids repeat? are they paired with empty answers?)
dup = df["id"].value_counts()
print(f"\nIDs that repeat: {(dup > 1).sum()}  (max repeats={int(dup.max())})")
if (dup > 1).any():
    sample_id = dup[dup > 1].index[0]
    print(f"  sample repeated id={sample_id}:")
    print(df[df["id"] == sample_id][["id","type","ans_empty","cot_empty"]].to_string(index=False))

# String length in chars.
prompt_chars = df["prompt"].fillna("").astype(str).str.len()
cot_chars = df["generated_cot"].fillna("").astype(str).str.len()
ans_chars = df["answer"].fillna("").astype(str).str.len()


def _stats(name, arr):
    a = np.asarray(arr)
    pct = np.percentile(a, [50, 75, 90, 95, 99, 100])
    print(f"{name:<14} mean={a.mean():9.1f}  p50={int(pct[0]):6d}  p75={int(pct[1]):6d}  "
          f"p90={int(pct[2]):6d}  p95={int(pct[3]):6d}  p99={int(pct[4]):6d}  max={int(pct[5]):6d}")


print("\n=== String length (chars) ===")
_stats("prompt",       prompt_chars)
_stats("generated_cot", cot_chars)
_stats("answer",       ans_chars)

# Token length using Nemotron tokenizer (mirrors notebook recipe).
print("\nTokenizing (Nemotron tokenizer) ...")
tok = Tokenizer.from_file(str(TOK))

users, assistants = [], []
for _, row in df.iterrows():
    user = str(row["prompt"]) + PROMPT_SUFFIX
    if row["ans_empty"]:
        assistant = str(row["generated_cot"]).rstrip() + "\n</think>"
    else:
        assistant = str(row["generated_cot"]).rstrip() + f"\n</think>\n\\boxed{{{row['answer']}}}"
    users.append(user)
    assistants.append(assistant)

u_lens = np.array([len(e.ids) for e in tok.encode_batch(users)])
a_lens = np.array([len(e.ids) for e in tok.encode_batch(assistants)])
total = u_lens + a_lens + 20  # ~chat template overhead

print("\n=== Token-length distribution (Nemotron tokenizer) ===")
_stats("user(prompt)",   u_lens)
_stats("assistant(CoT)", a_lens)
_stats("total",          total)

print("\nBuckets for total token_count:")
edges  = [0, 512, 1024, 2048, 3072, 4096, 6144, 8192, 16384, 10**9]
labels = ["<=512","<=1024","<=2048","<=3072","<=4096","<=6144","<=8192","<=16384",">16384"]
N = len(total)
counts, _ = np.histogram(total, bins=edges)
for lab, c in zip(labels, counts):
    print(f"  {lab:>8}: {c:6d}  ({100*c/N:5.2f}%)")
for limit in (4096, 6144, 8192):
    n = int((total > limit).sum())
    print(f"  >{limit}: {n} ({100*n/N:.2f}%)")

# Overlap with corpus.jsonl & rule_found.
corpus_ids = {json.loads(l)["problem_id"] for l in open(CORPUS)}
prob_status = {p["id"]: p["status"] for p in (json.loads(l) for l in open(PROBLEMS))}
rule_found_ids = {pid for pid, s in prob_status.items() if s == "rule_found"}

ids = set(df["id"])
print("\n=== Overlap with other sources ===")
print(f"  unique ids in this CSV       : {len(ids)}")
print(f"  in corpus.jsonl              : {len(ids & corpus_ids)}")
print(f"  NOT in corpus.jsonl (orphans): {len(ids - corpus_ids)}")
print(f"  in problems.jsonl            : {len(ids & set(prob_status))}")
print(f"  in rule_found set            : {len(ids & rule_found_ids)}")
print(f"  ids that are augmentations   : {len(ids - set(prob_status))}")

# Rows-by-id where id IS a real problem: what's its solver status?
real_rows = df[df["id"].isin(prob_status)]
real_rows = real_rows.assign(status=real_rows["id"].map(prob_status))
print("\nReal-problem rows per status:")
print(real_rows["status"].value_counts().to_string())
