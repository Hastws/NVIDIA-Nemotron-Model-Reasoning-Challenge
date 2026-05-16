"""Token length distribution for nemotron_cot_multitype.csv after chat-template wrapping.

We replicate the notebook's record-construction logic (strip stray \\boxed{}, append
gold answer, terminate with </think>) and then count tokens with the Nemotron
tokenizer (loaded from tonghuikang_repo/tokenizer.json).

Chat-template overhead is approximated with a small constant since we don't have
tokenizer_config.json locally. The bulk of length comes from prompt + CoT + answer.
"""
import re
import pandas as pd
import numpy as np
from tokenizers import Tokenizer

CSV  = "data/nemotron-cot-multitype/nemotron_cot_multitype.csv"
TOK  = "tonghuikang_repo/tokenizer.json"
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
# Rough wrapper overhead for Nemotron chat template (system+user+assistant special tokens).
CHAT_TEMPLATE_OVERHEAD = 20

tok = Tokenizer.from_file(TOK)
df  = pd.read_csv(CSV)
print(f"Rows in CSV: {len(df)}")

records = []
for _, row in df.iterrows():
    prompt = str(row["prompt"])
    answer = row["answer"]
    cot    = str(row["generated_cot"])
    if not cot or cot == "nan" or len(cot.strip()) < 5:
        continue
    cot_cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
    user = prompt + PROMPT_SUFFIX
    if answer is None or (isinstance(answer, float) and pd.isna(answer)) or str(answer).strip() == "":
        assistant = cot_cleaned + "\n</think>"
    else:
        assistant = cot_cleaned + f"\n</think>\n\\boxed{{{str(answer)}}}"
    records.append((str(row["type"]), user, assistant))

print(f"Valid SFT records: {len(records)}")

# Batch tokenize for speed.
users      = [r[1] for r in records]
assistants = [r[2] for r in records]
types      = [r[0] for r in records]

u_enc = tok.encode_batch(users)
a_enc = tok.encode_batch(assistants)

u_lens = np.array([len(e.ids) for e in u_enc])
a_lens = np.array([len(e.ids) for e in a_enc])
total  = u_lens + a_lens + CHAT_TEMPLATE_OVERHEAD

def stats(name, arr):
    pct = np.percentile(arr, [50, 75, 90, 95, 99, 99.5, 100])
    print(
        f"{name:<14} mean={arr.mean():8.1f}  std={arr.std():7.1f}  "
        f"min={arr.min():5d}  p50={int(pct[0]):5d}  p75={int(pct[1]):5d}  "
        f"p90={int(pct[2]):5d}  p95={int(pct[3]):5d}  p99={int(pct[4]):5d}  "
        f"p99.5={int(pct[5]):5d}  max={int(pct[6]):6d}"
    )

print("\n=== Token-length distribution (Nemotron tokenizer) ===")
stats("user(prompt)",   u_lens)
stats("assistant(CoT)", a_lens)
stats("total",          total)

print("\nBuckets for total length:")
edges = [0, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 10**9]
labels = ["<=512","<=1024","<=2048","<=3072","<=4096","<=6144","<=8192","<=12288","<=16384","<=32768",">32768"]
counts, _ = np.histogram(total, bins=edges)
N = len(total)
for lab, c in zip(labels, counts):
    print(f"  {lab:>10}: {c:6d}  ({100*c/N:5.2f}%)")

# MAX_SEQ_LEN = 8192 in the notebook. Show truncation pressure.
for limit in (4096, 6144, 8192, 16384):
    n = int((total > limit).sum())
    print(f"  超过 {limit:>5} tokens 的样本: {n} ({100*n/N:.2f}%)")

print("\n=== 按题型 (type) 分组的 total token 统计 ===")
df_t = pd.DataFrame({"type": types, "total": total, "user": u_lens, "assistant": a_lens})
print(df_t.groupby("type")["total"].describe(percentiles=[.5,.9,.95,.99]).round(1))
