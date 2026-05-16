"""Build two SFT CSV datasets from nemotron_cot_multitype.csv.

Dataset A — rule-found only:
    rows whose `id` is in problems.jsonl with status == 'rule_found'
    (these are problems where the upstream pipeline DID solve them).

Dataset B — rule-found + augmentations:
    Dataset A rows + every augmentation row from the multitype CSV (rows
    whose `answer` is empty/NaN — these are CoT-only training samples
    with new ids that don't exist in problems.jsonl).

Output schema matches problem_ids_matched.csv:
    id, prompt, answer, type, generated_cot
"""
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSV  = os.path.join(ROOT, "data/nemotron-cot-multitype/nemotron_cot_multitype.csv")
PROBLEMS = os.path.join(ROOT, "tonghuikang_repo/problems.jsonl")

OUT_A_DIR = os.path.join(ROOT, "data/nemotron-rule-found")
OUT_B_DIR = os.path.join(ROOT, "data/nemotron-rule-found-aug")
os.makedirs(OUT_A_DIR, exist_ok=True)
os.makedirs(OUT_B_DIR, exist_ok=True)

print("Loading source CSV...")
df = pd.read_csv(SRC_CSV)
print(f"  rows={len(df)}, cols={list(df.columns)}")

print("Loading problems.jsonl status flags...")
prob = pd.DataFrame([json.loads(l) for l in open(PROBLEMS)])
rule_found_ids = set(prob.loc[prob["status"] == "rule_found", "id"])
print(f"  rule_found problem ids: {len(rule_found_ids)}")

# Augmentation rows: empty / NaN answer column.
is_aug = df["answer"].isna() | (df["answer"].astype(str).str.strip() == "")
print(f"  augmentation rows in source: {int(is_aug.sum())}")

# --- Dataset A: rule-found only.
df_a = df[df["id"].isin(rule_found_ids)].reset_index(drop=True)
print(f"\nDataset A (rule-found only): {len(df_a)} rows")
print("  by type:", df_a["type"].value_counts().to_dict())

# --- Dataset B: rule-found + all augmentations (de-duplicated by id, just in case).
df_b = pd.concat([df_a, df[is_aug]], ignore_index=True)
df_b = df_b.drop_duplicates(subset=["id"]).reset_index(drop=True)
print(f"\nDataset B (rule-found + augmentations): {len(df_b)} rows")
print("  by type:", df_b["type"].value_counts().to_dict())

out_a = os.path.join(OUT_A_DIR, "nemotron_rule_found.csv")
out_b = os.path.join(OUT_B_DIR, "nemotron_rule_found_aug.csv")
df_a.to_csv(out_a, index=False)
df_b.to_csv(out_b, index=False)
print(f"\nWrote {out_a}  ({os.path.getsize(out_a)/1024/1024:.2f} MB)")
print(f"Wrote {out_b}  ({os.path.getsize(out_b)/1024/1024:.2f} MB)")
