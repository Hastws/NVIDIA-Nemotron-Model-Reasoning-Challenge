#!/usr/bin/env python3
"""Create curated 700-sample dataset with type-biased sampling.

Strategy: Over-represent hard types (cipher/bit_ops) where score gap is largest.
Use verified samples where available (from programmatic solver).

Composition:
  numeral:  80  (from verified pool - near ceiling already)
  gravity:  100 (from verified pool)
  unit_conv: 100 (from verified pool)
  cipher:   170 (from verified pool - biggest gap + headroom)
  bit_ops:  150 (from verified pool - second biggest gap)
  symbol:   100 (from original data - no solver available)
  Total:    700
"""
import polars as pl
import json
import os

DATA_DIR = "competition_data"
COT_DIR = "data"
OUT_DIR = "data"

# Load original data for type column and symbol
train = pl.read_csv(f"{DATA_DIR}/train.csv")
print(f"Original data: {len(train)} samples")

# Infer type from prompt (same logic used elsewhere)
def infer_type(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "numeral system" in prompt_lower or "base-" in prompt_lower or "roman" in prompt_lower:
        return "numeral"
    elif "gravitational" in prompt_lower or "gravity" in prompt_lower or "planet" in prompt_lower:
        return "gravity"
    elif "unit" in prompt_lower or "conversion" in prompt_lower or "convert" in prompt_lower:
        return "unit_conv"
    elif "cipher" in prompt_lower or "encrypt" in prompt_lower or "decrypt" in prompt_lower:
        return "cipher"
    elif "bit" in prompt_lower or "binary operation" in prompt_lower or "bitwise" in prompt_lower:
        return "bit_ops"
    elif "symbol" in prompt_lower or "transform" in prompt_lower:
        return "symbol"
    return "unknown"

# Add type column
train = train.with_columns(
    pl.col("prompt").map_elements(infer_type, return_dtype=pl.Utf8).alias("type")
)

type_counts = train.group_by("type").agg(pl.len().alias("count")).sort("type")
print("\nOriginal distribution:")
for row in type_counts.iter_rows():
    print(f"  {row[0]}: {row[1]}")

# Load verified IDs from programmatic solver data
verified_ids = set()
for jsonl_file in ["programmatic_cot.jsonl", "cipher_programmatic_cot.jsonl", "bit_ops_programmatic_cot.jsonl"]:
    fpath = os.path.join(COT_DIR, jsonl_file)
    if os.path.exists(fpath):
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                verified_ids.add(obj["id"])
        print(f"Loaded verified IDs from {jsonl_file}")

print(f"\nTotal verified IDs: {len(verified_ids)}")

# Sampling plan
sampling_plan = {
    "numeral": 80,
    "gravity": 100,
    "unit_conv": 100,
    "cipher": 170,
    "bit_ops": 150,
    "symbol": 100,
}

# For types with verified data: sample from verified pool
# For symbol: sample from original data
seed = 42
result_dfs = []

for type_name, n_samples in sampling_plan.items():
    type_df = train.filter(pl.col("type") == type_name)
    
    if type_name == "symbol":
        # No solver for symbol - use original data
        sampled = type_df.sample(n=min(n_samples, len(type_df)), seed=seed)
    else:
        # Use verified samples only
        verified_df = type_df.filter(pl.col("id").is_in(list(verified_ids)))
        if len(verified_df) >= n_samples:
            sampled = verified_df.sample(n=n_samples, seed=seed)
        else:
            print(f"WARNING: {type_name} only has {len(verified_df)} verified, need {n_samples}")
            sampled = verified_df
    
    result_dfs.append(sampled)
    print(f"  {type_name}: sampled {len(sampled)} (from {len(type_df)} available, {len(type_df.filter(pl.col('id').is_in(list(verified_ids))))} verified)")

# Combine and output
curated = pl.concat(result_dfs)
# Drop type column (not needed for training)
curated_out = curated.select(["id", "prompt", "answer"])
print(f"\nFinal curated dataset: {len(curated_out)} samples")

# Save
out_path = os.path.join(OUT_DIR, "sft_curated_700.csv")
curated_out.write_csv(out_path)
print(f"Saved to {out_path}")

# Also verify the distribution
print("\nFinal distribution:")
for type_name, n_samples in sampling_plan.items():
    actual = len(curated.filter(pl.col("type") == type_name))
    print(f"  {type_name}: {actual}/{n_samples}")
