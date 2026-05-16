#!/usr/bin/env python3
"""Split problem_ids_matched_v2.csv into two SFT datasets:
  A) v2_real     : 真问题 + programmatic (11 类), 无 augmentation
  B) v2_full_aug : 含 augmentation 的全集 (16 类)

输出到 data/v2_real/v2_real.csv  和  data/v2_full_aug/v2_full_aug.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "problem_ids_matched_v2.csv"

# augmentation 类型（无真实问题语义，仅训练表层模式）
AUG_TYPES = {"matching", "splitting", "concatenation", "spelling", "lstrip"}


def main() -> None:
    df = pd.read_csv(SRC).fillna("")
    print(f"Loaded {SRC.name}: rows={len(df)} unique_id={df['id'].nunique()}")

    # A) real-only
    real = df[~df["type"].isin(AUG_TYPES)].reset_index(drop=True)
    out_real_dir = ROOT / "data" / "v2_real"
    out_real_dir.mkdir(parents=True, exist_ok=True)
    out_real = out_real_dir / "v2_real.csv"
    real.to_csv(out_real, index=False)
    print(f"  v2_real     -> {out_real} (rows={len(real)})")
    print(real["type"].value_counts().to_string())

    # B) full v2 (含 augmentation)
    out_full_dir = ROOT / "data" / "v2_full_aug"
    out_full_dir.mkdir(parents=True, exist_ok=True)
    out_full = out_full_dir / "v2_full_aug.csv"
    df.to_csv(out_full, index=False)
    print(f"\n  v2_full_aug -> {out_full} (rows={len(df)})")
    print(df["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
