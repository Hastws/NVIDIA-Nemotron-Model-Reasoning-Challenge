#!/usr/bin/env python3
"""Convert tonghuikang_full.csv into a drop-in replacement for
problem_ids_matched.csv (same 5 columns).

Keeps two kinds of rows:
  1. Standard rows: have a non-empty `answer` and a non-empty `generated_cot`.
  2. Augmentation rows: `answer` is empty/NaN but `generated_cot` is non-empty.
     These are CoT-only training samples (no gold answer to supervise the
     final boxed token). They are kept verbatim with `answer = ""`. The
     downstream training code must branch on empty answer and emit
     `assistant = cot + "\n</think>"` (no `\boxed{...}`) for these rows.

Output schema (identical to problem_ids_matched.csv):
    id, prompt, answer, type, generated_cot
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "competition_data" / "tonghuikang_full" / "tonghuikang_full.csv"
OUT_DIR = ROOT / "data" / "nemotron-cot-multitype"
OUT_CSV = OUT_DIR / "nemotron_cot_multitype.csv"

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def recover_answer(cot: str) -> str | None:
    if not isinstance(cot, str) or not cot:
        return None
    matches = BOXED_RE.findall(cot)
    if not matches:
        return None
    ans = matches[-1].strip()
    return ans or None


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"Source rows: {len(df)}")
    print(f"Source null answers: {df['answer'].isna().sum()}")

    rows = []
    augmentation = 0
    dropped_no_cot = 0
    for _, row in df.iterrows():
        cot = row.get("generated_cot")
        if not isinstance(cot, str) or len(cot.strip()) < 5:
            dropped_no_cot += 1
            continue
        ans = row.get("answer")
        if ans is None or (isinstance(ans, float) and pd.isna(ans)) or str(ans).strip() == "":
            ans = ""
            augmentation += 1
        else:
            ans = str(ans)
        rows.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "answer": ans,
            "type": row["type"],
            "generated_cot": cot,
        })

    out = pd.DataFrame(rows, columns=["id", "prompt", "answer", "type", "generated_cot"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print()
    print(f"Augmentation rows kept (empty answer): {augmentation}")
    print(f"Dropped (CoT missing/too short): {dropped_no_cot}")
    print(f"Final rows: {len(out)}")
    print(f"  with answer:    {(out['answer'].astype(str).str.len() > 0).sum()}")
    print(f"  without answer: {(out['answer'].astype(str).str.len() == 0).sum()}")
    print()
    print("Type distribution:")
    for t, c in out["type"].value_counts().items():
        print(f"  {t:30s} {c:5d}")
    print()
    print(f"Wrote {OUT_CSV}  ({os.path.getsize(OUT_CSV) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
