"""Convert tonghuikang_repo/{reasoning,augmentations} into a single SFT CSV.

Output columns match `problem_ids_matched.csv` so the existing notebook just
needs DATASET_PATH changed.

Columns: id, prompt, answer, type, generated_cot
- For reasoning entries: prompt = train.csv prompt, answer = train.csv answer,
  type = problems.jsonl category, generated_cot = reasoning/<id>.txt
  (verbatim, the notebook strips trailing \\boxed{...} and re-appends the answer).
- For augmentation entries: prompt comes from the [prompt] section, answer is
  empty string (notebook will skip appending \\boxed{} since answer=""),
  generated_cot is the [completion] section followed by an explicit
  \\boxed{} marker so the notebook's regex strip leaves the body intact.

Run:
    uv run python scripts/build_tonghuikang_csv.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "tonghuikang_repo"
TRAIN_CSV = REPO / "train.csv"
PROBLEMS_INDEX = REPO / "problems.jsonl"
REASONING_DIR = REPO / "reasoning"
AUGMENTATIONS_DIR = REPO / "augmentations"

OUT_DIR = Path(__file__).resolve().parent.parent / "competition_data" / "tonghuikang_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "tonghuikang_full.csv"


def load_train_prompts() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    with TRAIN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = (row["prompt"], row["answer"])
    return out


def load_problem_categories() -> dict[str, str]:
    cats: dict[str, str] = {}
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cats[d["id"]] = d["category"]
    return cats


def parse_augmentation(text: str) -> tuple[str, str, str]:
    category = text.split("[category]\n", 1)[1].split("\n[prompt]\n", 1)[0]
    prompt = text.split("[prompt]\n", 1)[1].split("\n[completion]\n", 1)[0]
    completion = text.split("\n[completion]\n", 1)[1].rstrip("\n")
    return category, prompt, completion


def main() -> None:
    prompts = load_train_prompts()
    cats = load_problem_categories()

    rows: list[dict[str, str]] = []
    n_reasoning = 0
    n_aug = 0
    skipped = 0

    # ---- Reasoning entries (real problems with programmatic CoT) ----
    for path in sorted(REASONING_DIR.glob("*.txt")):
        pid = path.stem
        if pid not in prompts:
            skipped += 1
            continue
        prompt_text, answer = prompts[pid]
        category = cats.get(pid, "unknown")
        cot = path.read_text().rstrip("\n")
        rows.append(
            {
                "id": pid,
                "prompt": prompt_text,
                "answer": answer,
                "type": category,
                "generated_cot": cot,
            }
        )
        n_reasoning += 1

    # ---- Augmentation entries (synthetic skill problems, no \boxed answer) ----
    # The notebook does:
    #   cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
    #   assistant = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
    # For augmentations the original training format is:
    #   completion + "\n</think><|im_end|>"  (no \boxed)
    # To keep notebook code path identical we set answer="" and append a
    # placeholder \boxed{} that the strip regex will remove. The remaining
    # tail "\n</think>\n\\boxed{}" is acceptable: the model just learns to
    # close </think> and emit empty boxed for these tool tasks.
    # NOTE: augmentations are auxiliary skill drills; the empty boxed is fine
    # because at inference these category prompts never appear.
    for path in sorted(AUGMENTATIONS_DIR.glob("*.txt")):
        pid = path.stem
        try:
            category, prompt_text, completion = parse_augmentation(path.read_text())
        except Exception:
            skipped += 1
            continue
        cot = completion + "\n\\boxed{}"  # placeholder for regex strip
        rows.append(
            {
                "id": pid,
                "prompt": prompt_text,
                "answer": "",
                "type": category,
                "generated_cot": cot,
            }
        )
        n_aug += 1

    fieldnames = ["id", "prompt", "answer", "type", "generated_cot"]
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    from collections import Counter

    cnt = Counter(r["type"] for r in rows)
    print(f"Wrote {len(rows)} rows -> {OUT_PATH}")
    print(f"  reasoning: {n_reasoning}")
    print(f"  augment:   {n_aug}")
    if skipped:
        print(f"  skipped:   {skipped}")
    print()
    print("Per-type counts:")
    for k in sorted(cnt):
        print(f"  {k:30s} {cnt[k]:5d}")


if __name__ == "__main__":
    main()
