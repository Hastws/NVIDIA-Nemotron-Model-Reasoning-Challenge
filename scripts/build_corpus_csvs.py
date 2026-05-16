"""Build SFT CSVs directly from corpus.jsonl + per-problem segment files.

Source of truth:
  - /corpus.jsonl         (workspace root)  : 19263 entries (id, category, answer, masked/unmasked/total token counts, included)
  - tonghuikang_enhanced/corpus/<id>/synthetic.jsonl  : segment files (covers 17963/19263)
  - tonghuikang_repo/vocab.jsonl  : token_id -> token text mapping
  - tonghuikang_repo/problems.jsonl  : status field for rule_found filtering

Decoded layout per entry (matches corpus.py's recipe):
  prompt segment (masked)   : chat template + user prompt + suffix + assistant <think>
  cot segment   (unmasked)  : reasoning + "\\n</think>" + (real -> "\\n\\boxed{ans}") + "<|im_end|>"

Output schema = problem_ids_matched.csv:
  id, prompt, answer, type, generated_cot

We strip chat-template wrappers + the trailing "</think>...<|im_end|>" so the CSV
plays nicely with the existing notebook (which re-applies the chat template).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS_JSONL = ROOT / "data" / "corpus.jsonl"
SEG_DIR_PRIMARY = ROOT / "tonghuikang_enhanced" / "corpus"  # 17963/19263
SEG_DIR_FALLBACK = ROOT / "tonghuikang_repo" / "corpus"      # 14818/19263
VOCAB_PATH = ROOT / "tonghuikang_repo" / "vocab.jsonl"
PROBLEMS_JSONL = ROOT / "tonghuikang_repo" / "problems.jsonl"

OUT_FULL_DIR = ROOT / "data" / "nemotron-corpus-full"
OUT_RULE_DIR = ROOT / "data" / "nemotron-corpus-rule"
OUT_RULE_AUG_DIR = ROOT / "data" / "nemotron-corpus-rule-aug"
for d in (OUT_FULL_DIR, OUT_RULE_DIR, OUT_RULE_AUG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Token decoding (mirrors generate_csv.py logic)
# ---------------------------------------------------------------------------
BYTE_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def load_vocab() -> dict[int, str]:
    vocab: dict[int, str] = {}
    with open(VOCAB_PATH) as f:
        for line in f:
            entry = json.loads(line)
            vocab[int(entry["token_id"])] = entry["token"]
    return vocab


def decode_tokens(token_ids: list[int], vocab: dict[int, str]) -> str:
    parts: list[str] = []
    byte_buffer = bytearray()
    for tid in token_ids:
        text = vocab.get(tid, f"<unk:{tid}>")
        byte_matches = BYTE_RE.findall(text)
        if byte_matches and BYTE_RE.sub("", text) == "":
            for hex_str in byte_matches:
                byte_buffer.append(int(hex_str, 16))
            continue
        if byte_buffer:
            parts.append(byte_buffer.decode("utf-8", errors="replace"))
            byte_buffer.clear()
        # SentencePiece-style leading-space marker: convert U+2581 to space.
        text = text.replace("\u2581", " ")
        parts.append(text)
    if byte_buffer:
        parts.append(byte_buffer.decode("utf-8", errors="replace"))
    return "".join(parts)


def load_segments(problem_id: str) -> tuple[list[int], list[int]] | None:
    """Return (prompt_token_ids, cot_token_ids) or None if no segment file."""
    for base in (SEG_DIR_PRIMARY, SEG_DIR_FALLBACK):
        path = base / problem_id / "synthetic.jsonl"
        if path.exists():
            prompt_ids: list[int] = []
            cot_ids: list[int] = []
            with open(path) as f:
                for line in f:
                    seg = json.loads(line)
                    if seg["type"] == "masked":
                        prompt_ids.extend(seg["tokens"])
                    else:
                        cot_ids.extend(seg["tokens"])
            return prompt_ids, cot_ids
    return None


# ---------------------------------------------------------------------------
# Prompt / CoT post-processing
# ---------------------------------------------------------------------------
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def strip_prompt(decoded: str) -> str:
    """Extract user content from decoded chat-template tokens.

    The chat template wraps the user message between
        <|im_start|>user\n  ...  <|im_end|>
    and the prompt suffix may already be appended. We return only the user-visible
    body, with the boxed-answer suffix removed (the existing training notebook
    re-appends it via PROMPT_SUFFIX).
    """
    user_marker = "<|im_start|>user\n"
    end_marker = "<|im_end|>"
    if user_marker in decoded:
        body = decoded.split(user_marker, 1)[1]
        body = body.split(end_marker, 1)[0]
    else:
        body = decoded
    # Drop the boxed-answer suffix if the corpus baked it in.
    if body.endswith(PROMPT_SUFFIX):
        body = body[: -len(PROMPT_SUFFIX)]
    elif PROMPT_SUFFIX.strip() in body:
        body = body.replace(PROMPT_SUFFIX, "")
    return body.strip("\n")


_BOXED_TAIL = re.compile(r"\s*\n?</think>\s*(?:\n\s*\\boxed\{[^}]*\})?\s*<\|im_end\|>\s*$")


def strip_cot(decoded: str) -> str:
    """Strip the trailing `</think>[\\boxed{...}]<|im_end|>` so we keep only CoT."""
    return _BOXED_TAIL.sub("", decoded).rstrip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading vocab ...")
    vocab = load_vocab()
    print(f"  vocab size: {len(vocab)}")

    print("Loading corpus.jsonl ...")
    corpus = [json.loads(line) for line in open(CORPUS_JSONL)]
    print(f"  corpus rows: {len(corpus)}")

    print("Loading problems.jsonl status ...")
    problems = {p["id"]: p for p in (json.loads(line) for line in open(PROBLEMS_JSONL))}
    rule_found = {pid for pid, p in problems.items() if p["status"] == "rule_found"}
    print(f"  rule_found problems: {len(rule_found)}")

    skipped_no_segment = 0
    rows = []
    for entry in corpus:
        pid = entry["problem_id"]
        seg = load_segments(pid)
        if seg is None:
            skipped_no_segment += 1
            continue
        prompt_ids, cot_ids = seg
        prompt_text = strip_prompt(decode_tokens(prompt_ids, vocab))
        cot_text = strip_cot(decode_tokens(cot_ids, vocab))
        rows.append(
            {
                "id": pid,
                "prompt": prompt_text,
                "answer": entry.get("answer", ""),
                "type": entry["category"],
                "generated_cot": cot_text,
            }
        )

    df = pd.DataFrame(rows)
    print(f"\nDecoded rows: {len(df)}  (skipped {skipped_no_segment} programmatic entries with no segment file)")

    # Quick sanity check: print the first prompt + cot snippet.
    print("\n--- sample id ---")
    print(df.iloc[0]["id"])
    print("--- sample prompt (first 300 chars) ---")
    print(df.iloc[0]["prompt"][:300])
    print("--- sample CoT (first 300 chars) ---")
    print(df.iloc[0]["generated_cot"][:300])

    # 1) Full decoded corpus.
    full_csv = OUT_FULL_DIR / "nemotron_corpus_full.csv"
    df.to_csv(full_csv, index=False)
    print(f"\nWrote {full_csv}  ({full_csv.stat().st_size/1024/1024:.1f} MB)")

    # 2) rule_found only (real problems with status==rule_found).
    df_rule = df[df["id"].isin(rule_found)].reset_index(drop=True)
    rule_csv = OUT_RULE_DIR / "nemotron_corpus_rule.csv"
    df_rule.to_csv(rule_csv, index=False)
    print(f"Wrote {rule_csv}  ({rule_csv.stat().st_size/1024/1024:.1f} MB)  rows={len(df_rule)}")
    print("  by type:", df_rule["type"].value_counts().to_dict())

    # 3) rule_found + augmentations (rows whose id is NOT in problems.jsonl, i.e. CoT-only synthetic rows).
    is_aug = ~df["id"].isin(problems.keys())
    df_rule_aug = pd.concat([df_rule, df[is_aug]], ignore_index=True)
    df_rule_aug = df_rule_aug.drop_duplicates(subset=["id"]).reset_index(drop=True)
    rule_aug_csv = OUT_RULE_AUG_DIR / "nemotron_corpus_rule_aug.csv"
    df_rule_aug.to_csv(rule_aug_csv, index=False)
    print(f"Wrote {rule_aug_csv}  ({rule_aug_csv.stat().st_size/1024/1024:.1f} MB)  rows={len(df_rule_aug)}")
    print("  by type:", df_rule_aug["type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
