#!/usr/bin/env python3
"""Validate the difficulty-aware dataset - check quality, no duplicates, correct format."""
import csv
from collections import Counter

data_path = "data/sft_difficulty_aware.csv"

rows = []
with open(data_path) as f:
    rows = list(csv.DictReader(f))

print(f"Total rows: {len(rows)}")

# Check for duplicate IDs
ids = [r["id"] for r in rows]
dupes = [id for id, cnt in Counter(ids).items() if cnt > 1]
print(f"Duplicate IDs: {len(dupes)}")

# Check each problem has exactly ONE mode
print(f"Modes: {Counter(r['mode'] for r in rows)}")

# Verify: thinking mode should have non-empty thinking
issues = []
for r in rows:
    if r["mode"] in ("compact", "full_cot") and not r["thinking"].strip():
        issues.append(f"  Empty thinking: id={r['id']} mode={r['mode']} type={r['type']}")
    if r["mode"] == "answer_only" and r["thinking"].strip():
        issues.append(f"  Unexpected thinking in answer_only: id={r['id']}")

print(f"\nData quality issues: {len(issues)}")
for i in issues[:10]:
    print(i)

# Thinking length distribution by type × mode
print("\n=== Thinking length by type × mode ===")
from collections import defaultdict
stats = defaultdict(list)
for r in rows:
    key = (r["type"], r["mode"])
    stats[key].append(len(r["thinking"]))

for mode in ["compact", "full_cot"]:
    print(f"\n--- {mode} ---")
    for ptype in ["numeral", "gravity", "unit_conv", "cipher", "bit_ops", "symbol"]:
        lens = stats.get((ptype, mode), [])
        if lens:
            avg = sum(lens) / len(lens)
            print(f"  {ptype:>10s}: n={len(lens):5d}, avg={avg:6.0f}, min={min(lens):5d}, max={max(lens):5d}")

# Token estimate (rough: ~4 chars per token)
print("\n=== Token budget estimate (1 token ≈ 4 chars) ===")
for mode in ["answer_only", "compact", "full_cot"]:
    mode_rows = [r for r in rows if r["mode"] == mode]
    if not mode_rows:
        continue
    prompt_lens = [len(r["prompt"]) for r in mode_rows]
    think_lens = [len(r["thinking"]) for r in mode_rows]
    answer_lens = [len(r["answer"]) for r in mode_rows]
    total_chars = [len(r["prompt"]) + len(r["thinking"]) + len(r["answer"]) for r in mode_rows]
    avg_tokens = sum(total_chars) / len(total_chars) / 4
    max_tokens = max(total_chars) / 4
    print(f"  {mode:12s}: n={len(mode_rows):5d}, avg_tokens≈{avg_tokens:6.0f}, max_tokens≈{max_tokens:6.0f}")

# Check no same-problem has multiple modes (by construction, but verify)
id_modes = [(r["id"], r["mode"]) for r in rows]
id_counts = Counter(r["id"] for r in rows)
multi = {id: cnt for id, cnt in id_counts.items() if cnt > 1}
print(f"\nProblems with multiple modes: {len(multi)}")

print("\n✅ Validation complete")
