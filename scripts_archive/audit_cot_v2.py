#!/usr/bin/env python3
"""Comprehensive quality audit for data/cot_v2.jsonl"""

import json
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
COT_PATH = BASE / "data" / "cot_v2.jsonl"
TRAIN_PATH = BASE / "competition_data" / "train.csv"

REQUIRED_FIELDS = {"id", "type", "prompt", "answer", "thinking"}

# ── 1. Load data ────────────────────────────────────────────────────────────

print("=" * 80)
print("COT V2 QUALITY AUDIT")
print("=" * 80)

# Load CoT records
cot_records = []
with open(COT_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            cot_records.append(rec)
        except json.JSONDecodeError as e:
            print(f"  [JSON ERROR] Line {i}: {e}")

print(f"\nTotal CoT records loaded: {len(cot_records)}")

# Load train.csv gold answers
gold = {}
train_types = {}
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gold[row["id"]] = row["answer"]
        # train.csv may not have type column — check
        if "type" in row:
            train_types[row["id"]] = row["type"]

print(f"Total train.csv records: {len(gold)}")

# ── 2. Format consistency check ─────────────────────────────────────────────

print("\n" + "=" * 80)
print("SECTION 1: FORMAT CONSISTENCY")
print("=" * 80)

missing_fields = []
empty_thinking = []
short_thinking = []   # ≤ 1 line
format_issues = []

for rec in cot_records:
    rid = rec.get("id", "UNKNOWN")
    # Check required fields
    missing = REQUIRED_FIELDS - set(rec.keys())
    if missing:
        missing_fields.append((rid, missing))
    
    thinking = rec.get("thinking", "")
    # Empty thinking
    if not thinking or not thinking.strip():
        empty_thinking.append(rid)
    elif thinking.strip().count("\n") == 0 and len(thinking.strip()) < 20:
        short_thinking.append((rid, thinking.strip()))
    
    # Check for obvious format problems
    if thinking and ("\x00" in thinking or "\ufffd" in thinking):
        format_issues.append((rid, "contains null/replacement chars"))

print(f"\nRecords with missing fields: {len(missing_fields)}")
for rid, ms in missing_fields[:5]:
    print(f"  ID {rid}: missing {ms}")

print(f"Records with empty thinking: {len(empty_thinking)}")
for rid in empty_thinking[:5]:
    print(f"  ID {rid}")

print(f"Records with very short thinking (1 line, <20 chars): {len(short_thinking)}")
for rid, t in short_thinking[:5]:
    print(f"  ID {rid}: '{t}'")

print(f"Format issues (garbled/null chars): {len(format_issues)}")

# ── 3. Duplicate ID check ──────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SECTION 2: DUPLICATE ID CHECK")
print("=" * 80)

id_counts = Counter(rec["id"] for rec in cot_records)
dups = {k: v for k, v in id_counts.items() if v > 1}
print(f"Unique IDs: {len(id_counts)}")
print(f"Duplicate IDs: {len(dups)}")
if dups:
    for did, cnt in list(dups.items())[:10]:
        print(f"  ID {did}: appears {cnt} times")

# ── 4. Coverage analysis ────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SECTION 3: COVERAGE ANALYSIS")
print("=" * 80)

type_counts_cot = Counter(rec.get("type", "UNKNOWN") for rec in cot_records)
# Count train types if available, else estimate from ID presence
if train_types:
    type_counts_train = Counter(train_types.values())
else:
    type_counts_train = {}

print(f"\n{'Type':<15} {'CoT':>6} {'Train':>6} {'Coverage':>10}")
print("-" * 42)
for t in sorted(set(list(type_counts_cot.keys()) + list(type_counts_train.keys()))):
    cot_c = type_counts_cot.get(t, 0)
    train_c = type_counts_train.get(t, 0) if type_counts_train else "?"
    if isinstance(train_c, int) and train_c > 0:
        cov = f"{cot_c/train_c*100:.1f}%"
    else:
        cov = "?"
    print(f"{t:<15} {cot_c:>6} {str(train_c):>6} {cov:>10}")

print(f"\nTotal CoT: {len(cot_records)}")

# ── 5. Answer exact match verification ──────────────────────────────────────

print("\n" + "=" * 80)
print("SECTION 4: ANSWER EXACT MATCH vs TRAIN.CSV")
print("=" * 80)

match_count = 0
mismatch_count = 0
not_in_train = 0
mismatches_by_type = defaultdict(list)

for rec in cot_records:
    rid = rec["id"]
    cot_ans = rec.get("answer", "")
    if rid not in gold:
        not_in_train += 1
        continue
    gold_ans = gold[rid]
    if cot_ans.strip() == gold_ans.strip():
        match_count += 1
    else:
        mismatch_count += 1
        rtype = rec.get("type", "UNKNOWN")
        mismatches_by_type[rtype].append((rid, cot_ans, gold_ans))

total_checked = match_count + mismatch_count
print(f"\nExact match: {match_count}/{total_checked} ({match_count/total_checked*100:.2f}%)" if total_checked else "No records to check")
print(f"Mismatches: {mismatch_count}")
print(f"IDs not found in train.csv: {not_in_train}")

if mismatch_count > 0:
    print(f"\nMismatches by type:")
    for t, items in sorted(mismatches_by_type.items()):
        print(f"  {t}: {len(items)} mismatches")
        for rid, cot_a, gold_a in items[:3]:
            print(f"    ID {rid}: CoT='{cot_a[:60]}' vs Gold='{gold_a[:60]}'")

# ── 6. CoT quality analysis ─────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SECTION 5: COT QUALITY ANALYSIS (thinking length)")
print("=" * 80)

type_lens = defaultdict(list)
for rec in cot_records:
    t = rec.get("type", "UNKNOWN")
    thinking = rec.get("thinking", "")
    type_lens[t].append(len(thinking))

print(f"\n{'Type':<15} {'Count':>6} {'Mean':>8} {'Min':>6} {'Max':>6} {'Median':>8}")
print("-" * 55)
for t in sorted(type_lens.keys()):
    lens = sorted(type_lens[t])
    n = len(lens)
    mean_l = sum(lens) / n
    med = lens[n // 2]
    print(f"{t:<15} {n:>6} {mean_l:>8.0f} {lens[0]:>6} {lens[-1]:>6} {med:>8}")

# Check for extremely short thinking
print(f"\nRecords with thinking < 50 chars:")
for rec in cot_records:
    if len(rec.get("thinking", "")) < 50:
        print(f"  [{rec.get('type')}] ID {rec['id']}: '{rec.get('thinking','')[:80]}'")

# ── 7. Spot-check correctness (10 random per type) ─────────────────────────

print("\n" + "=" * 80)
print("SECTION 6: SPOT-CHECK — 10 RANDOM SAMPLES PER TYPE")
print("=" * 80)

by_type = defaultdict(list)
for rec in cot_records:
    by_type[rec.get("type", "UNKNOWN")].append(rec)

for t in sorted(by_type.keys()):
    samples = random.sample(by_type[t], min(10, len(by_type[t])))
    print(f"\n--- {t.upper()} ({len(by_type[t])} total, showing {len(samples)} samples) ---")
    for rec in samples:
        rid = rec["id"]
        thinking = rec.get("thinking", "")
        answer = rec.get("answer", "")
        gold_ans = gold.get(rid, "N/A")
        match_flag = "✓" if answer.strip() == gold_ans.strip() else "✗"
        lines = thinking.strip().split("\n")
        # Show first 2 and last 2 lines of thinking
        if len(lines) <= 6:
            preview = thinking.strip()
        else:
            preview = "\n".join(lines[:3]) + "\n  ...\n" + "\n".join(lines[-2:])
        print(f"\n  [{match_flag}] ID: {rid}")
        print(f"      Answer: {answer[:80]}")
        print(f"      Gold:   {gold_ans[:80]}")
        print(f"      Thinking ({len(thinking)} chars, {len(lines)} lines):")
        for pl in preview.split("\n")[:8]:
            print(f"        {pl}")

# ── 8. Specific correctness deep checks ─────────────────────────────────────

print("\n" + "=" * 80)
print("SECTION 7: DEEP CORRECTNESS CHECKS")
print("=" * 80)

# Check if thinking contains "Result:" at the end and it matches the answer
result_line_issues = []
for rec in cot_records:
    thinking = rec.get("thinking", "")
    answer = rec.get("answer", "")
    # Check if thinking ends with the answer
    last_lines = thinking.strip().split("\n")[-3:]
    last_text = " ".join(last_lines)
    if answer and answer not in last_text:
        result_line_issues.append((rec["id"], rec.get("type"), answer))

print(f"\nRecords where answer NOT found in last 3 lines of thinking: {len(result_line_issues)}")
if result_line_issues:
    for rid, rtype, ans in result_line_issues[:10]:
        print(f"  [{rtype}] ID {rid}: answer='{ans[:60]}'")

# Check gravity: look for numeric consistency in thinking
gravity_records = by_type.get("gravity", [])
gravity_issues = []
for rec in gravity_records[:50]:  # check first 50
    thinking = rec.get("thinking", "")
    answer = rec.get("answer", "")
    # Check if the final answer appears in thinking
    if answer not in thinking:
        gravity_issues.append(rec["id"])

print(f"\nGravity: answer not in thinking (out of {min(50, len(gravity_records))} checked): {len(gravity_issues)}")

# numeral: check pattern
numeral_records = by_type.get("numeral", [])
numeral_issues = []
for rec in numeral_records:
    thinking = rec.get("thinking", "")
    answer = rec.get("answer", "")
    if answer not in thinking:
        numeral_issues.append(rec["id"])
print(f"Numeral: answer not in thinking: {len(numeral_issues)}/{len(numeral_records)}")

# cipher: check that "Result:" or decrypted text appears
cipher_records = by_type.get("cipher", [])
cipher_issues = []
for rec in cipher_records:
    thinking = rec.get("thinking", "")
    answer = rec.get("answer", "")
    if answer not in thinking:
        cipher_issues.append(rec["id"])
print(f"Cipher: answer not in thinking: {len(cipher_issues)}/{len(cipher_records)}")

# bit_ops: check
bit_ops_records = by_type.get("bit_ops", [])
bitops_issues = []
for rec in bit_ops_records:
    thinking = rec.get("thinking", "")
    answer = rec.get("answer", "")
    if answer not in thinking:
        bitops_issues.append(rec["id"])
print(f"Bit_ops: answer not in thinking: {len(bitops_issues)}/{len(bit_ops_records)}")

# unit_conv: check
unit_conv_records = by_type.get("unit_conv", [])
uc_issues = []
for rec in unit_conv_records:
    thinking = rec.get("thinking", "")
    answer = rec.get("answer", "")
    if answer not in thinking:
        uc_issues.append(rec["id"])
print(f"Unit_conv: answer not in thinking: {len(uc_issues)}/{len(unit_conv_records)}")

# ── 9. Summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Total records:              {len(cot_records)}
Unique IDs:                 {len(id_counts)}
Duplicate IDs:              {len(dups)}
Missing fields:             {len(missing_fields)}
Empty thinking:             {len(empty_thinking)}
Very short thinking:        {len(short_thinking)}
Format issues (garbled):    {len(format_issues)}

Answer exact match rate:    {match_count}/{total_checked} = {match_count/total_checked*100:.2f}%
  Mismatches:               {mismatch_count}
  Not in train.csv:         {not_in_train}

Answer not in thinking:     {len(result_line_issues)} records
""")

print("Type breakdown:")
for t in sorted(type_counts_cot.keys()):
    print(f"  {t}: {type_counts_cot[t]} records")
