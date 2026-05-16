#!/usr/bin/env python3
"""Validate sft_merged_v1.csv for data quality issues."""
import csv, re, os, math
from collections import Counter, defaultdict

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
path = os.path.join(DATA, 'sft_merged_v1.csv')

with open(path) as f:
    rows = list(csv.DictReader(f))

print(f"Total rows: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")

issues = []

# 1. Check for missing/empty fields
print("\n=== 1. Missing fields ===")
for col in ['id', 'prompt', 'answer']:
    empty = sum(1 for r in rows if not r.get(col, '').strip())
    print(f"  {col}: {empty} empty")
    if empty:
        issues.append(f"{col} has {empty} empty values")

# 2. Check thinking field
has_thinking = sum(1 for r in rows if r.get('thinking', '').strip())
no_thinking = len(rows) - has_thinking
print(f"  thinking: {has_thinking} non-empty, {no_thinking} empty (answer-only)")

# 3. Check for duplicates
print("\n=== 2. Duplicate check ===")
# Same id+thinking combo = exact duplicate
combos = Counter((r['id'], r.get('thinking', '')[:50]) for r in rows)
exact_dupes = sum(v - 1 for v in combos.values() if v > 1)
print(f"  Exact duplicates (id+thinking): {exact_dupes}")

# Same id count (expected: some ids appear in multiple sources)
id_counts = Counter(r['id'] for r in rows)
multi = {k: v for k, v in id_counts.items() if v > 1}
print(f"  Unique IDs: {len(id_counts)}")
print(f"  IDs appearing >1 time: {len(multi)} (expected for multi-view)")
max_appear = max(id_counts.values())
print(f"  Max appearances of single ID: {max_appear}")
if max_appear > 3:
    worst = [k for k, v in id_counts.items() if v == max_appear][:3]
    print(f"  IDs with max appearances: {worst}")
    issues.append(f"Some IDs appear {max_appear} times")

# 4. Check for boxed in thinking (should NOT be there)
print("\n=== 3. Boxed leak check ===")
boxed_in_thinking = 0
boxed_samples = []
for r in rows:
    t = r.get('thinking', '')
    if '\\boxed' in t or 'boxed{' in t:
        boxed_in_thinking += 1
        if len(boxed_samples) < 3:
            boxed_samples.append((r['id'], t[:100]))
print(f"  Thinking contains \\boxed: {boxed_in_thinking}")
if boxed_in_thinking:
    issues.append(f"{boxed_in_thinking} rows have \\boxed in thinking")
    for sid, sample in boxed_samples:
        print(f"    {sid}: {sample}")

# 5. Check for DSL leak words in full CoT
print("\n=== 4. DSL leak check (in full CoT only) ===")
leak_words = ['dsl', 'machine solution', 'machine-generated', 'rewrite', 'rewriting']
leak_count = 0
for r in rows:
    t = r.get('thinking', '').lower()
    if len(t) > 200:  # likely full CoT, not compact
        for w in leak_words:
            if w in t:
                leak_count += 1
                break
print(f"  Full CoT with DSL leak words: {leak_count}")

# 6. Check prompt suffix (should NOT have boxed suffix already)
print("\n=== 5. Prompt format check ===")
has_suffix = sum(1 for r in rows if 'boxed' in r['prompt'].lower())
print(f"  Prompts containing 'boxed': {has_suffix} (should be 0, added by training script)")
if has_suffix:
    issues.append(f"{has_suffix} prompts already contain 'boxed'")

# 7. Check answer quality
print("\n=== 6. Answer quality ===")
empty_answer = sum(1 for r in rows if not r['answer'].strip())
print(f"  Empty answers: {empty_answer}")
avg_answer_len = sum(len(r['answer']) for r in rows) / len(rows)
print(f"  Avg answer length: {avg_answer_len:.1f} chars")

# 8. Check thinking length distribution
print("\n=== 7. Thinking length distribution ===")
thinking_lens = []
for r in rows:
    t = r.get('thinking', '')
    if t.strip():
        thinking_lens.append(len(t.split()))
if thinking_lens:
    print(f"  Non-empty thinking: {len(thinking_lens)}")
    thinking_lens.sort()
    print(f"  Min: {thinking_lens[0]}, Median: {thinking_lens[len(thinking_lens)//2]}, Max: {thinking_lens[-1]}")
    print(f"  P10: {thinking_lens[len(thinking_lens)//10]}, P90: {thinking_lens[9*len(thinking_lens)//10]}")
    very_short = sum(1 for x in thinking_lens if x < 5)
    very_long = sum(1 for x in thinking_lens if x > 300)
    print(f"  Very short (<5 words): {very_short}")
    print(f"  Very long (>300 words): {very_long}")
    if very_short:
        issues.append(f"{very_short} thinking entries < 5 words")

# 9. Check for Unicode/encoding issues
print("\n=== 8. Encoding check ===")
unicode_issues = 0
for r in rows:
    for col in ['prompt', 'answer', 'thinking']:
        v = r.get(col, '')
        if any(ord(c) > 0x4000 and ord(c) < 0xFFFF for c in v if ord(c) > 127):
            # Exclude CJK which might be OK in prompts
            pass
        # Check for null bytes or control chars
        if '\x00' in v or '\x01' in v:
            unicode_issues += 1
print(f"  Null/control chars: {unicode_issues}")

# 10. Type distribution sanity
print("\n=== 9. Verify answer matches original ===")
# Load original for spot check
import json
with open(os.path.join(DATA, 'train_annotated.csv')) as f:
    orig = {r['id']: r for r in csv.DictReader(f)}

mismatch = 0
for r in rows:
    o = orig.get(r['id'])
    if o and o['answer'].strip() != r['answer'].strip():
        mismatch += 1
        if mismatch <= 3:
            print(f"  MISMATCH {r['id']}: merged='{r['answer']}' vs orig='{o['answer']}'")
print(f"  Answer mismatches vs original: {mismatch}")
if mismatch:
    issues.append(f"{mismatch} answer mismatches")

# 11. Check CSV can be read back correctly
print("\n=== 10. CSV round-trip check ===")
with open(path) as f:
    reader = csv.DictReader(f)
    reread = list(reader)
print(f"  Re-read rows: {len(reread)} (expected {len(rows)})")
if len(reread) != len(rows):
    issues.append(f"CSV round-trip mismatch: {len(reread)} vs {len(rows)}")

# Summary
print(f"\n{'='*60}")
if issues:
    print(f"ISSUES FOUND: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("ALL CHECKS PASSED")
print(f"{'='*60}")
