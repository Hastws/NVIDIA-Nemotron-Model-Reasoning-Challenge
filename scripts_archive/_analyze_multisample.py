#!/usr/bin/env python3
"""Analyze nemotron3_nano_train_multisample_grid_n8.jsonl"""
import json
import sys
from collections import Counter, defaultdict

FILE = "data/nemotron3_nano_train_multisample_grid_n8.jsonl"

# --- Phase 1: Schema inspection ---
print("=" * 60)
print("PHASE 1: Schema Inspection")
print("=" * 60)

with open(FILE) as f:
    first = json.loads(f.readline())

print(f"Keys: {list(first.keys())}")
for k, v in first.items():
    if isinstance(v, str):
        print(f"  {k}: str, len={len(v)}")
        print(f"    preview: {repr(v[:200])}")
    elif isinstance(v, list):
        print(f"  {k}: list, len={len(v)}")
        if v:
            item = v[0]
            if isinstance(item, dict):
                print(f"    [0] keys: {list(item.keys())}")
                for ik, iv in item.items():
                    if isinstance(iv, str):
                        print(f"      {ik}: str, len={len(iv)}, preview={repr(iv[:120])}")
                    else:
                        print(f"      {ik}: {type(iv).__name__}, val={repr(iv)[:120]}")
            else:
                print(f"    [0]: {type(item).__name__}, val={repr(item)[:120]}")
    elif isinstance(v, dict):
        print(f"  {k}: dict, keys={list(v.keys())}")
        for ik, iv in v.items():
            print(f"    {ik}: {repr(iv)[:120]}")
    else:
        print(f"  {k}: {type(v).__name__}, val={repr(v)[:120]}")

# --- Phase 2: Full scan ---
print("\n" + "=" * 60)
print("PHASE 2: Full Dataset Analysis")
print("=" * 60)

total = 0
type_counter = Counter()
correct_counter = Counter()
correct_by_type = defaultdict(int)
total_by_type = defaultdict(int)
truncated_counter = Counter()
truncated_by_type = defaultdict(int)
think_lens = defaultdict(list)
answer_lens = defaultdict(list)
n_samples_per_row = []
temps_seen = set()
has_correct_any = 0  # rows where at least 1 sample is correct

# Track per-temp accuracy
correct_by_temp = defaultdict(int)
total_by_temp = defaultdict(int)

with open(FILE) as f:
    for line in f:
        d = json.loads(line)
        total += 1
        
        # Detect type from prompt or field
        qtype = d.get("type", "unknown")
        if qtype == "unknown" and "prompt" in d:
            # Try to infer type
            pass
        
        type_counter[qtype] += 1
        
        # Check if there are multiple samples
        samples = d.get("samples", [])
        if not samples:
            # Maybe the samples are at top level?
            # Check for alternative structures
            if "think" in d and "answer" in d:
                samples = [d]
            elif "responses" in d:
                samples = d["responses"]
        
        n_samples_per_row.append(len(samples))
        
        row_has_correct = False
        for s in samples:
            temp = s.get("temperature", s.get("temp", None))
            if temp is not None:
                temps_seen.add(temp)
                total_by_temp[temp] += 1
            
            correct = s.get("correct", False)
            if correct:
                correct_counter["correct"] += 1
                correct_by_type[qtype] += 1
                row_has_correct = True
                if temp is not None:
                    correct_by_temp[temp] += 1
            else:
                correct_counter["incorrect"] += 1
            
            finish = s.get("finish_reason", "")
            if finish == "length" or s.get("truncated", False):
                truncated_counter["truncated"] += 1
                truncated_by_type[qtype] += 1
            
            think = s.get("think", s.get("thinking", ""))
            answer = s.get("answer", s.get("response", ""))
            if isinstance(think, str):
                think_lens[qtype].append(len(think))
            if isinstance(answer, str):
                answer_lens[qtype].append(len(answer))
        
        total_by_type[qtype] += len(samples)
        if row_has_correct:
            has_correct_any += 1

print(f"\nTotal rows: {total}")
print(f"Total samples: {sum(n_samples_per_row)}")
print(f"Samples per row: min={min(n_samples_per_row)}, max={max(n_samples_per_row)}, avg={sum(n_samples_per_row)/len(n_samples_per_row):.1f}")
print(f"Temperatures seen: {sorted(temps_seen)}")
print(f"\nCorrect: {correct_counter['correct']}, Incorrect: {correct_counter['incorrect']}")
print(f"Overall accuracy: {correct_counter['correct']/(correct_counter['correct']+correct_counter['incorrect'])*100:.1f}%")
print(f"Rows with ≥1 correct: {has_correct_any}/{total} ({has_correct_any/total*100:.1f}%)")
print(f"Truncated: {truncated_counter['truncated']}")

print("\n--- Per-Type Breakdown ---")
print(f"{'Type':12s} {'Rows':>6s} {'Samples':>8s} {'Correct':>8s} {'Acc%':>7s} {'≥1Corr':>7s} {'Trunc':>7s} {'AvgThink':>10s}")
for t in sorted(type_counter.keys()):
    rows = type_counter[t]
    samps = total_by_type[t]
    corr = correct_by_type[t]
    trunc = truncated_by_type[t]
    acc = corr / samps * 100 if samps > 0 else 0
    avg_think = sum(think_lens[t]) / len(think_lens[t]) if think_lens[t] else 0
    # Count rows with ≥1 correct for this type (need recount)
    print(f"{t:12s} {rows:6d} {samps:8d} {corr:8d} {acc:6.1f}% {'-':>7s} {trunc:7d} {avg_think:10.0f}")

if temps_seen:
    print("\n--- Per-Temperature Accuracy ---")
    print(f"{'Temp':>8s} {'Total':>8s} {'Correct':>8s} {'Acc%':>7s}")
    for t in sorted(temps_seen):
        tot = total_by_temp[t]
        corr = correct_by_temp[t]
        acc = corr / tot * 100 if tot > 0 else 0
        print(f"{t:8.2f} {tot:8d} {corr:8d} {acc:6.1f}%")

# --- Phase 3: Think length distribution for correct samples ---
print("\n--- Think Length Stats (correct samples only) ---")
with open(FILE) as f:
    correct_think_by_type = defaultdict(list)
    for line in f:
        d = json.loads(line)
        qtype = d.get("type", "unknown")
        samples = d.get("samples", [])
        if not samples:
            if "think" in d and "answer" in d:
                samples = [d]
            elif "responses" in d:
                samples = d["responses"]
        for s in samples:
            if s.get("correct", False):
                think = s.get("think", s.get("thinking", ""))
                if isinstance(think, str):
                    correct_think_by_type[qtype].append(len(think))

print(f"{'Type':12s} {'Count':>7s} {'Min':>7s} {'P25':>7s} {'P50':>7s} {'P75':>7s} {'Max':>8s} {'AvgChars':>10s}")
for t in sorted(correct_think_by_type.keys()):
    lens = sorted(correct_think_by_type[t])
    n = len(lens)
    if n == 0:
        continue
    p25 = lens[n // 4]
    p50 = lens[n // 2]
    p75 = lens[3 * n // 4]
    avg = sum(lens) / n
    print(f"{t:12s} {n:7d} {lens[0]:7d} {p25:7d} {p50:7d} {p75:7d} {lens[-1]:8d} {avg:10.0f}")

# Total correct for training
total_correct = sum(len(v) for v in correct_think_by_type.values())
total_chars = sum(sum(v) for v in correct_think_by_type.values())
print(f"\nTotal correct samples: {total_correct}")
print(f"Total think chars: {total_chars:,} (~{total_chars//4:,} tokens)")
