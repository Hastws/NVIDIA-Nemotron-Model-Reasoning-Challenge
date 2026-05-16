#!/usr/bin/env python3
"""Deep analysis of multisample grid n8 — join with train.csv for type info"""
import json
from collections import Counter, defaultdict

FILE = "data/nemotron3_nano_train_multisample_grid_n8.jsonl"

# Load type mapping by inferring from prompts
id_to_type = {}

def infer_type(prompt):
    p = prompt.lower()
    if "bit manipulation" in p or "8-bit binary" in p:
        return "bit_ops"
    if "gravitational" in p or "gravity" in p or "free fall" in p or "free-fall" in p:
        return "gravity"
    if "numeral system" in p or "numerical system" in p or "wonderland numeral" in p:
        return "numeral"
    if "encryption" in p or "cipher" in p or "encrypted" in p or "secret language" in p or "ciphertext" in p:
        return "cipher"
    if ("unit" in p and ("convert" in p or "conversion" in p)) or "measurement system" in p:
        return "unit_conv"
    if "symbol" in p or "symbolic" in p or "equation" in p:
        return "symbol"
    return "unknown"

# First pass: build type mapping
with open(FILE) as f:
    for line in f:
        d = json.loads(line)
        qid = d["id"]
        if qid not in id_to_type:
            id_to_type[qid] = infer_type(d.get("prompt", ""))

print(f"Type mapping: {len(id_to_type)} questions")
print(f"Types: {Counter(id_to_type.values())}")

# Full scan with type resolution
type_counter = Counter()
correct_by_type = defaultdict(int)
total_by_type = defaultdict(int)
truncated_by_type = defaultdict(int)
correct_by_temp_type = defaultdict(lambda: defaultdict(int))
total_by_temp_type = defaultdict(lambda: defaultdict(int))
correct_think_lens = defaultdict(list)
correct_answer_lens = defaultdict(list)
all_think_lens = defaultdict(list)

# Per question: how many of 8 samples are correct?
question_correct_count = defaultdict(int)
question_total_count = defaultdict(int)
question_type = {}

# Unique questions
unique_ids = set()
temps_seen = set()

total = 0
with open(FILE) as f:
    for line in f:
        d = json.loads(line)
        total += 1
        qid = d["id"]
        unique_ids.add(qid)
        qtype = id_to_type.get(qid, "unknown")
        question_type[qid] = qtype
        temp = d.get("temperature", 0)
        temps_seen.add(temp)
        
        type_counter[qtype] += 1
        total_by_type[qtype] += 1
        total_by_temp_type[temp][qtype] += 1
        question_total_count[qid] += 1
        
        correct = d.get("correct", False)
        if correct:
            correct_by_type[qtype] += 1
            correct_by_temp_type[temp][qtype] += 1
            question_correct_count[qid] += 1
            correct_think_lens[qtype].append(len(d.get("think", "")))
            correct_answer_lens[qtype].append(len(d.get("answer", "")))
        
        finish = d.get("finish_reason", "")
        if finish == "length":
            truncated_by_type[qtype] += 1
        
        all_think_lens[qtype].append(len(d.get("think", "")))

print(f"\nTotal rows: {total}")
print(f"Unique questions: {len(unique_ids)}")
print(f"Samples per question: {total / len(unique_ids):.1f}")
print(f"Temperatures: {sorted(temps_seen)}")
print(f"Questions NOT in train.csv: {sum(1 for q in unique_ids if q not in id_to_type)}")

# Per-type breakdown
print("\n" + "=" * 80)
print("PER-TYPE ACCURACY")
print("=" * 80)
print(f"{'Type':12s} {'Total':>7s} {'Correct':>8s} {'Acc%':>7s} {'Trunc':>7s} {'Trunc%':>7s}")
for t in sorted(type_counter.keys()):
    tot = total_by_type[t]
    corr = correct_by_type[t]
    trunc = truncated_by_type[t]
    print(f"{t:12s} {tot:7d} {corr:8d} {corr/tot*100:6.1f}% {trunc:7d} {trunc/tot*100:6.1f}%")

total_corr = sum(correct_by_type.values())
total_all = sum(total_by_type.values())
total_trunc = sum(truncated_by_type.values())
print(f"{'TOTAL':12s} {total_all:7d} {total_corr:8d} {total_corr/total_all*100:6.1f}% {total_trunc:7d} {total_trunc/total_all*100:6.1f}%")

# Per-temp per-type accuracy
print("\n" + "=" * 80)
print("PER-TEMPERATURE × TYPE ACCURACY (%)")
print("=" * 80)
types_sorted = sorted(type_counter.keys())
header = f"{'Temp':>6s}" + "".join(f" {t:>10s}" for t in types_sorted) + f" {'ALL':>8s}"
print(header)
for temp in sorted(temps_seen):
    row = f"{temp:6.1f}"
    temp_total = 0
    temp_correct = 0
    for t in types_sorted:
        tot = total_by_temp_type[temp][t]
        corr = correct_by_temp_type[temp][t]
        acc = corr / tot * 100 if tot > 0 else 0
        row += f" {acc:9.1f}%"
        temp_total += tot
        temp_correct += corr
    row += f" {temp_correct/temp_total*100:7.1f}%"
    print(row)

# Per-question coverage: how many questions have at least 1 correct across all temps?
print("\n" + "=" * 80)
print("PER-QUESTION COVERAGE (≥1 correct sample)")
print("=" * 80)
questions_with_correct = defaultdict(int)
questions_total = defaultdict(int)
for qid in unique_ids:
    qt = question_type.get(qid, "unknown")
    questions_total[qt] += 1
    if question_correct_count[qid] > 0:
        questions_with_correct[qt] += 1

# Unique questions per type
n_unique_per_type = defaultdict(int)
for qid in unique_ids:
    n_unique_per_type[question_type.get(qid, "unknown")] += 1

print(f"{'Type':12s} {'Questions':>10s} {'≥1Correct':>10s} {'Coverage%':>10s}")
for t in sorted(questions_total.keys()):
    n = questions_total[t]
    c = questions_with_correct[t]
    print(f"{t:12s} {n:10d} {c:10d} {c/n*100:9.1f}%")

total_q = len(unique_ids)
total_qc = sum(questions_with_correct.values())
print(f"{'TOTAL':12s} {total_q:10d} {total_qc:10d} {total_qc/total_q*100:9.1f}%")

# Distribution of correct count per question
print("\n--- Correct samples per question distribution ---")
corr_dist = Counter()
for qid in unique_ids:
    corr_dist[question_correct_count[qid]] += 1
print(f"{'#Correct':>10s} {'#Questions':>12s} {'%':>7s}")
for k in sorted(corr_dist.keys()):
    print(f"{k:10d} {corr_dist[k]:12d} {corr_dist[k]/len(unique_ids)*100:6.1f}%")

# Think length stats for correct samples
print("\n" + "=" * 80)
print("THINK LENGTH STATS (correct samples only)")
print("=" * 80)
print(f"{'Type':12s} {'Count':>7s} {'Min':>7s} {'P25':>7s} {'P50':>7s} {'P75':>7s} {'Max':>8s} {'Mean':>8s}")
for t in sorted(correct_think_lens.keys()):
    lens = sorted(correct_think_lens[t])
    n = len(lens)
    if n == 0:
        continue
    p25 = lens[n // 4]
    p50 = lens[n // 2]
    p75 = lens[3 * n // 4]
    avg = sum(lens) / n
    print(f"{t:12s} {n:7d} {lens[0]:7d} {p25:7d} {p50:7d} {p75:7d} {lens[-1]:8d} {avg:8.0f}")

# Token budget estimation
print("\n" + "=" * 80)
print("TOKEN BUDGET ESTIMATION")
print("=" * 80)
total_correct = sum(len(v) for v in correct_think_lens.values())
total_think_chars = sum(sum(v) for v in correct_think_lens.values())
total_answer_chars = sum(sum(v) for v in correct_answer_lens.values())
print(f"Total correct samples: {total_correct}")
print(f"Total think chars: {total_think_chars:,} (~{total_think_chars//4:,} tokens)")
print(f"Total answer chars: {total_answer_chars:,} (~{total_answer_chars//4:,} tokens)")
print(f"E1 reference: ~48,000 tokens")
print(f"All correct think tokens: ~{total_think_chars//4:,} = {total_think_chars//4/48000:.0f}x E1")

# Strategy: select shortest correct per question
print("\n--- Strategy: Shortest correct think per question ---")
shortest_by_type = defaultdict(list)
with open(FILE) as f:
    question_samples = defaultdict(list)
    for line in f:
        d = json.loads(line)
        if d.get("correct", False):
            qid = d["id"]
            question_samples[qid].append(d)

for qid, samples in question_samples.items():
    qt = question_type.get(qid, "unknown")
    shortest = min(samples, key=lambda s: len(s.get("think", "")))
    shortest_by_type[qt].append(len(shortest.get("think", "")))

print(f"{'Type':12s} {'Count':>7s} {'AvgThink':>10s} {'TotalChars':>12s}")
grand_total_chars = 0
grand_total_count = 0
for t in sorted(shortest_by_type.keys()):
    lens = shortest_by_type[t]
    avg = sum(lens) / len(lens)
    tot = sum(lens)
    grand_total_chars += tot
    grand_total_count += len(lens)
    print(f"{t:12s} {len(lens):7d} {avg:10.0f} {tot:12,}")
print(f"{'TOTAL':12s} {grand_total_count:7d} {grand_total_chars/grand_total_count:10.0f} {grand_total_chars:12,}")
print(f"Estimated tokens: ~{grand_total_chars//4:,} ({grand_total_chars//4/48000:.1f}x E1)")

# Weighted-loss effective tokens (think_weight=0.1)
print(f"\nWeighted-loss (think=0.1, answer=1.0):")
eff_think = grand_total_chars // 4 * 0.1
eff_answer = grand_total_count * 20  # ~20 answer tokens per sample
print(f"  Effective think tokens: {eff_think:,.0f}")
print(f"  Effective answer tokens: {eff_answer:,.0f}")
print(f"  Total effective: {eff_think + eff_answer:,.0f} ({(eff_think + eff_answer)/48000:.1f}x E1)")
