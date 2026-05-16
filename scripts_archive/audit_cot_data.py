"""
Deep audit of sft_full_cot.csv — find all quality issues that limit performance.
"""
import csv
import re
from collections import Counter, defaultdict

def detect_type(p):
    p = p[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p): return "bit_ops"
    elif "encrypt" in p or "cipher" in p: return "cipher"
    elif "gravit" in p: return "gravity"
    elif "numeral" in p or "wonderland numbers" in p: return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p): return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p): return "symbol"
    return "unknown"

with open("data/sft_full_cot.csv") as f:
    rows = list(csv.DictReader(f))

# Also load full train.csv for comparison
with open("competition_data/train.csv") as f:
    train = {r["id"]: r for r in csv.DictReader(f)}

by_type = defaultdict(list)
for r in rows:
    by_type[detect_type(r["prompt"])].append(r)

print("=" * 60)
print("ISSUE 1: Thinking template diversity & rigidity")
print("=" * 60)
# Check if all thinking for same type starts the same way
for t in sorted(by_type):
    items = by_type[t]
    # First 50 chars of thinking
    first_lines = [r["thinking"].split("\n")[0][:80] for r in items[:20]]
    unique_starts = set(first_lines)
    print(f"\n{t} ({len(items)} samples) — {len(unique_starts)} unique first lines out of 20:")
    for s in list(unique_starts)[:5]:
        print(f"  '{s}'")

print("\n" + "=" * 60)
print("ISSUE 2: Thinking contains \\boxed{} or final answer?")
print("=" * 60)
for t in sorted(by_type):
    items = by_type[t]
    has_boxed = sum(1 for r in items if "\\boxed" in r["thinking"])
    has_answer_in_think = 0
    for r in items:
        ans = r["answer"].strip()
        if ans in r["thinking"]:
            has_answer_in_think += 1
    print(f"  {t}: boxed in thinking: {has_boxed}/{len(items)}, answer literal in thinking: {has_answer_in_think}/{len(items)}")

print("\n" + "=" * 60)
print("ISSUE 3: Thinking quality — does it actually reason or just state?")
print("=" * 60)
# Show full thinking for 1 sample per type
for t in sorted(by_type):
    items = by_type[t]
    r = items[0]
    print(f"\n--- {t} full thinking (sample 0, {len(r['thinking'])} chars) ---")
    print(r["thinking"][:600])
    print(f"--- answer: {r['answer']} ---")

print("\n" + "=" * 60)
print("ISSUE 4: Missing coverage — what's NOT in the data?")
print("=" * 60)
cot_ids = set(r["id"] for r in rows)
missing = [(k, v) for k, v in train.items() if k not in cot_ids]
missing_types = Counter(detect_type(v["prompt"]) for _, v in missing)
print(f"Total missing: {len(missing)}/9500")
for t in sorted(missing_types):
    print(f"  {t}: {missing_types[t]} samples missing")

print("\n" + "=" * 60)
print("ISSUE 5: Answer format consistency")
print("=" * 60)
for t in sorted(by_type):
    items = by_type[t]
    # Check answer patterns
    ans_types = Counter()
    for r in items:
        a = r["answer"].strip()
        if re.match(r"^[01]{8}$", a): ans_types["8bit_binary"] += 1
        elif re.match(r"^-?\d+\.\d+$", a): ans_types["decimal"] += 1
        elif re.match(r"^-?\d+$", a): ans_types["integer"] += 1
        elif re.match(r"^[IVXLCDM]+$", a): ans_types["roman"] += 1
        elif re.match(r"^[a-zA-Z\s]+$", a): ans_types["text"] += 1
        else: ans_types[f"other:{a[:20]}"] += 1
    print(f"  {t}: {dict(ans_types)}")

print("\n" + "=" * 60)
print("ISSUE 6: bit_ops 'ambiguous' markers in thinking")
print("=" * 60)
bit_items = by_type.get("bit_ops", [])
ambiguous_count = sum(1 for r in bit_items if "[ambiguous]" in r["thinking"])
print(f"  bit_ops with [ambiguous]: {ambiguous_count}/{len(bit_items)}")
# Show how ambiguous is resolved
for r in bit_items[:3]:
    amb = r["thinking"].count("[ambiguous]")
    print(f"    id={r['id'][:8]}... [ambiguous] count: {amb}")

print("\n" + "=" * 60)
print("ISSUE 7: Training format — how would this be fed to model?")
print("=" * 60)
# The KEY question: when training with enable_thinking=True + reasoning_content,
# how does apply_chat_template format it?
# Current v29 approach: answer-only, mask everything except \boxed{answer}
# Alternative: put thinking in reasoning_content field
print("Current v29: answer=\\boxed{ans}, thinking IGNORED (boxed-only loss)")
print("If we use thinking: need to format as reasoning_content in chat template")
print("Previous attempts (v21=0.56, v22=0.57) used this approach and FAILED")
print()
print("KEY QUESTION: Is the thinking format compatible with model's native style?")

# Check token count when thinking IS included
sample = by_type["bit_ops"][0]
full_text = f"<think>\n{sample['thinking']}\n</think>\n\\boxed{{{sample['answer']}}}"
print(f"\nSample bit_ops with thinking: {len(full_text)} chars (~{len(full_text)//4} tokens)")
sample2 = by_type["cipher"][0]
full_text2 = f"<think>\n{sample2['thinking']}\n</think>\n\\boxed{{{sample2['answer']}}}"
print(f"Sample cipher with thinking: {len(full_text2)} chars (~{len(full_text2)//4} tokens)")
