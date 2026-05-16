#!/usr/bin/env python3
"""Analyze bit_ops solver failure patterns to find improvement opportunities."""
import csv
import sys
sys.path.insert(0, '.')
from scripts.solve_bit_ops_v2 import solve_bit_ops, parse_bit_ops_prompt, solve_per_bit

# Load all bit_ops puzzles
puzzles = []
with open('competition_data/train.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = row['prompt'][:300].lower()
        if '8-bit binary' in p or ('bit' in p and 'binary' in p):
            puzzles.append(row)

print(f"Total bit_ops puzzles: {len(puzzles)}")

# Count examples per puzzle
example_counts = {}
for puz in puzzles:
    exs, _ = parse_bit_ops_prompt(puz['prompt'])
    n = len(exs) if exs else 0
    example_counts[n] = example_counts.get(n, 0) + 1
print("\nExample counts per puzzle:")
for k, v in sorted(example_counts.items()):
    print(f"  {k} examples: {v} puzzles")

# Categorize results
correct = []
wrong = []
failed = []

for row in puzzles:
    ans, info = solve_bit_ops(row['prompt'])
    gold = row['answer']
    if ans is None:
        failed.append((row, info))
    elif ans == gold:
        correct.append((row, ans, info))
    else:
        wrong.append((row, ans, info))

print(f"\nResults: correct={len(correct)}, wrong={len(wrong)}, failed={len(failed)}")

# Analyze wrong predictions
print("\n=== WRONG PREDICTIONS ANALYSIS ===")

# For wrong ones, check how many bits differ
bit_diffs = {}
for row, pred, info in wrong:
    gold = row['answer']
    diff = sum(1 for a, b in zip(pred, gold) if a != b)
    bit_diffs[diff] = bit_diffs.get(diff, 0) + 1
print("\nBit differences in wrong predictions:")
for k, v in sorted(bit_diffs.items()):
    print(f"  {k} bits differ: {v}")

# Show first 5 wrong examples with detail
print("\n--- Sample Wrong Predictions ---")
for i, (row, pred, info) in enumerate(wrong[:5]):
    exs, tgt = parse_bit_ops_prompt(row['prompt'])
    gold = row['answer']
    print(f"\nWrong #{i+1}:")
    print(f"  Examples: {len(exs)}")
    for inp, out in exs:
        print(f"    {inp} -> {out}")
    print(f"  Target:    {tgt}")
    print(f"  Predicted: {pred}")
    print(f"  Gold:      {gold}")
    # Show which bits differ
    diffs = [j for j in range(8) if pred[j] != gold[j]]
    print(f"  Diff bits: {diffs}")

# Analyze failed cases
print("\n=== FAILED CASES ANALYSIS ===")
fail_reasons = {}
for row, info in failed:
    fail_reasons[info] = fail_reasons.get(info, 0) + 1
print("Fail reasons:")
for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason}: {count}")

# For failed cases, show which output bits we CAN determine
print("\n--- Failed: partial bit analysis ---")
partial_stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
for row, info in failed[:50]:
    exs, tgt = parse_bit_ops_prompt(row['prompt'])
    if not exs or not tgt:
        partial_stats[0] += 1
        continue
    
    n = len(exs)
    inputs = [[int(ex[0][i]) for i in range(8)] for ex in exs]
    outputs = [[int(ex[1][i]) for i in range(8)] for ex in exs]
    
    # Check each bit independently
    solvable = 0
    for obit in range(8):
        out_col = [outputs[e][obit] for e in range(n)]
        for ibit in range(8):
            in_col = [inputs[e][ibit] for e in range(n)]
            if in_col == out_col or [1-x for x in in_col] == out_col:
                solvable += 1
                break
    partial_stats[solvable] += 1
print(f"Bits solvable at Level 1 (first 50 failed): {partial_stats}")

# NEW: Try global operations (rotation, reverse, NOT)
print("\n=== GLOBAL OPERATION ANALYSIS ===")
global_ops = {
    'identity': lambda s: s,
    'NOT': lambda s: ''.join('1' if c == '0' else '0' for c in s),
    'reverse': lambda s: s[::-1],
    'NOT_reverse': lambda s: ''.join('1' if c == '0' else '0' for c in s[::-1]),
}
for shift in range(1, 8):
    global_ops[f'rot_left_{shift}'] = lambda s, sh=shift: s[sh:] + s[:sh]
    global_ops[f'rot_right_{shift}'] = lambda s, sh=shift: s[-sh:] + s[:-sh]

global_correct = {}
for op_name, op_fn in global_ops.items():
    count = 0
    for row in puzzles:
        exs, tgt = parse_bit_ops_prompt(row['prompt'])
        if not exs or not tgt:
            continue
        # Check if this global op matches ALL examples
        if all(op_fn(inp) == out for inp, out in exs):
            pred = op_fn(tgt)
            if pred == row['answer']:
                count += 1
    if count > 0:
        global_correct[op_name] = count

print("Global operations that work:")
for op, count in sorted(global_correct.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count} correct")

# Check: XOR with constant
print("\n=== XOR WITH CONSTANT ANALYSIS ===")
xor_correct = 0
xor_total = 0
for row in puzzles:
    exs, tgt = parse_bit_ops_prompt(row['prompt'])
    if not exs or not tgt or len(exs) < 2:
        continue
    # Compute XOR constant from first example
    xor_const = ''.join(str(int(exs[0][0][i]) ^ int(exs[0][1][i])) for i in range(8))
    # Check if constant is consistent across all examples
    all_match = True
    for inp, out in exs[1:]:
        xc = ''.join(str(int(inp[i]) ^ int(out[i])) for i in range(8))
        if xc != xor_const:
            all_match = False
            break
    if all_match:
        xor_total += 1
        pred = ''.join(str(int(tgt[i]) ^ int(xor_const[i])) for i in range(8))
        if pred == row['answer']:
            xor_correct += 1
print(f"XOR constant: {xor_correct}/{xor_total} correct (out of {len(puzzles)} total)")

# Check: multi-function composition (e.g., rotate then XOR)
print("\n=== COMPOSITION ANALYSIS ===")
comp_correct = 0
for row in puzzles[:200]:  # Sample for speed
    exs, tgt = parse_bit_ops_prompt(row['prompt'])
    if not exs or not tgt:
        continue
    gold = row['answer']
    
    # Try: rotate + XOR_const
    for shift in range(1, 8):
        rotated = [ex[0][shift:] + ex[0][:shift] for ex in exs]
        xor_const = ''.join(str(int(rotated[0][i]) ^ int(exs[0][1][i])) for i in range(8))
        all_match = True
        for j in range(1, len(exs)):
            xc = ''.join(str(int(rotated[j][i]) ^ int(exs[j][1][i])) for i in range(8))
            if xc != xor_const:
                all_match = False
                break
        if all_match:
            r_tgt = tgt[shift:] + tgt[:shift]
            pred = ''.join(str(int(r_tgt[i]) ^ int(xor_const[i])) for i in range(8))
            if pred == gold:
                comp_correct += 1
                break
print(f"RotL + XOR composition: {comp_correct}/200 additional correct")
