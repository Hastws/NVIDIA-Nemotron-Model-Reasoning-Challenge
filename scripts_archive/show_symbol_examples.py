#!/usr/bin/env python3
"""Show pure symbol examples (not bit_ops, not cipher)."""
import csv

count = 0
with open("competition_data/train.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        p = row["prompt"][:300].lower()
        # Must be symbol, NOT bit_ops
        if "8-bit binary" in p or ("bit" in p and "binary" in p):
            continue
        if "encrypt" in p or "cipher" in p or "secret code" in p:
            continue
        if "gravit" in p or "numeral" in p or "unit" in p:
            continue
        if "transformation" in p:
            count += 1
            if count <= 8:
                print(f"--- Symbol Example {count} ---")
                print(row["prompt"])
                print(f"ANSWER: {row['answer']}")
                print()

print(f"Total symbol-type puzzles: {count}")
