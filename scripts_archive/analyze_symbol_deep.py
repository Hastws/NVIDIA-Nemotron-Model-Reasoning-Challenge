#!/usr/bin/env python3
"""
Deep analysis of symbol puzzles to find solvable patterns.
Hypothesis: each character maps to a string (possibly empty).
The equation is: concat(map(c) for c in input) == output
"""
import csv
import json
import re
from collections import defaultdict
from itertools import product

def detect_type(prompt):
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p):
        return "bit_ops"
    elif "encrypt" in p or "cipher" in p or "secret code" in p:
        return "cipher"
    elif "gravit" in p:
        return "gravity"
    elif "numeral" in p:
        return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p) or ("secret unit" in p):
        return "unit_conv"
    else:
        return "symbol"

def parse_symbol_prompt(prompt):
    """Parse symbol equation prompt."""
    lines = prompt.strip().split('\n')
    examples = []
    target = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("In Alice") or line.startswith("Below") or line.startswith("Now,"):
            if "Now," in line and ":" in line:
                target = line.split(":")[-1].strip()
            continue
        # Try to parse as equation: LHS = RHS
        if ' = ' in line:
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                examples.append((lhs, rhs))
    
    return examples, target

def try_char_substitution(examples, target):
    """Try to solve as character-by-character substitution (each char maps to a string)."""
    # Collect all characters
    all_chars = set()
    for lhs, rhs in examples:
        all_chars.update(lhs)
    
    # Simple approach: try to find mapping where each char maps to 0 or 1 chars
    # This is a constraint satisfaction problem
    
    # First check: is it possible that each char maps to exactly 0 or 1 chars?
    # If sum of mapped lengths must equal output length for each example
    
    n_chars = len(all_chars)
    
    # Try: build equations len(map(c1)) + len(map(c2)) + ... = len(rhs)
    # where lhs = c1c2c3...
    
    # For now, try the simplest case: each char maps to exactly 1 char (permutation)
    char_mapping = {}
    for lhs, rhs in examples:
        if len(lhs) == len(rhs):
            for c_in, c_out in zip(lhs, rhs):
                if c_in in char_mapping:
                    if char_mapping[c_in] != c_out:
                        return None, "Conflict in 1-to-1 mapping"
                else:
                    char_mapping[c_in] = c_out
    
    if char_mapping and target:
        # Verify against all examples
        all_match = True
        for lhs, rhs in examples:
            predicted = ''.join(char_mapping.get(c, '?') for c in lhs)
            if predicted != rhs:
                all_match = False
                break
        
        if all_match:
            result = ''.join(char_mapping.get(c, '?') for c in target)
            if '?' not in result:
                return result, "1-to-1 char substitution"
    
    return None, "Cannot solve"

# Analyze
total = 0
solvable = 0
categories = defaultdict(int)

with open("competition_data/train.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if detect_type(row["prompt"]) != "symbol":
            continue
        total += 1
        
        examples, target = parse_symbol_prompt(row["prompt"])
        
        if not examples or not target:
            categories["parse_failed"] += 1
            continue
        
        # Categorize the puzzle
        lhs_lens = [len(lhs) for lhs, rhs in examples]
        rhs_lens = [len(rhs) for lhs, rhs in examples]
        
        all_same_len = all(len(lhs) == len(rhs) for lhs, rhs in examples)
        
        # Check if input chars are digits or symbols
        all_digits = all(all(c.isdigit() or c in '/\\|+-*' for c in lhs) for lhs, _ in examples)
        
        if all_same_len:
            categories["same_length"] += 1
        elif all_digits:
            categories["digit_based"] += 1
        else:
            categories["variable_length"] += 1
        
        # Try to solve
        answer, method = try_char_substitution(examples, target)
        gold = row["answer"]
        
        if answer and answer == gold:
            solvable += 1
            if total <= 5:
                print(f"SOLVED: {examples} -> {target} = {answer} (gold: {gold})")

print(f"\nSymbol Summary:")
print(f"  Total: {total}")
print(f"  Solvable (1-to-1 char sub): {solvable}")
print(f"  Categories: {dict(categories)}")

# Deep dive into variable-length examples
print(f"\n--- Variable Length Examples ---")
count = 0
with open("competition_data/train.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if detect_type(row["prompt"]) != "symbol":
            continue
        
        examples, target = parse_symbol_prompt(row["prompt"])
        if not examples:
            continue
        
        all_same_len = all(len(lhs) == len(rhs) for lhs, rhs in examples)
        if not all_same_len and count < 5:
            count += 1
            print(f"\nExample {count}:")
            for lhs, rhs in examples:
                print(f"  '{lhs}' ({len(lhs)}) -> '{rhs}' ({len(rhs)})")
            print(f"  Target: '{target}', Gold: '{row['answer']}'")
