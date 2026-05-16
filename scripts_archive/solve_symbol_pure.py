"""Analyze pure symbol problems to find solving patterns."""
import polars as pl
import re
from collections import Counter
from itertools import product

df = pl.read_csv('competition_data/train.csv')
sym = df.filter(pl.col('prompt').str.contains('transformation rules'))

def parse_symbol(prompt):
    block = prompt.split('Now,')[0]
    lines = block.strip().split('\n')
    examples = []
    for line in lines:
        line = line.strip()
        if '=' in line and 'transformation' not in line.lower() and 'alice' not in line.lower() and 'secret' not in line.lower() and 'examples' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                examples.append((parts[0].strip(), parts[1].strip()))
    query_match = re.search(r'result for:\s*(.+)$', prompt, re.MULTILINE)
    query = query_match.group(1).strip() if query_match else None
    return examples, query

# KEY INSIGHT from example #4:
# %|*"| = %|"|   → remove char at position 2 (which is '*')
# \(*[^ = \([^   → remove char at position 2 (which is '*')  
# (%+[@ = (%[@   → remove char at position 2 (which is '+')
# |[*([ = |[([   → remove char at position 2 (which is '*')
# [^-[( = -^     → this doesn't follow...
#
# Wait - maybe the rule is: there's a set of "operator" chars that get REMOVED,
# and the remaining chars are the output?

# Hypothesis: certain ASCII chars are in an "operator set" - 
# they get removed and the rest are concatenated to form output

solved = 0
total_pure = 0
rule_types = Counter()

for i in range(len(sym)):
    row = sym.row(i, named=True)
    prompt = row['prompt']
    gold = row['answer']
    
    examples, query = parse_symbol(prompt)
    if not examples or not query:
        continue
    
    has_digit = any(c.isdigit() for c in examples[0][0])
    if has_digit:
        continue
    
    total_pure += 1
    
    # Strategy 1: Character removal - find which chars are "operators" (removed)
    # For each example, find which chars from input are missing in output
    # Try: the output is the input with certain positions/chars removed
    
    # Strategy 1a: Position-based removal
    # Try removing each subset of positions {0,1,2,3,4} and see if remaining matches output
    found = False
    for mask in range(1, 32):  # 5-bit mask for 5 positions
        positions_to_keep = [j for j in range(5) if not (mask & (1 << j))]
        positions_to_remove = [j for j in range(5) if mask & (1 << j)]
        
        all_match = True
        for lhs, rhs in examples:
            if len(lhs) != 5:
                all_match = False
                break
            predicted = ''.join(lhs[p] for p in positions_to_keep)
            if predicted != rhs:
                all_match = False
                break
        
        if all_match and len(query) == 5:
            pred = ''.join(query[p] for p in positions_to_keep)
            if pred == gold:
                solved += 1
                rule_types[f'remove_pos_{positions_to_remove}'] += 1
                found = True
                break
    
    if found:
        continue
    
    # Strategy 1b: Character set removal - certain chars are removed wherever they appear
    # Find the set of chars that appear in input but not output across all examples
    all_removed_chars = set()
    for lhs, rhs in examples:
        for ch in lhs:
            if ch not in rhs:
                all_removed_chars.add(ch)
    
    # Try: remove all chars in the removal set
    all_match = True
    for lhs, rhs in examples:
        predicted = ''.join(c for c in lhs if c not in all_removed_chars)
        if predicted != rhs:
            all_match = False
            break
    
    if all_match and query:
        pred = ''.join(c for c in query if c not in all_removed_chars)
        if pred == gold:
            solved += 1
            rule_types['char_removal'] += 1
            continue
    
    # Strategy 2: Character substitution (1-to-1 mapping)
    char_map = {}
    consistent = True
    for lhs, rhs in examples:
        if len(lhs) != len(rhs):
            consistent = False
            break
        for j in range(len(lhs)):
            if lhs[j] in char_map:
                if char_map[lhs[j]] != rhs[j]:
                    consistent = False
                    break
            else:
                char_map[lhs[j]] = rhs[j]
        if not consistent:
            break
    
    if consistent and query and char_map:
        pred = ''.join(char_map.get(c, '?') for c in query)
        if pred == gold:
            solved += 1
            rule_types['char_substitution'] += 1
            continue
    
    # Strategy 3: Reverse then something
    # Check if output is a reversed/shuffled version of a subset
    
    # Strategy 3a: Output is input reversed with some chars removed
    all_match = True
    for lhs, rhs in examples:
        reversed_lhs = lhs[::-1]
        # Check if rhs is a subsequence of reversed_lhs after removing some chars
        if not all(c in reversed_lhs for c in rhs):
            all_match = False
            break
    # Not a clean strategy...
    
    # Strategy 4: Position-based with char substitution
    # Each position p in output comes from position mapping[p] in input, 
    # possibly with a char shift
    
    if total_pure <= 2000 and not found:
        # Just track unsolved
        pass

print(f"\nPure symbol: Solved {solved}/{total_pure} ({100*solved/total_pure:.1f}%)")
print(f"\nRule types:")
for r, c in rule_types.most_common():
    print(f"  {r}: {c}")

# How many have variable output length?
var_len = 0
fixed_len = 0
for i in range(len(sym)):
    row = sym.row(i, named=True)
    examples, query = parse_symbol(row['prompt'])
    if not examples:
        continue
    has_digit = any(c.isdigit() for c in examples[0][0])
    if has_digit:
        continue
    
    out_lens = set(len(e[1]) for e in examples)
    if len(out_lens) == 1:
        fixed_len += 1
    else:
        var_len += 1

print(f"\nPure symbol output length: fixed={fixed_len}, variable={var_len}")

# For unsolved, show a few and try to understand the rule
print("\n\nUNSOLVED PURE SYMBOL EXAMPLES:")
count = 0
for i in range(len(sym)):
    if count >= 5:
        break
    row = sym.row(i, named=True)
    prompt = row['prompt']
    gold = row['answer']
    examples, query = parse_symbol(prompt)
    if not examples or not query:
        continue
    has_digit = any(c.isdigit() for c in examples[0][0])
    if has_digit:
        continue
    
    # Check if we already solved it
    found_sol = False
    for mask in range(1, 32):
        positions_to_keep = [j for j in range(5) if not (mask & (1 << j))]
        all_match = True
        for lhs, rhs in examples:
            if len(lhs) != 5:
                all_match = False
                break
            predicted = ''.join(lhs[p] for p in positions_to_keep)
            if predicted != rhs:
                all_match = False
                break
        if all_match and len(query) == 5:
            pred = ''.join(query[p] for p in positions_to_keep)
            if pred == gold:
                found_sol = True
                break
    
    if found_sol:
        continue
    
    print(f"\n#{i}: query={query!r} gold={gold!r}")
    for lhs, rhs in examples:
        # Show position-by-position analysis
        print(f"  {lhs!r} -> {rhs!r}  (in_len={len(lhs)}, out_len={len(rhs)})")
        # For each output char, find its origin in input
        origins = []
        for oc in rhs:
            positions = [j for j, ic in enumerate(lhs) if ic == oc]
            origins.append((oc, positions))
        print(f"    output char origins: {origins}")
    count += 1
