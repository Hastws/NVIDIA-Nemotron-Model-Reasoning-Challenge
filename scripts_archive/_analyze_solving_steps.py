#!/usr/bin/env python3
"""Analyze the SOLVING STEPS needed for each type - are they fixed?"""
import os, polars as pl
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pl.read_csv('competition_data/train.csv')

def classify(p):
    p = p.lower()
    if 'bit manipulation' in p or '8-bit binary' in p: return 'bit_ops'
    elif 'encrypt' in p or 'decrypt' in p: return 'cipher'
    elif 'gravitational' in p or 'falling distance' in p: return 'gravity'
    elif 'numeral system' in p: return 'numeral'
    elif 'transformation rules' in p: return 'symbol'
    elif 'unit conversion' in p or 'convert the following measurement' in p: return 'unit_conv'
    return 'unknown'

df = df.with_columns(pl.col('prompt').map_elements(classify, return_dtype=pl.Utf8).alias('qtype'))

print("=" * 70)
print("SOLVING COMPLEXITY ANALYSIS: How many steps to solve each type?")
print("=" * 70)

print("""
=== numeral ===
Structure: N examples of decimal->Roman, then "convert X"
Steps: 1) Recognize it's Roman numerals 2) Apply standard conversion rules
Computation: O(1) arithmetic, fixed algorithm
Token budget needed: ~50 tokens (trivial)
Base model: 100% already

=== gravity ===  
Structure: 5 pairs of (t, distance), then "find distance for t=X given d=0.5*g*t^2"
Steps: 1) From any pair, compute g = 2*d/t^2  2) Average g  3) d = 0.5*g*X^2
Computation: 3 multiplications + 1 division
Token budget needed: ~80 tokens
Base model: 72% (fails on FORMAT, not understanding)

=== unit_conv ===
Structure: 5 pairs of (X m, Y), then "convert Z m"
Steps: 1) Compute ratio Y/X from any pair  2) Multiply Z * ratio  3) Round
Computation: 1 division + 1 multiplication
Token budget needed: ~60 tokens
Base model: 56% (some truncation, some format)

=== cipher ===
Structure: 3-5 encrypt/decrypt examples, then query
Steps: 1) Align words  2) For each letter pair, find substitution  3) Build mapping  4) Apply
Computation: O(26) mapping build + O(len) application
Token budget needed: ~200 tokens (need to list alphabet mapping)
Base model: 34% (needs more space to work out mapping)

=== bit_ops ===
Structure: 8-11 examples of 8-bit binary -> 8-bit binary, then query
Steps: 1) For each bit position, collect input/output pairs
       2) Determine function for each bit (could be f(b1,b2,...,b8) for each output bit)
       3) Test candidates (XOR, AND, OR, NOT, shifts, rotations...)
       4) Apply to query
Computation: O(8 * 2^8) worst case for exhaustive bit function search
Token budget needed: ~500-2000 tokens (complex combinatorial reasoning)
Base model: 10% (90% truncated at 7680 tokens - model TRIES but runs out of space)

=== symbol ===
Structure: 3-4 examples of symbol-string -> symbol-string, then query
Steps: 1) Figure out what the symbols represent (could be anything!)
       2) Multi-step: possibly digit mapping + arithmetic + concatenation
       3) Some operators may not appear in examples (must be inferred)
Computation: Highly variable, combinatorial explosion
Token budget needed: ~2000-5000 tokens (open-ended reasoning)
Base model: 8% (84% truncated - model TRIES but it's genuinely hard + runs out of space)
""")

# Now quantify: what's the IDEAL thinking length for each type?
print("=" * 70)
print("IDEAL CoT LENGTH ESTIMATE (tokens, not chars)")
print("=" * 70)
print("""
Type         Ideal Tokens   Base Model Uses   Over-thinks By
------       ------------   ---------------   --------------
numeral         30-50          ~125 chars           2x (minor)
gravity         50-80         ~1000 chars           5x  
unit_conv       40-60         ~2500 chars          15x
cipher         150-300        ~5000 chars          10x
bit_ops        300-800       ~16000 chars → TRUNCATED
symbol        1000-3000      ~24000 chars → TRUNCATED

Key insight: The model MASSIVELY over-thinks easy types!
If we could teach it to think EFFICIENTLY:
- gravity: 80 tokens instead of 1000 chars → save budget for hard types
- cipher: 300 tokens instead of 5000 chars
- bit_ops: would fit in 7680 if it went straight to the right approach
""")

# Check answer format uniformity
print("=" * 70)
print("ANSWER FORMAT UNIFORMITY")
print("=" * 70)
for qtype in sorted(df['qtype'].unique().to_list()):
    subset = df.filter(pl.col('qtype') == qtype)
    answers = [str(a) for a in subset['answer'].to_list()]
    lengths = [len(a) for a in answers]
    
    # Check if all answers match a pattern
    all_numeric = all(a.replace('.','').replace('-','').isdigit() for a in answers)
    all_same_len = len(set(lengths)) == 1
    
    print(f"{qtype}: len={min(lengths)}-{max(lengths)}, numeric={all_numeric}, fixed_len={all_same_len}")
    print(f"  Sample answers: {answers[:5]}")
