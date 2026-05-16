"""
Deep analysis of each problem type in train.csv
Goal: Understand the math core, strip narrative fluff, see what's really being asked
"""
import polars as pl
from collections import Counter
import re

df = pl.read_csv('competition_data/train.csv')

# Classify
def classify(prompt):
    p = prompt.lower()
    for t, kws in {
        'bit_ops': ['bit manipulation'],
        'gravity': ['gravitational'],
        'unit_conv': ['unit conversion'],
        'cipher': ['encryption'],
        'numeral': ['numeral system'],
        'symbol': ['transformation rules'],
    }.items():
        for kw in kws:
            if kw in p:
                return t
    return 'unknown'

df = df.with_columns(pl.col("prompt").map_elements(classify, return_dtype=pl.Utf8).alias("type"))

print("=" * 60)
print("TYPE DISTRIBUTION")
print("=" * 60)
for t in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    subset = df.filter(pl.col("type") == t)
    print(f"  {t}: {len(subset)}")

# Analyze each type in detail
for tname in ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']:
    subset = df.filter(pl.col("type") == tname)
    print(f"\n{'=' * 60}")
    print(f"TYPE: {tname} ({len(subset)} samples)")
    print(f"{'=' * 60}")
    
    # Count example lines per problem
    n_examples = []
    for p in subset["prompt"].to_list()[:100]:
        lines = p.strip().split('\n')
        if tname == 'bit_ops':
            exs = [l for l in lines if '->' in l and 'Now' not in l]
        elif tname == 'gravity':
            exs = [l for l in lines if 'distance =' in l]
        elif tname == 'unit_conv':
            exs = [l for l in lines if 'becomes' in l]
        elif tname == 'cipher':
            exs = [l for l in lines if '->' in l and 'Now' not in l]
        elif tname == 'numeral':
            exs = [l for l in lines if '->' in l and 'Now' not in l]
        elif tname == 'symbol':
            exs = [l for l in lines if '=' in l and 'Now' not in l]
        n_examples.append(len(exs))
    
    print(f"  Examples per problem: min={min(n_examples)}, max={max(n_examples)}, avg={sum(n_examples)/len(n_examples):.1f}")
    
    # Answer analysis
    answers = subset["answer"].to_list()
    avg_len = sum(len(str(a)) for a in answers) / len(answers)
    print(f"  Answer avg length: {avg_len:.1f} chars")
    
    # Show 3 stripped examples (remove narrative)
    print(f"\n  --- Stripped Examples ---")
    for i, row in enumerate(subset.head(3).iter_rows(named=True)):
        p = row["prompt"]
        a = row["answer"]
        lines = p.strip().split('\n')
        
        # Strip first line (narrative) and extract pure examples + question
        core_lines = []
        for l in lines[1:]:  # skip "In Alice's Wonderland..." line
            l = l.strip()
            if l:
                core_lines.append(l)
        
        print(f"\n  Example {i+1}:")
        for l in core_lines:
            print(f"    {l}")
        print(f"    ANSWER: {a}")

    # For numeral: all answers should be Roman numerals
    if tname == 'numeral':
        roman_pattern = re.compile(r'^[IVXLCDM]+$')
        roman_count = sum(1 for a in answers if roman_pattern.match(str(a)))
        print(f"\n  Roman numeral answers: {roman_count}/{len(answers)} ({100*roman_count/len(answers):.1f}%)")
    
    # For gravity: check if answers are numeric
    if tname == 'gravity':
        numeric = sum(1 for a in answers if re.match(r'^[\d.]+$', str(a)))
        print(f"\n  Numeric answers: {numeric}/{len(answers)} ({100*numeric/len(answers):.1f}%)")
        vals = [float(a) for a in answers if re.match(r'^[\d.]+$', str(a))]
        print(f"  Value range: {min(vals):.2f} - {max(vals):.2f}")
    
    # For unit_conv: check if answers are numeric
    if tname == 'unit_conv':
        numeric = sum(1 for a in answers if re.match(r'^[\d.]+$', str(a)))
        print(f"\n  Numeric answers: {numeric}/{len(answers)} ({100*numeric/len(answers):.1f}%)")
        vals = [float(a) for a in answers if re.match(r'^[\d.]+$', str(a))]
        print(f"  Value range: {min(vals):.2f} - {max(vals):.2f}")
    
    # For cipher: answer word count
    if tname == 'cipher':
        word_counts = [len(str(a).split()) for a in answers]
        print(f"\n  Answer word count: min={min(word_counts)}, max={max(word_counts)}, avg={sum(word_counts)/len(word_counts):.1f}")

    # For bit_ops: answer format
    if tname == 'bit_ops':
        binary = sum(1 for a in answers if re.match(r'^[01]{8}$', str(a)))
        print(f"\n  8-bit binary answers: {binary}/{len(answers)} ({100*binary/len(answers):.1f}%)")
    
    # For symbol: answer format
    if tname == 'symbol':
        avg_ans_len = sum(len(str(a)) for a in answers) / len(answers)
        print(f"\n  Avg answer length: {avg_ans_len:.1f} chars")
        # Show some answers
        print(f"  Sample answers: {answers[:10]}")
