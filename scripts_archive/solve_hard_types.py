"""
Systematically solve bit_ops and symbol problems to determine solvability rates.
"""
import polars as pl
import re
from collections import Counter

df = pl.read_csv('competition_data/train.csv')

# ============================================================================
# BIT_OPS SOLVER
# ============================================================================
print("=" * 80)
print("BIT_OPS SOLVER")
print("=" * 80)

bit = df.filter(pl.col('prompt').str.contains('bit manipulation'))
print(f"Total: {len(bit)}")

def parse_bit_examples(prompt):
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        m = re.match(r'^([01]{8})\s*->\s*([01]{8})$', line)
        if m:
            examples.append((m.group(1), m.group(2)))
        m2 = re.match(r'.*output for:\s*([01]{8})', line)
        if m2:
            query = m2.group(1)
    return examples, query

def rev8(x):
    return int(f'{x:08b}'[::-1], 2)

def swap_pairs(x):
    return ((x & 0xAA) >> 1) | ((x & 0x55) << 1)

def try_solve_bit(examples, query):
    q = int(query, 2)
    
    simple_ops = {
        'NOT': lambda x: ~x & 0xff,
        'REV': rev8,
        'SWAP_NIB': lambda x: (x & 0xf) << 4 | (x >> 4),
        'SWAP_PAIR': swap_pairs,
    }
    for n in range(1, 8):
        simple_ops[f'ROL{n}'] = lambda x, n=n: ((x << n) | (x >> (8-n))) & 0xff
    
    # 1-step
    for name, op in simple_ops.items():
        if all(op(int(i,2)) == int(o,2) for i,o in examples):
            return f'{op(q):08b}', name
    
    # XOR constant
    xc = int(examples[0][0], 2) ^ int(examples[0][1], 2)
    if all(int(i,2) ^ xc == int(o,2) for i,o in examples):
        return f'{q ^ xc:08b}', f'XOR_{xc}'
    
    # 2-step composites (op1 then op2)
    for n1, op1 in simple_ops.items():
        for n2, op2 in simple_ops.items():
            if all(op2(op1(int(i,2))) == int(o,2) for i,o in examples):
                return f'{op2(op1(q)):08b}', f'{n1}->{n2}'

    # op + XOR, XOR + op
    for xc in range(256):
        for name, op in simple_ops.items():
            if all(op(int(i,2) ^ xc) == int(o,2) for i,o in examples):
                return f'{op(q ^ xc):08b}', f'XOR_{xc}->{name}'
            if all(op(int(i,2)) ^ xc == int(o,2) for i,o in examples):
                return f'{op(q) ^ xc:08b}', f'{name}->XOR_{xc}'
    
    # 3-step composites
    for n1, op1 in simple_ops.items():
        for n2, op2 in simple_ops.items():
            for n3, op3 in simple_ops.items():
                if all(op3(op2(op1(int(i,2)))) == int(o,2) for i,o in examples):
                    return f'{op3(op2(op1(q))):08b}', f'{n1}->{n2}->{n3}'
    
    # XOR + 2-step, 2-step + XOR
    for xc in range(256):
        for n1, op1 in simple_ops.items():
            for n2, op2 in simple_ops.items():
                if all(op2(op1(int(i,2) ^ xc)) == int(o,2) for i,o in examples):
                    return f'{op2(op1(q ^ xc)):08b}', f'XOR_{xc}->{n1}->{n2}'
                if all(op2(op1(int(i,2))) ^ xc == int(o,2) for i,o in examples):
                    return f'{op2(op1(q)) ^ xc:08b}', f'{n1}->{n2}->XOR_{xc}'
    
    # op + XOR + op
    for xc in range(256):
        for n1, op1 in simple_ops.items():
            for n2, op2 in simple_ops.items():
                val = op1(int(examples[0][0], 2)) ^ xc
                if op2(val) == int(examples[0][1], 2):
                    if all(op2(op1(int(i,2)) ^ xc) == int(o,2) for i,o in examples):
                        return f'{op2(op1(q) ^ xc):08b}', f'{n1}->XOR_{xc}->{n2}'
    
    return None, None

solved = 0
method_counts = Counter()
unsolved_indices = []
for i in range(len(bit)):
    row = bit.row(i, named=True)
    examples, query = parse_bit_examples(row['prompt'])
    gold = row['answer']
    sol, method = try_solve_bit(examples, query)
    if sol == gold:
        solved += 1
        # Normalize method
        method_counts[method.split('->')[0] if '->' not in method else 'composite'] += 1
    else:
        if len(unsolved_indices) < 5:
            unsolved_indices.append(i)
    if (i+1) % 200 == 0:
        print(f"  Progress: {i+1}/{len(bit)}, solved so far: {solved}")

print(f"\nBIT_OPS: Solved {solved}/{len(bit)} ({100*solved/len(bit):.1f}%)")
print(f"\nMethod distribution:")
for m, c in method_counts.most_common(20):
    print(f"  {m}: {c}")

# Show unsolved examples
print(f"\nUnsolved examples:")
for idx in unsolved_indices:
    row = bit.row(idx, named=True)
    examples, query = parse_bit_examples(row['prompt'])
    gold = row['answer']
    print(f"\n  #{idx}: query={query} gold={gold}")
    for inp, out in examples[:3]:
        i_val = int(inp, 2)
        o_val = int(out, 2)
        print(f"    {inp} -> {out}  xor={i_val^o_val:08b}")

# ============================================================================
# SYMBOL SOLVER
# ============================================================================
print("\n\n" + "=" * 80)
print("SYMBOL SOLVER")
print("=" * 80)

sym = df.filter(pl.col('prompt').str.contains('transformation rules'))
print(f"Total: {len(sym)}")

def parse_symbol_examples(prompt):
    """Extract equation examples and query from symbol prompt."""
    block = prompt.split('Now,')[0] if 'Now,' in prompt else prompt
    lines = block.strip().split('\n')
    examples = []
    for line in lines:
        line = line.strip()
        if '=' in line and 'transformation' not in line.lower() and 'alice' not in line.lower() and 'examples' not in line.lower() and 'secret' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2 and len(parts[0].strip()) > 0 and len(parts[1].strip()) > 0:
                examples.append((parts[0].strip(), parts[1].strip()))
    
    query_match = re.search(r'result for:\s*(.+)$', prompt, re.MULTILINE)
    query = query_match.group(1).strip() if query_match else None
    return examples, query

def is_numeric(s):
    return bool(re.match(r'^[\d.]+$', s))

def classify_and_solve_symbol(examples, query):
    """Try to solve symbol problems."""
    if not examples or not query:
        return None, 'no_data'
    
    # Check if it's a positional character mapping
    # Each position in input maps to something in output independently
    lhs_len = len(examples[0][0])
    
    # Check: is the output always derived from specific positions of input?
    # Try: output = specific chars from input (like cipher but position-based)
    
    # Check if it looks like arithmetic with operators
    has_operator = any(c in query for c in '+-*/|\\')
    has_digit = any(c.isdigit() for c in query)
    
    if has_digit and has_operator:
        # Might be custom arithmetic
        return None, 'arithmetic'
    
    # For pure symbol: check if it's a character-level substitution
    # Build char->char map
    char_map = {}
    consistent = True
    for lhs, rhs in examples:
        if len(lhs) != lhs_len:
            consistent = False
            break
    
    if consistent:
        # Variable output length - not simple substitution
        # Try: some chars are "kept", some are "removed", some are "transformed"
        pass
    
    return None, 'unknown'

# Quick stats on symbol problem structure
mixed_count = 0
pure_count = 0
input_lens = Counter()
output_len_varies = 0

for i in range(len(sym)):
    row = sym.row(i, named=True)
    examples, query = parse_symbol_examples(row['prompt'])
    gold = row['answer']
    
    if not examples:
        continue
    
    has_digit = any(c.isdigit() for c in examples[0][0])
    if has_digit:
        mixed_count += 1
    else:
        pure_count += 1
    
    lens_in = set(len(e[0]) for e in examples)
    lens_out = set(len(e[1]) for e in examples)
    
    if len(lens_in) == 1:
        input_lens[list(lens_in)[0]] += 1
    else:
        input_lens['variable'] += 1
    
    if len(lens_out) > 1:
        output_len_varies += 1

print(f"\nPure symbol: {pure_count}")
print(f"Mixed (has digits): {mixed_count}")
print(f"Output length varies: {output_len_varies}")
print(f"\nInput length distribution:")
for l, c in input_lens.most_common():
    print(f"  len={l}: {c}")

# Deep analysis of mixed/arithmetic type
print("\n\nMIXED/ARITHMETIC EXAMPLES:")
count = 0
for i in range(len(sym)):
    row = sym.row(i, named=True)
    examples, query = parse_symbol_examples(row['prompt'])
    gold = row['answer']
    
    if not examples:
        continue
    
    has_digit = any(c.isdigit() for c in examples[0][0])
    if has_digit and count < 10:
        print(f"\n  #{i}: query={query} gold={gold}")
        for lhs, rhs in examples:
            print(f"    {lhs} = {rhs}")
        count += 1

# Deep analysis of pure symbol - look for the underlying rule  
print("\n\nPURE SYMBOL DEEP ANALYSIS (first 10):")
for i in range(min(20, len(sym))):
    row = sym.row(i, named=True)
    examples, query = parse_symbol_examples(row['prompt'])
    gold = row['answer']
    
    if not examples:
        continue
    
    has_digit = any(c.isdigit() for c in examples[0][0])
    if has_digit:
        continue
    
    # For each example, check which input chars appear in output
    print(f"\n  #{i}: query={query!r} gold={gold!r}")
    for lhs, rhs in examples:
        # Find which lhs positions map to rhs
        kept = []
        for j, c in enumerate(lhs):
            if c in rhs:
                kept.append((j, c))
        removed = []
        for j, c in enumerate(lhs):
            if c not in rhs:
                removed.append((j, c))
        new_in_rhs = [c for c in rhs if c not in lhs]
        print(f"    {lhs!r} = {rhs!r}  kept_chars={[c for _,c in kept]} removed={[c for _,c in removed]} new={new_in_rhs}")
    
    if i >= 15:
        break
