"""
Deep analysis of bit_ops and symbol types to find common solving patterns.
"""
import polars as pl
import re
from collections import Counter

df = pl.read_csv('competition_data/train.csv')

# ============================================================================
# BIT_OPS ANALYSIS
# ============================================================================
print("=" * 80)
print("BIT_OPS ANALYSIS")
print("=" * 80)

bit = df.filter(pl.col('prompt').str.contains('bit manipulation'))
print(f"Total bit_ops: {len(bit)}")

# Parse examples from prompts
def parse_bit_examples(prompt):
    """Extract input->output pairs and the query from a bit_ops prompt."""
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

# Try to identify transformation type for each problem
def try_all_transforms(examples):
    """Try common bit transforms to see which one fits all examples."""
    results = []
    
    # 1. Rotate left by N
    for n in range(1, 8):
        if all(((int(i,2) << n) | (int(i,2) >> (8-n))) & 0xff == int(o,2) for i,o in examples):
            results.append(f"ROT_LEFT_{n}")
    
    # 2. Rotate right by N
    for n in range(1, 8):
        if all(((int(i,2) >> n) | (int(i,2) << (8-n))) & 0xff == int(o,2) for i,o in examples):
            results.append(f"ROT_RIGHT_{n}")
    
    # 3. XOR with constant
    if len(examples) >= 2:
        xor_const = int(examples[0][0], 2) ^ int(examples[0][1], 2)
        if all(int(i,2) ^ xor_const == int(o,2) for i,o in examples):
            results.append(f"XOR_0x{xor_const:02x}")
    
    # 4. NOT (bit flip)
    if all((~int(i,2) & 0xff) == int(o,2) for i,o in examples):
        results.append("NOT")
    
    # 5. Reverse bits
    def rev8(x):
        return int(f"{x:08b}"[::-1], 2)
    if all(rev8(int(i,2)) == int(o,2) for i,o in examples):
        results.append("REVERSE_BITS")
    
    # 6. Swap nibbles (swap upper 4 bits with lower 4 bits)
    if all((((int(i,2) & 0xf) << 4) | ((int(i,2) >> 4) & 0xf)) == int(o,2) for i,o in examples):
        results.append("SWAP_NIBBLES")
    
    # 7. Rotate left N then XOR constant
    for n in range(1, 8):
        rotated_0 = ((int(examples[0][0],2) << n) | (int(examples[0][0],2) >> (8-n))) & 0xff
        xor_c = rotated_0 ^ int(examples[0][1], 2)
        if all((((int(i,2) << n) | (int(i,2) >> (8-n))) & 0xff) ^ xor_c == int(o,2) for i,o in examples):
            results.append(f"ROT_LEFT_{n}_XOR_0x{xor_c:02x}")
    
    # 8. XOR constant then rotate left N
    for n in range(1, 8):
        xor_c_0 = int(examples[0][0], 2) ^ (((int(examples[0][1],2) >> n) | (int(examples[0][1],2) << (8-n))) & 0xff)
        # Verify: xor then rotate
        if all((((int(i,2) ^ xor_c_0) << n) | ((int(i,2) ^ xor_c_0) >> (8-n))) & 0xff == int(o,2) for i,o in examples):
            results.append(f"XOR_0x{xor_c_0:02x}_ROT_LEFT_{n}")
    
    # 9. Bit permutation (each output bit comes from a specific input bit)
    if len(examples) >= 8:
        perm = [None] * 8
        valid = True
        for bit_pos in range(8):
            # Which input bit maps to output bit_pos?
            candidates = set(range(8))
            for inp, out in examples:
                out_bit = int(out[7 - bit_pos])
                for src in list(candidates):
                    inp_bit = int(inp[7 - src])
                    if inp_bit != out_bit:
                        candidates.discard(src)
            if len(candidates) == 1:
                perm[bit_pos] = candidates.pop()
            elif len(candidates) == 0:
                valid = False
                break
            else:
                # Multiple candidates - need more examples
                perm[bit_pos] = sorted(candidates)
        if valid and all(isinstance(p, int) for p in perm):
            # Verify
            ok = True
            for inp, out in examples:
                predicted = ''
                for bit_pos in range(7, -1, -1):
                    predicted += inp[7 - perm[bit_pos]]
                if predicted != out:
                    ok = False
                    break
            if ok:
                results.append(f"BIT_PERM_{perm}")
    
    # 10. NOT then rotate
    for n in range(1, 8):
        if all(((((~int(i,2) & 0xff) << n) | ((~int(i,2) & 0xff) >> (8-n))) & 0xff) == int(o,2) for i,o in examples):
            results.append(f"NOT_ROT_LEFT_{n}")
    
    # 11. Rotate then NOT
    for n in range(1, 8):
        if all((~(((int(i,2) << n) | (int(i,2) >> (8-n))) & 0xff) & 0xff) == int(o,2) for i,o in examples):
            results.append(f"ROT_LEFT_{n}_NOT")
    
    # 12. Reverse bits then XOR
    if len(examples) >= 1:
        rev_0 = rev8(int(examples[0][0], 2))
        xor_c = rev_0 ^ int(examples[0][1], 2)
        if all(rev8(int(i,2)) ^ xor_c == int(o,2) for i,o in examples):
            results.append(f"REVERSE_XOR_0x{xor_c:02x}")
    
    # 13. Swap pairs of bits (swap bit 0&1, 2&3, 4&5, 6&7)
    def swap_pairs(x):
        return ((x & 0xAA) >> 1) | ((x & 0x55) << 1)
    if all(swap_pairs(int(i,2)) == int(o,2) for i,o in examples):
        results.append("SWAP_BIT_PAIRS")
    
    # 14. Composite: two operations from simple set
    # Only try if nothing found yet
    if not results:
        # Try all pairs of simple operations
        ops = {
            'NOT': lambda x: ~x & 0xff,
            'REV': lambda x: rev8(x),
            'SWAP_NIB': lambda x: ((x & 0xf) << 4) | ((x >> 4) & 0xf),
            'SWAP_PAIR': lambda x: swap_pairs(x),
        }
        for n in range(1, 8):
            ops[f'ROL{n}'] = lambda x, n=n: ((x << n) | (x >> (8-n))) & 0xff
            ops[f'ROR{n}'] = lambda x, n=n: ((x >> n) | (x << (8-n))) & 0xff
        
        for name1, op1 in ops.items():
            for name2, op2 in ops.items():
                if all(op2(op1(int(i,2))) == int(o,2) for i,o in examples):
                    results.append(f"COMPOSITE: {name1} -> {name2}")
        
        # XOR with constant + another op
        for xc in range(256):
            for name, op in ops.items():
                if all(op(int(i,2) ^ xc) == int(o,2) for i,o in examples):
                    results.append(f"XOR_0x{xc:02x} -> {name}")
                    break
                if all((op(int(i,2)) ^ xc) == int(o,2) for i,o in examples):
                    results.append(f"{name} -> XOR_0x{xc:02x}")
                    break
            if results:
                break
    
    return results

# Analyze first 20 bit_ops problems
transform_types = Counter()
for i in range(min(50, len(bit))):
    row = bit.row(i, named=True)
    examples, query = parse_bit_examples(row['prompt'])
    gold = row['answer']
    transforms = try_all_transforms(examples)
    
    # Verify answer
    verified = False
    if transforms and query:
        for t in transforms:
            if 'BIT_PERM' in t:
                verified = True
                break
    
    if i < 10:
        print(f"\n--- bit_ops #{i}: {len(examples)} examples, query={query}, gold={gold} ---")
        print(f"  Transforms found: {transforms}")
    
    for t in transforms:
        # Normalize
        name = t.split('_')[0] if 'COMPOSITE' not in t else 'COMPOSITE'
        if 'BIT_PERM' in t:
            name = 'BIT_PERM'
        elif 'COMPOSITE' in t:
            name = t
        transform_types[t] += 1

print(f"\n\nTransform type distribution (first 50):")
for t, c in transform_types.most_common(30):
    print(f"  {t}: {c}")

# ============================================================================
# SYMBOL ANALYSIS
# ============================================================================
print("\n\n" + "=" * 80)
print("SYMBOL ANALYSIS")
print("=" * 80)

sym = df.filter(pl.col('prompt').str.contains('transformation rules'))
print(f"Total symbol: {len(sym)}")

# Analyze symbol problem sub-types
def classify_symbol(prompt, answer):
    """Classify symbol problem variant."""
    # Check if it uses numbers
    examples_text = prompt.split('Now,')[0] if 'Now,' in prompt else prompt
    
    has_numbers = bool(re.search(r'\d', examples_text))
    has_special = bool(re.search(r'[!@#$%^&*(){}[\]|\\<>~`\'"+=\-/]', examples_text))
    has_letters = bool(re.search(r'[a-zA-Z]', examples_text.split('transformation')[0] if 'transformation' in examples_text else ''))
    
    # Parse examples
    lines = examples_text.strip().split('\n')
    eq_examples = []
    for line in lines:
        if '=' in line and '->' not in line:
            parts = line.strip().split('=')
            if len(parts) == 2:
                eq_examples.append((parts[0].strip(), parts[1].strip()))
    
    if has_numbers and not has_special:
        return 'numeric'
    elif has_numbers and has_special:
        return 'mixed'
    else:
        return 'pure_symbol'

# Analyze distribution
sym_types = Counter()
sym_example_counts = Counter()
for i in range(len(sym)):
    row = sym.row(i, named=True)
    stype = classify_symbol(row['prompt'], row['answer'])
    sym_types[stype] += 1
    
    # Count examples
    lines = row['prompt'].split('Now,')[0].strip().split('\n')
    eq_count = sum(1 for l in lines if '=' in l and len(l.strip()) > 3)
    sym_example_counts[eq_count] += 1

print(f"\nSymbol sub-types:")
for t, c in sym_types.most_common():
    print(f"  {t}: {c}")

print(f"\nExample count distribution:")
for n, c in sorted(sym_example_counts.items()):
    print(f"  {n} examples: {c}")

# Deep dive into a few symbol examples to understand the mapping
print("\n\nSYMBOL DEEP DIVE:")
for i in range(10):
    row = sym.row(i, named=True)
    prompt = row['prompt']
    answer = row['answer']
    stype = classify_symbol(prompt, answer)
    
    # Extract examples
    block = prompt.split('Now,')[0] if 'Now,' in prompt else prompt
    # Get the equation lines
    lines = block.strip().split('\n')
    eq_lines = [l.strip() for l in lines if '=' in l and len(l.strip()) > 3]
    
    # Get query
    query_match = re.search(r'result for:\s*(.+)$', prompt, re.MULTILINE)
    query = query_match.group(1).strip() if query_match else '?'
    
    print(f"\n--- symbol #{i} ({stype}) | query={query} | gold={answer} ---")
    for eq in eq_lines:
        print(f"  {eq}")
    
    # For pure_symbol: try character-level substitution
    if stype == 'pure_symbol':
        # Build char mapping from LHS to RHS
        char_map = {}
        for eq in eq_lines:
            parts = eq.split('=')
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                for j, ch in enumerate(lhs):
                    if j < len(rhs):
                        if ch not in char_map:
                            char_map[ch] = set()
                        char_map[ch].add(rhs[j])
        
        # Check if it's a simple substitution cipher
        is_subst = all(len(v) == 1 for v in char_map.values())
        if is_subst:
            mapping = {k: list(v)[0] for k, v in char_map.items()}
            predicted = ''.join(mapping.get(c, '?') for c in query)
            print(f"  SUBSTITUTION: {mapping}")
            print(f"  Predicted: {predicted} | Gold: {answer} | {'✓' if predicted == answer else '✗'}")
        else:
            print(f"  NOT simple substitution")
            # Check input/output lengths
            for eq in eq_lines:
                parts = eq.split('=')
                if len(parts) == 2:
                    print(f"    len: {len(parts[0].strip())} -> {len(parts[1].strip())}")
    
    elif stype == 'numeric':
        print("  (numeric type - may be arithmetic)")
