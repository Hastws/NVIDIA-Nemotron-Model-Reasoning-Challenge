import pandas as pd
import re

df = pd.read_csv('competition_data/train.csv')

# Classify
def classify(p):
    pl = p.lower()
    if 'cipher' in pl or 'encrypt' in pl or 'decrypt' in pl or 'encode' in pl or 'decode' in pl:
        return 'cipher'
    if 'gravit' in pl or 'planet' in pl:
        return 'gravity'
    if any(x in pl for x in ['base 2', 'base 8', 'base 10', 'base 16', 'base-2', 'binary', 'octal', 'hexadecimal']):
        return 'numeral'
    if any(x in pl for x in ['meter', 'mile', 'gallon', 'liter', 'pound', 'kilogram', 'inch', 'foot', 'feet',
                              'yard', 'ounce', 'celsius', 'fahrenheit', 'kelvin']):
        return 'unit_conv'
    if 'bit' in pl or 'xor' in pl or 'bitwise' in pl:
        return 'bit_ops'
    return 'symbol'

df['type'] = df['prompt'].apply(classify)
symbol_df = df[df['type'] == 'symbol'].copy()
print(f"Total symbol: {len(symbol_df)}")

# Parse examples from prompts
def parse_examples(prompt):
    """Extract the examples and query from a symbol prompt."""
    # Find lines with = that are examples
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if '=' in line and 'determine' not in line.lower() and 'result' not in line.lower() and 'transformation' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        if 'determine the result for:' in line.lower():
            query = line.split(':')[-1].strip()
    return examples, query

# Categorize: numeric vs symbolic operands
def is_numeric_type(examples):
    """Check if operands are numeric (like 55"38) vs pure symbolic (like }"+?")"""
    for lhs, rhs in examples:
        # Check if LHS contains digits
        if re.search(r'\d', lhs):
            return True
    return False

def get_operator(examples):
    """Try to extract the operator symbol between operands."""
    operators = set()
    for lhs, rhs in examples:
        # For numeric: pattern like 55"38 -> operator is "
        m = re.match(r'\d+(.)\d+', lhs)
        if m:
            operators.add(m.group(1))
        else:
            # For symbolic: find the middle character(s) — look for +-*/ style ops
            for op in ['+', '-', '*', '^', '(', ')', '#', "'", '"', '%', '$', '?', '}', '{', '[', ']']:
                # Check if this char appears in the middle consistently
                pass
    return operators

numeric_count = 0
symbolic_count = 0
num_examples_dist = []

for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    num_examples_dist.append(len(examples))
    if is_numeric_type(examples):
        numeric_count += 1
    else:
        symbolic_count += 1

print(f"\nNumeric operands: {numeric_count}")
print(f"Symbolic operands: {symbolic_count}")
print(f"Examples per problem: min={min(num_examples_dist)}, max={max(num_examples_dist)}, mean={sum(num_examples_dist)/len(num_examples_dist):.1f}")

# Deep dive: numeric type examples  
print("\n=== NUMERIC TYPE DEEP DIVE ===\n")
numeric_samples = []
for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if is_numeric_type(examples) and len(examples) >= 3:
        numeric_samples.append((examples, query, row['answer']))

# Try to reverse-engineer the rules for numeric types
print(f"Numeric samples with 3+ examples: {len(numeric_samples)}\n")

for i, (examples, query, answer) in enumerate(numeric_samples[:10]):
    print(f"--- Sample {i+1} ---")
    operators = {}
    for lhs, rhs in examples:
        m = re.match(r'(\d+)(.)(\d+)', lhs)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            print(f"  {a} {op} {b} = {rhs}")
            if op not in operators:
                operators[op] = []
            try:
                r = int(rhs)
                operators[op].append((a, b, r))
            except:
                operators[op].append((a, b, rhs))
    
    # Try common operations for each operator
    for op, triples in operators.items():
        if all(isinstance(t[2], int) for t in triples):
            # Try: a+b, a-b, a*b, a^b, |a-b|, a//b, a%b, concat, reverse
            for name, fn in [
                ('a+b', lambda a,b: a+b),
                ('a-b', lambda a,b: a-b),
                ('b-a', lambda a,b: b-a),
                ('a*b', lambda a,b: a*b),
                ('a//b', lambda a,b: a//b if b!=0 else -1),
                ('a%b', lambda a,b: a%b if b!=0 else -1),
                ('|a-b|', lambda a,b: abs(a-b)),
                ('a^2+b', lambda a,b: a**2+b),
                ('a+b^2', lambda a,b: a+b**2),
                ('concat_ab', lambda a,b: int(str(a)+str(b))),
                ('concat_ba', lambda a,b: int(str(b)+str(a))),
                ('a*b+a', lambda a,b: a*b+a),
                ('a*b-a', lambda a,b: a*b-a),
                ('a*b+b', lambda a,b: a*b+b),
                ('a*b-b', lambda a,b: a*b-b),
                ('(a+b)*2', lambda a,b: (a+b)*2),
                ('a^2-b^2', lambda a,b: a**2-b**2),
                ('a^2+b^2', lambda a,b: a**2+b**2),
                ('a*a+b*b', lambda a,b: a*a+b*b),
                ('reverse_digits_a+b', lambda a,b: int(str(a)[::-1])+b),
                ('digit_swap_a', lambda a,b: int(str(a)[::-1])),
            ]:
                try:
                    if all(fn(a,b) == r for a,b,r in triples):
                        print(f"  >>> RULE FOUND for '{op}': {name}")
                except:
                    pass
    
    # Parse query
    m = re.match(r'(\d+)(.)(\d+)', query) if query else None
    if m:
        print(f"  Query: {m.group(1)} {m.group(2)} {m.group(3)} => Expected: {answer}")
    print()

# Deep dive: symbolic type
print("\n=== SYMBOLIC TYPE DEEP DIVE ===\n")
symbolic_samples = []
for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if not is_numeric_type(examples) and len(examples) >= 3:
        symbolic_samples.append((examples, query, row['answer']))

print(f"Symbolic samples with 3+ examples: {len(symbolic_samples)}\n")
for i, (examples, query, answer) in enumerate(symbolic_samples[:8]):
    print(f"--- Sample {i+1} ---")
    for lhs, rhs in examples:
        print(f"  {lhs} = {rhs}")
    print(f"  Query: {query} => Expected: {answer}")
    
    # Check if it's a simple char-level transformation
    # e.g., each operand char maps to result char
    for lhs, rhs in examples:
        # Remove spaces, check length relationship
        lhs_clean = lhs.replace(' ', '')
        rhs_clean = rhs.replace(' ', '')
    
    # Check: is there an operator in the middle?
    # Pattern: XY op AB = CD  (where op is +, -, *, etc.)
    ops_found = set()
    for lhs, rhs in examples:
        for op in ['+', '-', '*']:
            if op in lhs:
                ops_found.add(op)
    if ops_found:
        print(f"  Operators found in LHS: {ops_found}")
    print()
