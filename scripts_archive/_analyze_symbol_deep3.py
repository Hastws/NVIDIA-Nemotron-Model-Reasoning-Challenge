import pandas as pd
import re
from collections import Counter

df = pd.read_csv('competition_data/train.csv')

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

def parse_examples(prompt):
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if '=' in line and 'determine' not in line.lower() and 'result' not in line.lower() and 'transformation' not in line.lower() and 'alice' not in line.lower() and 'equation' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        if 'determine the result for:' in line.lower():
            query = line.split(':')[-1].strip()
    return examples, query

def is_numeric_type(examples):
    for lhs, rhs in examples:
        if re.search(r'\d', lhs):
            return True
    return False

# ===== NUMERIC SYMBOL: try to find rules =====
print("="*60)
print("NUMERIC SYMBOL RULE DETECTION")
print("="*60)

basic_ops = [
    ('a+b', lambda a,b: a+b),
    ('a-b', lambda a,b: a-b),
    ('b-a', lambda a,b: b-a),
    ('a*b', lambda a,b: a*b),
    ('|a-b|', lambda a,b: abs(a-b)),
    ('concat_ab', lambda a,b: int(str(a)+str(b))),
    ('concat_ba', lambda a,b: int(str(b)+str(a))),
    ('a*b+a+b', lambda a,b: a*b+a+b),
    ('a^2+b^2', lambda a,b: a**2+b**2),
    ('a^2-b^2', lambda a,b: a**2-b**2),
    ('a^2*b', lambda a,b: a**2*b),
    ('a*b^2', lambda a,b: a*b**2),
    ('(a+b)^2', lambda a,b: (a+b)**2),
    ('(a-b)^2', lambda a,b: (a-b)**2),
    ('a*10+b', lambda a,b: a*10+b),
    ('reverse_a', lambda a,b: int(str(a)[::-1])),
    ('a//b', lambda a,b: a//b if b!=0 else -99999),
    ('a%b', lambda a,b: a%b if b!=0 else -99999),
]

# For numeric types:
# Each problem has multiple operators, each with its own rule
# Key insight: the operator SYMBOL defines the operation

numeric_total = 0
numeric_solved = 0
numeric_operator_rules = Counter()
unsolved_examples = []

for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if not is_numeric_type(examples) or len(examples) < 2:
        continue
    numeric_total += 1
    
    # Group examples by operator
    op_groups = {}
    for lhs, rhs in examples:
        m = re.match(r'(\d+)\s*(.)\s*(\d+)', lhs)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            if op not in op_groups:
                op_groups[op] = []
            try:
                r = int(rhs)
                op_groups[op].append((a, b, r))
            except:
                op_groups[op].append((a, b, rhs))
    
    # Parse query
    if not query:
        continue
    m = re.match(r'(\d+)\s*(.)\s*(\d+)', query)
    if not m:
        continue
    qa, qop, qb = int(m.group(1)), m.group(2), int(m.group(3))
    
    # Find rule for query operator  
    if qop in op_groups:
        triples = op_groups[qop]
        if all(isinstance(t[2], int) for t in triples):
            for name, fn in basic_ops:
                try:
                    if all(fn(a,b) == r for a,b,r in triples):
                        predicted = fn(qa, qb)
                        if str(predicted) == str(row['answer']):
                            numeric_solved += 1
                            numeric_operator_rules[name] += 1
                        else:
                            unsolved_examples.append((row['prompt'][:200], f"rule={name}, predicted={predicted}, actual={row['answer']}"))
                        break
                except:
                    pass

print(f"\nNumeric problems: {numeric_total}")
print(f"Solved by basic rules: {numeric_solved} ({numeric_solved/max(numeric_total,1)*100:.1f}%)")
print(f"\nRule distribution:")
for rule, count in numeric_operator_rules.most_common():
    print(f"  {rule}: {count}")

print(f"\nUnsolved/wrong predictions (first 10):")
for prompt, info in unsolved_examples[:10]:
    print(f"  {info}")
    print(f"  Prompt: {prompt[:100]}...")
    print()

# ===== SYMBOLIC: analyze character-level patterns =====
print("\n" + "="*60)
print("SYMBOLIC PATTERN ANALYSIS")
print("="*60)

# Key question: what's the structure?
# Pattern: "AB op CD = EF" where op is +, -, *
# or: "ABCDE = FGH" (no clear operator)

symbolic_with_ops = 0
symbolic_no_ops = 0
op_distribution = Counter()

for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if is_numeric_type(examples) or len(examples) < 2:
        continue
    
    has_op = False
    for lhs, rhs in examples:
        for op in ['+', '-', '*']:
            if op in lhs:
                has_op = True
                op_distribution[op] += 1
    
    if has_op:
        symbolic_with_ops += 1
    else:
        symbolic_no_ops += 1

print(f"\nSymbolic with arithmetic ops (+,-,*): {symbolic_with_ops}")
print(f"Symbolic without clear ops: {symbolic_no_ops}")
print(f"Op distribution: {dict(op_distribution)}")

# Analyze symbolic with operators more carefully
# Pattern: XY*AB = CD -- maybe operating on ASCII/char codes?
print("\n--- Symbolic with ops: character code analysis ---\n")
char_code_solved = 0
char_code_total = 0

for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if is_numeric_type(examples) or len(examples) < 3:
        continue
    
    # Check for operator-based patterns
    ops_in_lhs = set()
    for lhs, rhs in examples:
        for op in ['+', '-', '*']:
            if op in lhs:
                ops_in_lhs.add(op)
    
    if not ops_in_lhs:
        continue
    
    char_code_total += 1
    
    # Try: split LHS by operator, get char groups, apply op on char codes
    # Example: AB*CD = EF
    # Maybe: for each position i, result[i] = chr(ord(A[i]) op ord(C[i]))
    
    # Or maybe: it's about the length/structure
    # Let's check length patterns
    lhs_lens = []
    rhs_lens = []
    for lhs, rhs in examples:
        lhs_lens.append(len(lhs))
        rhs_lens.append(len(rhs))

# Check: no-operator symbolic — is it a substitution cipher on chars?
print("\n--- No-op symbolic: substitution analysis ---\n")

no_op_total = 0
for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if is_numeric_type(examples) or len(examples) < 3:
        continue
    
    ops_in_lhs = set()
    for lhs, rhs in examples:
        for op in ['+', '-', '*']:
            if op in lhs:
                ops_in_lhs.add(op)
    
    if ops_in_lhs:
        continue  # skip ones with operators
    
    no_op_total += 1
    if no_op_total <= 5:
        print(f"--- No-op sample {no_op_total} ---")
        for lhs, rhs in examples:
            print(f"  '{lhs}' => '{rhs}'  (len {len(lhs)} => {len(rhs)})")
        print(f"  Query: '{query}' => '{row['answer']}'")
        
        # Check: is there a char-to-char mapping?
        char_map = {}
        consistent = True
        for lhs, rhs in examples:
            if len(lhs) == len(rhs):
                for c1, c2 in zip(lhs, rhs):
                    if c1 in char_map:
                        if char_map[c1] != c2:
                            consistent = False
                    char_map[c1] = c2
        if consistent and char_map and all(len(l)==len(r) for l,r in examples):
            print(f"  -> Consistent char mapping! {char_map}")
            # Verify on query
            if query and all(c in char_map for c in query):
                predicted = ''.join(char_map.get(c, '?') for c in query)
                print(f"  -> Predicted: '{predicted}', Actual: '{row['answer']}', Match: {predicted == row['answer']}")
        print()

print(f"\nTotal no-op symbolic: {no_op_total}")
