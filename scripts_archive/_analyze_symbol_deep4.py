import pandas as pd
import re
from collections import Counter

df = pd.read_csv('competition_data/train.csv')

def classify(p):
    pl = p.lower()
    if 'cipher' in pl or 'encrypt' in pl or 'decrypt' in pl or 'encode' in pl or 'decode' in pl: return 'cipher'
    if 'gravit' in pl or 'planet' in pl: return 'gravity'
    if any(x in pl for x in ['base 2', 'base 8', 'base 10', 'base 16', 'binary', 'octal', 'hexadecimal']): return 'numeral'
    if any(x in pl for x in ['meter', 'mile', 'gallon', 'liter', 'pound', 'kilogram', 'inch', 'foot', 'feet', 'yard', 'ounce', 'celsius', 'fahrenheit', 'kelvin']): return 'unit_conv'
    if 'bit' in pl or 'xor' in pl or 'bitwise' in pl: return 'bit_ops'
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

# ============================================
# FOCUS: Symbolic with operators (+, -, *)
# These seem to have 2-char operands with operator in middle
# Like: AB*CD = EF
# ============================================
print("="*60)
print("SYMBOLIC WITH OPERATORS: DEEP ANALYSIS")
print("="*60)

# Let's look at a bunch very carefully
count = 0
for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if len(examples) < 3:
        continue
    
    # Check if ALL examples have operators in the LHS
    has_all_ops = True
    parsed_examples = []
    for lhs, rhs in examples:
        # Try: operandA op operandB format
        # Operators: +, -, *
        found = False
        for op in ['+', '-', '*']:
            if op in lhs:
                parts = lhs.split(op, 1)
                if len(parts) == 2:
                    parsed_examples.append((parts[0], op, parts[1], rhs))
                    found = True
                    break
        if not found:
            has_all_ops = False
    
    if not has_all_ops or not parsed_examples:
        continue
    
    count += 1
    if count <= 15:
        print(f"\n--- Problem {count} ---")
        for left, op, right, result in parsed_examples:
            # Show as char codes
            left_codes = [ord(c) for c in left]
            right_codes = [ord(c) for c in right]
            result_codes = [ord(c) for c in result]
            print(f"  '{left}' {op} '{right}' = '{result}'")
            print(f"    codes: {left_codes} {op} {right_codes} = {result_codes}")
        
        # Parse query too 
        if query:
            for op in ['+', '-', '*']:
                if op in query:
                    qparts = query.split(op, 1)
                    print(f"  Query: '{qparts[0]}' {op} '{qparts[1]}' => '{row['answer']}'")
                    break
        
        # Now try to figure out the rule:
        # Hypothesis 1: char-by-char operation on ASCII codes
        # For *, maybe it's some form of multiplication on char codes
        # For +, maybe it's addition on char codes (mod something)
        # For -, maybe subtraction
        
        # Check if operands have consistent lengths
        left_lens = [len(p[0]) for p in parsed_examples]
        right_lens = [len(p[2]) for p in parsed_examples]
        result_lens = [len(p[3]) for p in parsed_examples]
        print(f"    Left lens: {left_lens}, Right lens: {right_lens}, Result lens: {result_lens}")
        
        # Group by operator
        op_groups = {}
        for left, op, right, result in parsed_examples:
            if op not in op_groups:
                op_groups[op] = []
            op_groups[op].append((left, right, result))
        
        for op, group in op_groups.items():
            print(f"\n    Operator '{op}' ({len(group)} examples):")
            
            # Try: result = left (skip chars matching right?) 
            # Try: result length = |len(left) - len(right)|? or len(left)+len(right)?
            for left, right, result in group:
                # Check various hypotheses
                # H1: result = chars in left but not in right (set difference)
                left_set = set(left)
                right_set = set(right)
                diff_lr = ''.join(c for c in left if c not in right_set)
                diff_rl = ''.join(c for c in right if c not in left_set)
                
                # H2: result = remove chars from left that appear in right
                removed = list(left)
                rchars = list(right)
                for rc in rchars:
                    if rc in removed:
                        removed.remove(rc)
                h2_result = ''.join(removed)
                
                print(f"      '{left}' {op} '{right}' = '{result}'  | set_diff_LR='{diff_lr}' set_diff_RL='{diff_rl}' remove='{h2_result}'")

if count == 0:
    print("No symbolic problems with operators found!")
else:
    print(f"\nTotal symbolic with operators parsed: {count}")

# ============================================
# SPECIAL FOCUS: the "mixed" numeric examples
# where answers contain non-digit chars
# ============================================
print("\n" + "="*60)
print("NUMERIC WITH NON-DIGIT ANSWERS")  
print("="*60)

count2 = 0
for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if len(examples) < 3:
        continue
    
    # Check if numeric type
    has_digits = any(re.search(r'\d', lhs) for lhs, rhs in examples)
    if not has_digits:
        continue
    
    answer = str(row['answer'])
    if not re.match(r'^-?\d+\.?\d*$', answer):
        count2 += 1
        if count2 <= 10:
            print(f"\n--- Non-digit answer {count2}: '{answer}' ---")
            for lhs, rhs in examples:
                print(f"  {lhs} = {rhs}")
            print(f"  Query: {query}")

print(f"\nTotal numeric with non-digit answers: {count2}")
