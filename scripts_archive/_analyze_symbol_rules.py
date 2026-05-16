"""
Focus on cracking the char-code arithmetic in symbol problems.

Key observation from Problem 2:
  '%|' * '"|' = '%|"|'   -> concat(left, right)!
  '\(' * '[^' = '\([^'   -> concat(left, right)!
  '(%' + '[@' = '(%[@'   -> concat(left, right) too!
  '|[' * '([' = '|[(['   -> concat(left, right)!
  '[^' - '[(' = '-^'     -> NOT concat

Problem 7:
  '#"' * '/[' = '#"/['   -> concat!
  '`$' + '%/' = '`$%/'   -> concat!

Problem 8:
  '`$' + '%/' = '`$%/'   -> concat!

So SOME operators map to concat. What do others map to?

Let me test more hypotheses on the ASCII codes.
"""
import pandas as pd
import re

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

# Define a charset (printable ASCII 32-126)
CHAR_BASE = 33  # '!' is the lowest we see
CHAR_RANGE = 94  # 33 to 126

def char_to_num(c):
    return ord(c) - CHAR_BASE

def num_to_char(n):
    return chr((n % CHAR_RANGE) + CHAR_BASE)

# Hypotheses for 2-char operands:
# Treat each 2-char string as a "number" in some base
# Or treat char-by-char

def try_charwise_ops(left, right, result):
    """Try char-by-char operations and return matching rule name or None."""
    if len(left) != 2 or len(right) != 2:
        return None
    
    l0, l1 = char_to_num(left[0]), char_to_num(left[1])
    r0, r1 = char_to_num(right[0]), char_to_num(right[1])
    
    # Hypothesis: treat as base-94 number
    left_val = l0 * CHAR_RANGE + l1
    right_val = r0 * CHAR_RANGE + r1
    
    # Try basic ops on the base-94 number
    for name, fn in [
        ('base94_add', lambda: left_val + right_val),
        ('base94_sub', lambda: left_val - right_val),
        ('base94_sub_rev', lambda: right_val - left_val),
        ('base94_mul', lambda: left_val * right_val),
        ('base94_concat', lambda: left_val * (CHAR_RANGE**len(right)) + right_val),
    ]:
        try:
            val = fn()
            # Convert back to chars
            if val < 0:
                val = val % (CHAR_RANGE ** 4)  # wrap around
            chars = []
            v = val
            if v == 0:
                chars = [CHAR_BASE]
            while v > 0:
                chars.append((v % CHAR_RANGE) + CHAR_BASE)
                v //= CHAR_RANGE
            predicted = ''.join(chr(c) for c in reversed(chars))
            if predicted == result:
                return name
        except:
            pass
    
    # Try char-by-char modular arithmetic
    for name, fn in [
        ('charwise_add', lambda i: (([l0,l1][i] + [r0,r1][i]) % CHAR_RANGE)),
        ('charwise_sub', lambda i: (([l0,l1][i] - [r0,r1][i]) % CHAR_RANGE)),
        ('charwise_sub_rev', lambda i: (([r0,r1][i] - [l0,l1][i]) % CHAR_RANGE)),
        ('charwise_xor', lambda i: ([l0,l1][i] ^ [r0,r1][i])),
    ]:
        if len(result) == 2:
            try:
                pred = ''.join(num_to_char(fn(i)) for i in range(2))
                if pred == result:
                    return name
            except:
                pass
    
    # concat
    if left + right == result:
        return 'concat_lr'
    if right + left == result:
        return 'concat_rl'
    
    return None

# Test on all symbolic problems with operators
rule_found_count = 0
total_tested = 0
rule_counter = {}
correct_predictions = 0
total_problems = 0

for idx, row in symbol_df.iterrows():
    examples, query = parse_examples(row['prompt'])
    if len(examples) < 3:
        continue
    
    # Parse into operator groups
    parsed = []
    for lhs, rhs in examples:
        for op in ['+', '-', '*']:
            if op in lhs:
                parts = lhs.split(op, 1)
                if len(parts) == 2:
                    parsed.append((parts[0], op, parts[1], rhs))
                    break
    
    if not parsed:
        continue
    
    # Group by operator and find rules
    op_groups = {}
    for left, op, right, result in parsed:
        if op not in op_groups:
            op_groups[op] = []
        op_groups[op].append((left, right, result))
    
    op_rules = {}
    for op, group in op_groups.items():
        # Try each rule on all examples for this op
        candidate_rules = {}
        for left, right, result in group:
            rule = try_charwise_ops(left, right, result)
            if rule:
                candidate_rules[rule] = candidate_rules.get(rule, 0) + 1
        
        # Pick the rule that matches ALL examples
        for rule_name, count in candidate_rules.items():
            if count == len(group):
                op_rules[op] = rule_name
                rule_found_count += 1
                rule_counter[rule_name] = rule_counter.get(rule_name, 0) + 1
                break
    
    total_tested += len(op_groups)
    
    # Try to predict the answer
    if query:
        for op in ['+', '-', '*']:
            if op in query:
                qparts = query.split(op, 1)
                if op in op_rules:
                    total_problems += 1
                    rule = op_rules[op]
                    # Apply rule to predict
                    ql, qr = qparts[0], qparts[1]
                    if rule == 'concat_lr':
                        predicted = ql + qr
                    elif rule == 'concat_rl':
                        predicted = qr + ql
                    elif len(ql) == 2 and len(qr) == 2:
                        l0, l1 = char_to_num(ql[0]), char_to_num(ql[1])
                        r0, r1 = char_to_num(qr[0]), char_to_num(qr[1])
                        left_val = l0 * CHAR_RANGE + l1
                        right_val = r0 * CHAR_RANGE + r1
                        
                        if rule == 'base94_add':
                            val = left_val + right_val
                        elif rule == 'base94_sub':
                            val = left_val - right_val
                        elif rule == 'base94_sub_rev':
                            val = right_val - left_val
                        elif rule == 'base94_mul':
                            val = left_val * right_val
                        elif rule == 'charwise_add':
                            predicted = ''.join(num_to_char((char_to_num(ql[i]) + char_to_num(qr[i])) % CHAR_RANGE) for i in range(2))
                            if predicted == str(row['answer']):
                                correct_predictions += 1
                            continue
                        elif rule == 'charwise_sub':
                            predicted = ''.join(num_to_char((char_to_num(ql[i]) - char_to_num(qr[i])) % CHAR_RANGE) for i in range(2))
                            if predicted == str(row['answer']):
                                correct_predictions += 1
                            continue
                        elif rule == 'charwise_sub_rev':
                            predicted = ''.join(num_to_char((char_to_num(qr[i]) - char_to_num(ql[i])) % CHAR_RANGE) for i in range(2))
                            if predicted == str(row['answer']):
                                correct_predictions += 1
                            continue
                        elif rule == 'charwise_xor':
                            predicted = ''.join(num_to_char(char_to_num(ql[i]) ^ char_to_num(qr[i])) for i in range(2))
                            if predicted == str(row['answer']):
                                correct_predictions += 1
                            continue
                        else:
                            continue
                        
                        if val < 0:
                            val = val % (CHAR_RANGE ** 4)
                        chars = []
                        v = val
                        if v == 0:
                            chars = [CHAR_BASE]
                        while v > 0:
                            chars.append((v % CHAR_RANGE) + CHAR_BASE)
                            v //= CHAR_RANGE
                        predicted = ''.join(chr(c) for c in reversed(chars))
                        
                        if predicted == str(row['answer']):
                            correct_predictions += 1
                    else:
                        if rule in ('concat_lr', 'concat_rl'):
                            pass  # already handled
                        continue
                break

print(f"Operator groups tested: {total_tested}")
print(f"Rules found (match all examples): {rule_found_count}")
print(f"\nRule distribution:")
for rule, count in sorted(rule_counter.items(), key=lambda x: -x[1]):
    print(f"  {rule}: {count}")

print(f"\nPrediction accuracy: {correct_predictions}/{total_problems} ({correct_predictions/max(total_problems,1)*100:.1f}%)")
