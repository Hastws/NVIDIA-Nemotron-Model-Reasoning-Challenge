#!/usr/bin/env python3
"""
Deep digit-level analysis of numeric symbol problems.
Many symbol problems treat each digit separately.
"""
import csv
from collections import defaultdict

rows = [r for r in csv.DictReader(open('competition_data/train.csv'))
        if 'symbol' in r['prompt'].lower() or 'equation' in r['prompt'].lower()]

OP_CHARS = set('+-*/|\\^&')

def split_by_op(expr):
    for i, c in enumerate(expr):
        if c in OP_CHARS and i > 0 and i < len(expr) - 1:
            return expr[:i], c, expr[i+1:]
    return None

def parse_examples(prompt):
    lines = prompt.strip().split('\n')
    examples = []
    query = None
    for line in lines:
        line = line.strip()
        if 'determine the result for:' in line.lower():
            query = line.split(':')[-1].strip()
        elif '=' in line and 'alice' not in line.lower() and 'equation' not in line.lower() \
                and 'transformation' not in line.lower() and 'determine' not in line.lower():
            parts = line.split('=', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                if lhs and rhs:
                    examples.append((lhs, rhs))
    return examples, query

# Focus on one specific unsolved problem to understand the pattern
# 34/44=1, 41/32=9, 34|25=69, 87\64=8853
# gold for 69/52 = 17/
# Hmm... 17/ has a slash in it, so this is treating digits like characters!

# Let me check: are the operands and results treated as CHARACTER STRINGS?
# For 34/44=1: '3','4' op '4','4' = '1'
# For 69/52: '6','9' op '5','2' = '1','7','/'

# Wait, if '/' appears in the answer (17/), these are NOT numeric - they're ASCII!
# The "numbers" are actually ASCII character codes or positions

# Let me re-examine: treat each char as ord(c) - ord('0') or just as chars
# 34/44 → result='1'  
# Using ASCII: ord('3')=51, ord('4')=52, ord('/')=47, ord('4')=52, ord('4')=52
# hmm... many possibilities

# Let me try treating them as BASE-94 strings (like the symbol ASCII problems)
CHAR_BASE = 33
CHAR_RANGE = 94

def str_to_b94(s):
    val = 0
    for c in s:
        val = val * CHAR_RANGE + (ord(c) - CHAR_BASE)
    return val

def b94_to_str(val):
    if val == 0:
        return chr(CHAR_BASE)
    if val < 0:
        val = val % (CHAR_RANGE ** 10)
    chars = []
    while val > 0:
        chars.append(chr((val % CHAR_RANGE) + CHAR_BASE))
        val //= CHAR_RANGE
    return ''.join(reversed(chars)) if chars else chr(CHAR_BASE)

# Now test: 34 / 44 = 1
# b94('34') = (ord('3')-33)*94 + (ord('4')-33) = 18*94 + 19 = 1711
# b94('44') = (ord('4')-33)*94 + (ord('4')-33) = 19*94 + 19 = 1805
# b94('1') = ord('1')-33 = 15
# 1711 - 1805 = -94... % 94^2 = ...
# 1711 + 1805 = 3516... b94_to_str(3516) = 3516//94=37 r=38, 37//94=0 r=37
# chr(38+33)=chr(71)='G', chr(37+33)=chr(70)='F' → "GF" ≠ "1"

# Hmm, let me try charwise:
# '3' cw_sub '4' = chr(((51-33)-(52-33)) % 94 + 33) = chr((-1)%94 + 33) = chr(93+33)=chr(126)='~'
# Not '1'

# Maybe it's digit-by-digit mod 10?
# '34' and '44': digit pairs: (3,4) and (4,4)
# result = '1': only 1 digit... 3+4=7, 4+4=8 → "78" ≠ "1"
# (3-4) mod 10 = 9, (4-4) = 0 → "90" ≠ "1"
# 3*4=12, 4*4=16 → "1216" ≠ "1"
# 3^4=7, 4^4=0 → "70" ≠ "1"

# Maybe the output length is just sum of digits? 3+4+4+4=15 ≠ 1
# Maybe product of digits? 3*4*4*4=192 ≠ 1

# I notice: 34|25=69, which looks like 3+2=5, 4+5=9 → charwise digit addition? 
# wait: 3+2=5, 4+5=9 → "59" but result is "69"
# nope. 3|2=3, 4|5=5 → bitwise OR → "35" ≠ "69"
# 6=3*2, 9=... hmm

# Let me check 34|25=69 more carefully
# 3*2=6, 4*... what? 4*? no
# (3+2)*something? 
# Actually for | (pipe), maybe it's string concatenation-like?

# Let me just print all problems organized by operator to see patterns
numeric_probs = defaultdict(list)
for r in rows:
    examples, query = parse_examples(r['prompt'])
    if not examples or not query:
        continue
    
    query_split = split_by_op(query)
    if not query_split:
        continue
    
    ql, qo, qr = query_split
    
    # Check if mixed (has non-printable, non-digit, non-alpha in results)
    numeric_probs[qo].append((r, examples, query, ql, qr))

for op in sorted(numeric_probs.keys()):
    probs = numeric_probs[op]
    print(f"\n=== Operator '{op}' ({len(probs)} problems) ===")
    
    # Show 3 examples
    for r, examples, query, ql, qr in probs[:3]:
        gold = r['answer'].strip()
        print(f"  Query: {query} = {gold}")
        for lhs, rhs in examples[:4]:
            print(f"    {lhs} = {rhs}")
        print()
