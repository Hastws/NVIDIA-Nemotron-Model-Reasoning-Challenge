"""深入分析 symbol 题型结构"""
import csv
import re

def parse_equation_puzzle(prompt):
    lines = prompt.strip().split('\n')
    pairs = []
    query = None
    for line in lines:
        line = line.strip()
        if line.startswith('In Alice') or line.startswith('Below') or not line:
            continue
        if 'determine the result for:' in line.lower():
            q = line.split(':', 1)[-1].strip()
            query = q
        elif ' = ' in line and 'example' not in line.lower():
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs, query

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    
    pure_sym = []
    numeric = []
    
    for row in reader:
        if 'equation' not in row['prompt'].lower()[:200]:
            continue
        
        pairs, query = parse_equation_puzzle(row['prompt'])
        if not pairs:
            continue
        
        has_digits = any(c.isdigit() for p in pairs for c in p[0] + p[1])
        
        entry = {
            'pairs': pairs,
            'query': query,
            'answer': row['answer'],
            'prompt': row['prompt']
        }
        
        if has_digits:
            if len(numeric) < 8:
                numeric.append(entry)
        else:
            if len(pure_sym) < 8:
                pure_sym.append(entry)

print("========== PURE SYMBOL (non-numeric) ==========")
for i, ex in enumerate(pure_sym[:5]):
    print(f"\n--- Example {i+1} ---")
    for inp, out in ex['pairs']:
        print(f"  '{inp}' → '{out}'  (len {len(inp)} → {len(out)})")
    print(f"  Query: '{ex['query']}' → Answer: '{ex['answer']}'")
    print(f"  Len pattern: {[f'{len(p[0])}→{len(p[1])}' for p in ex['pairs']]}")

print("\n\n========== NUMERIC WITH OPERATORS ==========")
for i, ex in enumerate(numeric[:5]):
    print(f"\n--- Example {i+1} ---")
    for inp, out in ex['pairs']:
        print(f"  '{inp}' → '{out}'  (len {len(inp)} → {len(out)})")
    print(f"  Query: '{ex['query']}' → Answer: '{ex['answer']}'")
