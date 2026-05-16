"""分析 symbol 题型的规律性"""
import csv
import re

def parse_symbol(prompt):
    """提取 symbol 题的输入输出对和查询"""
    # 格式: "X = Y\n" ... "determine the result for: Z"
    lines = prompt.strip().split('\n')
    pairs = []
    query = None
    for line in lines:
        line = line.strip()
        if ' = ' in line and 'determine' not in line.lower():
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
        if 'determine the result for:' in line.lower():
            q = line.split(':', 1)[-1].strip()
            query = q
    return pairs, query

def classify_symbol_puzzle(pairs, query):
    """尝试分类 symbol 题型"""
    if not pairs:
        return "empty"
    
    lhs0 = pairs[0][0]
    rhs0 = pairs[0][1]
    
    # Check if numeric
    all_numeric_lhs = all(re.match(r'^[\d\s\+\-\*\/\{\}\|\&\^\%\#\@\!\~\<\>\=\(\)]+$', p[0].replace(' ', '')) for p in pairs)
    all_numeric_rhs = all(re.match(r'^[\d]+$', p[1].strip()) for p in pairs)
    
    if all_numeric_rhs:
        return "numeric_output"
    else:
        return "symbolic_output"

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    total = 0
    categories = {}
    examples = {}
    
    for row in reader:
        if 'symbol transformation rule' not in row['prompt']:
            continue
        total += 1
        
        pairs, query = parse_symbol(row['prompt'])
        cat = classify_symbol_puzzle(pairs, query)
        categories[cat] = categories.get(cat, 0) + 1
        
        if cat not in examples or len(examples[cat]) < 3:
            examples.setdefault(cat, [])
            examples[cat].append({
                'pairs': pairs[:3],
                'query': query,
                'answer': row['answer']
            })

print(f"Symbol 总数: {total}")
print(f"\n分类分布:")
for cat, c in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {c}")

print(f"\n=== 样例 ===")
for cat, exs in examples.items():
    print(f"\n--- {cat} ---")
    for ex in exs[:2]:
        print(f"  Pairs: {ex['pairs'][:3]}")
        print(f"  Query: {ex['query']}")
        print(f"  Answer: {ex['answer']}")
        print()
