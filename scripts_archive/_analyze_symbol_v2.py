"""分析 symbol (equation) 题型: 是否也是字符替换密码?"""
import csv
import re

def parse_equation_puzzle(prompt):
    """解析 equation 题目"""
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
        elif ' = ' in line:
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs, query

def try_char_substitution(pairs):
    """尝试建立字符级替换映射"""
    # 方法: 假设每个输入字符映射到 0 或 1 个输出字符
    # 对齐输入和输出
    char_map = {}
    consistent = True
    
    for inp, out in pairs:
        # 每个输入字符映射到一段输出
        # 先试最简单的: 每个输入字符映射到恰好0或1个输出字符
        # 这是一个约束满足问题
        
        # 简化: 先假设每个字符映射到恰好1个字符 (input_len == output_len)
        if len(inp) == len(out):
            for i, c in enumerate(inp):
                if c in char_map:
                    if char_map[c] != out[i]:
                        consistent = False
                        break
                else:
                    char_map[c] = out[i]
            if not consistent:
                break
    
    return char_map, consistent

def try_char_sub_variable_len(pairs):
    """尝试字符替换 (每个字符→0或1个字符)"""
    from itertools import product
    
    # 收集所有出现的字符
    all_chars = set()
    for inp, out in pairs:
        all_chars.update(inp)
    
    # 对每对, 约束: sum of (0 or 1 output char per input char) = output
    # 先建立可能的映射
    char_map = {}  # c -> set of possible outputs (including '' for deletion)
    
    for c in all_chars:
        char_map[c] = set()  # will be populated
    
    # For each pair, try to deduce mappings
    # This is NP-hard in general, but with small alphabets it's feasible
    # Start by looking at pairs where input_len == output_len (1-to-1 mapping guaranteed)
    for inp, out in pairs:
        if len(inp) == len(out):
            for i, c in enumerate(inp):
                char_map[c].add(out[i])
    
    # Also look at pairs where output_len < input_len (some chars must map to empty)
    for inp, out in pairs:
        if len(inp) > len(out):
            diff = len(inp) - len(out)
            # 'diff' input chars map to '' (empty)
            pass  # complex to resolve without more info
    
    return char_map

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    total = 0
    equal_len = 0  # all pairs have equal input/output length
    consistent_sub = 0  # consistent 1-to-1 char substitution
    verified_correct = 0
    
    categorized = {'pure_symbol': 0, 'numeric_with_ops': 0, 'mixed': 0}
    
    for row in reader:
        if 'equation' not in row['prompt'].lower()[:200]:
            continue
        total += 1
        
        pairs, query = parse_equation_puzzle(row['prompt'])
        if not pairs or not query:
            continue
        
        # Check if all pairs have equal input/output length
        all_equal = all(len(p[0]) == len(p[1]) for p in pairs)
        if all_equal:
            equal_len += 1
            
            # Try char substitution
            char_map, consistent = try_char_substitution(pairs)
            if consistent:
                consistent_sub += 1
                
                # Verify: can we predict the answer?
                if all(c in char_map for c in query):
                    predicted = ''.join(char_map[c] for c in query)
                    if predicted == row['answer']:
                        verified_correct += 1
                    elif total <= 3:
                        print(f"Map mismatch: pred={predicted} actual={row['answer']}")
                        print(f"  Pairs: {pairs[:2]}")
                        print(f"  Query: {query}")
                        print(f"  Map: {char_map}")
        
        # Categorize
        has_digits = any(c.isdigit() for p in pairs for c in p[0] + p[1])
        has_ops = any(c in '+-*/\\|><`{}~^&%#@!' for p in pairs for c in p[0])
        if has_digits and has_ops:
            categorized['numeric_with_ops'] += 1
        elif not has_digits:
            categorized['pure_symbol'] += 1
        else:
            categorized['mixed'] += 1

print(f"\n=== Symbol (Equation) 分析 ===")
print(f"总数: {total}")
print(f"所有对等长: {equal_len}/{total}")
print(f"一致的1-1替换: {consistent_sub}/{total}")
print(f"验证正确: {verified_correct}/{total}")
print(f"\n分类: {categorized}")
