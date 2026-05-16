"""深入分析 bit_ops: 逐位映射 + 多步组合"""
import csv
import re
from itertools import product

def parse_bit_ops(prompt):
    pairs = re.findall(r'(\d{8}) -> (\d{8})', prompt)
    query_match = re.search(r'output for: (\d{8})', prompt)
    if not query_match:
        return None, None
    return pairs, query_match.group(1)

def analyze_bitwise_position(pairs):
    """分析每个位位置的映射关系 (是否依赖其他位)"""
    n = len(pairs)
    if n < 4:
        return None
    
    # For each output bit position, check if it depends on only one input position
    for out_pos in range(8):
        for in_pos in range(8):
            # Check if out_bit[out_pos] = in_bit[in_pos] for all pairs
            direct = all(pairs[i][0][in_pos] == pairs[i][1][out_pos] for i in range(n))
            inverted = all(pairs[i][0][in_pos] != pairs[i][1][out_pos] for i in range(n))
            if direct or inverted:
                continue  # Found a mapping
        # If no single input bit maps, this output bit depends on multiple inputs
    return None  # placeholder

def try_permutation_based(pairs):
    """尝试位置排列 + 可选取反"""
    n = len(pairs)
    # For each output bit position, find which input position it maps from (+ optional NOT)
    mapping = []  # list of (input_pos, invert)
    
    for out_pos in range(8):
        found = False
        for in_pos in range(8):
            direct = all(pairs[i][0][in_pos] == pairs[i][1][out_pos] for i in range(n))
            if direct:
                mapping.append((in_pos, False))
                found = True
                break
            inverted = all(pairs[i][0][in_pos] != pairs[i][1][out_pos] for i in range(n))
            if inverted:
                mapping.append((in_pos, True))
                found = True
                break
        if not found:
            return None  # This output bit can't be explained by a single input bit
    
    return mapping

def apply_mapping(inp_str, mapping):
    """根据映射生成输出"""
    result = []
    for in_pos, invert in mapping:
        bit = inp_str[in_pos]
        if invert:
            bit = '0' if bit == '1' else '1'
        result.append(bit)
    return ''.join(result)

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    total = 0
    perm_solved = 0
    unsolved_examples = []
    
    for row in reader:
        if 'bit manipulation rule' not in row['prompt']:
            continue
        total += 1
        
        pairs, query = parse_bit_ops(row['prompt'])
        if not pairs:
            continue
        
        mapping = try_permutation_based(pairs)
        if mapping:
            # Verify with answer
            predicted = apply_mapping(query, mapping)
            if predicted == row['answer']:
                perm_solved += 1
            else:
                print(f"映射匹配但答案不对: predicted={predicted}, actual={row['answer']}")
        else:
            if len(unsolved_examples) < 5:
                unsolved_examples.append({
                    'pairs': pairs,
                    'query': query,
                    'answer': row['answer']
                })

print(f"\n位排列+取反 可解: {perm_solved}/{total} ({perm_solved/total*100:.1f}%)")
print(f"不可解: {total - perm_solved}")

# 分析不可解样例
print(f"\n=== 不可解样例分析 ===")
for ex in unsolved_examples[:3]:
    print(f"\nPairs:")
    for i, o in ex['pairs']:
        print(f"  {i} -> {o}")
    print(f"Query: {ex['query']}")
    print(f"Answer: {ex['answer']}")
    
    # 分析每个output bit
    print("  Output bit analysis:")
    for out_pos in range(8):
        possible_sources = []
        for in_pos in range(8):
            direct = all(ex['pairs'][i][0][in_pos] == ex['pairs'][i][1][out_pos] for i in range(len(ex['pairs'])))
            inverted = all(ex['pairs'][i][0][in_pos] != ex['pairs'][i][1][out_pos] for i in range(len(ex['pairs'])))
            if direct:
                possible_sources.append(f"bit{in_pos}")
            if inverted:
                possible_sources.append(f"~bit{in_pos}")
        
        if not possible_sources:
            # Try XOR of 2 input bits
            for a in range(8):
                for b in range(a+1, 8):
                    xor_match = all(
                        (int(ex['pairs'][i][0][a]) ^ int(ex['pairs'][i][0][b])) == int(ex['pairs'][i][1][out_pos])
                        for i in range(len(ex['pairs']))
                    )
                    if xor_match:
                        possible_sources.append(f"bit{a}^bit{b}")
                    # AND
                    and_match = all(
                        (int(ex['pairs'][i][0][a]) & int(ex['pairs'][i][0][b])) == int(ex['pairs'][i][1][out_pos])
                        for i in range(len(ex['pairs']))
                    )
                    if and_match:
                        possible_sources.append(f"bit{a}&bit{b}")
                    # OR
                    or_match = all(
                        (int(ex['pairs'][i][0][a]) | int(ex['pairs'][i][0][b])) == int(ex['pairs'][i][1][out_pos])
                        for i in range(len(ex['pairs']))
                    )
                    if or_match:
                        possible_sources.append(f"bit{a}|bit{b}")

        print(f"    out[{out_pos}]: {possible_sources if possible_sources else 'COMPLEX'}")
