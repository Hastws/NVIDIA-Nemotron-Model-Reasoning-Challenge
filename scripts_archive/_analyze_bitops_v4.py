"""修正版 bit_ops 求解器 — 找所有合法映射，仅当预测唯一时才算可解"""
import csv
import re

def parse_bit_ops(prompt):
    pairs = re.findall(r'(\d{8}) -> (\d{8})', prompt)
    query_match = re.search(r'output for: (\d{8})', prompt)
    if not query_match:
        return None, None
    return pairs, query_match.group(1)

def solve_bitops(pairs, query):
    """
    对每个 output bit[j], 枚举所有可能的表达式 f(bit[a], bit[b]):
    - 1-input: identity(bit[a]) 或 not(bit[a])
    - 2-input: 16种布尔函数 (AND, OR, XOR, NAND, NOR, XNOR, etc.)
    
    对 query 求出每个 output bit 的所有可能值。
    如果某个 bit 的所有合法映射都给出相同预测，则该 bit 确定。
    如果所有 8 个 bit 都确定，则整个 puzzle 可解。
    """
    n = len(pairs)
    bit_predictions = []  # for each output bit: set of possible values
    
    for out_j in range(8):
        out_vals = [int(pairs[i][1][out_j]) for i in range(n)]
        possible_outputs = set()
        
        # 1-input: identity or NOT
        for in_a in range(8):
            in_vals = [int(pairs[i][0][in_a]) for i in range(n)]
            # identity
            if all(in_vals[i] == out_vals[i] for i in range(n)):
                possible_outputs.add(int(query[in_a]))
            # NOT
            if all((1 - in_vals[i]) == out_vals[i] for i in range(n)):
                possible_outputs.add(1 - int(query[in_a]))
        
        # 2-input: 枚举所有 (a, b, truth_table)
        for in_a in range(8):
            vals_a = [int(pairs[i][0][in_a]) for i in range(n)]
            for in_b in range(in_a + 1, 8):
                vals_b = [int(pairs[i][0][in_b]) for i in range(n)]
                
                # 16 种 2-input boolean functions
                for tt in range(16):
                    # Check if this truth table matches all pairs
                    match = True
                    for i in range(n):
                        idx = vals_a[i] * 2 + vals_b[i]
                        predicted = (tt >> idx) & 1
                        if predicted != out_vals[i]:
                            match = False
                            break
                    
                    if match:
                        # Apply to query
                        qa = int(query[in_a])
                        qb = int(query[in_b])
                        q_idx = qa * 2 + qb
                        pred = (tt >> q_idx) & 1
                        possible_outputs.add(pred)
        
        if len(possible_outputs) == 1:
            bit_predictions.append(str(possible_outputs.pop()))
        elif len(possible_outputs) == 0:
            return None  # 无法解释这个 output bit
        else:
            return None  # 该 output bit 有歧义 (0和1都可能)
    
    return ''.join(bit_predictions)

# 统计
with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    total = 0
    solved_correct = 0
    solved_wrong = 0
    ambiguous = 0
    
    for row in reader:
        if 'bit manipulation rule' not in row['prompt']:
            continue
        total += 1
        
        pairs, query = parse_bit_ops(row['prompt'])
        if not pairs:
            continue
        
        predicted = solve_bitops(pairs, query)
        if predicted is not None:
            if predicted == row['answer']:
                solved_correct += 1
            else:
                solved_wrong += 1
                if solved_wrong <= 3:
                    print(f"WRONG: pred={predicted} actual={row['answer']}")
                    print(f"  Pairs: {len(pairs)}, Query: {query}")
        else:
            ambiguous += 1

print(f"\n=== Bit_ops 完整分析 (含歧义过滤) ===")
print(f"总数: {total}")
print(f"确定且正确: {solved_correct}/{total} ({solved_correct/total*100:.1f}%)")
print(f"确定但错误: {solved_wrong} (可能有更复杂的操作)")
print(f"歧义(多个结果): {ambiguous}")
