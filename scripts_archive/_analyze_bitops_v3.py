"""完整 bit_ops 求解器: 每个 output bit = f(bit_i, bit_j) 的所有组合"""
import csv
import re
from itertools import product

def parse_bit_ops(prompt):
    pairs = re.findall(r'(\d{8}) -> (\d{8})', prompt)
    query_match = re.search(r'output for: (\d{8})', prompt)
    if not query_match:
        return None, None
    return pairs, query_match.group(1)

# 所有可能的 2-input boolean functions (16种)
def make_2input_fns():
    """生成所有 16 种 2-input boolean functions"""
    fns = []
    for truth_table in range(16):
        # truth_table encodes f(0,0), f(0,1), f(1,0), f(1,1)
        def fn(a, b, tt=truth_table):
            idx = a * 2 + b
            return (tt >> idx) & 1
        fns.append((truth_table, fn))
    return fns

TWO_INPUT_FNS = make_2input_fns()

# 常见的名称
FN_NAMES = {
    0: "ZERO", 1: "AND", 2: "A&~B", 3: "A", 
    4: "~A&B", 5: "B", 6: "XOR", 7: "OR",
    8: "NOR", 9: "XNOR", 10: "~B", 11: "A|~B",
    12: "~A", 13: "~A|B", 14: "NAND", 15: "ONE"
}

def solve_bitops_full(pairs, query):
    """对每个 output bit, 尝试所有 (input_pos_a, input_pos_b, 2-input-function) 组合"""
    n = len(pairs)
    mapping = []
    
    for out_pos in range(8):
        found = False
        # 先试1-input (单个位 + 可选取反)
        for in_pos in range(8):
            direct = all(int(pairs[i][0][in_pos]) == int(pairs[i][1][out_pos]) for i in range(n))
            if direct:
                mapping.append(('1in', in_pos, None, 3))  # identity = fn 3 (just A)
                found = True
                break
            inverted = all(int(pairs[i][0][in_pos]) != int(pairs[i][1][out_pos]) for i in range(n))
            if inverted:
                mapping.append(('1in', in_pos, None, 12))  # NOT A = fn 12
                found = True
                break
        if found:
            continue
        
        # 2-input: 尝试所有 (a, b, fn) 组合
        for a in range(8):
            for b in range(8):
                if a == b:
                    continue
                for tt, fn in TWO_INPUT_FNS:
                    if tt in (0, 15, 3, 5, 12, 10):  # skip trivial (constant, identity, inversion)
                        continue
                    match = True
                    for i in range(n):
                        va = int(pairs[i][0][a])
                        vb = int(pairs[i][0][b])
                        expected = int(pairs[i][1][out_pos])
                        if fn(va, vb) != expected:
                            match = False
                            break
                    if match:
                        mapping.append(('2in', a, b, tt))
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            return None  # 无法解出这个 output bit
    
    # 验证: 用 mapping 对 query 做预测
    result = []
    for entry in mapping:
        if entry[0] == '1in':
            _, pos, _, tt = entry
            bit = int(query[pos])
            fn = TWO_INPUT_FNS[tt][1]
            result.append(str(fn(bit, 0)))  # b doesn't matter for 1-input fns
        else:
            _, a, b, tt = entry
            va = int(query[a])
            vb = int(query[b])
            fn = TWO_INPUT_FNS[tt][1]
            result.append(str(fn(va, vb)))
    
    return ''.join(result)

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    total = 0
    solved = 0
    correct = 0
    wrong = 0
    unsolvable = 0
    
    for row in reader:
        if 'bit manipulation rule' not in row['prompt']:
            continue
        total += 1
        
        pairs, query = parse_bit_ops(row['prompt'])
        if not pairs:
            continue
        
        predicted = solve_bitops_full(pairs, query)
        if predicted is not None:
            solved += 1
            if predicted == row['answer']:
                correct += 1
            else:
                wrong += 1
                if wrong <= 5:
                    print(f"WRONG: predicted={predicted}, actual={row['answer']}")
        else:
            unsolvable += 1

print(f"\n=== Bit_ops 完整分析 ===")
print(f"总数: {total}")
print(f"可解(2-input gate): {solved}/{total} ({solved/total*100:.1f}%)")
print(f"  正确: {correct}")
print(f"  映射冲突: {wrong}")
print(f"不可解: {unsolvable}")
