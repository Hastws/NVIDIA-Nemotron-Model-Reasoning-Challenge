"""验证 numeric symbol 的算术运算假设"""
import csv
import re
from itertools import combinations

def parse_eq(prompt):
    """解析equation prompt为pairs和query"""
    lines = prompt.split('\n')
    pairs = []
    query = None
    for line in lines:
        line = line.strip()
        if 'determine the result for:' in line.lower():
            query = line.split(':', 1)[-1].strip()
        elif ' = ' in line and 'example' not in line.lower() and 'Alice' not in line:
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs, query

def extract_num_op(s):
    """从 '34/44' 提取 (34, '/', 44)"""
    # 找2位数+操作符+2位数的模式
    m = re.match(r'^(\d{1,4})([\W_])(\d{1,4})$', s)
    if m:
        return int(m.group(1)), m.group(2), int(m.group(3))
    return None

# 可能的操作
def try_ops(a, b, result_str):
    """给定a, b和结果字符串，找出哪个操作匹配"""
    ops = []
    try:
        r = int(result_str)
        if a + b == r: ops.append('+')
        if a - b == r: ops.append('-')
        if a * b == r: ops.append('*')
        if b != 0 and a // b == r: ops.append('//')
        if b != 0 and a % b == r: ops.append('%')
        if abs(a - b) == r: ops.append('|a-b|')
        # reverse subtract
        if b - a == r: ops.append('b-a')
        # concatenation
        if str(a) + str(b) == result_str: ops.append('concat')
        # digit operations
        da = [int(d) for d in str(a)]
        db = [int(d) for d in str(b)]
        if sum(da) + sum(db) == r: ops.append('digitsum')
        if sum(da) * sum(db) == r: ops.append('digitmul')
        # interleave
        sa, sb = str(a), str(b)
        if len(sa) == 2 and len(sb) == 2:
            interleave1 = sa[0]+sb[0]+sa[1]+sb[1]
            interleave2 = sb[0]+sa[0]+sb[1]+sa[1]
            if interleave1 == result_str: ops.append('interleave1')
            if interleave2 == result_str: ops.append('interleave2')
            # cross products
            cross1 = str(int(sa[0])*int(sb[0])) + str(int(sa[1])*int(sb[1]))
            if cross1 == result_str: ops.append('cross_mul')
        # power
        if a > 0 and b > 0 and b <= 10:
            if a**b == r: ops.append('pow')
    except (ValueError, ZeroDivisionError):
        pass
    
    # String concat
    if str(a) + str(b) == result_str: ops.append('str_concat')
    if str(b) + str(a) == result_str: ops.append('str_rev_concat')
    
    return ops

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    total_numeric = 0
    fully_solved = 0
    partial_solved = 0
    
    for row in reader:
        if 'equation' not in row['prompt'].lower()[:200]:
            continue
        
        pairs, query = parse_eq(row['prompt'])
        if not pairs:
            continue
        
        # Check numeric
        all_numeric = True
        parsed = []
        for inp, out in pairs:
            trio = extract_num_op(inp)
            if trio is None:
                all_numeric = False
                break
            parsed.append((trio, out))
        
        if not all_numeric or not parsed:
            continue
        total_numeric += 1
        
        # 对每个操作符号，尝试找到对应的数学操作
        op_symbols = set(trio[1] for trio, _ in parsed)
        
        # 对每对，找出操作符号→数学操作的映射
        op_map = {}
        solvable = True
        for (a, op_sym, b), result_str in parsed:
            possible = try_ops(a, b, result_str)
            if op_sym not in op_map:
                op_map[op_sym] = set(possible) if possible else set()
            else:
                op_map[op_sym] = op_map[op_sym] & set(possible) if possible else set()
        
        all_unique = all(len(v) == 1 for v in op_map.values())
        any_found = all(len(v) >= 1 for v in op_map.values())
        
        if all_unique:
            fully_solved += 1
        elif any_found:
            partial_solved += 1
        else:
            if total_numeric - fully_solved - partial_solved <= 5:
                print(f"Unsolved: ops={op_map}")
                for (a, op, b), r in parsed[:3]:
                    print(f"  {a}{op}{b} = {r}")

print(f"\n=== Numeric Symbol 算术分析 ===")
print(f"数字型总数: {total_numeric}")
print(f"完全可解(每个op唯一): {fully_solved}/{total_numeric} ({fully_solved/total_numeric*100:.1f}%)")
print(f"部分可解(op有多个候选): {partial_solved}")
print(f"不可解: {total_numeric - fully_solved - partial_solved}")
