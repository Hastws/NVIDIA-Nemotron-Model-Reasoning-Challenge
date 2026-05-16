"""Bit_ops: 尝试整字节组合操作 (1-3步链式)"""
import csv
import re

def parse_bit_ops(prompt):
    pairs = re.findall(r'(\d{8}) -> (\d{8})', prompt)
    query_match = re.search(r'output for: (\d{8})', prompt)
    if not query_match:
        return None, None
    return pairs, query_match.group(1)

# 基本操作 (8-bit)
def make_ops():
    ops = []
    # NOT
    ops.append(("NOT", lambda x: x ^ 0xFF))
    # Rotate left/right 1-7
    for r in range(1, 8):
        ops.append((f"ROL{r}", lambda x, r=r: ((x << r) | (x >> (8-r))) & 0xFF))
        ops.append((f"ROR{r}", lambda x, r=r: ((x >> r) | (x << (8-r))) & 0xFF))
    # Reverse bits
    ops.append(("REV", lambda x: int(f'{x:08b}'[::-1], 2)))
    # Swap nibbles
    ops.append(("SWAP4", lambda x: ((x >> 4) | ((x & 0xF) << 4)) & 0xFF))
    # XOR with constant
    for c in range(1, 256):
        ops.append((f"XOR{c:02x}", lambda x, c=c: x ^ c))
    # AND with constant
    for c in range(1, 255):
        ops.append((f"AND{c:02x}", lambda x, c=c: x & c))
    # OR with constant
    for c in range(1, 255):
        ops.append((f"OR{c:02x}", lambda x, c=c: x | c))
    return ops

OPS = make_ops()
print(f"Basic operations: {len(OPS)}")

def try_single_op(pairs):
    """尝试单步操作"""
    results = []
    for name, fn in OPS:
        if all(fn(int(i, 2)) == int(o, 2) for i, o in pairs):
            results.append((name, fn))
    return results

def try_two_ops(pairs):
    """尝试两步组合操作 (only non-XOR/AND/OR first, then any)"""
    # 先用非参数化操作 (NOT, ROT, REV, SWAP) 作为第一步
    non_param = [(n, f) for n, f in OPS if not any(n.startswith(p) for p in ['XOR', 'AND', 'OR'])]
    
    for name1, fn1 in non_param:
        # 第一步变换所有输入
        transformed = [(f'{fn1(int(i, 2)):08b}', o) for i, o in pairs]
        # 第二步找匹配操作
        for name2, fn2 in OPS:
            if all(fn2(int(t, 2)) == int(o, 2) for t, o in transformed):
                return (f"{name1}→{name2}", lambda x, f1=fn1, f2=fn2: f2(f1(x)))
    
    # 也试: ANY → non_param
    for name1, fn1 in OPS:
        transformed = [(f'{fn1(int(i, 2)):08b}', o) for i, o in pairs]
        for name2, fn2 in non_param:
            if all(fn2(int(t, 2)) == int(o, 2) for t, o in transformed):
                return (f"{name1}→{name2}", lambda x, f1=fn1, f2=fn2: f2(f1(x)))
    
    return None

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    total = 0
    single_solved = 0
    two_solved = 0
    unsolved = 0
    correct = 0
    
    for row in reader:
        if 'bit manipulation rule' not in row['prompt']:
            continue
        total += 1
        
        pairs, query = parse_bit_ops(row['prompt'])
        if not pairs:
            continue
        
        # Try single op
        singles = try_single_op(pairs)
        if singles:
            pred = f'{singles[0][1](int(query, 2)):08b}'
            if pred == row['answer']:
                single_solved += 1
                correct += 1
            continue
        
        # Try two ops
        result = try_two_ops(pairs)
        if result:
            name, fn = result
            pred = f'{fn(int(query, 2)):08b}'
            if pred == row['answer']:
                two_solved += 1
                correct += 1
            else:
                if two_solved + single_solved < 5:
                    print(f"2-step WRONG: {name}, pred={pred}, actual={row['answer']}")
            continue
        
        unsolved += 1

print(f"\n=== Bit_ops 整字节组合分析 ===")
print(f"总数: {total}")
print(f"单步可解: {single_solved}")
print(f"两步可解: {two_solved}")
print(f"总正确: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"不可解: {unsolved}")
