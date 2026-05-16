"""分析 bit_ops 题型的规律性"""
import csv
import re
from collections import defaultdict

def parse_bit_ops(prompt):
    """提取 bit_ops 的输入输出对和查询"""
    pairs = re.findall(r'(\d{8}) -> (\d{8})', prompt)
    query_match = re.search(r'output for: (\d{8})', prompt)
    if not query_match:
        return None, None
    return pairs, query_match.group(1)

def try_operations(pairs):
    """尝试常见位操作，看哪个能匹配所有对"""
    results = []
    
    for name, op_fn in [
        ("NOT", lambda x: x ^ 0xFF),
        ("ROT_L1", lambda x: ((x << 1) | (x >> 7)) & 0xFF),
        ("ROT_L2", lambda x: ((x << 2) | (x >> 6)) & 0xFF),
        ("ROT_L3", lambda x: ((x << 3) | (x >> 5)) & 0xFF),
        ("ROT_R1", lambda x: ((x >> 1) | (x << 7)) & 0xFF),
        ("ROT_R2", lambda x: ((x >> 2) | (x << 6)) & 0xFF),
        ("ROT_R3", lambda x: ((x >> 3) | (x << 5)) & 0xFF),
        ("REVERSE", lambda x: int(f'{x:08b}'[::-1], 2)),
        ("SWAP_NIBBLE", lambda x: ((x & 0xF0) >> 4) | ((x & 0x0F) << 4)),
    ]:
        if all(op_fn(int(i, 2)) == int(o, 2) for i, o in pairs):
            results.append(name)
    
    # try XOR with constant
    for const in range(256):
        if all((int(i, 2) ^ const) == int(o, 2) for i, o in pairs):
            results.append(f"XOR_{const:02x}")
    
    # try NOT then rotate, rotate then NOT, etc.
    for r in range(1, 8):
        rot_l = lambda x, r=r: ((x << r) | (x >> (8-r))) & 0xFF
        rot_r = lambda x, r=r: ((x >> r) | (x << (8-r))) & 0xFF
        if all(rot_l(int(i, 2)) == int(o, 2) for i, o in pairs):
            results.append(f"ROT_L{r}")
        if all(rot_r(int(i, 2)) == int(o, 2) for i, o in pairs):
            results.append(f"ROT_R{r}")
        # NOT + ROT
        if all(rot_l(int(i, 2) ^ 0xFF) == int(o, 2) for i, o in pairs):
            results.append(f"NOT_ROT_L{r}")
        if all(rot_r(int(i, 2) ^ 0xFF) == int(o, 2) for i, o in pairs):
            results.append(f"NOT_ROT_R{r}")
        # ROT + NOT
        if all((rot_l(int(i, 2)) ^ 0xFF) == int(o, 2) for i, o in pairs):
            results.append(f"ROT_L{r}_NOT")
        if all((rot_r(int(i, 2)) ^ 0xFF) == int(o, 2) for i, o in pairs):
            results.append(f"ROT_R{r}_NOT")
    
    # REVERSE + XOR
    for const in range(256):
        rev = lambda x: int(f'{x:08b}'[::-1], 2)
        if all((rev(int(i, 2)) ^ const) == int(o, 2) for i, o in pairs):
            results.append(f"REV_XOR_{const:02x}")
    
    # XOR + REVERSE
    for const in range(256):
        rev = lambda x: int(f'{x:08b}'[::-1], 2)
        if all(rev(int(i, 2) ^ const) == int(o, 2) for i, o in pairs):
            results.append(f"XOR_{const:02x}_REV")

    return results

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    total = 0
    simple_solved = 0
    op_dist = defaultdict(int)
    unsolved = []
    
    for row in reader:
        if 'bit manipulation rule' not in row['prompt']:
            continue
        total += 1
        
        pairs, query = parse_bit_ops(row['prompt'])
        if not pairs:
            continue
        
        ops = try_operations(pairs)
        if ops:
            # Verify answer
            first_op = ops[0]
            simple_solved += 1
            op_dist[ops[0].split('_')[0]] += 1
        else:
            unsolved.append(row['id'][:8])

print(f"Bit_ops 总数: {total}")
print(f"简单操作可解: {simple_solved}/{total} ({simple_solved/total*100:.1f}%)")
print(f"不可解: {total - simple_solved}")
print(f"\n操作类型分布:")
for op, c in sorted(op_dist.items(), key=lambda x: -x[1]):
    print(f"  {op}: {c}")
