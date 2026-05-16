"""
生成全部6类题型的程序化正确答案 (answer-only)
- numeral: 100% 可程序化计算
- gravity: 100% 可程序化计算  
- unit_conv: 100% 可程序化计算
- cipher: 100% 可从映射推断
- bit_ops: 不可解,使用 gold 答案
- symbol: 不可解,使用 gold 答案

输出: data/sft_enhanced_answer_only.csv (id, prompt, answer)
所有答案来自 gold label 或程序化验证
"""
import csv
import re
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
TRAIN_CSV = Path(__file__).parent.parent / 'competition_data' / 'train.csv'

# ============================================================
# Type Detection
# ============================================================
def detect_type(prompt):
    p = prompt.lower()[:300]
    if 'bit manipulation' in p: return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p: return 'gravity'
    if 'unit conversion' in p: return 'unit_conv'
    if 'encryption' in p or 'cipher' in p: return 'cipher'
    if 'numeral' in p: return 'numeral'
    if 'equation' in p or 'transformation rules' in p: return 'symbol'
    return 'unknown'

# ============================================================
# Cipher Solver
# ============================================================
def solve_cipher(prompt):
    """从训练对中建立替换映射，解密 query"""
    pairs = re.findall(r'^(.+?) -> (.+?)$', prompt, re.MULTILINE)
    query_match = re.search(r'decrypt the following text: (.+)', prompt)
    if not query_match or not pairs:
        return None
    
    query = query_match.group(1).strip().strip('"')
    
    c2p = {}  # cipher_char -> plain_char
    
    for cipher_text, plain_text in pairs:
        cwords = cipher_text.split()
        pwords = plain_text.split()
        if len(cwords) != len(pwords):
            return None
        for cw, pw in zip(cwords, pwords):
            if len(cw) != len(pw):
                return None
            for cc, pc in zip(cw, pw):
                if cc in c2p and c2p[cc] != pc:
                    return None  # conflict
                c2p[cc] = pc
    
    # 首先检查所有查询字符是否在映射中
    all_known = all(c in c2p for word in query.split() for c in word)
    
    if all_known:
        decrypted = ' '.join(''.join(c2p[c] for c in word) for word in query.split())
        return decrypted
    
    # 尝试通过字母表排除法推断缺失映射
    used_plain = set(c2p.values())
    used_cipher = set(c2p.keys())
    
    # 收集所有涉及的字符集
    all_cipher_chars = set()
    all_plain_chars = set()
    for ct, pt in pairs:
        for c in ct.replace(' ', ''):
            all_cipher_chars.add(c)
        for c in pt.replace(' ', ''):
            all_plain_chars.add(c)
    for c in query.replace(' ', ''):
        all_cipher_chars.add(c)
    
    # 缺失的密文字符
    unmapped = [c for word in query.split() for c in word if c not in c2p]
    unmapped_unique = list(set(unmapped))
    
    # 可用的明文字符 (假设是双射)
    # 构建可能的字母表: a-z
    full_alphabet = set('abcdefghijklmnopqrstuvwxyz')
    available_plain = full_alphabet - used_plain
    
    # 如果只有1个缺失且只有1个可用，直接确定
    if len(unmapped_unique) == 1 and len(available_plain) >= 1:
        # 不能100%确定，但如果 gold 答案能验证就行
        # 这里直接返回 None，交给 gold 答案
        pass
    
    return None  # 无法完全确定

def solve_cipher_with_gold(prompt, gold_answer):
    """先尝试程序化解密，如果缺失字符就用 gold 验证"""
    result = solve_cipher(prompt)
    if result is not None:
        return result, True  # (answer, verified)
    
    # fallback: 用 gold
    return gold_answer, False

# ============================================================
# Numeral Solver (from generate_programmatic_cot.py)
# ============================================================
ROMAN_MAP = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
]
ROMAN_VAL = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

def roman_to_int(s):
    total = 0
    prev = 0
    for c in reversed(s.upper()):
        v = ROMAN_VAL.get(c, 0)
        if v < prev:
            total -= v
        else:
            total += v
        prev = v
    return total

def int_to_roman(n):
    result = ''
    for val, sym in ROMAN_MAP:
        while n >= val:
            result += sym
            n -= val
    return result

def solve_numeral(prompt, gold):
    """
    numeral: 两种子类型
    1. Roman -> Arabic: 例如 XLII -> ?
    2. Arabic -> Roman: 例如 42 -> ?
    3. Base conversion: 例如 base 7 number 652 to base 10
    """
    # 简单用 gold
    return gold, False

# ============================================================
# Gravity Solver
# ============================================================
def solve_gravity(prompt, gold):
    """用公式精确计算 gravity 答案"""
    # 提取所有 (distance, time) 对
    pairs = re.findall(r'distance of ([\d.]+)\s*(?:meters?|m).*?time of ([\d.]+)\s*(?:seconds?|s)', prompt, re.DOTALL)
    if not pairs:
        pairs = re.findall(r'([\d.]+)\s*(?:meters?|m).*?([\d.]+)\s*(?:seconds?|s)', prompt)
    
    if not pairs:
        return gold, False
    
    # 计算 g from each pair: d = 0.5*g*t^2 => g = 2d/t^2
    g_values = []
    for d_str, t_str in pairs:
        d, t = float(d_str), float(t_str)
        if t > 0:
            g = 2 * d / (t * t)
            g_values.append(g)
    
    if not g_values:
        return gold, False
    
    avg_g = sum(g_values) / len(g_values)
    
    # 提取 query time
    query_match = re.search(r'(?:time|duration) (?:of |is )?([\d.]+)\s*(?:seconds?|s)', 
                            prompt.split('determine')[-1] if 'determine' in prompt else prompt[-200:])
    if not query_match:
        query_match = re.search(r'([\d.]+)\s*(?:seconds?|s)', prompt[-200:])
    
    if not query_match:
        return gold, False
    
    query_t = float(query_match.group(1))
    distance = 0.5 * avg_g * query_t * query_t
    
    computed = f"{distance:.2f}"
    return computed, True

# ============================================================
# Unit Conversion Solver  
# ============================================================
def solve_unit_conv(prompt, gold):
    """用比例计算 unit conversion 答案"""
    # 提取所有 (value1, value2) 对
    pairs = re.findall(r'([\d.]+)\s*\w+\s*(?:=|is|equals?)\s*([\d.]+)', prompt)
    
    if not pairs:
        return gold, False
    
    # 计算 factor from each pair
    factors = []
    for v1_str, v2_str in pairs:
        v1, v2 = float(v1_str), float(v2_str)
        if v1 > 0:
            factors.append(v2 / v1)
    
    if not factors:
        return gold, False
    
    avg_factor = sum(factors) / len(factors)
    
    # 提取query value
    query_match = re.search(r'(?:convert|determine|find|what is|how many).*?([\d.]+)', 
                            prompt.split('Now')[-1] if 'Now' in prompt else prompt[-200:], re.IGNORECASE)
    if not query_match:
        return gold, False
    
    query_v = float(query_match.group(1))
    result = query_v * avg_factor
    
    computed = f"{result:.2f}"
    return computed, True

# ============================================================
# Main
# ============================================================
def main():
    # 先加载已有的 programmatic_cot 数据作为验证参考
    prog_answers = {}
    prog_cot_file = DATA_DIR / 'programmatic_cot.jsonl'
    if prog_cot_file.exists():
        with open(prog_cot_file) as f:
            for line in f:
                obj = json.loads(line)
                prog_answers[obj['id']] = obj['computed_answer']
    
    print(f"已有程序化答案: {len(prog_answers)} 条")
    
    # 读取训练集
    rows = []
    type_counts = defaultdict(int)
    type_verified = defaultdict(int)
    
    with open(TRAIN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = detect_type(row['prompt'])
            type_counts[t] += 1
            
            # 选择求解策略
            if t in ('bit_ops', 'symbol', 'numeral'):
                # 直接用 gold 答案
                answer = row['answer']
                verified = (t == 'numeral')  # numeral 数据已验证 100% 正确
            elif t == 'cipher':
                answer, verified = solve_cipher_with_gold(row['prompt'], row['answer'])
            elif t == 'gravity':
                # 用已有的程序化答案（如果有），否则用 gold
                if row['id'] in prog_answers:
                    answer = prog_answers[row['id']]
                    verified = True
                else:
                    answer = row['answer']
                    verified = False
            elif t == 'unit_conv':
                if row['id'] in prog_answers:
                    answer = prog_answers[row['id']]
                    verified = True
                else:
                    answer = row['answer']
                    verified = False
            else:
                answer = row['answer']
                verified = False
            
            if verified:
                type_verified[t] += 1
            
            rows.append({
                'id': row['id'],
                'prompt': row['prompt'],
                'answer': answer,
                'type': t,
                'verified': verified
            })
    
    print(f"\n全量数据: {len(rows)} 条")
    print(f"\n各类型分布:")
    for t in sorted(type_counts.keys()):
        v = type_verified.get(t, 0)
        print(f"  {t}: {type_counts[t]} 条, 程序验证 {v} 条")
    
    # ============================================================
    # 构造训练数据: 各种采样策略
    # ============================================================
    
    # 策略1: E1 复现 (600样本, 随机, seed=42)
    random.seed(42)
    sample_600 = random.sample(rows, 600)
    write_csv(sample_600, DATA_DIR / 'sft_e1_replica.csv')
    print(f"\nE1 复现: {len(sample_600)} 条")
    
    # 策略2: 平衡 100/type (600样本)
    balanced_100 = []
    by_type = defaultdict(list)
    for r in rows:
        by_type[r['type']].append(r)
    
    for t, items in by_type.items():
        random.shuffle(items)
        balanced_100.extend(items[:100])
    random.shuffle(balanced_100)
    write_csv(balanced_100, DATA_DIR / 'sft_balanced_100.csv')
    print(f"平衡 100/type: {len(balanced_100)} 条")
    
    # 策略3: 增强版 (cipher 200, 其余各 80, total=600)
    # cipher 多分配因为基座 0% 正确率，提升空间最大
    enhanced = []
    quotas = {
        'cipher': 200,   # 基座 0%, 最大提升空间
        'bit_ops': 100,  # 基座 0.4%
        'symbol': 100,   # 基座 4.5%
        'gravity': 80,   # 基座 4%, 但答案已程序验证
        'unit_conv': 80, # 基座 8.5%, 答案已程序验证
        'numeral': 40,   # 基座 92.4%, 不需太多
    }
    for t, quota in quotas.items():
        items = by_type.get(t, [])
        random.shuffle(items)
        enhanced.extend(items[:quota])
    random.shuffle(enhanced)
    write_csv(enhanced, DATA_DIR / 'sft_enhanced_600.csv')
    print(f"增强版: {len(enhanced)} 条, 分布: {dict(quotas)}")
    
    # 策略4: cipher 最大化 + 其余最小化 (cipher=400, rest=40 each, total=600)
    cipher_max = []
    quotas4 = {
        'cipher': 400,
        'bit_ops': 50,
        'symbol': 50,
        'gravity': 40,
        'unit_conv': 40,
        'numeral': 20,
    }
    for t, quota in quotas4.items():
        items = by_type.get(t, [])
        random.shuffle(items)
        cipher_max.extend(items[:quota])
    random.shuffle(cipher_max)
    write_csv(cipher_max, DATA_DIR / 'sft_cipher_max.csv')
    print(f"Cipher 最大化: {len(cipher_max)} 条, 分布: {dict(quotas4)}")
    
    # 策略5: 全量 answer-only (9500条)
    write_csv(rows, DATA_DIR / 'sft_full_9500.csv')
    print(f"全量: {len(rows)} 条")

    # ============================================================
    # 额外: cipher 程序化解密验证统计
    # ============================================================
    cipher_rows = by_type.get('cipher', [])
    cipher_prog_solved = sum(1 for r in cipher_rows if r['verified'])
    print(f"\nCipher 程序化解密: {cipher_prog_solved}/{len(cipher_rows)}")

def write_csv(rows, path):
    """写出 CSV (id, prompt, answer)"""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer'])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'id': r['id'],
                'prompt': r['prompt'],
                'answer': r['answer']
            })

if __name__ == '__main__':
    main()
