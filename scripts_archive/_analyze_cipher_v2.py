"""分析 cipher 题型：用完整映射推断 + 缺失字符猜测"""
import csv
import re
from collections import defaultdict

def solve_cipher(prompt, answer):
    """尝试程序化解密"""
    pairs = re.findall(r'^(.+?) -> (.+?)$', prompt, re.MULTILINE)
    query_match = re.search(r'decrypt the following text: (.+)', prompt)
    if not query_match or not pairs:
        return None, "no_match"
    
    query = query_match.group(1).strip().strip('"')
    
    # 构建正向映射和反向映射
    c2p = {}  # cipher -> plain
    p2c = {}  # plain -> cipher
    conflict = False
    
    for cipher_text, plain_text in pairs:
        cwords = cipher_text.split()
        pwords = plain_text.split()
        if len(cwords) != len(pwords):
            return None, "word_count_mismatch"
        for cw, pw in zip(cwords, pwords):
            if len(cw) != len(pw):
                return None, "word_len_mismatch"
            for cc, pc in zip(cw, pw):
                if cc in c2p and c2p[cc] != pc:
                    return None, "conflict"
                if pc in p2c and p2c[pc] != cc:
                    return None, "reverse_conflict"
                c2p[cc] = pc
                p2c[pc] = cc
    
    # 尝试解密
    decrypted = []
    all_known = True
    for word in query.split():
        dw = ''
        for c in word:
            if c in c2p:
                dw += c2p[c]
            else:
                all_known = False
                dw += '?'
        decrypted.append(dw)
    
    result = ' '.join(decrypted)
    
    if result == answer:
        return result, "exact"
    
    # 检查是否缺失字符可以通过排除法推断
    if not all_known:
        # 已知的映射
        used_plain = set(c2p.values())
        used_cipher = set(c2p.keys())
        
        # 用 answer 来验证：把 answer 和 result 中的 ? 对照
        if len(result) == len(answer):
            for i, (r, a) in enumerate(zip(result, answer)):
                if r == '?' and a != ' ':
                    # 找到 query 中对应的密文字符
                    qi = 0
                    for word_idx, word in enumerate(query.split()):
                        for ci, c in enumerate(word):
                            if qi == i - query[:i].count(' ') + result[:i].count(' ') - result[:i].count(' '):
                                pass
                            qi += 1
            
            # 简单: 直接把 answer 当作"如果映射完整会得到什么"来验证
            new_c2p = dict(c2p)
            q_chars = list(query.replace(' ', ''))
            a_chars = list(answer.replace(' ', ''))
            
            if len(q_chars) == len(a_chars):
                new_conflict = False
                for qc, ac in zip(q_chars, a_chars):
                    if qc in new_c2p:
                        if new_c2p[qc] != ac:
                            new_conflict = True
                            break
                    else:
                        new_c2p[qc] = ac
                
                if not new_conflict:
                    return answer, "inferred"
    
    return result, "partial"


with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    stats = defaultdict(int)
    total = 0
    
    for row in reader:
        if 'secret encryption rules' not in row['prompt']:
            continue
        total += 1
        result, status = solve_cipher(row['prompt'], row['answer'])
        stats[status] += 1

print(f"Cipher 总数: {total}")
print(f"\n解密状态:")
for s, c in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c}/{total} ({c/total*100:.1f}%)")
print(f"\n可程序化解决 (exact+inferred): {stats['exact']+stats['inferred']}/{total} ({(stats['exact']+stats['inferred'])/total*100:.1f}%)")
