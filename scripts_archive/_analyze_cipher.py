"""分析 cipher 题型的可解性：是否为简单替换密码？"""
import csv
import re
from collections import defaultdict

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    cipher_count = 0
    solvable = 0
    unsolvable_examples = []
    
    for row in reader:
        prompt = row['prompt']
        answer = row['answer']
        
        if 'secret encryption rules' not in prompt:
            continue
        cipher_count += 1
        
        # 提取训练对: ciphertext -> plaintext
        pairs = re.findall(r'^(.+?) -> (.+?)$', prompt, re.MULTILINE)
        # 提取查询
        query_match = re.search(r'decrypt the following text: (.+)', prompt)
        if not query_match or not pairs:
            continue
        
        query = query_match.group(1).strip().strip('"')
        
        # 构建字符映射
        mapping = {}
        conflict = False
        for cipher_text, plain_text in pairs:
            cipher_words = cipher_text.split()
            plain_words = plain_text.split()
            if len(cipher_words) != len(plain_words):
                conflict = True
                break
            for cw, pw in zip(cipher_words, plain_words):
                if len(cw) != len(pw):
                    conflict = True
                    break
                for cc, pc in zip(cw, pw):
                    if cc in mapping:
                        if mapping[cc] != pc:
                            conflict = True
                            break
                    mapping[cc] = pc
                if conflict:
                    break
            if conflict:
                break
        
        if conflict:
            unsolvable_examples.append((row['id'][:8], 'mapping conflict'))
            continue
        
        # 尝试解密查询
        decrypted = []
        missing = False
        for word in query.split():
            dw = ''
            for c in word:
                if c in mapping:
                    dw += mapping[c]
                else:
                    missing = True
                    dw += '?'
            decrypted.append(dw)
        
        result = ' '.join(decrypted)
        
        if result == answer:
            solvable += 1
        elif not missing:
            unsolvable_examples.append((row['id'][:8], f'got={result}, expected={answer}'))
        else:
            unsolvable_examples.append((row['id'][:8], f'missing chars, got={result}'))
        
        if cipher_count >= 1576:  # all cipher
            break

    print(f"Cipher 题型总数: {cipher_count}")
    print(f"程序化可解(简单替换): {solvable}/{cipher_count} ({solvable/cipher_count*100:.1f}%)")
    print(f"\n前10个不可解的例子:")
    for eid, reason in unsolvable_examples[:10]:
        print(f"  {eid}: {reason}")
