"""
生成 E1 增强版: 
- 与 E1 完全相同的 seed=42 采样策略
- 但 gravity/unit_conv 答案替换为程序化计算的精确答案
- cipher/bit_ops/symbol/numeral 保持 gold 答案不变

输出: data/sft_e1_enhanced.csv
"""
import csv
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
TRAIN_CSV = Path(__file__).parent.parent / 'competition_data' / 'train.csv'

def detect_type(prompt):
    p = prompt.lower()[:300]
    if 'bit manipulation' in p: return 'bit_ops'
    if 'gravitational' in p or 'gravity' in p: return 'gravity'
    if 'unit conversion' in p: return 'unit_conv'
    if 'encryption' in p or 'cipher' in p: return 'cipher'
    if 'numeral' in p: return 'numeral'
    if 'equation' in p or 'transformation rules' in p: return 'symbol'
    return 'unknown'

# 加载程序化精确答案
prog_answers = {}
with open(DATA_DIR / 'programmatic_cot.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        prog_answers[obj['id']] = obj['computed_answer']

print(f"程序化答案: {len(prog_answers)} 条")

# 读全部训练数据
all_rows = []
with open(TRAIN_CSV) as f:
    for row in csv.DictReader(f):
        all_rows.append(row)

# seed=42 采样 600 条 (复现 E1)
random.seed(42)
sample = random.sample(all_rows, 600)

# 替换 gravity/unit_conv 答案
replaced = 0
type_stats = {}
for row in sample:
    t = detect_type(row['prompt'])
    type_stats[t] = type_stats.get(t, 0) + 1
    
    if t in ('gravity', 'unit_conv') and row['id'] in prog_answers:
        old_ans = row['answer']
        new_ans = prog_answers[row['id']]
        if old_ans != new_ans:
            replaced += 1
        row['answer'] = new_ans

print(f"\n采样: {len(sample)} 条")
print(f"类型分布: {type_stats}")
print(f"替换了 {replaced} 个答案 (gravity/unit_conv 程序化计算)")

# 写出
out_path = DATA_DIR / 'sft_e1_enhanced.csv'
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer'])
    writer.writeheader()
    for row in sample:
        writer.writerow({'id': row['id'], 'prompt': row['prompt'], 'answer': row['answer']})

print(f"\n输出: {out_path}")

# 额外: 生成增强版 (cipher 加权) 
random.seed(42)  # 重新设 seed 保证可复现
all_by_type = {}
for row in all_rows:
    t = detect_type(row['prompt'])
    all_by_type.setdefault(t, []).append(row)

# 策略: cipher 额外加入 (从 E1 未选中的 cipher 里再选)
e1_ids = set(r['id'] for r in sample)
extra_cipher = [r for r in all_by_type.get('cipher', []) if r['id'] not in e1_ids]
random.shuffle(extra_cipher)

# E1 + 100 extra cipher = 700 条
enhanced_700 = list(sample) + extra_cipher[:100]
random.shuffle(enhanced_700)

out_700 = DATA_DIR / 'sft_e1_plus_cipher100.csv'
with open(out_700, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer'])
    writer.writeheader()
    for row in enhanced_700:
        writer.writerow({'id': row['id'], 'prompt': row['prompt'], 'answer': row['answer']})

print(f"E1+cipher100: {len(enhanced_700)} 条 → {out_700}")

# E1 + 200 extra cipher = 800 条
enhanced_800 = list(sample) + extra_cipher[:200]
random.shuffle(enhanced_800)

out_800 = DATA_DIR / 'sft_e1_plus_cipher200.csv'
with open(out_800, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer'])
    writer.writeheader()
    for row in enhanced_800:
        writer.writerow({'id': row['id'], 'prompt': row['prompt'], 'answer': row['answer']})

print(f"E1+cipher200: {len(enhanced_800)} 条 → {out_800}")
