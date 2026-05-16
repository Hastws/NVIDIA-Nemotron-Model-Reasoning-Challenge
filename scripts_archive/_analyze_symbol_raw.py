"""分析 symbol: 直接用原始 prompt 文本"""
import csv

with open('/Users/hastws/work_space/NVIDIA Nemotron 模型推理挑战赛/competition_data/train.csv') as f:
    reader = csv.DictReader(f)
    count = 0
    for row in reader:
        if 'equation' not in row['prompt'].lower()[:200]:
            continue
        if count < 3:
            print(f"=== RAW PROMPT {count+1} ===")
            print(repr(row['prompt'][:600]))
            print(f"ANSWER: {repr(row['answer'])}")
            print()
            count += 1
        elif count < 6:
            # 数字型
            has_digits = any(c.isdigit() for c in row['prompt'][:400])
            if has_digits:
                print(f"=== NUMERIC PROMPT {count+1} ===")
                print(repr(row['prompt'][:600]))
                print(f"ANSWER: {repr(row['answer'])}")
                print()
                count += 1
