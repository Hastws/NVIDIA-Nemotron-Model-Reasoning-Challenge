"""
生成 cipher 加量训练数据集 (answer-only)

输出:
  1. data/sft_e1_plus_cipher400.csv   — E1(600) + 额外400 cipher = 1000条
  2. data/sft_e1_plus_cipher_all.csv   — E1(600) + 全部剩余 cipher ≈ 2100条
  3. data/sft_cipher_focused.csv       — cipher全量 + 其他类型少量保底 ≈ 2126条
"""
import csv
import random
from collections import Counter
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


def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'answer'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'id': row['id'], 'prompt': row['prompt'], 'answer': row['answer']})


def print_stats(name, rows):
    types = Counter(detect_type(r['prompt']) for r in rows)
    print(f"\n{'='*60}")
    print(f"  {name}: {len(rows)} 条")
    print(f"{'='*60}")
    for t in sorted(types.keys()):
        print(f"  {t:12s}: {types[t]:5d}  ({types[t]/len(rows)*100:5.1f}%)")
    # 检查 id 唯一性
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), f"重复 id! {len(ids)} rows, {len(set(ids))} unique"


def main():
    # 读全部训练数据
    all_rows = []
    with open(TRAIN_CSV) as f:
        for row in csv.DictReader(f):
            all_rows.append(row)
    print(f"训练集总量: {len(all_rows)}")

    # 按类型分组
    by_type = {}
    for row in all_rows:
        t = detect_type(row['prompt'])
        by_type.setdefault(t, []).append(row)
    print("各类型分布:")
    for t in sorted(by_type.keys()):
        print(f"  {t:12s}: {len(by_type[t])}")

    # ── E1 base: seed=42 采样 600 条 ──
    random.seed(42)
    e1_sample = random.sample(all_rows, 600)
    e1_ids = set(r['id'] for r in e1_sample)
    print_stats("E1 base (seed=42)", e1_sample)

    # E1 中的 cipher 数量
    e1_cipher_ids = set(r['id'] for r in e1_sample if detect_type(r['prompt']) == 'cipher')
    print(f"\nE1 中 cipher: {len(e1_cipher_ids)} 条")

    # 全部 cipher (不在 E1 中的)
    all_cipher = by_type.get('cipher', [])
    extra_cipher = [r for r in all_cipher if r['id'] not in e1_ids]
    print(f"全部 cipher: {len(all_cipher)} 条, E1 外剩余: {len(extra_cipher)} 条")

    # 用固定 seed shuffle extra_cipher 保证可复现
    random.seed(42)
    random.shuffle(extra_cipher)

    # ── Dataset 1: E1 + 400 extra cipher = 1000 条 ──
    ds1 = list(e1_sample) + extra_cipher[:400]
    random.seed(100)
    random.shuffle(ds1)
    out1 = DATA_DIR / 'sft_e1_plus_cipher400.csv'
    write_csv(ds1, out1)
    print_stats(f"sft_e1_plus_cipher400 → {out1}", ds1)

    # ── Dataset 2: E1 + 全部剩余 cipher ──
    ds2 = list(e1_sample) + extra_cipher
    random.seed(200)
    random.shuffle(ds2)
    out2 = DATA_DIR / 'sft_e1_plus_cipher_all.csv'
    write_csv(ds2, out2)
    print_stats(f"sft_e1_plus_cipher_all → {out2}", ds2)

    # ── Dataset 3: cipher-focused (cipher全量 + 其他少量保底) ──
    # cipher: 全量
    ds3_rows = list(all_cipher)
    ds3_ids = set(r['id'] for r in ds3_rows)

    # 其他类型按配额采样
    quotas = {'gravity': 200, 'unit_conv': 200, 'numeral': 50, 'bit_ops': 50, 'symbol': 50}
    random.seed(300)
    for t, n in quotas.items():
        pool = by_type.get(t, [])
        picked = random.sample(pool, min(n, len(pool)))
        for r in picked:
            if r['id'] not in ds3_ids:
                ds3_rows.append(r)
                ds3_ids.add(r['id'])

    random.seed(301)
    random.shuffle(ds3_rows)
    out3 = DATA_DIR / 'sft_cipher_focused.csv'
    write_csv(ds3_rows, out3)
    print_stats(f"sft_cipher_focused → {out3}", ds3_rows)

    print("\n✅ 全部生成完毕!")


if __name__ == '__main__':
    main()
