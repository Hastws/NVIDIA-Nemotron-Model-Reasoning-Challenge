"""分析为什么卡在 0.68 到不了 0.75"""
import json
from collections import defaultdict

type_greedy = defaultdict(lambda: [0, 0])

with open('data/cot_t0.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        t = obj.get('type', 'unknown')
        samples = obj.get('samples', [])
        type_greedy[t][1] += 1
        
        for s in samples:
            temp = s.get('temperature', 999)
            if temp <= 0.5:
                if s.get('correct', False):
                    type_greedy[t][0] += 1
                break

print('=== 基座模型单次低温正确率 (最接近评测 temp=0.0) ===')
types = ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']
rates = {}
for t in types:
    c, n = type_greedy[t]
    pct = c/n*100 if n > 0 else 0
    rates[t] = pct
    print(f'  {t}: {c}/{n} = {pct:.1f}%')

avg_base = sum(rates.values()) / 6
print(f'\n均匀加权基座平均: {avg_base:.1f}% (实测 Baseline-Zero = 52%)')

# 倒推 E1 = 0.68 的各类分数
# E1 训练 600 样本(每类~100条), 总分 0.68
# 提升 = 0.68 - 0.52 = 0.16 (16个点)
# 16个点 = 6类各提升多少?

print(f'\n=== 倒推 E1(0.68) 的分类分数 ===')
print(f'基座→E1 总提升: +16个点')
print(f'假设各类训练增益:')

# 合理假设: easy types 有格式修复+微量提升, hard types 有更大提升 (从零学起)
scenarios = {
    'A: 均匀提升': {t: rates[t]+16 for t in types},
    'B: easy+8, hard+24': {
        'numeral': rates['numeral']+5,  # 已经高,提升少
        'gravity': rates['gravity']+10,
        'unit_conv': rates['unit_conv']+8,
        'cipher': rates['cipher']+25,
        'bit_ops': rates['bit_ops']+20,
        'symbol': rates['symbol']+25,
    },
    'C: easy+3, hard+30': {
        'numeral': min(rates['numeral']+3, 100),
        'gravity': min(rates['gravity']+5, 100),
        'unit_conv': min(rates['unit_conv']+3, 100),
        'cipher': min(rates['cipher']+30, 100),
        'bit_ops': min(rates['bit_ops']+30, 100),
        'symbol': min(rates['symbol']+35, 100),
    },
}

for name, est in scenarios.items():
    avg = sum(est.values()) / 6
    print(f'\n  {name}: avg={avg:.1f}%')
    for t in types:
        print(f'    {t}: {rates[t]:.0f}% → {est[t]:.0f}%')

# 关键: 从 0.68 到 0.75 需要什么?
print(f'\n=== 从 0.68 到 0.75 需要什么? ===')
print(f'差距: 7个点 = 6类中提升总量 42个点')
print(f'\n可能的路径:')
print(f'  路径1: cipher +25 (18→43) + bit_ops +10 + symbol +7 = +42')
print(f'  路径2: cipher +30 + gravity +5 + unit_conv +5 + 零和 = +40')
print(f'  路径3: 全面提升每类 +7 = +42')
print(f'\n关键瓶颈:')
print(f'  numeral/gravity/unit_conv 已接近天花板, 提升空间 <5%')
print(f'  cipher 从 {rates["cipher"]:.0f}% 提升空间最大')
print(f'  bit_ops ({rates["bit_ops"]:.0f}%) 和 symbol ({rates["symbol"]:.0f}%) 提升极难')
