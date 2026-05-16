"""修正版: 考虑 T0 采样用的是 API (max_tokens=3584) vs 评测 (max_tokens=7680)"""
import json
from collections import defaultdict

# T0 原始采样 (max_tokens=3584): 很多题被截断了!
type_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'truncated': 0})

with open('data/cot_t0.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        t = obj.get('type', 'unknown')
        samples = obj.get('samples', [])
        type_stats[t]['total'] += 1
        
        for s in samples:
            temp = s.get('temperature', 999)
            if temp <= 0.5:
                if s.get('correct', False):
                    type_stats[t]['correct'] += 1
                if s.get('finish_reason') == 'length':
                    type_stats[t]['truncated'] += 1
                break

# T0v2 重采样 (max_tokens=7680): 更准确
type_v2 = defaultdict(lambda: {'correct': 0, 'total': 0})

with open('data/cot_t0_v2.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        t = obj.get('type', 'unknown')
        type_v2[t]['total'] += 1
        if obj.get('correct_count', 0) > 0:
            # 至少1次正确 / 3次采样 ≈ 单次正确概率的上界
            type_v2[t]['correct'] += 1

types = ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']

print('=== 为什么卡在 0.68? 全面诊断 ===\n')

print('基座模型各类正确率:')
print(f'{"类型":<12} {"T0低温单次":<14} {"截断率":<10} {"T0v2(7680)":<14}')
print('-' * 52)

t0_rates = {}
for t in types:
    s = type_stats[t]
    c_rate = s['correct']/s['total']*100 if s['total'] > 0 else 0
    trunc = s['truncated']/s['total']*100 if s['total'] > 0 else 0
    t0_rates[t] = c_rate
    
    v2 = type_v2.get(t, {'correct': 0, 'total': 0})
    v2_rate = v2['correct']/v2['total']*100 if v2['total'] > 0 else 0
    
    print(f'{t:<12} {c_rate:>5.1f}%{"":<8} {trunc:>5.1f}%{"":<4} {v2_rate:>5.1f}%')

# 关键发现: 评测用 max_tokens=7680, 截断问题大幅缓解
print(f'\n=== 关键: 截断是隐形杀手 ===')
print('T0 用 max_tokens=3584 → gravity/unit_conv 截断严重 → 单次正确率被低估')  
print('评测用 max_tokens=7680 → 基座实际能力比 T0 数据显示的更高')
print('所以 Baseline-Zero=0.52 > T0均值=39.3% 是合理的')

# 估算评测环境下的真实基线 (考虑截断修复)
print(f'\n=== 评测环境估算 (max_tokens=7680, temp=0.0) ===')
# numeral: 不受截断影响, ≈100%
# gravity: T0v2 (7680) = 89.2% (3shot), greedy 可能 ~75-80%
# unit_conv: T0v2 = 93.9% (3shot), greedy 可能 ~80-85%
# cipher: 截断严重但仍很低, ~10-15%
# bit_ops: 基本无解 ~3-5% 
# symbol: ~8-12%

est_base = {
    'numeral': 98,
    'gravity': 75,
    'unit_conv': 80,
    'cipher': 12,
    'bit_ops': 5,
    'symbol': 10,
}
base_avg = sum(est_base.values()) / 6
print(f'估算基座评测分 (enable_thinking): {base_avg:.1f}% ≈ 46.7%')
print('实测 Baseline-Zero = 52% → 说明模型可能在 cipher/symbol 有更高基线')

# 修正基座: 拟合52%
# 52% = (num + grav + unit + cipher + bit + sym) / 6
# 312 = num + grav + unit + cipher + bit + sym
# 已知 num≈98, grav≈75, unit≈80, 余: cipher+bit+sym ≈ 59
# 假设 cipher≈25, bit≈10, sym≈24 (比估算稍高)
est_52 = {
    'numeral': 98,
    'gravity': 75,
    'unit_conv': 80,
    'cipher': 25,
    'bit_ops': 10,
    'symbol': 24,
}
print(f'\n拟合 0.52 的各类分数:')
for t in types:
    print(f'  {t}: ~{est_52[t]}%')
print(f'  合计: {sum(est_52.values())/6:.1f}%')

# E1 = 0.68 → 提升 16个点
# 16 * 6 = 96个点 分配到6类
est_e1 = {
    'numeral': 99,    # +1 (天花板)
    'gravity': 85,    # +10 (格式修复+训练)
    'unit_conv': 90,  # +10 (格式修复+训练)
    'cipher': 45,     # +20 (从低基线提升最大)
    'bit_ops': 20,    # +10 (难题也有少许提升)
    'symbol': 35,     # +11 (难题少许提升)
}
e1_avg = sum(est_e1.values()) / 6
delta = {t: est_e1[t] - est_52[t] for t in types}
print(f'\n估算 E1(0.68) 各类分数:')
for t in types:
    print(f'  {t}: {est_52[t]}% → {est_e1[t]}% (+{delta[t]})')
print(f'  合计: {e1_avg:.1f}%')

# 0.75 目标
print(f'\n=== 从 0.68 → 0.75 需要什么 ===')
need = 75 * 6 - sum(est_e1.values())
print(f'需要总计再提升 {need} 个点 (分布在6类上)')
print(f'\n每类可用提升空间:')
for t in types:
    headroom = 100 - est_e1[t]
    difficulty = 'easy' if headroom > 30 else 'medium' if headroom > 10 else 'hard'
    print(f'  {t}: {est_e1[t]}% → 100%, 剩余 {headroom}% [{difficulty}]')

print(f'\n最现实的路径:')
target = {
    'numeral': 99,    # +0 (已满)
    'gravity': 90,    # +5 
    'unit_conv': 94,  # +4
    'cipher': 65,     # +20 ← 主攻方向!
    'bit_ops': 25,    # +5 (微提)
    'symbol': 40,     # +5 (微提)
}
tgt_avg = sum(target.values()) / 6
for t in types:
    d = target[t] - est_e1[t]
    if d > 0:
        print(f'  {t}: {est_e1[t]}% → {target[t]}% (+{d})')
print(f'  预期总分: {tgt_avg:.1f}%')

target2 = {
    'numeral': 99,
    'gravity': 92,
    'unit_conv': 95, 
    'cipher': 75,    # ← cipher 是最大单点杠杆
    'bit_ops': 30,
    'symbol': 45,
}
tgt2_avg = sum(target2.values()) / 6
print(f'\n激进路径 (cipher 全面突破):')
for t in types:
    d = target2[t] - est_e1[t]
    if d > 0:
        print(f'  {t}: {est_e1[t]}% → {target2[t]}% (+{d})')
print(f'  预期总分: {tgt2_avg:.1f}%')
