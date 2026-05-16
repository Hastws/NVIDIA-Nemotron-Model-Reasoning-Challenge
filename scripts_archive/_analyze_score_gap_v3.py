"""精确拟合: 从已知分数反推各类准确率"""

types = ['numeral', 'gravity', 'unit_conv', 'cipher', 'bit_ops', 'symbol']

# === 已知数据点 ===
# Baseline-Zero (空LoRA + enable_thinking) = 0.52 → 总和 = 312
# E1 (600随机SFT + enable_thinking) = 0.68 → 总和 = 408
# SFT baseline (600随机SFT, no thinking) = 0.66 → 总和 = 396
# E5 balanced (均衡采样) = 0.66 → 总和 = 396
# E3 3epochs = 0.63 → 总和 = 378 (过拟合退化)

# === T0v2 数据 (max_tokens=7680, 3次采样): "至少答对1次" ===
# 这是基座能力上界 (多试几次总会碰上正确的)
t0v2_pass = {
    'numeral': 100.0,   # 几乎全对
    'gravity': 89.7,    # 高 
    'unit_conv': 93.5,  # 高
    'cipher': 0,        # T0v2没重采cipher,但T0数据+重采=18%
    'bit_ops': 9.7,     # 很低
    'symbol': 11.8,     # 很低
}

# T0 截断率 (max_tokens=3584): 说明哪些类型的思考需要更多空间
truncation = {
    'numeral': 0.0,
    'gravity': 15.5,
    'unit_conv': 57.8,
    'cipher': 86.4,
    'bit_ops': 96.2,
    'symbol': 90.4,
}

print('='*60)
print('  为什么600道题卡在0.68, 到不了0.75?')
print('='*60)

print('\n📊 1. 截断率揭示了真相:')
print(f'{"类型":<12} {"T0截断率":<12} {"含义"}')
print('-'*55)
for t in types:
    tr = truncation[t]
    if tr < 10:
        meaning = '思考短 → 基座就会'
    elif tr < 30:
        meaning = '偶尔截断 → 给足空间能解'
    elif tr < 60:
        meaning = '经常截断 → 需要长思考'
    else:
        meaning = '几乎全截断 → 模型拼命想但想不通'
    print(f'{t:<12} {tr:>5.1f}%{"":<6} {meaning}')

print('\n📊 2. 两组拟合 (假设6类等权):')
print()

# 拟合 Baseline-Zero = 0.52 (总和 312)
# numeral: 已知~100%, 取99
# gravity: T0v2=89.7% (3shot), greedy单次估 ~78%
# unit_conv: T0v2=93.5%, greedy估 ~82%
# 99+78+82 = 259, 余53给cipher+bit+sym
# cipher基座估 ~22% (T0低温11.2% 但评测tokens更多)
# bit_ops ~8%, symbol ~23%
base = {'numeral': 99, 'gravity': 78, 'unit_conv': 82, 'cipher': 22, 'bit_ops': 8, 'symbol': 23}
print(f'  Baseline-Zero拟合 (目标=52.0%):')
base_sum = sum(base.values())
base_avg = base_sum/6
for t in types:
    print(f'    {t:<12} {base[t]:>3}%')
print(f'    {"合计":<12} {base_avg:.1f}% (目标52%)')

# 拟合 E1 = 0.68 (总和 408)
# 训练600条 (每类~100条) 带来的提升:
# numeral: 99→99 (不需要训练)
# gravity: 78→90 (+12, 格式对齐+规律学习)
# unit_conv: 82→93 (+11, 格式对齐+规律学习)
# cipher: 22→50 (+28, 学到部分解密模式)
# bit_ops: 8→30 (+22, 记住一些输入→输出模式)
# symbol: 23→46 (+23, 记住一些变换模式)
e1 = {'numeral': 99, 'gravity': 90, 'unit_conv': 93, 'cipher': 50, 'bit_ops': 30, 'symbol': 46}
print(f'\n  E1拟合 (目标=68.0%):')
e1_sum = sum(e1.values())
e1_avg = e1_sum/6
delta_total = e1_sum - base_sum
for t in types:
    d = e1[t] - base[t]
    print(f'    {t:<12} {base[t]:>3}% → {e1[t]:>3}% (+{d:>2})')
print(f'    {"合计":<12} {e1_avg:.1f}% (目标68%)')
print(f'    总提升: +{delta_total} (分6类 = +{delta_total/6:.0f}/类)')

print('\n📊 3. 从0.68→0.75需要什么?')
need_total = 450 - e1_sum   # 75*6 = 450
print(f'  需要总计再提升: +{need_total} (平均每类 +{need_total/6:.0f})')
print()
print(f'  各类剩余空间 & 难度:')
for t in types:
    room = 100 - e1[t]
    if room <= 5:
        bar = '▓' * 1
        label = '🔴 已封顶'
    elif room <= 15:
        bar = '▓' * 3
        label = '🟡 空间小'  
    elif room <= 40:
        bar = '▓' * 6
        label = '🟢 有空间'
    else:
        bar = '▓' * 10
        label = '🟢 大空间'
    print(f'    {t:<12} {e1[t]:>3}% → 100%  余{room:>2}%  {bar} {label}')

print(f'\n📊 4. 核心矛盾:')
print()
easy_room = (100-e1['numeral']) + (100-e1['gravity']) + (100-e1['unit_conv'])
hard_room = (100-e1['cipher']) + (100-e1['bit_ops']) + (100-e1['symbol'])
print(f'  简单3类 (numeral/gravity/unit_conv) 剩余空间: {easy_room}% ({easy_room/3:.0f}%/类)')
print(f'  困难3类 (cipher/bit_ops/symbol) 剩余空间: {hard_room}% ({hard_room/3:.0f}%/类)')
print(f'  要凑 +{need_total} 点:')
print(f'    - 简单3类最多贡献: ~{easy_room}% (全做对也才+{easy_room})')
print(f'    - 还差: {need_total - easy_room}% 必须来自困难3类')
print()
print(f'  但困难3类的问题是:')
print(f'    cipher:  100%可编程求解, 但模型需要>100条训练样本才能看到足够多的替换模式')
print(f'    bit_ops: 3输入布尔函数, 只有9.5%可唯一确定. 大部分题目是多解的 → 60%+不现实')
print(f'    symbol:  每题规则独特, 2%可编程求解. 模型只能靠"猜"和"记忆" → 50%+很难')

print(f'\n📊 5. 现实路径估算:')
print()

scenarios = {
    '保守 (cipher+已验证数据)': {
        'numeral': 99, 'gravity': 92, 'unit_conv': 95, 
        'cipher': 65, 'bit_ops': 32, 'symbol': 48
    },
    '乐观 (cipher全面突破+更多训练)': {
        'numeral': 99, 'gravity': 94, 'unit_conv': 96,
        'cipher': 80, 'bit_ops': 35, 'symbol': 50
    },
    '理论极限 (每类都做到最好)': {
        'numeral': 100, 'gravity': 96, 'unit_conv': 98,
        'cipher': 90, 'bit_ops': 40, 'symbol': 55
    },
}

for name, est in scenarios.items():
    avg = sum(est.values()) / 6
    delta = avg - 68
    print(f'  {name}:')
    for t in types:
        d = est[t] - e1[t]
        if d > 0:
            print(f'    {t}: {e1[t]}→{est[t]} (+{d})')
    print(f'    → 预期: {avg:.1f}% (提升 +{delta:.1f})')
    print()

print(f'📊 6. 结论: 为什么600道题只能0.68?')
print(f'')
print(f'  1) 简单3类占50%分值 → 已接近满分 → 几乎无提升空间')
print(f'  2) cipher占16.7%分值 → 是最大单点杠杆 → E1约50%, 理论可到80-90%')
print(f'  3) bit_ops+symbol占33.3%分值 → 基座能力极差(~8-23%)')  
print(f'     → LoRA训练后~30-46% → 再往上极难 (多解问题/独特规则)')
print(f'  4) 600随机样本: 每类仅~100条, 对困难类远远不够')
print(f'  5) 0.75 需要困难3类平均≈55%, 当前~42% → 差 +13%/类')
print(f'     而这13%恰好是最难获取的部分')
print(f'')
print(f'  ⚡ 唯一的突破口: CIPHER')
print(f'  cipher 100%可编程求解 → 可以生成无限正确训练样本')
print(f'  把cipher从50%提到80% → 单这一项就值 +5分 (0.68→0.73)')
print(f'  再加上gravity/unit_conv的微提(+3分) → 0.76 可触及')
