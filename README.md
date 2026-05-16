# NVIDIA Nemotron Model Reasoning Challenge

> Kaggle: [nvidia-nemotron-model-reasoning-challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
> 团队: **Nemotron in wonderland** | 当前: **0.86** | 榜首: **0.87** | 差距: **0.01**

基于 **Nemotron-3-Nano-30B-A3B-BF16** (MoE)，通过 LoRA SFT 提升 6 类推理任务表现。

## 题型

| 题型 | 描述 |
|------|------|
| `bit_ops` | 8-bit 二进制变换规则推断 |
| `cipher` | 加密/密码规则推断 |
| `gravity` | 重力模拟 / 物理规则 |
| `numeral` | 自定义数字系统运算 |
| `symbol` | 符号变换方程 |
| `unit_conv` | 自定义单位换算 |

## 目录结构

```
├── README.md
├── .gitignore
├── kaggle_scripts/              # 主力训练 Notebook
│   ├── sft-no-lm-head/          # ★ 当前 0.86 版本 (soft prompt-loss)
│   ├── sft-unsloth/             # 参考 0.87 版本 (lm_head + thinking tags)
│   ├── sft/                     # SFTTrainer 基线
│   ├── grpo/                    # GRPO 实验
│   └── sft_old/
├── kaggle_kernels/              # Kaggle 公开 Kernel 存档 (ipynb only)
│   ├── ryanholbrook_*/          # 官方 Submission Demo
│   ├── dgxchen_*/               # Training w/ Unsloth 0.85 LB
│   ├── huikang_*/               # Tong Hui Kang 系列
│   ├── konbu17_*/               # CoT SFT LoRA
│   ├── amanatar_*/              # SFT→GRPO v3
│   ├── torpidoff_*/             # Full Pipeline
│   ├── afr1ste_*/               # 0.86 Tinker Adapter Guide
│   └── my_runs/                 # 自己的历史实验 kernel
├── competition_data/            # 竞赛原始数据 + baseline eval
│   ├── train.csv / test.csv
│   └── base_model_eval.jsonl    # 300 行基线评测
├── data/                        # 训练数据集 (多版本)
├── data_archive/                # 历史数据集迭代
├── scripts/                     # 工具脚本
├── scripts_archive/             # 历史脚本
├── tonghuikang_data/            # Tong Hui Kang 数据
├── tonghuikang_enhanced/        # Tong Hui Kang 增强数据
├── adapter/                     # 参考 adapter
├── adapter_output/              # 训练产出 adapter (gitignored)
├── kaggle_input/                # Kaggle 离线缓存 (gitignored, 18GB)
├── kaggle_upload/               # 提交暂存区 (gitignored)
├── offline_packages/            # .whl 离线包 (gitignored)
└── notebook_archive/            # 历史 notebook
```

## 榜单 (2026-05-15)

| Rank | Team | Score |
|------|------|-------|
| 1 | Y \| M \| F | 0.87 |
| 2 | Researcher 7919 | 0.87 |
| 3 | Lora is all you need | 0.87 |
| 4 | Diptyajit Das | 0.86 |
| 5 | Mattpelor | 0.86 |
| **6** | **Nemotron in wonderland** | **0.86** |
| ... | (共 20+ 支 0.86) | 0.86 |

## 0.86 → 0.87 关键差异

| 项 | 0.87 (dgxchen/unsloth) | 0.86 (我们的) |
|---|---|---|
| LoRA target | **含 `lm_head`** | 去掉了 lm_head |
| chat template | `enable_thinking=True` | `enable_thinking=False` |
| assistant 结构 | `<cot>\n</think>\n\boxed{ans}` | think 全 strip |
| epochs / sched | 1 / linear / no clip | 2 / cosine / clip=1.0 |
| prompt-loss | 全 token | soft mask 0.1 |

## 0.86 版本配置

```yaml
Model:     Nemotron-3-Nano-30B-A3B-BF16 (MoE)
LoRA:      r=32, α=32, dropout=0
Target:    q/k/v/o/in/out/up/down_proj (不含 lm_head)
Optim:     lr=2e-4, cosine, warmup=0.03, weight_decay=0.01
Train:     epochs=2, bs=2×16(eff=32), max_seq=8192, bf16
Data:      7830 rows (nemotron-cot-tong), stratified-by-type
PromptLoss: weight=0.1 on user prefix, mask via "<|im_start|>assistant\n"
```

## Baseline 诊断 (300 行 x 50/类型)

| 类型 | 准确率 | 截断率 | 备注 |
|------|--------|--------|------|
| bit_ops | 0.10 | 90% | 长度截断是主因 |
| symbol | 0.08 | 84% | 同上 |
| cipher | 0.34 | 54% | |
| unit_conv | 0.56 | 44% | |
| gravity | 0.72 | 2% | |
| numeral | 1.00 | 0% | 全对 |
| **Overall** | **0.467** | **~46%** | thinking_len p99=30K |

## 上分路线

1. **回滚关键差异** (lm_head + `</think>` + 1 epoch) → 预期 ≥0.87
2. 增大推理 max_tokens ≥ 32K 解决截断
3. bit_ops/symbol 短-CoT rejection sampling 数据增强
4. Self-consistency K=3 投票
5. LoRA rank sweep (r=64/128)

