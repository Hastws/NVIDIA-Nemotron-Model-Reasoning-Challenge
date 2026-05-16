"""
多模型多温度一致性采样 — 用于生成高质量 SFT 训练数据

═══ 核心理念 ═══
"基座自己能稳定做对的题才是好训练数据" → 一致性 > 数量
历史教训: 9000 全量样本训练(0.63) 远不如 600 精选样本(0.68)

═══ 策略概述 ═══
  1. T0 基座模型 (nano-30b) 每题 3 次采样，测试 self-consistency
     → 多温度 [0.4, 0.8, 1.2] 提供决策信号: 不同温度下都对 = 真会
  2. T1 中等模型 (super-120b) 仅对 T0 失败题补采 2 次
     → 作为 teacher，提供 T0 学不会的题的 CoT 示范
  3. 三重质量过滤:
     a) 正确次数 (correct_count)
     b) 答案字符串一致性 (防 "0.5" vs "1/2" 格式矛盾)
     c) Majority agreement (多数预测一致性)
  4. Score-based CoT 选择: 不选最短，选推理质量最高的
     → 完整输出 > 有 thinking > 长度适中

═══ 模型层级 ═══
  T0 (基座): nano-30b        — 每题 3 次采样 (与比赛评测同模型)
  T1 (中等): super-120b      — 仅补 T0 没全对的题
  T2 (已砍): ultra-253b      — 超纲题基座学不来，已移除

═══ 数据质量分级 ═══
  gold:     T0 ≥ 2/3 correct + 答案一致 + agreement ≥ 0.67
            → 最高质量，基座稳定掌握
  silver:   T0 = 1/3 correct，或 T0 ≥ 2/3 但一致性不达标
            → 中等质量，基座偶尔做对
  silver+:  T1 ≥ 2/2 correct (teacher CoT)
            → 基座做不对但大模型能教
  discard:  全模型都做不对 → 不可教，丢弃

═══ 每类题数据量 ═══
  目标: 500 ~ 1000 条/类
  难度混合比例: gold : silver : silver+ = 5 : 3 : 2
  (gold 优先保证数据质量，silver/silver+ 补齐数量和覆盖面)

═══ 运行流程 ═══
  Phase 1: python scripts/multi_model_cot.py --phase t0 --limit 100  # 测试跑
  Phase 1: python scripts/multi_model_cot.py --phase t0 --resume     # 全量+断点续跑
  Phase 2: python scripts/multi_model_cot.py --phase t1              # T1 补难题
  Phase 3: python scripts/multi_model_cot.py --phase merge           # 合并+分级+输出

═══ 输出文件 ═══
  data/cot_t0.jsonl       — T0 原始采样结果 (每行一题，含 3 个样本)
  data/cot_t1.jsonl       — T1 原始采样结果 (每行一题，含 2 个样本)
  data/merged_cot.jsonl   — 合并后全量数据 (含质量标签)
  data/sft_cot_data.jsonl — 最终 SFT 训练集 (经数据量控制)
"""
import os
import re
import csv
import json
import math
import time
import argparse
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — 所有关键参数集中于此，修改前请对照官方 metric 脚本
# ═══════════════════════════════════════════════════════════════════════════════
API_BASE = "https://integrate.api.nvidia.com/v1"  # NVIDIA Integrate API 端点

# 官方评测 prompt 后缀 — 必须与 Kaggle metric 脚本完全一致!
# 来源: competition_notebooks/nemotron-baseline-evaluation.ipynb → user_content 拼接
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# 模型层级配置
# 设计原则:
#   - T0 用比赛同款基座模型，数据-模型对齐最重要
#   - T1 用更大模型作 teacher，但只补 T0 失败题 (避免教超纲内容)
#   - temperatures 用 spread 策略: 低温(确定性) + 中温(平衡) + 高温(探索)
#     不同温度下都答对 = 真正掌握，而非恰好碰对
MODEL_TIERS = {
    "t0": {
        "model": "nvidia/nemotron-3-nano-30b-a3b",
        "desc": "基座模型 (与比赛评测一致)",
        "samples_per_question": 3,       # 每题采样 3 次，测 self-consistency
        "temperatures": [0.4, 0.8, 1.2], # 低/中/高 spread (v3: 从[0.6,1.0,1.0]拉开)
        "max_tokens": 7680,              # ⚠️ 对齐官方 Eval Page (2026-03-24 确认)
        "enable_thinking": True,         # 官方评测也开 thinking
    },
    "t1": {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "desc": "中等模型 (120B, 补难题)",
        "samples_per_question": 2,       # 2 次采样, ≥ 2/2 才算可靠
        "temperatures": [0.6, 0.8],      # 偏保守，teacher 不需要太大探索
        "max_tokens": 7680,              # 对齐官方 Eval Page (2026-03-24 确认)
        "enable_thinking": True,
    },
}

# 每类题数据量控制
# 历史教训: 全量9500训练=0.63, 精选600=0.68 → 质量远比数量重要
# 第一轮: 6 种题型 × 80~150 = 总量 480~900 条 (对齐 E1 的 600)
# 后续可根据评测结果上调
PER_TYPE_MIN = 80    # 少于此数告警 (数据不够需扩充)
PER_TYPE_MAX = 150   # 超过此数截断 (防过多低质量数据)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_CSV = os.path.join(os.path.dirname(__file__), "..", "competition_data", "train.csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILS — 工具函数
# ═══════════════════════════════════════════════════════════════════════════════
def detect_type(prompt):
    """根据 prompt 前 300 字符的关键词检测题目类型。

    6 种题型:
      bit_ops   — 8位二进制位操作 (shift, XOR, AND, OR, NOT, rotate)
      cipher    — 文本加密/解密 (替换密码、凯撒密码等)
      gravity   — 重力常数推断 (物理公式计算)
      numeral   — 进制转换 (罗马数字、Wonderland Numbers 等)
      unit_conv — 单位换算 (长度、重量、温度等)
      symbol    — 符号方程变换 (规则推理、符号替换)

    注意: 基于关键词的启发式方法，非 100% 准确。
    如果 prompt 里没有明确关键词会返回 'unknown'。
    """
    p = prompt[:300].lower()
    if "8-bit binary" in p or ("bit" in p and "binary" in p):
        return "bit_ops"
    elif "encrypt" in p or "cipher" in p:
        return "cipher"
    elif "gravit" in p:
        return "gravity"
    elif "numeral" in p or "wonderland numbers" in p:
        return "numeral"
    elif ("unit" in p and "conversion" in p) or ("convert" in p and "measurement" in p):
        return "unit_conv"
    elif "transformation" in p and ("equation" in p or "rule" in p):
        return "symbol"
    return "unknown"


def extract_boxed(text):
    """从文本中提取最后一个非空 \\boxed{} 内容。

    与官方 extract_final_answer() 的 boxed 提取逻辑一致:
      - 匹配所有 \\boxed{...} (含末尾未闭合的 \\boxed{...)
      - 优先取最后一个非空 match

    注意: 这里只做 boxed 提取，不做 fallback ("Final answer is:" 等)。
    因为训练数据采样时我们严格要求 \\boxed{} 格式，
    fallback 提取不可靠，可能引入错误标注。
    """
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return None


def answers_match(pred, gold):
    """判断预测答案是否与标准答案匹配。

    完全对齐官方 verify() 的逻辑:
      - 数值: rel_tol=1e-2, abs_tol=1e-5 (允许约1%的相对误差)
      - 字符串: 大小写不敏感的精确匹配

    这个函数用于训练数据筛选，必须和评测标准一致，
    否则我们以为"对了"的数据在评测时可能"错了"。
    """
    if pred is None:
        return False
    pred, gold = pred.strip(), gold.strip()
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, OverflowError):
        return pred.lower() == gold.lower()


def normalize_pred(pred):
    """归一化预测值字符串，解决 "42" vs "42.0" 的格式差异问题。

    用于 answers_consistent 和 agreement_ratio 计算前的预处理，
    避免纯字符串比较导致语义相同但格式不同的答案被判为不一致。

    规则:
      - 能转 float 的 → str(float(x))  (如 "42.0" → "42.0", "42" → "42.0")
      - 不能转的 → strip().lower()      (如 "Cat" → "cat")
    """
    if pred is None:
        return None
    pred = pred.strip()
    try:
        return str(float(pred))
    except (ValueError, OverflowError):
        return pred.lower()


def score_cot(sample):
    """对一个正确样本的 CoT 质量打分。分数越高 = 越好。

    用于在多个正确样本中选出最佳 CoT 作为训练目标。
    设计原则: 完整性 > 实质推理 > 长度适中 > 格式规范

    评分维度:
      完整性:   finish_reason="stop" +5 / "length" -3
      推理深度: thinking > 200 chars +3 / > 50 +1 / < 50 -2
      长度甜区: 300~2500 chars +2 / > 3000 -1 / < 100 -2
      格式规范: \\boxed{} 在 content 中 +2 / 仅在 thinking 中 -2

    理论分数范围: [-9, +12]
    """
    score = 0
    thinking = sample.get("thinking") or ""
    response = sample.get("response") or ""
    think_len = len(thinking)
    total_len = think_len + len(response)

    # [维度1] 完整输出 (没被截断) 是最重要的信号
    # 被截断意味着 \boxed{} 可能不完整，训练这样的 CoT 有害
    if sample.get("finish_reason") == "stop":
        score += 5
    elif sample.get("finish_reason") == "length":
        score -= 3  # 被截断的 CoT 可能不完整

    # [维度2] 有实质推理过程 (thinking 字段长度)
    # 太短的 thinking 可能是瞎猜碰对，不是真推理
    if think_len > 200:
        score += 3   # 有充实的推理链
    elif think_len > 50:
        score += 1   # 有简短推理
    else:
        score -= 2   # 太短，可能是瞎猜

    # [维度3] 总长度适中 (thinking + response)
    # 太短 = 可能没推理; 太长 = 啰嗦，训练时可能撑爆 token
    # 甜区 300~6000 基于 max_tokens=7680 估算 (2026-03-24 更新)
    if 300 <= total_len <= 6000:
        score += 2   # 甜区: 有内容但不啰嗦
    elif total_len > 7000:
        score -= 1   # 啰嗦
    elif total_len < 100:
        score -= 2   # 太短

    # [维度4] \boxed{} 出现位置
    # 好的输出: content 末尾有 \boxed{} → 训练时模型学到正确的输出格式
    # 差的输出: \boxed{} 只在 thinking 里 → 模型学到错误的格式习惯
    if extract_boxed(response):
        score += 2   # content 中有 \boxed{}: 格式规范
    elif extract_boxed(thinking):
        score -= 2   # 仅 thinking 中有: 格式不规范

    return score


def call_api(client, model, prompt, temperature, max_tokens, enable_thinking, retries=3):
    """调用 NVIDIA Integrate API 生成回答。

    请求格式严格对齐官方评测:
      - 无 system message (官方也没有)
      - user content = prompt + METRIC_SUFFIX
      - top_p=1.0 (官方固定)
      - enable_thinking 通过 extra_body.chat_template_kwargs 传递

    返回: (thinking, content, finish_reason)
      - thinking: reasoning_content 字段 (模型内部推理链)
      - content:  message.content 字段 (最终输出，含 \\boxed{})
      - finish_reason: "stop"(正常完成) / "length"(被截断) / "error"(异常)

    重试策略: 指数退避 (2s, 4s, 8s)，最多 retries 次
    """
    user_content = prompt + METRIC_SUFFIX
    messages = [{"role": "user", "content": user_content}]

    extra = {}
    if enable_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": True}

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
                extra_body=extra if extra else None,
                timeout=180,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            thinking = getattr(choice.message, 'reasoning_content', None) or ""
            finish = choice.finish_reason
            return thinking, content, finish
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    ⚠️ Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                return "", f"[ERROR] {e}", "error"


def load_rows():
    """加载 train.csv 全部行, 返回 list[dict]。
    每行包含: id, prompt, answer
    """
    train_path = os.path.abspath(TRAIN_CSV)
    with open(train_path, "r") as f:
        return list(csv.DictReader(f))


def load_done_ids(path):
    """从 JSONL 文件中加载已完成的 sample IDs (用于断点续跑)。
    静默忽略损坏行, 保证鲁棒性。
    """
    done = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["id"])
                except:
                    pass
    return done


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE: 单模型多次采样 (T0 / T1 共用)
#  每题 N 次采样 → 统计 self-consistency → 记录最佳 CoT
# ═══════════════════════════════════════════════════════════════════════════════
def _sample_one_temp(client, model, prompt, gold, temp, max_tokens, enable_think):
    """单次采样: 调用 API + 提取答案 + 判定正误。线程安全，无共享状态。

    答案提取严格只从 content (response) 中提取 \\boxed{}:
      - 官方评测只看 content，不看 thinking/reasoning_content
      - 如果 \\boxed{} 仅出现在 thinking 中，评测时会失分
      - 因此不做 thinking fallback，保证训练数据与评测行为对齐
    """
    t0 = time.time()
    thinking, content, finish = call_api(
        client, model, prompt, temp, max_tokens, enable_think
    )
    elapsed = time.time() - t0

    # 严格只从 content 提取，不 fallback 到 thinking
    # (thinking 里的 \boxed{} 在官方评测中不计分)
    pred = extract_boxed(content)
    match = answers_match(pred, gold)

    return {
        "temperature": temp,
        "thinking": thinking,
        "response": content,
        "predicted": pred,
        "correct": match,
        "finish_reason": finish,
        "elapsed": round(elapsed, 2),
    }


def _process_one_question(client, model, prompt, gold, temps, n_samples, max_tokens, enable_think):
    """处理单道题: 并发采样 N 个温度 → 一致性判断 → 选最佳 CoT。线程安全。"""
    # 同一题的 N 次采样并发执行 (I/O 密集, 线程并发有效)
    with ThreadPoolExecutor(max_workers=n_samples) as inner_pool:
        futures = []
        for s_idx in range(n_samples):
            temp = temps[s_idx] if s_idx < len(temps) else temps[-1]
            futures.append(inner_pool.submit(
                _sample_one_temp, client, model, prompt, gold,
                temp, max_tokens, enable_think
            ))
        samples = [f.result() for f in futures]  # 保持温度顺序

    correct_count = sum(1 for s in samples if s["correct"])
    any_correct = correct_count > 0
    all_correct = correct_count == n_samples
    consistency = correct_count / n_samples

    # 答案字符串一致性
    correct_preds = [s["predicted"] for s in samples if s["correct"] and s["predicted"]]
    normalized_correct = [normalize_pred(p) for p in correct_preds]
    answers_consistent = len(set(normalized_correct)) <= 1
    if any_correct and not answers_consistent:
        all_correct = False

    # Majority agreement
    all_preds = [s["predicted"] for s in samples if s["predicted"]]
    normalized_all = [normalize_pred(p) for p in all_preds]
    if normalized_all:
        cnt = Counter(normalized_all)
        major_pred, major_count = cnt.most_common(1)[0]
        agreement_ratio = round(major_count / n_samples, 4)
    else:
        major_pred = None
        agreement_ratio = 0.0

    if any_correct and agreement_ratio < 0.67:
        all_correct = False

    # Score-based CoT 选择
    best = None
    best_score = -999
    for s in samples:
        if not s["correct"]:
            continue
        sc = score_cot(s)
        if sc > best_score:
            best_score = sc
            best = s

    return {
        "samples": samples,
        "correct_count": correct_count,
        "any_correct": any_correct,
        "all_correct": all_correct,
        "consistency": round(consistency, 4),
        "agreement_ratio": agreement_ratio,
        "answers_consistent": answers_consistent,
        "best": best,
    }


def run_sampling_phase(tier_key, args, client, rows, only_ids=None):
    """对指定模型层级执行多次采样，生成 cot_{tier_key}.jsonl。

    并发策略 (两层):
      - 外层: --workers 道题同时处理 (默认 4)
      - 内层: 每题 N 个温度的采样并发执行
      理论: 同时在飞 workers × n_samples 个 API 请求
      例: --workers 4 + T0 3次采样 = 12 个并发请求

    核心流程 (对每一题):
      1. 按 temperatures 列表并发采样 N 次
      2. 提取每次的 \\boxed{} 答案，与 gold 比对
      3. 统计 correct_count, answers_consistent, agreement_ratio
      4. 用 score_cot() 选出最佳正确样本
      5. 写入 JSONL (支持 --resume 断点续跑)
    """
    tier = MODEL_TIERS[tier_key]
    model = tier["model"]
    n_samples = tier["samples_per_question"]
    temps = tier["temperatures"]
    max_tokens = tier["max_tokens"]
    enable_think = tier["enable_thinking"]
    workers = getattr(args, 'workers', 4)

    output_path = os.path.abspath(os.path.join(DATA_DIR, f"cot_{tier_key}.jsonl"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resume
    done_ids = set()
    if args.resume:
        done_ids = load_done_ids(output_path)
        print(f"Resume: {len(done_ids)} IDs already done")

    # 过滤范围
    if only_ids is not None:
        rows = [r for r in rows if r["id"] in only_ids]
    rows = rows[args.start:]
    if args.limit > 0:
        rows = rows[:args.limit]

    # 跳过已完成的
    rows = [r for r in rows if r["id"] not in done_ids]

    total = len(rows)
    correct_any = 0
    correct_all = 0
    processed = 0
    type_stats = defaultdict(lambda: {"total": 0, "correct_any": 0, "correct_all": 0})
    write_lock = threading.Lock()

    print(f"\n{'='*80}")
    print(f"Phase: {tier_key.upper()} — {tier['desc']}")
    print(f"Model:   {model}")
    print(f"Samples: {n_samples} per question, temps={temps}")
    print(f"Tokens:  {max_tokens}, thinking={enable_think}")
    print(f"Workers: {workers} (concurrent questions)")
    print(f"Range:   {total} questions")
    print(f"Output:  {output_path}")
    print(f"{'='*80}\n")

    open_mode = "a" if args.resume else "w"
    with open(output_path, open_mode) as out_f:

        def handle_row(row):
            """处理单题并写入结果。在 worker 线程中执行。"""
            sid = row["id"]
            prompt = row["prompt"]
            gold = row["answer"]
            ptype = detect_type(prompt)

            result = _process_one_question(
                client, model, prompt, gold, temps, n_samples, max_tokens, enable_think
            )

            best = result["best"]
            record = {
                "id": sid,
                "type": ptype,
                "prompt": prompt,
                "gold": gold,
                "tier": tier_key,
                "model": model,
                "n_samples": n_samples,
                "correct_count": result["correct_count"],
                "consistency": result["consistency"],
                "agreement_ratio": result["agreement_ratio"],
                "answers_consistent": result["answers_consistent"],
                "samples": result["samples"],
                "best_thinking": best["thinking"] if best else "",
                "best_response": best["response"] if best else "",
                "best_predicted": best["predicted"] if best else None,
            }

            return ptype, gold, result, record

        # 题目间并发: workers 道题同时处理
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(handle_row, row): row for row in rows}

            for future in as_completed(future_map):
                try:
                    ptype, gold, result, record = future.result()
                except Exception as exc:
                    row = future_map[future]
                    print(f"❌ ERROR processing {row['id']}: {exc}")
                    continue

                # 加锁: 更新统计 + 写文件 + 打印 (保证一致性)
                with write_lock:
                    processed += 1

                    if result["any_correct"]:
                        correct_any += 1
                    if result["all_correct"]:
                        correct_all += 1

                    type_stats[ptype]["total"] += 1
                    if result["any_correct"]:
                        type_stats[ptype]["correct_any"] += 1
                    if result["all_correct"]:
                        type_stats[ptype]["correct_all"] += 1

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()

                    acc_any = correct_any / processed
                    acc_all = correct_all / processed
                    icon = "🟢" if result["all_correct"] else ("🟡" if result["any_correct"] else "🔴")
                    print(f"{icon} [{processed}/{total}] {ptype:10s} "
                          f"consist={result['correct_count']}/{n_samples} "
                          f"agree={result['agreement_ratio']:.2f} "
                          f"gold={gold:20s} "
                          f"acc_any={acc_any:.3f} acc_all={acc_all:.3f}")

                    # 每 100 条汇总
                    if processed % 100 == 0:
                        print(f"\n{'='*80}")
                        print(f"📊 {tier_key.upper()} CHECKPOINT: {processed}/{total}")
                        print(f"   any_correct: {acc_any:.4f} ({correct_any}/{processed})")
                        print(f"   all_correct: {acc_all:.4f} ({correct_all}/{processed})")
                        for t in sorted(type_stats):
                            s = type_stats[t]
                            a_any = s["correct_any"] / s["total"] if s["total"] else 0
                            a_all = s["correct_all"] / s["total"] if s["total"] else 0
                            print(f"   {t:12s}: any={a_any:.3f} all={a_all:.3f} ({s['total']})")
                        print(f"{'='*80}\n")

    # 最终汇总
    print(f"\n{'='*80}")
    print(f"🏁 {tier_key.upper()} FINAL — {model}")
    print(f"   Processed: {processed}")
    print(f"   any_correct: {correct_any}/{processed} = {correct_any/max(processed,1):.4f}")
    print(f"   all_correct: {correct_all}/{processed} = {correct_all/max(processed,1):.4f}")
    for t in sorted(type_stats):
        s = type_stats[t]
        a_any = s["correct_any"] / s["total"] if s["total"] else 0
        a_all = s["correct_all"] / s["total"] if s["total"] else 0
        print(f"   {t:12s}: any={a_any:.3f} all={a_all:.3f} ({s['total']})")
    print(f"   Output: {output_path}")
    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE: T1 — 只跑 T0 没全对的难题
#  T1 的输入 = T0 中 correct_count < n_samples 的题目 ID
#  目的: 用更大模型为这些题提供 teacher CoT
# ═══════════════════════════════════════════════════════════════════════════════
def get_failed_ids(tier_key):
    """加载某 tier 没有全部答对的 sample IDs (correct_count < n_samples)。
    这些是"基座不稳定"的题，需要 T1 teacher 来补。
    """
    path = os.path.join(DATA_DIR, f"cot_{tier_key}.jsonl")
    failed = set()
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return failed
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec["correct_count"] < rec["n_samples"]:
                    failed.add(rec["id"])
            except:
                pass
    return failed


def get_zero_correct_ids(tier_key):
    """加载某 tier 一次都没答对的 sample IDs (correct_count == 0)。
    这些是"连碰对都没碰对"的超难题。
    """
    path = os.path.join(DATA_DIR, f"cot_{tier_key}.jsonl")
    zero = set()
    if not os.path.exists(path):
        return zero
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec["correct_count"] == 0:
                    zero.add(rec["id"])
            except:
                pass
    return zero


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE: MERGE — 合并所有 tier 并分级
#  输入: data/cot_t0.jsonl + data/cot_t1.jsonl
#  输出: data/merged_cot.jsonl (全量) + data/sft_cot_data.jsonl (经数据量控制)
# ═══════════════════════════════════════════════════════════════════════════════
def run_merge(gold_only=False):
    """合并 t0/t1 数据，按一致性分级，控制每类数据量。

    分级逻辑 (从高到低):
      gold    ← T0 ≥ 2/3 + 答案一致 + agreement ≥ 0.67
      silver  ← T0 = 1/3，或 T0 ≥ 2/3 但一致性不达标
      silver+ ← T1 ≥ 2/2 (teacher CoT)
      discard ← 其余 (不写入)

    数据量控制:
      每类 80~150 条，gold:silver:silver+ = 5:3:2
      溢出级别的配额自动转给其他级别

    Args:
      gold_only: 若为 True，仅保留 gold 级别数据 (最高质量, 对齐 E1 策略)
    """

    # 加载所有 tier 数据
    all_data = {}  # id -> {t0: record, t1: record}
    for tier_key in ["t0", "t1"]:
        path = os.path.join(DATA_DIR, f"cot_{tier_key}.jsonl")
        if not os.path.exists(path):
            print(f"Skipping {tier_key}: {path} not found")
            continue
        count = 0
        with open(path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    sid = rec["id"]
                    if sid not in all_data:
                        all_data[sid] = {}
                    all_data[sid][tier_key] = rec
                    count += 1
                except:
                    pass
        print(f"Loaded {count} records from {tier_key}")

    # 分级
    gold_data = []      # T0 ≥ 2/3 correct + 答案一致
    silver_data = []    # T0 = 1/3 correct, 或 T0 ≥ 2/3 但答案不一致
    silver_plus = []    # T1 ≥ 2/2 correct (teacher CoT, 高门槛)

    for sid, tiers in all_data.items():
        t0 = tiers.get("t0")
        t1 = tiers.get("t1")

        # 基础信息从任意 tier 提取
        ref = t0 or t1
        prompt = ref["prompt"]
        gold = ref["gold"]
        ptype = ref["type"]

        # T0 一致性
        t0_correct = t0["correct_count"] if t0 else 0
        t0_total = t0["n_samples"] if t0 else 0
        t0_ans_consistent = t0.get("answers_consistent", True) if t0 else True
        t0_agreement = t0.get("agreement_ratio", 0.0) if t0 else 0.0

        # 选最佳 CoT: 优先 T0 的 score-based best，但有两个例外:
        #   1. T0 没有正确样本 → 用 T1 的 best
        #   2. T0 = 1/3 且 T1 ≥ 2/2 → T1 更可靠，用 T1 的 CoT
        t0_has_best = t0 and t0.get("best_predicted") is not None
        t1_has_best = t1 and t1.get("best_predicted") is not None
        t1_reliable = t1 and t1.get("correct_count", 0) >= 2  # T1 ≥ 2/2

        best_thinking = ""
        best_response = ""
        best_tier = None

        if t0_has_best and not (t0_correct <= 1 and t1_reliable and t1_has_best):
            # T0 有正确样本，且不满足"T0 弱 + T1 强"的替换条件
            best_thinking = t0["best_thinking"]
            best_response = t0["best_response"]
            best_tier = "t0"
        elif t1_has_best:
            # T0 没做对，或 T0=1/3 但 T1 更可靠 → 用 T1 的 CoT
            best_thinking = t1["best_thinking"]
            best_response = t1["best_response"]
            best_tier = "t1"
        elif t0_has_best:
            # fallback: T0 有但 T1 没有
            best_thinking = t0["best_thinking"]
            best_response = t0["best_response"]
            best_tier = "t0"

        if best_tier is None:
            continue  # 所有模型都没做对，丢弃

        entry = {
            "id": sid,
            "type": ptype,
            "prompt": prompt,
            "gold": gold,
            "best_thinking": best_thinking,
            "best_response": best_response,
            "best_tier": best_tier,
            "t0_correct": t0_correct,
            "t0_total": t0_total,
        }

        if t0_correct >= 2 and t0_total >= 3 and t0_ans_consistent and t0_agreement >= 0.67:
            # T0 ≥ 2/3 且答案一致 且 majority agreement 高: 最高质量
            entry["quality"] = "gold"
            gold_data.append(entry)
        elif t0_correct >= 2 and t0_total >= 3:
            # T0 ≥ 2/3 但答案不一致 或 agreement 低: 降级为 silver
            entry["quality"] = "silver"
            silver_data.append(entry)
        elif t0_correct == 1 and t0_total >= 3:
            # T0 = 1/3: 基座偶尔做对
            # 但如果 T1 ≥ 2/2 且 CoT 来自 T1 → 升级为 silver+ (teacher 更可靠)
            if t1_reliable and best_tier == "t1":
                entry["quality"] = "silver+"
                silver_plus.append(entry)
            else:
                entry["quality"] = "silver"
                silver_data.append(entry)
        elif t1 and t1["correct_count"] >= 2:
            # T1 ≥ 2/2: teacher 稳定做对才收录
            entry["quality"] = "silver+"
            silver_plus.append(entry)
        # else: 丢弃 (T1 只对 1/2 = 不可靠, 直接扔掉)

    # Gold-only 模式: 仅保留最高质量数据，与 E1 (0.68) 策略对齐
    # 历史教训: CoT 训练=0.63 < answer-only=0.68, silver+ 可能引入分布偏移
    if gold_only:
        silver_data = []
        silver_plus = []
        print("\n⚡ GOLD-ONLY mode: discarding silver/silver+ data")

    all_entries = gold_data + silver_data + silver_plus

    # 输出完整合并数据
    merged_path = os.path.join(DATA_DIR, "merged_cot.jsonl")
    with open(merged_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 统计
    type_stats = defaultdict(lambda: {"gold": 0, "silver": 0, "silver+": 0, "total": 0})
    for entry in all_entries:
        t = entry["type"]
        q = entry["quality"]
        type_stats[t][q] += 1
        type_stats[t]["total"] += 1

    print(f"\n{'='*80}")
    print(f"🏁 MERGE RESULTS (before volume control)")
    print(f"   Gold:    {len(gold_data):5d} (T0 ≥ 2/3 + consistent)")
    print(f"   Silver:  {len(silver_data):5d} (T0 = 1/3 or inconsistent)")
    print(f"   Silver+: {len(silver_plus):5d} (T1 ≥ 2/2)")
    print(f"   Total:   {len(all_entries):5d}")
    print(f"\n   By type (raw):")
    for t in sorted(type_stats):
        s = type_stats[t]
        print(f"   {t:12s}: gold={s['gold']:4d}  silver={s['silver']:4d}  silver+={s['silver+']:4d}  total={s['total']:4d}")
    print(f"\n   Output: {merged_path}")
    print(f"{'='*80}")

    # ═══════════════ 每类数据量控制: 500 ~ 1000, 难度混合 ═══════════════
    # 目标: 每类题 PER_TYPE_MIN ~ PER_TYPE_MAX 条
    # 比例: gold(最可靠) : silver(偶尔对) : silver+(teacher) = 5 : 3 : 2
    # 为什么要混合? 纯 gold 太简单, 模型学不到难题; 纯 silver+ 噪声太大
    # 5:3:2 是经验值, 保证主体是高质量数据, 同时覆盖一些中等难度
    RATIO = {"gold": 5, "silver": 3, "silver+": 2}
    ratio_total = sum(RATIO.values())

    type_buckets = defaultdict(lambda: {"gold": [], "silver": [], "silver+": []})
    for entry in all_entries:
        type_buckets[entry["type"]][entry["quality"]].append(entry)

    sft_entries = []
    print(f"\n📦 Volume control: {PER_TYPE_MIN} ~ {PER_TYPE_MAX} per type")
    print(f"   Difficulty mix: gold:silver:silver+ = {RATIO['gold']}:{RATIO['silver']}:{RATIO['silver+']}")
    for ptype in sorted(type_buckets):
        bucket = type_buckets[ptype]
        available = {q: len(bucket[q]) for q in RATIO}
        total_available = sum(available.values())

        # 目标总量: 在 MIN~MAX 之间，取 min(available, MAX)
        target_total = min(total_available, PER_TYPE_MAX)

        # 按比例分配各级别的配额
        selected = []
        for quality in ["gold", "silver", "silver+"]:
            quota = int(target_total * RATIO[quality] / ratio_total)
            # 如果本级不够，只取已有的 (剩余配额给其他级别)
            take = min(quota, available[quality])
            selected.extend(bucket[quality][:take])

        # 配额剩余: 如果某级别不够用，把剩余配额按优先级填给其他级别
        remaining = target_total - len(selected)
        if remaining > 0:
            already_ids = {e["id"] for e in selected}
            for quality in ["gold", "silver", "silver+"]:
                for e in bucket[quality]:
                    if remaining <= 0:
                        break
                    if e["id"] not in already_ids:
                        selected.append(e)
                        already_ids.add(e["id"])
                        remaining -= 1

        count = len(selected)
        tag = "✅" if count >= PER_TYPE_MIN else "⚠️"
        g = len([e for e in selected if e["quality"] == "gold"])
        s = len([e for e in selected if e["quality"] == "silver"])
        sp = len([e for e in selected if e["quality"] == "silver+"])
        print(f"   {tag} {ptype:12s}: {count:4d} selected (gold={g}, silver={s}, silver+={sp})")
        sft_entries.extend(selected)

    # 输出 SFT 训练集
    sft_path = os.path.join(DATA_DIR, "sft_cot_data.jsonl")
    with open(sft_path, "w") as f:
        for entry in sft_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n   SFT total: {len(sft_entries)} → {sft_path}")
    print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — 命令行入口
#  执行顺序: t0 (全量基座采样) → t1 (补难题) → merge (分级+输出)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Multi-model multi-temperature CoT sampling")
    parser.add_argument("--phase", required=True, choices=["t0", "t1", "merge"],
                        help="t0=基座采样, t1=中等模型补难题, merge=合并分级")
    parser.add_argument("--start", type=int, default=0,
                        help="从第 N 行开始 (0-indexed, 用于分段跑)")
    parser.add_argument("--limit", type=int, default=0,
                        help="最多处理 N 条 (0=全部, 用于测试)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑: 跳过已有结果的 ID")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发题目数 (默认 4, 总并发=workers×samples_per_question)")
    parser.add_argument("--gold-only", action="store_true",
                        help="merge 阶段仅保留 gold 级别数据 (最高质量, 对齐 E1)")
    args = parser.parse_args()

    if args.phase == "merge":
        run_merge(gold_only=args.gold_only)
        return

    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        return

    client = OpenAI(base_url=API_BASE, api_key=api_key)
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from train.csv")

    if args.phase == "t0":
        # T0: 基座模型跑全量
        run_sampling_phase("t0", args, client, rows)
    elif args.phase == "t1":
        # T1: 中等模型只跑 T0 没全对的
        failed = get_failed_ids("t0")
        if not failed:
            print("No failed IDs from T0 — run T0 first!")
            return
        print(f"T1 targets: {len(failed)} questions that T0 didn't consistently solve")
        run_sampling_phase("t1", args, client, rows, only_ids=failed)


if __name__ == "__main__":
    main()
