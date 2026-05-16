#!/usr/bin/env python3
"""Generate the Colab training hub notebook."""
import json, os

def cell(cell_type, source, **kw):
    c = {
        "cell_type": cell_type,
        "metadata": kw.get("metadata", {}),
        "source": source.split("\n") if isinstance(source, str) else source,
    }
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

cells = []

# ========== Cell 1: Title ==========
cells.append(cell("markdown", """\
# 🚀 Nemotron Colab Training Hub

在 Colab 上训练 LoRA adapter 并快速评测，对比多种方法。

**支持的方法 (改 `METHOD` 即可切换)**:
| Method | 描述 | 对应历史实验 |
|--------|------|-------------|
| `v2_answer_only` | 随机600, answer-only, thinking=True, 全文loss | V2=0.68 |
| `stratified_600` | 分层100/类型, answer-only | v33=0.58 |
| `boxed_only` | 随机600, loss仅在\\\\boxed{}上 | v29=0.66 |
| `typed_cot` | 按题型程序化CoT (reasoning_content) | v36 |
| `compact_rules` | ultra-compact规则SFT | v38 stage2 |
| `verified_7741` | 7741验证样本大数据量 | v32=0.62 |
| `thinking_off` | 随机600, thinking=False | baseline=0.66 |

**GPU 要求**: Nemotron-30B 需要 A100 80GB。设 `USE_SMALL_MODEL=True` 可用小模型先验流程。"""))

# ========== Cell 2: Install ==========
cells.append(cell("code", """\
%%capture
!pip install -U datasets trl peft accelerate kagglehub sentencepiece protobuf
# mamba_ssm / causal_conv1d — Nemotron 模型需要; 如果安装失败会自动 fallback
!pip install mamba_ssm causal_conv1d 2>/dev/null || echo "⚠️ mamba_ssm install failed, will use mock fallback"
"""))

# ========== Cell 3: Config ==========
cells.append(cell("code", """\
# =====================================================================
#  ⚙️  实验配置 — 只改这个 cell 就能切换方法
# =====================================================================

METHOD = "v2_answer_only"   # 👈 改这里切换方法
# 可选: v2_answer_only | stratified_600 | boxed_only | typed_cot
#       compact_rules | verified_7741 | thinking_off

USE_SMALL_MODEL = False     # True = 用 Qwen2.5-0.5B 跑通流程; False = Nemotron-30B
EVAL_SAMPLES   = 200        # 训练后评测样本数 (从训练集外采样)
LORA_RANK      = 32
MAX_SEQ_LEN    = 1024
GRAD_ACCUM     = 4

# ---- 方法预设 (一般不需要改) ----
METHOD_PRESETS = {
    "v2_answer_only": dict(
        data="train.csv", n=600, sample="random", suffix="short",
        loss="full", thinking=True, lr=2e-4, epochs=1, cot_field=None,
    ),
    "stratified_600": dict(
        data="train.csv", n=600, sample="stratified", suffix="official",
        loss="full", thinking=True, lr=2e-4, epochs=1, cot_field=None,
    ),
    "boxed_only": dict(
        data="train.csv", n=600, sample="random", suffix="official",
        loss="boxed_only", thinking=True, lr=2e-4, epochs=1, cot_field=None,
    ),
    "typed_cot": dict(
        data="sft_typed_cot_600.csv", n=0, sample="all", suffix="short",
        loss="full", thinking=True, lr=2e-4, epochs=1, cot_field="thinking",
    ),
    "compact_rules": dict(
        data="sft_compact_rules.csv", n=600, sample="random", suffix="short",
        loss="full", thinking=True, lr=2e-4, epochs=1, cot_field="thinking",
    ),
    "verified_7741": dict(
        data="sft_ao_7741.csv", n=0, sample="all", suffix="official",
        loss="full", thinking=True, lr=1e-4, epochs=1, cot_field=None,
    ),
    "thinking_off": dict(
        data="train.csv", n=600, sample="random", suffix="official",
        loss="full", thinking=False, lr=2e-4, epochs=1, cot_field=None,
    ),
}

CFG = METHOD_PRESETS[METHOD]
print(f"✅ Method = {METHOD}")
for k, v in CFG.items():
    print(f"   {k}: {v}")
"""))

# ========== Cell 4: Mount Drive + Data ==========
cells.append(cell("code", """\
import os

# --- Google Drive ---
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

# 👉 把项目文件夹路径改成你自己的 Drive 路径
DRIVE_PROJECT = "/content/drive/MyDrive/nemotron_competition"

# 如果 Drive 里没数据，也可以直接上传到 /content
DATA_DIR = os.path.join(DRIVE_PROJECT, "data") if USE_DRIVE else "/content/data"
TRAIN_CSV = os.path.join(DRIVE_PROJECT, "competition_data", "train.csv") if USE_DRIVE else "/content/train.csv"

OUTPUT_DIR = "/content/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kaggle credentials (用于下载模型)
# 方法1: 在 Colab 的 Secrets 里设置 KAGGLE_USERNAME 和 KAGGLE_KEY
# 方法2: 直接在这里填
# os.environ["KAGGLE_USERNAME"] = "your_username"
# os.environ["KAGGLE_KEY"] = "your_key"

print(f"DATA_DIR   = {DATA_DIR}")
print(f"TRAIN_CSV  = {TRAIN_CSV}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")

# 验证数据存在
for f in [TRAIN_CSV]:
    if os.path.exists(f):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} NOT FOUND — 请上传数据!")
"""))

# ========== Cell 5: Imports + Utilities ==========
cells.append(cell("code", """\
import os, sys, gc, re, json, types, stat, shutil, zipfile, random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- Constants ----
SHORT_SUFFIX = "\\nPut your final answer inside \\\\boxed{}."
OFFICIAL_SUFFIX = "\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`"

BOX_RE = re.compile(r"\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}")
BOX_OPEN_RE = re.compile(r"\\\\boxed\\{([^}]*)$")


def infer_type(prompt: str) -> str:
    p = prompt.lower()
    if "binary" in p or "bit" in p: return "bit_ops"
    if "cipher" in p or "encrypt" in p or "decrypt" in p: return "cipher"
    if "gravity" in p or "fall" in p or "m/s" in p: return "gravity"
    if "roman" in p or "numeral" in p: return "numeral"
    if "convert" in p or "unit" in p: return "unit_conv"
    if "symbol" in p or "equation" in p: return "symbol"
    return "unknown"


def extract_boxed(text: str) -> str:
    if not text:
        return "NOT_FOUND"
    matches = BOX_RE.findall(text)
    if matches:
        val = matches[-1].strip()
        return val if val else "NOT_FOUND"
    m = BOX_OPEN_RE.search(text.strip())
    if m:
        val = m.group(1).strip()
        return val if val else "NOT_FOUND"
    return "NOT_FOUND"


def is_correct(pred: str, gold: str, tol: float = 1e-2) -> bool:
    p, g = str(pred).strip(), str(gold).strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) <= tol
    except Exception:
        return False


print("✅ Utilities loaded")
"""))

# ========== Cell 6: Download Model ==========
cells.append(cell("code", """\
import kagglehub

if USE_SMALL_MODEL:
    MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"🔬 Using small model for flow validation: {MODEL_PATH}")
else:
    MODEL_PATH = kagglehub.model_download(
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    )
    print(f"✅ Nemotron model downloaded to: {MODEL_PATH}")
"""))

# ========== Cell 7: Prepare Training Data ==========
cells.append(cell("code", """\
# ---- Load & sample ----
data_path = os.path.join(DATA_DIR, CFG["data"]) if CFG["data"] != "train.csv" else TRAIN_CSV
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows from {CFG['data']}")

if CFG["sample"] == "random" and CFG["n"] > 0:
    df = df.sample(n=min(CFG["n"], len(df)), random_state=SEED)
elif CFG["sample"] == "stratified" and CFG["n"] > 0:
    df["_type"] = df["prompt"].apply(infer_type)
    per_type = CFG["n"] // df["_type"].nunique()
    df = (
        df.groupby("_type", group_keys=False)
        .apply(lambda x: x.sample(n=min(per_type, len(x)), random_state=SEED))
        .reset_index(drop=True)
    )
    df = df.drop(columns=["_type"])

print(f"Training samples: {len(df)}")

# ---- Hold out eval set ----
all_train = pd.read_csv(TRAIN_CSV)
train_ids = set(df["id"].tolist())
eval_pool = all_train[~all_train["id"].isin(train_ids)]
eval_df = eval_pool.sample(n=min(EVAL_SAMPLES, len(eval_pool)), random_state=SEED + 1)
print(f"Eval samples: {len(eval_df)} (held out from training)")

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

suffix = SHORT_SUFFIX if CFG["suffix"] == "short" else OFFICIAL_SUFFIX
cot_field = CFG["cot_field"]
enable_thinking = CFG["thinking"]


def build_text(row):
    user_msg = str(row["prompt"]) + suffix
    answer = str(row["answer"])
    thinking = ""
    if cot_field and cot_field in row.index:
        thinking = str(row[cot_field]) if pd.notna(row[cot_field]) else ""

    if thinking.strip():
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"\\\\boxed{{{answer}}}",
             "reasoning_content": thinking},
        ]
    else:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"\\\\boxed{{{answer}}}"},
        ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    except Exception:
        think_block = f"<think>\\n{thinking}\\n</think>\\n" if thinking.strip() else "<think>\\n</think>\\n"
        text = (
            f"<|im_start|>user\\n{user_msg}<|im_end|>\\n"
            f"<|im_start|>assistant\\n{think_block}\\\\boxed{{{answer}}}<|im_end|>"
        )
    return text


# ---- Build HF Dataset ----
if CFG["loss"] == "boxed_only":
    THINK_CLOSE_ID = tokenizer.convert_tokens_to_ids("</think>")
    print(f"</think> token ID: {THINK_CLOSE_ID}")

    records = []
    for _, row in df.iterrows():
        text = build_text(row)
        ids = tokenizer.encode(text, add_special_tokens=False,
                               truncation=True, max_length=MAX_SEQ_LEN)
        prefix_len = len(ids)
        for i, tid in enumerate(ids):
            if tid == THINK_CLOSE_ID:
                prefix_len = i + 1
                break
        labels = [-100] * prefix_len + ids[prefix_len:]
        records.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
        })
    hf_dataset = Dataset.from_dict({
        "input_ids": [r["input_ids"] for r in records],
        "attention_mask": [r["attention_mask"] for r in records],
        "labels": [r["labels"] for r in records],
    })
    n_loss = sum(1 for x in hf_dataset[0]["labels"] if x != -100)
    print(f"Boxed-only: {n_loss} loss tokens in example 0")
    print(f"Loss tokens: {tokenizer.decode([t for t in hf_dataset[0]['labels'] if t != -100])}")
    SKIP_PREPARE = True
else:
    df["text"] = df.apply(build_text, axis=1)
    hf_dataset = Dataset.from_pandas(df[["text"]])
    print(f"Full-text loss. Example (first 300 chars):")
    print(hf_dataset[0]["text"][:300])
    SKIP_PREPARE = False

print(f"\\n✅ Dataset ready: {len(hf_dataset)} examples, loss={CFG['loss']}")
"""))

# ========== Cell 8: Load Model + LoRA ==========
cells.append(cell("code", """\
# ---- Mock mamba3 modules (Nemotron needs this if mamba_ssm is incomplete) ----
if not USE_SMALL_MODEL:
    for _name in [
        "mamba_ssm.ops.cute", "mamba_ssm.ops.cute.mamba3",
        "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
        "mamba_ssm.ops.tilelang", "mamba_ssm.ops.tilelang.mamba3",
        "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo",
    ]:
        if _name not in sys.modules:
            _m = types.ModuleType(_name)
            _m.__path__ = []
            sys.modules[_name] = _m

    # rmsnorm fallback (pure Python, no CUDA kernel needed)
    def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                         group_size=None, norm_before_gate=True, upcast=True):
        dtype = x.dtype
        if upcast: x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps) * weight.float()
        if bias is not None: out = out + bias.float()
        if z is not None: out = out * F.silu(z.float())
        return out.to(dtype)

    for name, mod in list(sys.modules.items()):
        if hasattr(mod, "rmsnorm_fn"):
            mod.rmsnorm_fn = _pure_rmsnorm_fn

# ---- Load Model ----
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=dtype
)
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

# Force slow path (bypass broken CUDA kernels on some setups)
if not USE_SMALL_MODEL:
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name:
            mod.is_fast_path_available = False
            print(f"  Patched {name}: is_fast_path_available = False")
    for name, mod in list(sys.modules.items()):
        if hasattr(mod, "rmsnorm_fn"):
            mod.rmsnorm_fn = _pure_rmsnorm_fn

# ---- LoRA ----
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

gpu_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
print(f"GPU memory after model+LoRA: {gpu_gb:.1f} GB")
"""))

# ========== Cell 9: Train ==========
cells.append(cell("code", """\
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=CFG["epochs"],
    learning_rate=CFG["lr"],
    logging_steps=5,
    bf16=torch.cuda.is_available(),
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)

if SKIP_PREPARE:
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
else:
    training_args.dataset_text_field = "text"
    data_collator = None

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
    **({"data_collator": data_collator} if data_collator else {}),
)

print(f"🏋️ Training: method={METHOD}, samples={len(hf_dataset)}, "
      f"epochs={CFG['epochs']}, lr={CFG['lr']}, loss={CFG['loss']}")
trainer.train()
print("✅ Training complete!")

# Cleanup trainer
del trainer; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""))

# ========== Cell 10: Evaluate ==========
cells.append(cell("code", """\
eval_suffix = OFFICIAL_SUFFIX  # 评测统一用官方 suffix

model.eval()
results = []
print(f"🔍 Evaluating on {len(eval_df)} held-out samples...")

for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
    prompt = str(row["prompt"]) + eval_suffix
    messages = [{"role": "user", "content": prompt}]

    try:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception:
        chat = f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"

    inputs = tokenizer(chat, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=False)
    pred = extract_boxed(text)
    gold = str(row["answer"])
    qtype = infer_type(str(row["prompt"]))

    results.append({
        "id": row["id"], "type": qtype, "gold": gold,
        "pred": pred, "correct": is_correct(pred, gold),
        "raw": text[:500],
    })

res_df = pd.DataFrame(results)
print("✅ Evaluation complete!")
"""))

# ========== Cell 11: Results ==========
cells.append(cell("code", """\
overall_acc = res_df["correct"].mean()
print(f"\\n{'='*50}")
print(f"  METHOD: {METHOD}")
print(f"  Overall Accuracy: {overall_acc:.4f} ({res_df['correct'].sum()}/{len(res_df)})")
print(f"{'='*50}")

# By type
type_acc = (
    res_df.groupby("type")["correct"]
    .agg(["mean", "sum", "count"])
    .rename(columns={"mean": "acc", "sum": "correct", "count": "total"})
    .sort_values("acc", ascending=False)
)
print("\\nAccuracy by type:")
print(type_acc.to_string())

# Failure analysis
not_found = (res_df["pred"] == "NOT_FOUND").sum()
print(f"\\nNOT_FOUND: {not_found}/{len(res_df)} ({not_found/len(res_df)*100:.1f}%)")

# Show some failures
failures = res_df[~res_df["correct"]].head(5)
if len(failures) > 0:
    print("\\nSample failures:")
    for _, r in failures.iterrows():
        print(f"  [{r['type']}] gold={r['gold']}, pred={r['pred']}")
"""))

# ========== Cell 12: Save ==========
cells.append(cell("code", """\
# ---- Save adapter ----
model.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size/1024:.1f} KB)")

# ---- Save results ----
res_path = os.path.join(OUTPUT_DIR, f"eval_{METHOD}.csv")
res_df.to_csv(res_path, index=False)
print(f"\\nResults saved to {res_path}")

# ---- Package submission.zip (optional) ----
zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in ["adapter_model.safetensors", "adapter_config.json"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)

print(f"\\n📦 submission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
with zipfile.ZipFile(zip_path, "r") as zf:
    print(f"Contents: {zf.namelist()}")

# ---- Copy to Drive for persistence ----
if USE_DRIVE:
    drive_out = os.path.join(DRIVE_PROJECT, "adapters", METHOD)
    os.makedirs(drive_out, exist_ok=True)
    for fname in os.listdir(OUTPUT_DIR):
        src = os.path.join(OUTPUT_DIR, fname)
        dst = os.path.join(drive_out, fname)
        shutil.copy2(src, dst)
    print(f"✅ Copied to Drive: {drive_out}")
"""))

# ========== Cell 13: Notes ==========
cells.append(cell("markdown", """\
## 📝 使用说明

### 快速开始
1. **上传数据到 Drive**: 把 `competition_data/train.csv` 和 `data/` 文件夹上传到 `DRIVE_PROJECT` 路径
2. **设置 Kaggle 凭证**: 在 Colab Secrets 或代码中设置 `KAGGLE_USERNAME` + `KAGGLE_KEY`
3. **选择方法**: 修改 Cell 3 的 `METHOD` 变量
4. **跑 All Cells**: Runtime → Run all

### 方法对比流程
1. 先用 `USE_SMALL_MODEL=True` 验证流程能跑通
2. 然后 `USE_SMALL_MODEL=False`，依次测试各 METHOD
3. 每次训练后结果自动保存到 Drive，可以对比

### 方法推荐顺序
1. `v2_answer_only` — 已验证最佳 baseline
2. `boxed_only` — boxed-only loss 变体
3. `typed_cot` — typed CoT 实验
4. `compact_rules` — ultra-compact 规则
5. 对比以上结果后再决定是否尝试 `verified_7741` 等

### 注意事项
- Nemotron 30B 在 BF16 需要 ~60GB 显存，确保使用 **A100 80GB** 运行时
- 每次切换 METHOD 后需要 **重启运行时** (Runtime → Restart) 重新加载模型
- `eval_df` 自动从训练数据中排除，保证评测公平性
- 评测用 `transformers.generate()` 而非 vLLM，结果与 Kaggle 评测会有差异
"""))

# ========== Assemble notebook ==========
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "gpuClass": "premium",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(__file__), "nemotron_colab_training_hub.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ Generated: {out_path}")
print(f"   Cells: {len(cells)}")
