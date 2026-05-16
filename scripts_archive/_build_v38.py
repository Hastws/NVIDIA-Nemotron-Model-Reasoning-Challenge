#!/usr/bin/env python3
"""Build v38 notebook: 3-stage pipeline (boxed-only SFT → CoT SFT → GRPO).
Each stage uses N_SAMPLES (configurable), default 10 for quick validation.
"""
import json, copy

NB_PATH = 'nvidia-nemotron-3stage-v38.ipynb'

def make_cell(cell_type, source, cell_id=None):
    # Create a notebook cell.
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.split('\n') if isinstance(source, str) else source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    if cell_id:
        cell["id"] = cell_id
    # Fix: source lines should have \n except last
    lines = source.split('\n') if isinstance(source, str) else source
    fixed = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            fixed.append(line + '\n' if not line.endswith('\n') else line)
        else:
            fixed.append(line.rstrip('\n'))
    cell["source"] = fixed
    return cell

cells = []

# =============================================================================
# Cell 0: Markdown header
# =============================================================================
cells.append(make_cell("markdown", "## 3-Stage Pipeline: Boxed-Only SFT → CoT SFT → GRPO"))

# =============================================================================
# Cell 1: pip install
# =============================================================================
cells.append(make_cell("code", 
    '!pip install -q --no-index --find-links /kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages datasets trl --ignore-installed'))

# =============================================================================
# Cell 2: version check
# =============================================================================
cells.append(make_cell("code", """import datasets
import trl
print("datasets:", datasets.__version__)
print("trl:", trl.__version__)"""))

# =============================================================================
# Cell 3: Imports + env fixes + hyperparams
# =============================================================================
cells.append(make_cell("code", r"""import os
import sys
import stat
import shutil
import gc
import zipfile
import types
import re
import polars as pl
import torch
import torch.nn.functional as F
import kagglehub
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Block Mamba3 import chain ---
_mock_class = type("_Mock", (), {})
for _name, _attrs in [
    ("mamba_ssm.modules.mamba3", {"Mamba3": _mock_class}),
    ("mamba_ssm.ops.cute", {}),
    ("mamba_ssm.ops.cute.mamba3", {}),
    ("mamba_ssm.ops.cute.mamba3.mamba3_step_fn", {"mamba3_step_fn": lambda *a, **kw: None}),
]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = []
        for _k, _v in _attrs.items():
            setattr(_m, _k, _v)
        sys.modules[_name] = _m

# --- Triton rmsnorm fallback ---
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)

for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn

src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
dst = "/tmp/ptxas-blackwell"
if os.path.exists(src):
    shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    import triton.backends.nvidia as nv_backend
    src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
    dst_bin = "/tmp/triton_nvidia_bin"
    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
    for f in os.listdir(dst_bin):
        fp = os.path.join(dst_bin, f)
        if os.path.isfile(fp):
            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
    os.environ["TRITON_PTXAS_PATH"] = dst

# ===================== HYPERPARAMETERS =====================
# Stage 1: Boxed-only SFT (answer-only, loss only on \boxed{})
S1_SAMPLES = 600         # answer-only samples for boxed-only loss
S1_EPOCHS = 1
S1_LR = 2e-4

# Stage 2: Ultra-compact rule SFT (7-32 char rules, full-text loss)
S2_SAMPLES = 600         # compact rule samples (from 7945 pool)
S2_EPOCHS = 1
S2_LR = 5e-5            # lower LR to preserve Stage 1 gains

# Stage 3: DISABLED (no GRPO, pure SFT test)
# S3_SAMPLES = 0

# Shared
LORA_RANK = 32
MAX_SEQ_LEN = 1024
GRAD_ACCUM = 4

OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("✓ Hyperparameters set")"""))

# =============================================================================
# Cell 4: Markdown - Stage 1
# =============================================================================
cells.append(make_cell("markdown", """## Stage 1: Boxed-Only Loss SFT
Answer-only training with loss ONLY on `\\boxed{answer}` tokens.
This teaches format without disturbing the model's thinking ability."""))

# =============================================================================
# Cell 5: Stage 1 - Data loading + boxed-only loss
# =============================================================================
cells.append(make_cell("code", r"""MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

# </think> special token ID — used to find where boxed answer starts
THINK_CLOSE_ID = tokenizer.convert_tokens_to_ids("</think>")
print(f"</think> token ID: {THINK_CLOSE_ID}")

# Load answer-only data (random 600 from train.csv, same as V2/E1)
train_df = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')
train_df = train_df.sample(n=min(S1_SAMPLES, len(train_df)), seed=42)
print(f"Stage 1 data: {len(train_df)} samples")

hf_s1 = Dataset.from_pandas(train_df.to_pandas())

def build_boxed_only_example(example):
    # Pre-tokenize with labels: mask everything up to and including </think>.
    prompt = example["prompt"]
    answer = example["answer"]
    user_msg = prompt + SUFFIX
    assistant_msg = f"\\boxed{{{answer}}}"
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=True,
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)
    
    # Find </think> token and mask everything up to and including it
    prefix_len = len(input_ids)  # default: mask all (safety)
    for i, tid in enumerate(input_ids):
        if tid == THINK_CLOSE_ID:
            prefix_len = i + 1
            break
    labels = [-100] * prefix_len + input_ids[prefix_len:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }

hf_s1 = hf_s1.map(build_boxed_only_example, remove_columns=hf_s1.column_names)
n_loss = sum(1 for x in hf_s1[0]["labels"] if x != -100)
print(f"Stage 1 ready: {len(hf_s1)} examples, {n_loss} loss tokens in example 0")
print(f"Loss tokens: {tokenizer.decode([t for t in hf_s1[0]['labels'] if t != -100])}")"""))

# =============================================================================
# Cell 6: Model loading + LoRA
# =============================================================================
cells.append(make_cell("code", r"""model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
print(f"Model loaded. Vocab size: {len(tokenizer)}")

for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=16, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()"""))

# =============================================================================
# Cell 7: Stage 1 Training
# =============================================================================
cells.append(make_cell("code", r"""import triton.backends.nvidia.compiler as nv_compiler
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

tokenizer.model_max_length = MAX_SEQ_LEN

s1_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=S1_EPOCHS,
    learning_rate=S1_LR,
    logging_steps=5,
    bf16=True,
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
    dataset_kwargs={"skip_prepare_dataset": True},
)

s1_trainer = SFTTrainer(
    model=model,
    train_dataset=hf_s1,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    args=s1_args,
)

print("=== Stage 1: Boxed-Only SFT ===")
s1_trainer.train()
print("✓ Stage 1 complete")

del s1_trainer
gc.collect()
torch.cuda.empty_cache()
print(f"GPU after Stage 1: {torch.cuda.memory_allocated()/1024**3:.1f} GB")"""))

# =============================================================================
# Cell 8: Markdown - Stage 2
# =============================================================================
cells.append(make_cell("markdown", """## Stage 2: CoT SFT
Rule-generated programmatic CoT. Full-text loss (standard SFT).
Lower LR to preserve Stage 1 format training."""))

# =============================================================================
# Cell 9: Stage 2 - CoT data + training
# =============================================================================
cells.append(make_cell("code", r"""# Load ultra-compact rule data (7-32 char rules per type, 7945 total)
cot_df = pl.read_csv('/kaggle/input/prog-cot-training-data/sft_compact_rules.csv')
cot_df = cot_df.sample(n=min(S2_SAMPLES, len(cot_df)), seed=42)
print(f"Stage 2 data: {len(cot_df)} samples")

hf_s2 = Dataset.from_pandas(cot_df.to_pandas())

SHORT_SUFFIX = "\nPut your final answer inside \\boxed{}."

def build_cot_text(example):
    # Use reasoning_content field to avoid double <think> blocks.
    # enable_thinking=True auto-inserts <think>...</think> around reasoning_content.
    prompt = example["prompt"]
    answer = example["answer"]
    thinking = example.get("thinking", "") or ""
    user_msg = prompt + SHORT_SUFFIX
    
    try:
        if thinking.strip():
            # Use reasoning_content so template puts it inside <think> properly
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"\\boxed{{{answer}}}", "reasoning_content": thinking},
            ]
        else:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
            ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True
        )
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{thinking}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
        )
    return {"text": text}

hf_s2 = hf_s2.map(build_cot_text, remove_columns=hf_s2.column_names)
print(f"Stage 2 ready: {len(hf_s2)} examples")
print(f"Example (first 300 chars):\n{hf_s2[0]['text'][:300]}")

s2_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=S2_EPOCHS,
    learning_rate=S2_LR,
    logging_steps=5,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)

s2_trainer = SFTTrainer(
    model=model,
    train_dataset=hf_s2,
    processing_class=tokenizer,
    args=s2_args,
)

print("=== Stage 2: CoT SFT ===")
s2_trainer.train()
print("✓ Stage 2 complete")

del s2_trainer
gc.collect()
torch.cuda.empty_cache()
print(f"GPU after Stage 2: {torch.cuda.memory_allocated()/1024**3:.1f} GB")"""))

# =============================================================================
# Stage 3 DISABLED for v20 (pure 2-stage SFT test)
# =============================================================================
if False:  # GRPO disabled
    cells.append(make_cell("markdown", """## Stage 3: GRPO Reinforcement Learning
    Explore & exploit: model generates multiple completions per prompt,
    gets rewarded for correct `\\boxed{}` answers."""))

# =============================================================================
# Cell 11: Stage 3 - GRPO setup (DISABLED)
# =============================================================================
if False:  # GRPO disabled for v20
    cells.append(make_cell("code", r"""# --- Mock missing optional deps for TRL ---
import types as _types

def _create_mock_module(name, attrs=None):
    mod = _types.ModuleType(name)
    mod.__path__ = []
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod

_mock_class = type("_Mock", (), {})
_all_mocks = {
    "mergekit": {}, "mergekit.config": {"MergeConfiguration": _mock_class},
    "mergekit.merge": {"MergeOptions": _mock_class, "run_merge": lambda *a, **kw: None},
    "mergekit.architecture": {}, "mergekit.io": {}, "mergekit.io.tasks": {},
    "mergekit.io.lazy_tensor_loader": {}, "mergekit.common": {},
    "mergekit.graph": {}, "mergekit.merge_methods": {},
    "mergekit.options": {}, "mergekit.plan": {}, "mergekit.sparsify": {},
    "llm_blender": {"Blender": _mock_class},
    "weave": {"EvaluationLogger": _mock_class}, "weave.trace": {},
    "weave.trace.context": {"weave_client_context": _mock_class},
    "liger_kernel": {}, "liger_kernel.transformers": {},
}
for pkg_name, attrs in _all_mocks.items():
    if pkg_name not in sys.modules:
        sys.modules[pkg_name] = _create_mock_module(pkg_name, attrs)

sys.modules["weave"].trace = sys.modules["weave.trace"]
sys.modules["weave.trace"].context = sys.modules["weave.trace.context"]
sys.modules["mergekit"].config = sys.modules["mergekit.config"]
sys.modules["mergekit"].merge = sys.modules["mergekit.merge"]
sys.modules["mergekit"].io = sys.modules["mergekit.io"]
sys.modules["mergekit.io"].tasks = sys.modules["mergekit.io.tasks"]
sys.modules["mergekit.io"].lazy_tensor_loader = sys.modules["mergekit.io.lazy_tensor_loader"]
print(f"✓ Mocked {len(_all_mocks)} optional TRL dependencies")

from trl import GRPOTrainer, GRPOConfig

# --- Classify type from prompt (train.csv has no type column) ---
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

_type_keywords = {
    'bit_ops': ['bitwise', 'bit operation', 'bit shift', 'XOR', 'AND, OR, NOT'],
    'gravity': ['gravitational', 'gravity', 'celestial', 'planet', 'gravitational constant'],
    'unit_conv': ['unit conversion', 'convert the following measurement', 'secret unit'],
    'cipher': ['encryption', 'cipher', 'encrypt', 'decrypt', 'encoded', 'secret code'],
    'numeral': ['numeral system', 'Roman numeral', 'ancient numeral', 'number system'],
    'symbol': ['symbol', 'symbolic', 'equation', 'transformation rule', 'symbol manipulation'],
}

def _classify_type(prompt):
    p_lower = prompt.lower()
    for t, kws in _type_keywords.items():
        for kw in kws:
            if kw.lower() in p_lower:
                return t
    return "unknown"

# Load full training data for GRPO (different from SFT data)
grpo_df = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')
grpo_df = grpo_df.with_columns(
    pl.col("prompt").map_elements(_classify_type, return_dtype=pl.Utf8).alias("type")
)
print("Type distribution:")
print(grpo_df.group_by("type").len().sort("type"))

# Strategic sampling
type_quotas = {
    "numeral": 50, "gravity": 150, "unit_conv": 150,
    "cipher": 175, "bit_ops": 150, "symbol": 125,
}
grpo_frames = []
for t, n in type_quotas.items():
    subset = grpo_df.filter(pl.col("type") == t)
    actual_n = min(n, len(subset))
    grpo_frames.append(subset.sample(n=actual_n, seed=42))
grpo_df = pl.concat(grpo_frames).sample(fraction=1.0, seed=42)
grpo_df = grpo_df.head(S3_SAMPLES)

print(f"\nGRPO dataset: {len(grpo_df)} samples")
for t in type_quotas:
    cnt = len(grpo_df.filter(pl.col("type") == t))
    if cnt > 0:
        print(f"  {t}: {cnt}")

def format_grpo_prompt(example):
    user_msg = example["prompt"] + METRIC_SUFFIX
    messages = [{"role": "user", "content": user_msg}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except Exception:
        text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    return {"prompt": text, "answer": example["answer"]}

grpo_dataset = Dataset.from_pandas(grpo_df.to_pandas())
grpo_dataset = grpo_dataset.map(format_grpo_prompt, remove_columns=[c for c in grpo_dataset.column_names if c not in ["prompt", "answer"]])
print(f"✓ GRPO prompts formatted")

# --- Reward function ---
def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'\\boxed\{([^}]*?)$', text)
    if m:
        return m.group(1).strip()
    return None

def answers_match(pred, gold):
    if pred is None:
        return False
    p, g = pred.strip().lower(), gold.strip().lower()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) <= 1e-2
    except (ValueError, OverflowError):
        return False

_debug_count = 0
def reward_func(completions, answer, **kwargs):
    global _debug_count
    rewards = []
    for comp, gold in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        pred = extract_boxed_answer(text)
        if _debug_count < 5:
            print(f"[GRPO #{_debug_count}] pred={pred} | gold={gold} | tail=...{text[-100:]}")
            _debug_count += 1
        if pred is None:
            rewards.append(-1.0)
        elif answers_match(pred, gold):
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards

print("✓ GRPO data and reward function ready")"""))

# =============================================================================
# Cell 12: Stage 3 - GRPO Training (DISABLED)
# =============================================================================
if False:  # GRPO disabled for v20
    cells.append(make_cell("code", r"""if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

grpo_config = GRPOConfig(
    output_dir="/kaggle/working/grpo",
    num_train_epochs=S3_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=S3_LR,
    num_generations=S3_NUM_GEN,
    max_completion_length=S3_MAX_COMPLETION,
    max_prompt_length=S3_MAX_PROMPT,
    temperature=0.7,
    beta=0.0,
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_func],
    args=grpo_config,
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
)

print(f"=== Stage 3: GRPO ===")
print(f"  Samples: {len(grpo_dataset)}, Gens/prompt: {S3_NUM_GEN}")
print(f"  Max completion: {S3_MAX_COMPLETION}, LR: {S3_LR}")
grpo_trainer.train()
print("✓ Stage 3 GRPO complete")"""))

# =============================================================================
# Cell 13: Markdown - Save
# =============================================================================
cells.append(make_cell("markdown", "## Save and Package Submission"))

# =============================================================================
# Cell 14: Save + zip
# =============================================================================
cells.append(make_cell("code", r"""model.save_pretrained(OUTPUT_DIR)
print(f"Adapter files saved to {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size/1024:.1f} KB)")

zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        zf.write(fpath, fname)

print(f"\nCreated {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")

with zipfile.ZipFile(zip_path, 'r') as zf:
    names = zf.namelist()
    print(f"Contents: {names}")
    assert "adapter_config.json" in names, "Missing adapter_config.json!"
    print("✓ submission.zip ready!")"""))

# =============================================================================
# Build notebook
# =============================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells,
}

with open(NB_PATH, 'w') as f:
    json.dump(notebook, f, indent=1)

# Verify
with open(NB_PATH) as f:
    nb = json.load(f)
print(f"✓ Created {NB_PATH}: {len(nb['cells'])} cells")
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])
    first = src.split('\n')[0][:70]
    print(f"  {i:2d} [{ct:8s}] {first}")
