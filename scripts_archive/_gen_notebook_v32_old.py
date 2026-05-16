#!/usr/bin/env python3
"""Generate v32 training notebook (.ipynb) programmatically.
This avoids manual JSON editing errors.
"""
import json, os

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in source.rstrip("\n").split("\n")]}

def code_cell(source):
    lines = source.rstrip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]  # last line no trailing \n
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}

cells = []

# ────────────────── Cell 1: Title ──────────────────
cells.append(md_cell("""# v32: Full-Scale Verified Training (9500 samples)

**Strategy**: Answer-only SFT with all 9500 verified training samples.
- Data: `sft_cot_v2_hybrid.csv` — all 6 types, naturally balanced (~1576/type)
- Format: Same as E1 (0.68 best) — `enable_thinking=True`, answer = `\\boxed{...}`
- Loss: Standard full-text loss (proven better than boxed-only by +0.02)
- LoRA: rank=32, alpha=16, all-linear, dropout=0 (large data → no regularization needed)
- LR: 1e-4 (halved from E1 due to 15× more data)"""))

# ────────────────── Cell 2: pip install ──────────────────
cells.append(code_cell(
    "!pip install -q --no-index --find-links "
    "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages "
    "datasets trl --ignore-installed"))

# ────────────────── Cell 3: import check ──────────────────
cells.append(code_cell("""import datasets, trl
print(f"datasets: {datasets.__version__} | trl: {trl.__version__}")"""))

# ────────────────── Cell 4: Imports + Triton Fixes + Config ──────────────────
cells.append(code_cell(r"""import os, sys, stat, shutil, gc, zipfile
import polars as pl
import torch
import torch.nn.functional as F
import kagglehub
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ═══════════════ Kaggle / Triton Environment Fixes ═══════════════
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

# ╔══════════════════════════════════════════════════════════════════╗
# ║  v32 EXPERIMENT CONFIGURATION                                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Change DATA_SOURCE to switch experiments:                      ║
# ║    "cot_v2_hybrid" → 9500 samples (full, balanced)             ║
# ║    "original"      → 600 random (E1 baseline reproduction)     ║
# ╚══════════════════════════════════════════════════════════════════╝
DATA_SOURCE = "cot_v2_hybrid"  # ← THE KEY SWITCH
SUBSAMPLE_SIZE = 600           # only used when DATA_SOURCE="original"

LORA_RANK = 32
LORA_ALPHA = 16        # scale = alpha/rank = 0.5 (E1 proven)
LORA_DROPOUT = 0.0     # 0 for large data (1 epoch = no overfitting risk)
MAX_SEQ_LEN = 1024
NUM_EPOCHS = 1
GRAD_ACCUM = 4
LR = 1e-4              # halved from E1's 2e-4 due to 15x more data

OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Config: data={DATA_SOURCE}, lr={LR}, epochs={NUM_EPOCHS}, "
      f"rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")"""))

# ────────────────── Cell 5: Data section header ──────────────────
cells.append(md_cell("## Data Loading & Formatting"))

# ────────────────── Cell 6: Data loading + formatting ──────────────────
cells.append(code_cell(r"""# Download model
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

COMP_DATA = '/kaggle/input/nvidia-nemotron-3-reasoning-challenge'
COT_DATA = '/kaggle/input/prog-cot-training-data'

# ─── Load data ───
if DATA_SOURCE == "cot_v2_hybrid":
    train_df = pl.read_csv(f'{COT_DATA}/sft_cot_v2_hybrid.csv')
elif DATA_SOURCE == "original":
    train_df = pl.read_csv(f'{COMP_DATA}/train.csv')
    train_df = train_df.sample(n=min(SUBSAMPLE_SIZE, len(train_df)), seed=42)
else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}")

print(f"Data: {DATA_SOURCE} | Samples: {len(train_df)}")
if 'type' in train_df.columns:
    print("Type distribution:")
    for row in train_df['type'].value_counts().sort('type').iter_rows():
        print(f"  {row[0]}: {row[1]}")

hf_dataset = Dataset.from_pandas(train_df.to_pandas())

# ─── Tokenizer ───
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── Prompt suffix (aligned with evaluation metric) ───
SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

def build_training_text(example):
    """Answer-only format - proven by E1 (0.68).
    Model gets <think></think>\\boxed{answer} and learns to fill thinking on its own."""
    user_msg = example["prompt"] + SUFFIX
    assistant_msg = f"\\boxed{{{example['answer']}}}"
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=True,
    )
    return {"text": text}

# ─── Template verification ───
_sample = build_training_text({"prompt": "What is 2+2?", "answer": "4"})
print(f"\nTemplate preview:\n{_sample['text']}")
_ids = tokenizer.encode(_sample["text"])
print(f"Template tokens: {len(_ids)}")

# ─── Apply to dataset ───
hf_dataset = hf_dataset.map(
    build_training_text,
    remove_columns=hf_dataset.column_names,
)
print(f"\nDataset ready: {len(hf_dataset)} examples")

# ─── Token length distribution ───
lengths = [len(tokenizer.encode(ex["text"])) for ex in hf_dataset]
import numpy as np
print(f"Token lengths: mean={np.mean(lengths):.0f}, max={max(lengths)}, "
      f"p95={np.percentile(lengths, 95):.0f}, p99={np.percentile(lengths, 99):.0f}")
over = sum(1 for l in lengths if l > MAX_SEQ_LEN)
if over > 0:
    print(f"⚠ {over}/{len(lengths)} examples exceed MAX_SEQ_LEN={MAX_SEQ_LEN}!")
else:
    print(f"✓ All {len(lengths)} examples fit within MAX_SEQ_LEN={MAX_SEQ_LEN}")"""))

# ────────────────── Cell 7: Model section header ──────────────────
cells.append(md_cell("## Model Loading & LoRA Configuration"))

# ────────────────── Cell 8: Model + LoRA ──────────────────
cells.append(code_cell(r"""# ─── Mock broken CUDA modules ───
from unittest.mock import MagicMock
_mock_modules = [
    "cutlass", "cutlass.cute", "cutlass.utils",
    "mamba_ssm.ops.cute", "mamba_ssm.ops.cute.mamba3",
    "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
    "mamba_ssm.ops.tilelang", "mamba_ssm.ops.tilelang.mamba3",
    "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo",
]
for mod_name in _mock_modules:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# ─── Load Model ───
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
print(f"Model loaded. Vocab size: {len(tokenizer)}")

# Force slow path — bypass broken CUDA kernels
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

# ─── LoRA Layer Analysis (for team review) ───
print("\n" + "="*60)
print("MODEL ARCHITECTURE — LoRA TARGET ANALYSIS")
print("="*60)
from collections import Counter
layer_types = Counter()
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    # Extract module type for analysis
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p in ('weight', 'bias'):
            key = parts[i-1] if i > 0 else name
            layer_types[key] += param.numel()
            break

print(f"\nTotal parameters: {total_params:,}")
print("\nParameter distribution by layer type:")
for lt, count in layer_types.most_common(15):
    pct = count / total_params * 100
    print(f"  {lt:30s}: {count:>12,} ({pct:5.1f}%)")

# ─── Apply LoRA ───
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",  # covers attention + MLP + Mamba projections
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Show which modules got LoRA adapters
lora_modules = set()
for name, _ in model.named_parameters():
    if 'lora_' in name:
        # Extract the module path (remove .lora_A/B.weight)
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p.startswith('lora_'):
                mod_name = parts[i-1]
                lora_modules.add(mod_name)
                break
print(f"\nLoRA applied to {len(lora_modules)} unique module types: {sorted(lora_modules)}")"""))

# ────────────────── Cell 9: Training section header ──────────────────
cells.append(md_cell("## Training"))

# ────────────────── Cell 10: SFTTrainer + train ──────────────────
cells.append(code_cell(r"""import triton.backends.nvidia.compiler as nv_compiler
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=5,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)

total_steps = len(hf_dataset) // GRAD_ACCUM * NUM_EPOCHS
print(f"Training: {len(hf_dataset)} examples × {NUM_EPOCHS} epoch(s)")
print(f"Effective batch: {1 * GRAD_ACCUM} | Total steps: ~{total_steps}")
print(f"LR: {LR} | Max seq: {MAX_SEQ_LEN}")
print("Starting training...")
trainer.train()"""))

# ────────────────── Cell 11: Save section header ──────────────────
cells.append(md_cell("## Save & Package Submission"))

# ────────────────── Cell 12: Save adapter + zip ──────────────────
cells.append(code_cell(r"""# Save adapter
trainer.model.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size/1024:.1f} KB)")

# Package submission
zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        zf.write(fpath, fname)

print(f"\nCreated {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")

# Verify
with zipfile.ZipFile(zip_path, 'r') as zf:
    names = zf.namelist()
    print(f"Contents: {names}")
    assert "adapter_config.json" in names, "Missing adapter_config.json!"
    assert "adapter_model.safetensors" in names, "Missing adapter_model.safetensors!"
    print("✓ submission.zip is valid and ready to submit!")"""))

# ────────────────── Assemble notebook ──────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out_path = "nvidia-nemotron-sfttrainer-v32.ipynb"
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Generated: {out_path}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
