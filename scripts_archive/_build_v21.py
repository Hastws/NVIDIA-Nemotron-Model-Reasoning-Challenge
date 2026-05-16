#!/usr/bin/env python3
"""Build v21 notebook: Single-stage compact-rules with boxed-only loss.
Data: sft_compact_rules.csv (has thinking column with 7-32 char rules)
Loss: only on \boxed{answer} tokens (thinking masked to -100)
"""
import json

NB_PATH = 'nvidia-nemotron-compact-boxed-v21.ipynb'

def make_cell(cell_type, source, cell_id=None):
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

# Cell 0: Markdown
cells.append(make_cell("markdown", "## v21: Compact Rules + Boxed-Only Loss\nSingle-stage SFT: compact rules in thinking, loss only on `\\boxed{answer}`."))

# Cell 1: pip install
cells.append(make_cell("code",
    '!pip install -q --no-index --find-links /kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages datasets trl --ignore-installed'))

# Cell 2: version check
cells.append(make_cell("code", """import datasets
import trl
print("datasets:", datasets.__version__)
print("trl:", trl.__version__)"""))

# Cell 3: imports + hyperparams
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
SAMPLES = 600
EPOCHS = 1
LR = 2e-4
LORA_RANK = 32
MAX_SEQ_LEN = 1024
GRAD_ACCUM = 4

OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("✓ Hyperparameters set")"""))

# Cell 4: Data loading + tokenizer + build examples
cells.append(make_cell("code", r"""MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

# </think> token ID — used to find where boxed answer starts
THINK_CLOSE_ID = tokenizer.convert_tokens_to_ids("</think>")
print(f"</think> token ID: {THINK_CLOSE_ID}")

# Load compact rules data (7-32 char thinking per type)
cot_df = pl.read_csv('/kaggle/input/prog-cot-training-data/sft_compact_rules.csv')
cot_df = cot_df.sample(n=min(SAMPLES, len(cot_df)), seed=42)
print(f"Data: {len(cot_df)} samples")
print(f"Type distribution:")
print(cot_df.group_by("type").len().sort("type"))

hf_ds = Dataset.from_pandas(cot_df.to_pandas())

def build_compact_boxed_example(example):
    # Format: user prompt + assistant with <think>rule</think>\boxed{answer}
    # Loss: ONLY on \boxed{answer} tokens (everything up to </think> is masked)
    prompt = example["prompt"]
    answer = str(example["answer"])
    thinking = example.get("thinking", "") or ""
    user_msg = prompt + SUFFIX

    # Build message with reasoning_content if thinking exists
    if thinking.strip():
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}", "reasoning_content": thinking},
        ]
    else:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
        ]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception:
        # Fallback: manual ChatML
        if thinking.strip():
            text = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n{thinking}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
            )
        else:
            text = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n<think></think>\\boxed{{{answer}}}<|im_end|>"
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

hf_ds = hf_ds.map(build_compact_boxed_example, remove_columns=hf_ds.column_names)
n_loss = sum(1 for x in hf_ds[0]["labels"] if x != -100)
n_total = len(hf_ds[0]["input_ids"])
print(f"\nReady: {len(hf_ds)} examples")
print(f"Example 0: {n_total} tokens total, {n_loss} loss tokens")
print(f"Loss tokens: {tokenizer.decode([t for t in hf_ds[0]['labels'] if t != -100])}")
# Show a few more
for i in [1, 2, 3]:
    loss_toks = [t for t in hf_ds[i]["labels"] if t != -100]
    print(f"Example {i}: {len(hf_ds[i]['input_ids'])} tokens, {len(loss_toks)} loss -> {tokenizer.decode(loss_toks)[:80]}")"""))

# Cell 5: Model loading + LoRA
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

# Cell 6: Training
cells.append(make_cell("code", r"""import triton.backends.nvidia.compiler as nv_compiler
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

tokenizer.model_max_length = MAX_SEQ_LEN

train_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
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

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    args=train_args,
)

print("=== Training: Compact Rules + Boxed-Only Loss ===")
print(f"  Samples: {len(hf_ds)}, Epochs: {EPOCHS}, LR: {LR}")
trainer.train()
print("✓ Training complete")

del trainer
gc.collect()
torch.cuda.empty_cache()
print(f"GPU after training: {torch.cuda.memory_allocated()/1024**3:.1f} GB")"""))

# Cell 7: Save + zip
cells.append(make_cell("markdown", "## Save and Package Submission"))

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

# Build notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": cells,
}

with open(NB_PATH, 'w') as f:
    json.dump(notebook, f, indent=1)

with open(NB_PATH) as f:
    nb = json.load(f)

print(f"✓ Created {NB_PATH}: {len(nb['cells'])} cells")
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:70]
    print(f"  {i:2d} [{c['cell_type']:8s}] {src}")
