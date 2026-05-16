#!/usr/bin/env python3
"""Build the 2-stage SFT training notebook for Kaggle submission."""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

# ============================================================
# Cell 1: Title
# ============================================================
cells.append(new_markdown_cell("""\
# 2-Stage SFT Training: Reasoning + Format

**Strategy:**
- **Stage 1**: 15K mixed data (compact rules + full CoT + answer-only) → learn reasoning patterns
- **Stage 2**: 600 answer-only samples at low LR → polish `\\boxed{}` output format

**Critical**: User prompt suffix matches official evaluation **EXACTLY**.
"""))

# ============================================================
# Cell 2: Install packages
# ============================================================
cells.append(new_code_cell("""\
!pip install -q --no-index --find-links /kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages datasets trl --ignore-installed
"""))

# ============================================================
# Cell 3: Imports + Triton fixes
# ============================================================
cells.append(new_code_cell("""\
import os
import sys
import stat
import shutil
import gc
import zipfile
import time
import json
import polars as pl
import pandas as pd
import torch
import torch.nn.functional as F
import kagglehub
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# --- Triton / Kaggle environment fixes ---
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

# Copy PTXAS binaries to writable temp
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
    print("✅ Triton patched")
else:
    print("⚠️ ptxas-blackwell not found, skipping Triton patch")

print("✅ Imports and environment ready")
"""))

# ============================================================
# Cell 4: Configuration
# ============================================================
cells.append(new_markdown_cell("""\
## Configuration

All hyperparameters are defined here. The `PROMPT_SUFFIX` is copied verbatim from the **official evaluation metric script** — do not modify it.
"""))

cells.append(new_code_cell("""\
# =============================================
#  🔧 HYPERPARAMETERS — EDIT HERE
# =============================================

# --- Stage 1: Learn reasoning (15K mixed data) ---
STAGE1_LR        = 2e-4
STAGE1_EPOCHS    = 1
STAGE1_MAX_SEQ   = 2048    # CoT can be long
STAGE1_GRAD_ACCUM = 4

# --- Stage 2: Format polish (answer-only, low LR) ---
STAGE2_ENABLED   = True
STAGE2_N_SAMPLES = 600
STAGE2_LR        = 4e-5    # 1/5 of Stage 1
STAGE2_EPOCHS    = 1
STAGE2_MAX_SEQ   = 512     # answer-only is short
STAGE2_GRAD_ACCUM = 4

# --- LoRA ---
LORA_RANK        = 32
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05

# =============================================
#  🔴 OFFICIAL PROMPT SUFFIX — DO NOT MODIFY
# =============================================
# Source: competition_notebooks/nemotron-baseline-evaluation.ipynb
# Lines: user_content = item.prompt + '\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`'
PROMPT_SUFFIX = '\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`'

OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Configuration ===")
print(f"Stage 1: LR={STAGE1_LR}, epochs={STAGE1_EPOCHS}, max_seq={STAGE1_MAX_SEQ}")
print(f"Stage 2: enabled={STAGE2_ENABLED}, n={STAGE2_N_SAMPLES}, LR={STAGE2_LR}, epochs={STAGE2_EPOCHS}")
print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Prompt suffix: {repr(PROMPT_SUFFIX)}")
"""))

# ============================================================
# Cell 5: Data loading
# ============================================================
cells.append(new_markdown_cell("## Data Loading & Statistics"))

cells.append(new_code_cell("""\
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
COMP_DATA = '/kaggle/input/nvidia-nemotron-3-reasoning-challenge'
COT_DATA = '/kaggle/input/prog-cot-training-data'

# Load the merged dataset
train_df = pl.read_csv(f'{COT_DATA}/sft_merged_v1.csv')

print(f"{'='*60}")
print(f"  Loaded: sft_merged_v1.csv — {len(train_df)} rows")
print(f"{'='*60}")

# Statistics
pdf = train_df.to_pandas()
has_thinking = pdf['thinking'].fillna('').str.strip().str.len() > 0
n_with = has_thinking.sum()
n_without = (~has_thinking).sum()
print(f"\\n  With thinking: {n_with} ({n_with/len(pdf)*100:.1f}%)")
print(f"  Without thinking (answer-only): {n_without} ({n_without/len(pdf)*100:.1f}%)")

# Classify thinking types
short_mask = has_thinking & (pdf['thinking'].str.len() < 50)
long_mask = has_thinking & (pdf['thinking'].str.len() >= 50)
print(f"  - Compact rules (<50 chars): {short_mask.sum()}")
print(f"  - Full CoT (≥50 chars): {long_mask.sum()}")

# Show thinking length distribution
print(f"\\n  Thinking length stats (non-empty):")
lengths = pdf.loc[has_thinking, 'thinking'].str.len()
print(f"    min={lengths.min()}, median={lengths.median():.0f}, mean={lengths.mean():.0f}, max={lengths.max()}")

# Check for any data issues
print(f"\\n  --- Sanity checks ---")
print(f"  Empty prompt: {(pdf['prompt'].fillna('').str.len() == 0).sum()}")
print(f"  Empty answer: {(pdf['answer'].fillna('').astype(str).str.len() == 0).sum()}")
boxed_in_thinking = pdf.loc[has_thinking, 'thinking'].str.contains(r'\\\\boxed', regex=True, na=False).sum()
print(f"  \\\\boxed in thinking: {boxed_in_thinking}")
print(f"  Columns: {list(pdf.columns)}")
"""))

# ============================================================
# Cell 6: Prompt format verification + training text builder
# ============================================================
cells.append(new_markdown_cell("""\
## Prompt Format & Training Text

**Critical**: The user message suffix must match the official evaluation metric script EXACTLY.
The official code is:
```python
user_content = item.prompt + '\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`'
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': user_content}],
    tokenize=False, add_generation_prompt=True, enable_thinking=True,
)
```
"""))

cells.append(new_code_cell("""\
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =============================================
#  Prompt suffix verification
# =============================================
print("=== PROMPT SUFFIX VERIFICATION ===")
print(f"Our PROMPT_SUFFIX: {repr(PROMPT_SUFFIX)}")

# Reproduce what the official eval does
official_suffix = '\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`'
assert PROMPT_SUFFIX == official_suffix, f"❌ MISMATCH!\\nOurs:     {repr(PROMPT_SUFFIX)}\\nOfficial: {repr(official_suffix)}"
print("✅ Prompt suffix matches official evaluation exactly")

# Show what the full user message looks like
sample_prompt = "What is 2 + 2?"
user_content = sample_prompt + PROMPT_SUFFIX
print(f"\\nFull user message example:\\n---\\n{user_content}\\n---")

# Show what official eval generates for inference
official_inference_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_content}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
print(f"\\nOfficial inference prompt (enable_thinking=True):\\n---\\n{official_inference_prompt}\\n---")

# =============================================
#  build_training_text function
# =============================================
def build_training_text(example):
    \"\"\"Build ChatML training text that matches official evaluation prompt format.
    
    For rows WITH thinking:
      <|im_start|>user\\n{prompt}{SUFFIX}<|im_end|>\\n
      <|im_start|>assistant\\n<think>\\n{thinking}\\n</think>\\n\\\\boxed{answer}<|im_end|>
    
    For rows WITHOUT thinking (answer-only):
      Uses apply_chat_template with enable_thinking=False to produce:
      <|im_start|>user\\n{prompt}{SUFFIX}<|im_end|>\\n
      <|im_start|>assistant\\n<think></think>\\\\boxed{answer}<|im_end|>
    \"\"\"
    prompt = example["prompt"]
    answer = str(example["answer"])
    thinking = example.get("thinking", None)
    
    user_msg = prompt + PROMPT_SUFFIX
    
    if thinking and str(thinking).strip():
        # CoT / compact rule path — manually build ChatML for exact control
        text = (
            f"<|im_start|>user\\n{user_msg}<|im_end|>\\n"
            f"<|im_start|>assistant\\n<think>\\n{str(thinking).strip()}\\n</think>\\n\\\\boxed{{{answer}}}<|im_end|>"
        )
    else:
        # Answer-only path — use tokenizer template for consistency
        assistant_msg = f"\\\\boxed{{{answer}}}"
        try:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except Exception:
            text = (
                f"<|im_start|>user\\n{user_msg}<|im_end|>\\n"
                f"<|im_start|>assistant\\n<think></think>\\\\boxed{{{answer}}}<|im_end|>"
            )
    
    return {"text": text}

# =============================================
#  Format verification with actual data
# =============================================
print("\\n=== FORMAT VERIFICATION ===")
sample_df = train_df.to_pandas()

# Test with a thinking row
think_rows = sample_df[sample_df['thinking'].fillna('').str.strip().str.len() > 0]
if len(think_rows) > 0:
    row = think_rows.iloc[0].to_dict()
    result = build_training_text(row)
    print(f"\\n--- THINKING ROW (id={row['id']}) ---")
    text = result['text']
    print(text[:600])
    if len(text) > 600:
        print(f"... ({len(text)} chars total)")
    # Verify key patterns
    assert '<think>\\n' in text, "Missing <think> tag"
    assert '\\n</think>\\n' in text, "Missing </think> tag"
    assert '\\\\boxed{' in text, "Missing \\\\boxed{}"
    assert PROMPT_SUFFIX.lstrip('\\n') in text, "Missing prompt suffix in user message"
    print("✅ Thinking row format OK")

# Test with an answer-only row
ao_rows = sample_df[sample_df['thinking'].fillna('').str.strip().str.len() == 0]
if len(ao_rows) > 0:
    row = ao_rows.iloc[0].to_dict()
    result = build_training_text(row)
    print(f"\\n--- ANSWER-ONLY ROW (id={row['id']}) ---")
    text = result['text']
    print(text[:600])
    assert '\\\\boxed{' in text, "Missing \\\\boxed{}"
    assert PROMPT_SUFFIX.lstrip('\\n') in text, "Missing prompt suffix in user message"
    print("✅ Answer-only row format OK")

print("\\n✅ All format checks passed!")
"""))

# ============================================================
# Cell 7: Build full dataset + token analysis
# ============================================================
cells.append(new_code_cell("""\
# Convert full dataset
hf_dataset = Dataset.from_pandas(train_df.to_pandas())
hf_dataset = hf_dataset.map(
    build_training_text,
    remove_columns=hf_dataset.column_names,
)
print(f"Dataset formatted: {len(hf_dataset)} rows")

# Token length analysis
print("\\n=== TOKEN LENGTH ANALYSIS ===")
token_lengths = []
for i in range(len(hf_dataset)):
    toks = tokenizer(hf_dataset[i]['text'], add_special_tokens=False)
    token_lengths.append(len(toks['input_ids']))

import numpy as np
tl = np.array(token_lengths)
print(f"  Total samples: {len(tl)}")
print(f"  Min tokens:    {tl.min()}")
print(f"  Median tokens: {np.median(tl):.0f}")
print(f"  Mean tokens:   {tl.mean():.0f}")
print(f"  P95 tokens:    {np.percentile(tl, 95):.0f}")
print(f"  P99 tokens:    {np.percentile(tl, 99):.0f}")
print(f"  Max tokens:    {tl.max()}")
truncated = (tl > STAGE1_MAX_SEQ).sum()
print(f"\\n  Truncated at {STAGE1_MAX_SEQ}: {truncated} ({truncated/len(tl)*100:.1f}%)")
if truncated > 0:
    print(f"  ⚠️  {truncated} samples will be truncated. Consider increasing STAGE1_MAX_SEQ.")
else:
    print(f"  ✅ No truncation at max_seq={STAGE1_MAX_SEQ}")

# Show 3 samples: short, medium, long
sorted_idx = np.argsort(tl)
for label, idx in [("SHORTEST", sorted_idx[0]), ("MEDIAN", sorted_idx[len(sorted_idx)//2]), ("LONGEST", sorted_idx[-1])]:
    print(f"\\n--- {label} ({tl[idx]} tokens) ---")
    print(hf_dataset[int(idx)]['text'][:400])
    if len(hf_dataset[int(idx)]['text']) > 400:
        print(f"... [truncated, {len(hf_dataset[int(idx)]['text'])} chars]")
"""))

# ============================================================
# Cell 8: Model loading + LoRA
# ============================================================
cells.append(new_markdown_cell("## Model Loading & LoRA Setup"))

cells.append(new_code_cell("""\
# Mock cutlass/mamba3 to prevent import errors
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
print(f"Mock modules injected: {len(_mock_modules)}")

# Load model
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
print(f"Model loaded in {time.time()-t0:.1f}s. Vocab size: {len(tokenizer)}")

# Patch: force slow path to bypass broken CUDA kernels
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

# Setup LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Log model architecture summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\nTotal params: {total_params/1e9:.2f}B")
print(f"Trainable params: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.2f}%)")
"""))

# ============================================================
# Cell 9: Stage 1 Training
# ============================================================
cells.append(new_markdown_cell("""\
## Stage 1: Learn Reasoning

Train on all 15K samples (compact rules + full CoT + answer-only). This teaches the model to:
1. Recognize different problem types
2. Apply reasoning strategies within `<think>` tags  
3. Output answers in `\\boxed{}` format
"""))

cells.append(new_code_cell("""\
import triton.backends.nvidia.compiler as nv_compiler
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

print(f"{'='*60}")
print(f"  STAGE 1: Reasoning Training")
print(f"  Samples: {len(hf_dataset)}, LR: {STAGE1_LR}, Epochs: {STAGE1_EPOCHS}")
print(f"  Max Seq: {STAGE1_MAX_SEQ}, Grad Accum: {STAGE1_GRAD_ACCUM}")
print(f"  Effective batch: {STAGE1_GRAD_ACCUM} (batch_size=1 × accum={STAGE1_GRAD_ACCUM})")
total_steps = (len(hf_dataset) // STAGE1_GRAD_ACCUM) * STAGE1_EPOCHS
print(f"  Estimated steps: ~{total_steps}")
print(f"{'='*60}")

stage1_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=STAGE1_GRAD_ACCUM,
    num_train_epochs=STAGE1_EPOCHS,
    learning_rate=STAGE1_LR,
    logging_steps=5,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=STAGE1_MAX_SEQ,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)

stage1_trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=stage1_args,
)

t0 = time.time()
stage1_result = stage1_trainer.train()
stage1_time = time.time() - t0

print(f"\\n{'='*60}")
print(f"  STAGE 1 COMPLETE")
print(f"  Time: {stage1_time/60:.1f} min")
print(f"  Final loss: {stage1_result.training_loss:.4f}")
print(f"{'='*60}")
"""))

# ============================================================
# Cell 10: Stage 2 Training
# ============================================================
cells.append(new_markdown_cell("""\
## Stage 2: Format Polish (Optional)

Light fine-tuning with answer-only data at low LR. Reinforces `\\boxed{}` output format without disrupting learned reasoning.

Set `STAGE2_ENABLED = False` above to skip this stage.
"""))

cells.append(new_code_cell("""\
if STAGE2_ENABLED:
    print(f"{'='*60}")
    print(f"  STAGE 2: Format Polish")
    print(f"{'='*60}")
    
    # Prepare Stage 2 dataset from answer-only rows
    full_df = train_df.to_pandas()
    ao_mask = full_df['thinking'].fillna('').str.strip().str.len() == 0
    answer_only_df = full_df[ao_mask]
    
    n_sample = min(STAGE2_N_SAMPLES, len(answer_only_df))
    stage2_df = answer_only_df.sample(n=n_sample, random_state=42)
    
    print(f"  Answer-only pool: {len(answer_only_df)}")
    print(f"  Sampled for Stage 2: {n_sample}")
    print(f"  LR: {STAGE2_LR}, Epochs: {STAGE2_EPOCHS}, Max Seq: {STAGE2_MAX_SEQ}")
    
    stage2_dataset = Dataset.from_pandas(stage2_df)
    stage2_dataset = stage2_dataset.map(
        build_training_text,
        remove_columns=stage2_dataset.column_names,
    )
    
    # Show a Stage 2 sample
    print(f"\\n--- Stage 2 sample ---")
    print(stage2_dataset[0]['text'][:300])
    
    stage2_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=STAGE2_GRAD_ACCUM,
        num_train_epochs=STAGE2_EPOCHS,
        learning_rate=STAGE2_LR,
        logging_steps=5,
        bf16=True,
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=STAGE2_MAX_SEQ,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    
    stage2_trainer = SFTTrainer(
        model=model,  # Same model object — continues from Stage 1
        train_dataset=stage2_dataset,
        processing_class=tokenizer,
        args=stage2_args,
    )
    
    t0 = time.time()
    stage2_result = stage2_trainer.train()
    stage2_time = time.time() - t0
    
    print(f"\\n{'='*60}")
    print(f"  STAGE 2 COMPLETE")
    print(f"  Time: {stage2_time/60:.1f} min")
    print(f"  Final loss: {stage2_result.training_loss:.4f}")
    print(f"{'='*60}")
else:
    print("Stage 2 SKIPPED (STAGE2_ENABLED=False)")
"""))

# ============================================================
# Cell 11: Save + Package
# ============================================================
cells.append(new_markdown_cell("## Save & Package Submission"))

cells.append(new_code_cell("""\
# Save adapter weights and config
model.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:40s} {size/1024:.1f} KB")

# Package into submission.zip
zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)  # files at zip root

print(f"\\nCreated {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")

# Verify submission
with zipfile.ZipFile(zip_path, 'r') as zf:
    names = zf.namelist()
    print(f"Contents: {names}")
    assert "adapter_config.json" in names, "❌ Missing adapter_config.json!"
    assert "adapter_model.safetensors" in names, "❌ Missing adapter_model.safetensors!"
    
    # Print adapter config
    with zf.open("adapter_config.json") as cf:
        config = json.loads(cf.read())
        print(f"\\nAdapter config:")
        print(f"  r (rank):     {config.get('r', '?')}")
        print(f"  lora_alpha:   {config.get('lora_alpha', '?')}")
        print(f"  target_modules: {config.get('target_modules', '?')[:80]}")

print("\\n✅ submission.zip is valid and ready to submit!")
"""))

# ============================================================
# Cell 12: Training summary
# ============================================================
cells.append(new_code_cell("""\
print("=" * 60)
print("  TRAINING SUMMARY")
print("=" * 60)
print(f"  Stage 1: {len(hf_dataset)} samples, loss={stage1_result.training_loss:.4f}, time={stage1_time/60:.1f}min")
if STAGE2_ENABLED:
    print(f"  Stage 2: {n_sample} samples, loss={stage2_result.training_loss:.4f}, time={stage2_time/60:.1f}min")
    print(f"  Total time: {(stage1_time + stage2_time)/60:.1f} min")
else:
    print(f"  Stage 2: SKIPPED")
    print(f"  Total time: {stage1_time/60:.1f} min")
print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
print(f"  Prompt suffix verified: ✅")
print(f"  Output: /kaggle/working/submission.zip")
print("=" * 60)
"""))

# ============================================================
# Assemble notebook
# ============================================================
nb.cells = cells

output_path = "nvidia-nemotron-2stage-sft.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✅ Notebook created: {output_path}")
print(f"   Cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'code')} code, {sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
