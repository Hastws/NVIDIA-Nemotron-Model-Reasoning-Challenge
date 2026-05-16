#!/usr/bin/env python3
"""Build v22 notebook: Weighted-loss self-distilled CoT SFT.
Data: sft_distill.csv (3211 correct self-distilled samples with think + answer)
Loss: think tokens get weight THINK_WEIGHT (0.1), answer tokens get weight 1.0
"""
import json

NB_PATH = 'nvidia-nemotron-weighted-loss-v22.ipynb'

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
cells.append(make_cell("markdown", """## v22: Weighted-Loss Self-Distilled CoT SFT
Self-distilled CoT training with weighted loss:
- Think tokens: low weight (0.1) — light guidance toward good reasoning
- Answer tokens: full weight (1.0) — strong signal on correct answers
- Data: 3211 correct self-distilled samples (temp=0.7, base model's own CoT)"""))

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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

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
THINK_WEIGHT = 0.1       # Low weight on think tokens
ANSWER_WEIGHT = 1.0      # Full weight on answer tokens
EPOCHS = 1
LR = 5e-5                # Conservative LR (3211 samples × ~81 effective tokens = ~260K)
LORA_RANK = 32
MAX_SEQ_LEN = 4096       # Match eval max_model_len
GRAD_ACCUM = 4
MAX_SAMPLES = 0          # 0 = use all samples

OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Hyperparameters: THINK_WEIGHT={THINK_WEIGHT}, LR={LR}, MAX_SEQ_LEN={MAX_SEQ_LEN}")"""))

# Cell 4: Data loading + tokenizer + build examples with weighted loss
cells.append(make_cell("code", r"""MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

# Key token IDs
THINK_CLOSE_ID = tokenizer.convert_tokens_to_ids("</think>")
print(f"</think> token ID: {THINK_CLOSE_ID}")

# Load self-distilled data
cot_df = pl.read_csv('/kaggle/input/prog-cot-training-data/sft_distill.csv')
if MAX_SAMPLES > 0 and len(cot_df) > MAX_SAMPLES:
    # Balanced sampling across types
    sampled = []
    per_type = MAX_SAMPLES // cot_df['type'].n_unique()
    for t in cot_df['type'].unique().sort().to_list():
        t_df = cot_df.filter(pl.col('type') == t)
        n = min(per_type, len(t_df))
        sampled.append(t_df.sample(n=n, seed=42))
    cot_df = pl.concat(sampled).sample(fraction=1.0, seed=42)  # shuffle
print(f"Data: {len(cot_df)} samples")
print(cot_df.group_by("type").agg(
    pl.len().alias("count"),
    pl.col("thinking").str.len_chars().mean().alias("avg_think_chars")
).sort("type"))

hf_ds = Dataset.from_pandas(cot_df.to_pandas())

def build_weighted_example(example):
    prompt = example["prompt"]
    answer = str(example["answer"])
    thinking = example.get("thinking", "") or ""
    user_msg = prompt + SUFFIX

    # Build message with reasoning_content
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

    # Find boundaries:
    # 1. Find where assistant response starts (after the last "assistant\n")
    # 2. Find </think> token
    # Everything before assistant response: labels=-100, weight=0
    # Think tokens (up to and including </think>): labels=token_id, weight=THINK_WEIGHT
    # Answer tokens (after </think>): labels=token_id, weight=ANSWER_WEIGHT

    # Find assistant start: look for <|im_start|>assistant pattern
    # The tokenizer encodes this differently, so find the </think> token instead
    # and use that as the boundary between think and answer regions

    # Find the LAST </think> token (in case the thinking content contains the text "</think>")
    think_close_pos = -1
    for i, tid in enumerate(input_ids):
        if tid == THINK_CLOSE_ID:
            think_close_pos = i

    if think_close_pos == -1:
        # No </think> found — treat all as answer (fallback)
        # Mask prompt (heuristic: find last newline-assistant pattern)
        labels = [-100] * len(input_ids)
        loss_weights = [0.0] * len(input_ids)
        # Find start of answer content (rough: last 30% of tokens)
        ans_start = max(0, len(input_ids) - 30)
        for i in range(ans_start, len(input_ids)):
            labels[i] = input_ids[i]
            loss_weights[i] = ANSWER_WEIGHT
    else:
        # Find where user message ends / assistant starts
        # Look for the <think> token (ID should be one less or nearby)
        think_open_id = tokenizer.convert_tokens_to_ids("<think>")
        assistant_start = 0
        for i, tid in enumerate(input_ids):
            if tid == think_open_id:
                assistant_start = i
                break

        labels = [-100] * len(input_ids)
        loss_weights = [0.0] * len(input_ids)

        # Think region: from <think> to </think> (inclusive)
        for i in range(assistant_start, think_close_pos + 1):
            labels[i] = input_ids[i]
            loss_weights[i] = THINK_WEIGHT

        # Answer region: after </think> to end
        for i in range(think_close_pos + 1, len(input_ids)):
            labels[i] = input_ids[i]
            loss_weights[i] = ANSWER_WEIGHT

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_weights": loss_weights,
    }

hf_ds = hf_ds.map(build_weighted_example, remove_columns=hf_ds.column_names)

# Diagnostic: check a few examples
for i in range(min(5, len(hf_ds))):
    ids = hf_ds[i]["input_ids"]
    labs = hf_ds[i]["labels"]
    wts = hf_ds[i]["loss_weights"]
    n_think = sum(1 for w in wts if w == THINK_WEIGHT)
    n_answer = sum(1 for w in wts if w == ANSWER_WEIGHT)
    n_masked = sum(1 for l in labs if l == -100)
    eff = n_think * THINK_WEIGHT + n_answer * ANSWER_WEIGHT
    ans_text = tokenizer.decode([ids[j] for j in range(len(ids)) if wts[j] == ANSWER_WEIGHT])
    print(f"Ex {i}: {len(ids)} toks | masked={n_masked} | think={n_think}(w={THINK_WEIGHT}) | answer={n_answer}(w=1.0) | eff={eff:.0f} | ans: {ans_text[:80]}")

# Total effective tokens
total_eff = sum(
    sum(w for w in ex["loss_weights"] if w > 0)
    for ex in hf_ds
)
print(f"\nTotal effective token-weight: {total_eff:,.0f}")
print(f"E1 reference: ~48,000")
print(f"Ratio: {total_eff/48000:.1f}x")
print(f"\nReady: {len(hf_ds)} examples")"""))

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

# Cell 6: Custom Trainer + Data Collator + Training
cells.append(make_cell("code", r"""import triton.backends.nvidia.compiler as nv_compiler
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

tokenizer.model_max_length = MAX_SEQ_LEN

# --- Custom Data Collator that handles loss_weights ---
def weighted_data_collator(features):
    max_len = max(len(f["input_ids"]) for f in features)
    pad_id = tokenizer.pad_token_id

    batch_ids, batch_mask, batch_labels, batch_weights = [], [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        batch_ids.append(f["input_ids"] + [pad_id] * pad_len)
        batch_mask.append(f["attention_mask"] + [0] * pad_len)
        batch_labels.append(f["labels"] + [-100] * pad_len)
        batch_weights.append(f["loss_weights"] + [0.0] * pad_len)

    return {
        "input_ids": torch.tensor(batch_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "loss_weights": torch.tensor(batch_weights, dtype=torch.float32),
    }

# --- Custom Trainer with weighted loss ---
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weights = inputs.pop("loss_weights")
        outputs = model(**inputs)

        # Get logits and compute per-token loss
        logits = outputs.logits
        labels = inputs["labels"]

        # Shift: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()

        # Per-token cross entropy
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        flat_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        per_token_loss = flat_loss.view(shift_labels.shape)

        # Apply weights (labels=-100 already produces 0 loss from CrossEntropyLoss ignore)
        # But we still mask explicitly for safety
        mask = (shift_labels != -100).float()
        weighted_loss = (per_token_loss * shift_weights * mask).sum()
        normalizer = (shift_weights * mask).sum()

        loss = weighted_loss / normalizer.clamp(min=1.0)

        return (loss, outputs) if return_outputs else loss

train_args = TrainingArguments(
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
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    remove_unused_columns=False,
)

trainer = WeightedLossTrainer(
    model=model,
    train_dataset=hf_ds,
    processing_class=tokenizer,
    data_collator=weighted_data_collator,
    args=train_args,
)

print(f"=== Weighted-Loss Self-Distilled CoT SFT ===")
print(f"  Samples: {len(hf_ds)}, Epochs: {EPOCHS}, LR: {LR}")
print(f"  Think weight: {THINK_WEIGHT}, Answer weight: {ANSWER_WEIGHT}")
print(f"  Steps: {len(hf_ds) // GRAD_ACCUM}")
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
    src = ''.join(c['source'])[:80]
    print(f"  {i:2d} [{c['cell_type']:8s}] {src}")
