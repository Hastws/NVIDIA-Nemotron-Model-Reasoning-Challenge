#!/usr/bin/env python3
"""Generate the Unsloth training notebook for Kaggle."""

import json
import sys
import os

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def code(text):
    cells.append({
        "cell_type": "code", "metadata": {},
        "source": [text],
        "outputs": [], "execution_count": None
    })


# ================================================================
# Cell 0: Title
# ================================================================
md("""# NVIDIA Nemotron Unsloth Training — V125

> **Unsloth Framework**: Uses `unsloth/Nemotron-3-Nano-30B-A3B` for training with gate_proj/x_proj LoRA on Mamba layers.
> Includes Unsloth→HF PEFT adapter conversion pipeline for submission.
>
> **Key differences from HF PEFT (V123)**:
> - Unsloth splits Mamba `in_proj` into `gate_proj` + `x_proj` → separate LoRA → more parameters on Mamba
> - Unsloth fuses MoE expert weights (w1/w2) → efficient 3D LoRA
> - `alpha=32, dropout=0.0` aligned with 0.85 solution
> - Same boxed loss weight 5x strategy""")

# ================================================================
# Cell 1: Install Unsloth
# ================================================================
code(r"""import subprocess, sys, os, glob

# --- Step 1: Install Unsloth ---
print("Installing Unsloth...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "unsloth"],
    capture_output=True, text=True, timeout=300
)
if result.returncode != 0:
    print(f"--no-deps failed, trying full: {result.stderr[-200:]}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "unsloth"],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"pip failed, trying git: {result.stderr[-200:]}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "unsloth @ git+https://github.com/unslothai/unsloth.git"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            raise RuntimeError(f"Cannot install unsloth: {result.stderr[-500:]}")
print("Unsloth installed")

# --- Step 2: Install trl + datasets from offline packages ---
offline_dirs = [
    "/kaggle/input/sft-offline-packages",
    "/kaggle/input/datasets/hastws/sft-offline-packages",
]
offline_dir = None
for d in offline_dirs:
    if os.path.isdir(d):
        whl_files = [f for f in os.listdir(d) if f.endswith('.whl')]
        if whl_files:
            offline_dir = d
            print(f"Found offline packages at: {d} ({len(whl_files)} wheels)")
            break

if offline_dir:
    whls = sorted(glob.glob(os.path.join(offline_dir, "*.whl")))
    print(f"Installing {len(whls)} wheels...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + whls,
        capture_output=True, text=True, timeout=180
    )
    if result.returncode == 0:
        print("Installed all offline wheels")
    else:
        for whl in whls:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", whl],
                capture_output=True, text=True, timeout=60
            )
            name = os.path.basename(whl).split('-')[0]
            print(f"  {'OK' if r.returncode == 0 else 'FAIL'}: {name}")

# --- Step 3: flash_attn ---
try:
    import flash_attn
    print(f"flash_attn={flash_attn.__version__}")
except ImportError:
    fa_whls = glob.glob("/kaggle/input/**/*flash_attn*.whl", recursive=True)
    if fa_whls:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", fa_whls[0]],
                       capture_output=True, text=True, timeout=120)
        print(f"Installed flash_attn from {os.path.basename(fa_whls[0])}")

# Verify
for pkg in ["trl", "datasets", "unsloth"]:
    try:
        mod = __import__(pkg)
        print(f"  {pkg}={getattr(mod, '__version__', '?')}")
    except ImportError as e:
        if pkg == "unsloth":
            raise RuntimeError(f"FATAL: unsloth not importable: {e}")
        print(f"  WARNING: {pkg}: {e}")

print("\nPackage setup complete")
""")

# ================================================================
# Cell 2: Imports & Environment
# ================================================================
code(r"""import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import stat
import shutil
import gc
import zipfile
import time
import json
import math
import re
import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import kagglehub
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"GPU memory: {mem / 1024**3:.1f} GB")

# --- Triton / Blackwell environment fixes ---
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
    print("Triton patched for Blackwell")
else:
    print("ptxas-blackwell not found, skipping Triton patch")

print("Imports and environment ready")
""")

# ================================================================
# Cell 3: Config
# ================================================================
md("""## Configuration

Key changes from V123 (HF PEFT):
- `alpha=32` (was 64) — aligned with 0.85 solution
- `dropout=0.0` (was 0.05) — no regularization
- `target_modules="all-linear"` — Unsloth string mode covers gate_proj, x_proj, experts, lm_head""")

# ================================================================
# Cell 4: Hyperparameters
# ================================================================
code(r"""# =============================================
#  HYPERPARAMETERS
# =============================================

# --- Training ---
STAGE1_LR        = 1e-4
STAGE1_EPOCHS    = 2
STAGE1_MAX_SEQ   = 2048
STAGE1_BATCH     = 1
STAGE1_GRAD_ACCUM = 4
STAGE1_PACKING   = False
STAGE1_ANSWER_ONLY = False

# --- LoRA (Unsloth) ---
LORA_RANK        = 32
LORA_ALPHA       = 32       # Aligned with 0.85 (was 64)
LORA_DROPOUT     = 0.0      # Aligned with 0.85 (was 0.05)

# --- Boxed Loss Weight ---
BOXED_LOSS_WEIGHT = 5.0

# --- Type Filter ---
TRAIN_TYPES      = []

# --- Holdout ---
HOLDOUT_ENABLED  = False
HOLDOUT_N_PER_TYPE = 10

# --- Prompt Suffix (official) ---
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EVAL_MAX_TOKENS  = 3584

print(f"Training: LR={STAGE1_LR}, epochs={STAGE1_EPOCHS}, max_seq={STAGE1_MAX_SEQ}")
print(f"Batch: {STAGE1_BATCH} x {STAGE1_GRAD_ACCUM} = {STAGE1_BATCH * STAGE1_GRAD_ACCUM}")
print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Boxed loss weight: {BOXED_LOSS_WEIGHT}")
print(f"Train types: {TRAIN_TYPES if TRAIN_TYPES else 'ALL'}")
""")

# ================================================================
# Cell 5: Data Loading
# ================================================================
md("## Data Loading")

code(r"""print("=== DATA LOADING ===")

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
COMP_DATA = '/kaggle/input/nvidia-nemotron-3-reasoning-challenge'

# --- Find training data ---
TARGET_FILE = 'sft_thinking.csv'
_COT_CANDIDATES = [
    '/kaggle/input/prog-cot-training-data',
    '/kaggle/input/datasets/hastws/prog-cot-training-data',
]
COT_DATA = None
for d in _COT_CANDIDATES:
    if os.path.isfile(os.path.join(d, TARGET_FILE)):
        COT_DATA = d
        break
if COT_DATA is None:
    import glob as _glob
    matches = _glob.glob(f'/kaggle/input/**/{TARGET_FILE}', recursive=True)
    if matches:
        COT_DATA = os.path.dirname(matches[0])
    else:
        raise FileNotFoundError(f"{TARGET_FILE} not found under /kaggle/input/")

print(f"COT_DATA = {COT_DATA}")
train_df = pl.read_csv(f'{COT_DATA}/{TARGET_FILE}')
print(f"Loaded: {len(train_df)} rows")

# Type distribution
pdf = train_df.to_pandas()
if 'type' in pdf.columns:
    print("\nType distribution:")
    for t in sorted(pdf['type'].unique()):
        print(f"  {t}: {(pdf['type'] == t).sum()}")

# --- Type inference ---
import re as _re_type
_NUM_EQ_RE = _re_type.compile(r'^(\d+)([^\d])(\d+)$')

def _is_numeric_equation(prompt):
    for line in prompt.strip().split('\n'):
        line = line.strip()
        if ' = ' in line and 'alice' not in line.lower() and 'equation' not in line.lower() \
                and 'transformation' not in line.lower() and 'determine' not in line.lower():
            lhs = line.split(' = ', 1)[0].strip()
            if _NUM_EQ_RE.match(lhs):
                return True
    return False

def _infer_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or '8-bit binary' in p:
        return 'bit_ops'
    elif 'numeral system' in p:
        return 'numeral'
    elif 'encrypt' in p or 'decrypt' in p:
        return 'cipher'
    elif 'gravitational' in p:
        return 'gravity'
    elif 'unit' in p and 'convert' in p:
        return 'unit_conv'
    elif 'symbol' in p or 'transformation rule' in p:
        return 'eq_numeric' if _is_numeric_equation(prompt) else 'eq_symbolic'
    return 'unknown'
""")

# ================================================================
# Cell 6: Format Training Text
# ================================================================
md("## Format Training Text")

code(r"""from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer vocab size: {len(tokenizer)}")

def _has_thinking(thinking):
    if thinking is None:
        return False
    if isinstance(thinking, float) and math.isnan(thinking):
        return False
    s = str(thinking).strip()
    return len(s) > 0 and s.lower() != 'nan'

def build_training_text(example):
    prompt = example["prompt"]
    answer = str(example["answer"])
    user_msg = prompt + PROMPT_SUFFIX

    if STAGE1_ANSWER_ONLY:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think></think>\\boxed{{{answer}}}<|im_end|>"
        )
    else:
        thinking = example.get("thinking", None)
        if _has_thinking(thinking):
            text = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n{str(thinking).strip()}\n</think>\n"
                f"\\boxed{{{answer}}}<|im_end|>"
            )
        else:
            text = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n<think></think>\\boxed{{{answer}}}<|im_end|>"
            )
    return {"text": text}

# --- Build dataset ---
stage1_pdf = train_df.to_pandas()

# Type-based filtering
if TRAIN_TYPES:
    if 'type' not in stage1_pdf.columns:
        stage1_pdf['_type'] = stage1_pdf['prompt'].apply(_infer_type)
        stage1_pdf = stage1_pdf[stage1_pdf['_type'].isin(TRAIN_TYPES)].reset_index(drop=True)
    else:
        stage1_pdf = stage1_pdf[stage1_pdf['type'].isin(TRAIN_TYPES)].reset_index(drop=True)
    print(f"Type filter: {TRAIN_TYPES} -> {len(stage1_pdf)} rows")

# Holdout split
holdout_df = None
if HOLDOUT_ENABLED:
    type_col = stage1_pdf['type'] if 'type' in stage1_pdf.columns else stage1_pdf['prompt'].apply(_infer_type)
    holdout_parts = []
    for t in sorted(type_col.unique()):
        t_df = stage1_pdf[type_col == t]
        holdout_parts.append(t_df.sample(n=min(HOLDOUT_N_PER_TYPE, len(t_df)), random_state=42))
    holdout_df = pd.concat(holdout_parts).reset_index(drop=True)
    holdout_df['type'] = holdout_df['prompt'].apply(_infer_type)
    stage1_pdf = stage1_pdf.drop(holdout_df.index).reset_index(drop=True)
    print(f"Holdout: {len(holdout_df)}, Training: {len(stage1_pdf)}")

hf_dataset = Dataset.from_pandas(stage1_pdf)
hf_dataset = hf_dataset.map(build_training_text, remove_columns=hf_dataset.column_names)
print(f"Dataset: {len(hf_dataset)} rows")

# Token length analysis
token_lengths = []
for i in range(len(hf_dataset)):
    toks = tokenizer(hf_dataset[i]['text'], add_special_tokens=False)
    token_lengths.append(len(toks['input_ids']))
tl = np.array(token_lengths)
print(f"Token lengths: min={tl.min()}, median={np.median(tl):.0f}, max={tl.max()}")
print(f"  >{STAGE1_MAX_SEQ}: {(tl > STAGE1_MAX_SEQ).sum()} ({(tl > STAGE1_MAX_SEQ).mean()*100:.1f}%)")
""")

# ================================================================
# Cell 7: Model Loading with Unsloth
# ================================================================
md("""## Model Loading with Unsloth

Uses `FastLanguageModel` from Unsloth. This splits Mamba's `in_proj` into `gate_proj` + `x_proj`
with separate LoRA on each — more fine-grained than HF PEFT. Also fuses MoE expert weights for 3D LoRA.""")

code(r"""from unsloth import FastLanguageModel
from unittest.mock import MagicMock

# Mock problematic CUDA modules
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

t0 = time.time()

# Load with Unsloth — downloads unsloth/Nemotron-3-Nano-30B-A3B from HuggingFace
# This provides the split gate_proj/x_proj + fused expert handling
print("Loading model with Unsloth...")
model, unsloth_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Nemotron-3-Nano-30B-A3B",
    max_seq_length=STAGE1_MAX_SEQ,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
print(f"Model loaded in {time.time()-t0:.1f}s")

# Patch: force slow path for Blackwell
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        if hasattr(mod, 'is_fast_path_available'):
            mod.is_fast_path_available = False
            print(f"Patched {name}: is_fast_path_available = False")

# Setup LoRA with Unsloth
print(f"\nLoRA: r={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules="all-linear",
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Print trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params/1e9:.2f}B")
print(f"Trainable params: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.3f}%)")

# List LoRA modules for verification
print("\n=== LoRA Module Coverage ===")
_lora_modules = set()
for name, param in model.named_parameters():
    if param.requires_grad and 'lora' in name.lower():
        parts = name.split('.')
        for i, p in enumerate(parts):
            if 'lora' in p.lower():
                module_path = '.'.join(parts[max(0,i-2):i])
                _lora_modules.add(module_path)
                break
print(f"Unique LoRA target types: {len(_lora_modules)}")
for m in sorted(_lora_modules)[:20]:
    print(f"  {m}")
if len(_lora_modules) > 20:
    print(f"  ... and {len(_lora_modules) - 20} more")
""")

# ================================================================
# Cell 8: Training
# ================================================================
md("## Training with SFTTrainer + Boxed Loss Weight")

code(r"""import triton.backends.nvidia.compiler as nv_compiler
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

eff_batch = STAGE1_BATCH * STAGE1_GRAD_ACCUM
total_steps = (len(hf_dataset) // eff_batch) * STAGE1_EPOCHS
print(f"{'='*60}")
print(f"  TRAINING: Unsloth + SFTTrainer")
print(f"  Samples: {len(hf_dataset)}, LR: {STAGE1_LR}, Epochs: {STAGE1_EPOCHS}")
print(f"  Max Seq: {STAGE1_MAX_SEQ}, Effective batch: {eff_batch}")
print(f"  Boxed loss weight: {BOXED_LOSS_WEIGHT}")
print(f"  Steps: ~{total_steps}")
print(f"{'='*60}")

stage1_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=STAGE1_BATCH,
    gradient_accumulation_steps=STAGE1_GRAD_ACCUM,
    num_train_epochs=STAGE1_EPOCHS,
    learning_rate=STAGE1_LR,
    logging_steps=10,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=STAGE1_MAX_SEQ,
    packing=STAGE1_PACKING,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)

# ── Weighted Boxed Loss ──
_compute_loss_fn = None
if BOXED_LOSS_WEIGHT > 1.0:
    _boxed_marker_ids = tokenizer.encode("\\boxed{", add_special_tokens=False)
    _boxed_weight = float(BOXED_LOSS_WEIGHT)
    print(f"[loss] \\boxed{{ marker: {len(_boxed_marker_ids)} IDs: {_boxed_marker_ids}")

    def _weighted_boxed_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_len = shift_labels.shape

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(batch_size, seq_len)

        weights = torch.ones(batch_size, seq_len, device=per_token_loss.device)
        marker = torch.tensor(_boxed_marker_ids, device=shift_labels.device)
        marker_len = len(_boxed_marker_ids)

        for bi in range(batch_size):
            last_pos = -1
            for i in range(seq_len - marker_len + 1):
                if torch.equal(shift_labels[bi, i:i+marker_len], marker):
                    last_pos = i
            if last_pos >= 0:
                weights[bi, last_pos:] = _boxed_weight

        mask = (shift_labels != -100).float()
        weighted_loss = (per_token_loss * weights * mask).sum() / (weights * mask).sum()
        return weighted_loss

    _compute_loss_fn = _weighted_boxed_loss
    print(f"[loss] Weighted boxed loss: {_boxed_weight}x after last \\boxed{{")

stage1_trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=stage1_args,
    compute_loss_func=_compute_loss_fn,
)

t0 = time.time()
stage1_result = stage1_trainer.train()
stage1_time = time.time() - t0

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"  Time: {stage1_time/60:.1f} min")
print(f"  Final loss: {stage1_result.training_loss:.4f}")
print(f"{'='*60}")
""")

# ================================================================
# Cell 9: Save Unsloth Adapter & Convert
# ================================================================
md("""## Save & Convert Unsloth → HF PEFT

1. **Key rename**: `base_model.model.model` → `base_model.model.backbone`
2. **Expert unfuse**: `experts.w1` → `experts.{i}.up_proj`, `experts.w2` → `experts.{i}.down_proj`
3. **Mamba merge**: `gate_proj` + `x_proj` → `in_proj` via SVD (rank-64 → rank-32, ~98% preserved)
4. **Config rewrite**: explicit target_modules for HF PEFT""")

code(r"""# --- Save raw Unsloth adapter ---
UNSLOTH_DIR = "/kaggle/working/unsloth_adapter"
os.makedirs(UNSLOTH_DIR, exist_ok=True)
model.save_pretrained(UNSLOTH_DIR)
print(f"Unsloth adapter saved to {UNSLOTH_DIR}")
for f in sorted(os.listdir(UNSLOTH_DIR)):
    size = os.path.getsize(os.path.join(UNSLOTH_DIR, f))
    print(f"  {f:40s} {size/1024:.1f} KB")

# --- Load metric/ model key shapes for conversion ---
from safetensors import safe_open
from safetensors.torch import save_file
import glob as _glob

model_keys = set()
for model_sf in _glob.glob(os.path.join(MODEL_PATH, "*.safetensors")):
    with safe_open(model_sf, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor_slice = f.get_slice(key)
            model_keys.add((key, tuple(tensor_slice.get_shape()), tensor_slice.get_dtype()))
print(f"Loaded {len(model_keys)} model key shapes from metric/ model")

# --- Key rename function ---
def trained_adapter_key_rename(key_name):
    return key_name.replace("base_model.model.model", "base_model.model.backbone")

# --- Convert Unsloth → HF PEFT ---
print("\n=== Converting Unsloth -> HF PEFT ===")

adapter_tensors = {}
with safe_open(os.path.join(UNSLOTH_DIR, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
    for key in f.keys():
        adapter_tensors[key] = f.get_tensor(key)
print(f"Loaded {len(adapter_tensors)} adapter tensors")

# Collect base names
base_names = set()
for key in adapter_tensors:
    base = re.sub(r"\.lora_[AB]\.weight$", "", key)
    base_names.add(base)
print(f"Found {len(base_names)} LoRA base names")

# Identify Mamba layers needing gate_proj+x_proj -> in_proj merge
mamba_merge_layers = {}
for base in base_names:
    for proj in ("gate_proj", "x_proj"):
        if f".{proj}" in base:
            layer_path = base.rsplit(f".{proj}", 1)[0]
            mamba_merge_layers.setdefault(layer_path, {})[proj] = base
mamba_merge_bases = set()
for projs in mamba_merge_layers.values():
    mamba_merge_bases.update(projs.values())
print(f"Mamba merge: {len(mamba_merge_layers)} layers")

model_key_shapes = {k: s for k, s, _ in model_keys}

# Build converted tensors
tensors = {}

for base in sorted(base_names):
    lora_A = adapter_tensors[f"{base}.lora_A.weight"]
    lora_B = adapter_tensors[f"{base}.lora_B.weight"]
    renamed = trained_adapter_key_rename(base)

    # Skip empty w3 experts
    if ".experts.w3" in base and lora_A.numel() == 0:
        continue

    # Skip gate_proj/x_proj (handled in Mamba merge below)
    if base in mamba_merge_bases:
        continue

    # Expert unfusing: w1 -> per-expert up_proj, w2 -> per-expert down_proj
    if ".experts.w1" in base or ".experts.w2" in base:
        if lora_A.shape[0] == 1:
            lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
        elif lora_B.shape[0] == 1:
            lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()

        num_experts = lora_A.shape[0]
        proj_name = "up_proj" if ".w1" in base else "down_proj"

        for i in range(num_experts):
            exp_renamed = re.sub(
                r"\.experts\.w[12]",
                f".experts.{i}.{proj_name}",
                renamed,
            )
            tensors[f"{exp_renamed}.lora_A.weight"] = lora_A[i].contiguous()
            tensors[f"{exp_renamed}.lora_B.weight"] = lora_B[i].contiguous()
        continue

    # Direct rename
    tensors[f"{renamed}.lora_A.weight"] = lora_A
    tensors[f"{renamed}.lora_B.weight"] = lora_B

# Mamba: gate_proj + x_proj -> in_proj via SVD
print("\nMamba gate_proj + x_proj -> in_proj SVD merge:")
for layer_path, projs in sorted(mamba_merge_layers.items()):
    renamed_layer = trained_adapter_key_rename(layer_path)
    in_proj_base = f"{renamed_layer}.in_proj"

    model_in_proj_key = (
        renamed_layer.replace("base_model.model.", "") + ".in_proj.weight"
    )
    in_proj_dim = model_key_shapes[model_in_proj_key][0]

    gate_A = adapter_tensors[f"{projs['gate_proj']}.lora_A.weight"].float()
    gate_B = adapter_tensors[f"{projs['gate_proj']}.lora_B.weight"].float()
    x_A = adapter_tensors[f"{projs['x_proj']}.lora_A.weight"].float()
    x_B = adapter_tensors[f"{projs['x_proj']}.lora_B.weight"].float()
    rank = gate_A.shape[0]

    A_cat = torch.cat([gate_A, x_A], dim=0)
    B_block = torch.zeros(in_proj_dim, 2 * rank)
    B_block[:gate_B.shape[0], :rank] = gate_B
    B_block[gate_B.shape[0]:gate_B.shape[0] + x_B.shape[0], rank:] = x_B

    Q_B, R_B = torch.linalg.qr(B_block)
    Q_A, R_A = torch.linalg.qr(A_cat.T)
    core = R_B @ R_A.T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)

    k = rank
    new_B = (Q_B @ U[:, :k]) * S[:k].unsqueeze(0)
    new_A = Vh[:k, :] @ Q_A.T

    kept = S[:k].sum().item()
    total = S.sum().item()
    print(f"  {layer_path}: SVD {kept:.2f}/{total:.2f} ({kept/total*100:.1f}%)")

    tensors[f"{in_proj_base}.lora_A.weight"] = new_A
    tensors[f"{in_proj_base}.lora_B.weight"] = new_B

print(f"\nConverted {len(adapter_tensors)} -> {len(tensors)} tensors")
save_file(tensors, os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))
print(f"Saved to {OUTPUT_DIR}/adapter_model.safetensors")
""")

# ================================================================
# Cell 10: Write config & Package
# ================================================================
code(r"""# --- Write HF PEFT adapter_config.json ---
adapter_config = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": str(MODEL_PATH),
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": LORA_RANK,
    "rank_pattern": {},
    "revision": None,
    "target_modules": [
        "k_proj", "o_proj", "in_proj", "q_proj",
        "up_proj", "v_proj", "down_proj", "out_proj", "lm_head",
    ],
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False,
}

config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
with open(config_path, "w") as f:
    json.dump(adapter_config, f, indent=2)
print(f"Wrote adapter_config.json")
print(f"  r={adapter_config['r']}, alpha={adapter_config['lora_alpha']}")
print(f"  target_modules={adapter_config['target_modules']}")

# --- Package submission.zip ---
zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in ["adapter_model.safetensors", "adapter_config.json"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)
            print(f"  Added {fname} ({os.path.getsize(fpath)/1024/1024:.1f} MB)")

print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")

# Verify
with zipfile.ZipFile(zip_path, 'r') as zf:
    names = zf.namelist()
    assert "adapter_config.json" in names
    assert "adapter_model.safetensors" in names
    print(f"Contents: {names}")

print("submission.zip ready!")
""")

# ================================================================
# Cell 11: Holdout Evaluation
# ================================================================
md("## Holdout Evaluation (Optional)")

# Read holdout cell from existing notebook
nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'nvidia-nemotron-sfttrainer-training.ipynb')
try:
    with open(nb_path) as f:
        existing_nb = json.load(f)
    holdout_src = ''.join(existing_nb['cells'][18]['source'])
    code(holdout_src)
except Exception as e:
    print(f"Warning: Could not read holdout cell: {e}")
    code("# Holdout evaluation not available\nprint('Holdout evaluation skipped')")

# ================================================================
# Build notebook
# ================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
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
    "cells": cells
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'nvidia-nemotron-unsloth-training.ipynb')
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Created {out_path}")
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    ct = c['cell_type']
    src = ''.join(c['source'])
    preview = src[:70].replace('\n', ' | ')
    print(f"  Cell {i:2d} [{ct:8s}] {len(src):5d} chars: {preview}")
