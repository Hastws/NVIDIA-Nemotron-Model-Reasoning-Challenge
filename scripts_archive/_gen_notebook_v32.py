#!/usr/bin/env python3
"""Generate v32 training notebook (.ipynb) programmatically.
Uses list-of-strings approach to avoid triple-quote nesting issues.
"""
import json
import os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}

cells = []

# Cell 1: Title
cells.append(md([
    "# v32: Full-Scale Verified Training (9500 samples)\n",
    "\n",
    "**Strategy**: Answer-only SFT with all 9500 verified training samples.\n",
    "- Data: `sft_cot_v2_hybrid.csv` - all 6 types, naturally balanced (~1576/type)\n",
    "- Format: Same as E1 (0.68 best) - `enable_thinking=True`, answer = `\\boxed{...}`\n",
    "- Loss: Standard full-text loss (proven better than boxed-only by +0.02)\n",
    "- LoRA: rank=32, alpha=16, all-linear, dropout=0\n",
    "- LR: 1e-4 (halved from E1 due to 15x more data)"
]))

# Cell 2: pip install
cells.append(code([
    "!pip install -q --no-index --find-links /kaggle/input/datasets/dennisfong/"
    "nvidia-nemotron-offline-packages/offline_packages datasets trl --ignore-installed"
]))

# Cell 3: import check
cells.append(code([
    "import datasets, trl\n",
    "print(f'datasets: {datasets.__version__} | trl: {trl.__version__}')"
]))

# Cell 4: Imports + Triton + Config
c4 = []
c4.append("import os, sys, stat, shutil, gc, zipfile\n")
c4.append("import polars as pl\n")
c4.append("import torch\n")
c4.append("import torch.nn.functional as F\n")
c4.append("import kagglehub\n")
c4.append("from datasets import Dataset\n")
c4.append("from transformers import AutoModelForCausalLM, AutoTokenizer\n")
c4.append("from peft import LoraConfig, get_peft_model, TaskType\n")
c4.append("from trl import SFTTrainer, SFTConfig\n")
c4.append("\n")
c4.append("# === Kaggle / Triton Environment Fixes ===\n")
c4.append("def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,\n")
c4.append("                     group_size=None, norm_before_gate=True, upcast=True):\n")
c4.append("    dtype = x.dtype\n")
c4.append("    if upcast:\n")
c4.append("        x = x.float()\n")
c4.append("    variance = x.pow(2).mean(-1, keepdim=True)\n")
c4.append("    x_normed = x * torch.rsqrt(variance + eps)\n")
c4.append("    out = x_normed * weight.float()\n")
c4.append("    if bias is not None:\n")
c4.append("        out = out + bias.float()\n")
c4.append("    if z is not None:\n")
c4.append("        out = out * F.silu(z.float())\n")
c4.append("    return out.to(dtype)\n")
c4.append("\n")
c4.append("for name, mod in list(sys.modules.items()):\n")
c4.append("    if hasattr(mod, 'rmsnorm_fn'):\n")
c4.append("        mod.rmsnorm_fn = _pure_rmsnorm_fn\n")
c4.append("\n")
c4.append('src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script')
c4.append('/triton/backends/nvidia/bin/ptxas-blackwell"\n')
c4.append('dst = "/tmp/ptxas-blackwell"\n')
c4.append("if os.path.exists(src):\n")
c4.append("    shutil.copy2(src, dst)\n")
c4.append("    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)\n")
c4.append("    import triton.backends.nvidia as nv_backend\n")
c4.append('    src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")\n')
c4.append('    dst_bin = "/tmp/triton_nvidia_bin"\n')
c4.append("    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)\n")
c4.append("    for f in os.listdir(dst_bin):\n")
c4.append("        fp = os.path.join(dst_bin, f)\n")
c4.append("        if os.path.isfile(fp):\n")
c4.append("            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)\n")
c4.append('    nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")\n')
c4.append('    os.environ["TRITON_PTXAS_PATH"] = dst\n')
c4.append("\n")
c4.append("# ================================================================\n")
c4.append("#   v32 EXPERIMENT CONFIGURATION\n")
c4.append("# ================================================================\n")
c4.append('#   "cot_v2_hybrid" = 9500 samples (full, balanced)\n')
c4.append('#   "original"      = 600 random (E1 baseline)\n')
c4.append("# ================================================================\n")
c4.append('DATA_SOURCE = "cot_v2_hybrid"\n')
c4.append("SUBSAMPLE_SIZE = 600\n")
c4.append("\n")
c4.append("LORA_RANK = 32\n")
c4.append("LORA_ALPHA = 16        # scale = alpha/rank = 0.5 (E1 proven)\n")
c4.append("LORA_DROPOUT = 0.0     # 0 for large data (1 epoch = no overfitting risk)\n")
c4.append("MAX_SEQ_LEN = 1024\n")
c4.append("NUM_EPOCHS = 1\n")
c4.append("GRAD_ACCUM = 4\n")
c4.append("LR = 1e-4              # halved from E1's 2e-4 due to 15x more data\n")
c4.append("\n")
c4.append('OUTPUT_DIR = "/kaggle/working/adapter"\n')
c4.append("os.makedirs(OUTPUT_DIR, exist_ok=True)\n")
c4.append("print(f'Config: data={DATA_SOURCE}, lr={LR}, epochs={NUM_EPOCHS}, '\n")
c4.append("      f'rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}')")
cells.append(code(c4))

# Cell 5: Data header
cells.append(md(["## Data Loading & Formatting"]))

# Cell 6: Data loading + formatting
c6 = []
c6.append("MODEL_PATH = kagglehub.model_download(\n")
c6.append('    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")\n')
c6.append("\n")
c6.append("COMP_DATA = '/kaggle/input/nvidia-nemotron-3-reasoning-challenge'\n")
c6.append("COT_DATA = '/kaggle/input/prog-cot-training-data'\n")
c6.append("\n")
c6.append('if DATA_SOURCE == "cot_v2_hybrid":\n')
c6.append("    train_df = pl.read_csv(f'{COT_DATA}/sft_cot_v2_hybrid.csv')\n")
c6.append('elif DATA_SOURCE == "original":\n')
c6.append("    train_df = pl.read_csv(f'{COMP_DATA}/train.csv')\n")
c6.append("    train_df = train_df.sample(n=min(SUBSAMPLE_SIZE, len(train_df)), seed=42)\n")
c6.append("else:\n")
c6.append("    raise ValueError(f'Unknown DATA_SOURCE: {DATA_SOURCE}')\n")
c6.append("\n")
c6.append("print(f'Data: {DATA_SOURCE} | Samples: {len(train_df)}')\n")
c6.append("if 'type' in train_df.columns:\n")
c6.append("    print('Type distribution:')\n")
c6.append("    for row in train_df['type'].value_counts().sort('type').iter_rows():\n")
c6.append("        print(f'  {row[0]}: {row[1]}')\n")
c6.append("\n")
c6.append("hf_dataset = Dataset.from_pandas(train_df.to_pandas())\n")
c6.append("\n")
c6.append("tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)\n")
c6.append("if tokenizer.pad_token is None:\n")
c6.append("    tokenizer.pad_token = tokenizer.eos_token\n")
c6.append("\n")
c6.append("# Prompt suffix (aligned with evaluation metric)\n")
c6.append("SUFFIX = '\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`'\n")
c6.append("\n")
c6.append("def build_training_text(example):\n")
c6.append("    user_msg = example['prompt'] + SUFFIX\n")
c6.append("    assistant_msg = f'\\\\boxed{{{example[\"answer\"]}}}'\n")
c6.append("    messages = [\n")
c6.append('        {"role": "user", "content": user_msg},\n')
c6.append('        {"role": "assistant", "content": assistant_msg},\n')
c6.append("    ]\n")
c6.append("    text = tokenizer.apply_chat_template(\n")
c6.append("        messages, tokenize=False, add_generation_prompt=False,\n")
c6.append("        enable_thinking=True,\n")
c6.append("    )\n")
c6.append("    return {'text': text}\n")
c6.append("\n")
c6.append("# Template verification\n")
c6.append("_sample = build_training_text({'prompt': 'What is 2+2?', 'answer': '4'})\n")
c6.append("print('Template preview:')\n")
c6.append("print(_sample['text'])\n")
c6.append("_ids = tokenizer.encode(_sample['text'])\n")
c6.append("print(f'Template tokens: {len(_ids)}')\n")
c6.append("\n")
c6.append("# Apply to dataset\n")
c6.append("hf_dataset = hf_dataset.map(\n")
c6.append("    build_training_text,\n")
c6.append("    remove_columns=hf_dataset.column_names,\n")
c6.append(")\n")
c6.append("print(f'Dataset ready: {len(hf_dataset)} examples')\n")
c6.append("\n")
c6.append("# Token length distribution\n")
c6.append("lengths = [len(tokenizer.encode(ex['text'])) for ex in hf_dataset]\n")
c6.append("import numpy as np\n")
c6.append("print(f'Token lengths: mean={np.mean(lengths):.0f}, max={max(lengths)}, '\n")
c6.append("      f'p95={np.percentile(lengths, 95):.0f}, p99={np.percentile(lengths, 99):.0f}')\n")
c6.append("over = sum(1 for l in lengths if l > MAX_SEQ_LEN)\n")
c6.append("if over > 0:\n")
c6.append("    print(f'WARNING: {over}/{len(lengths)} exceed MAX_SEQ_LEN={MAX_SEQ_LEN}!')\n")
c6.append("else:\n")
c6.append("    print(f'All {len(lengths)} examples fit within MAX_SEQ_LEN={MAX_SEQ_LEN}')")
cells.append(code(c6))

# Cell 7: Model header
cells.append(md(["## Model Loading & LoRA Configuration"]))

# Cell 8: Model + LoRA
c8 = []
c8.append("from unittest.mock import MagicMock\n")
c8.append("_mock_modules = [\n")
c8.append('    "cutlass", "cutlass.cute", "cutlass.utils",\n')
c8.append('    "mamba_ssm.ops.cute", "mamba_ssm.ops.cute.mamba3",\n')
c8.append('    "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",\n')
c8.append('    "mamba_ssm.ops.tilelang", "mamba_ssm.ops.tilelang.mamba3",\n')
c8.append('    "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo",\n')
c8.append("]\n")
c8.append("for mod_name in _mock_modules:\n")
c8.append("    if mod_name not in sys.modules:\n")
c8.append("        sys.modules[mod_name] = MagicMock()\n")
c8.append("\n")
c8.append("model = AutoModelForCausalLM.from_pretrained(\n")
c8.append("    MODEL_PATH, device_map='auto', trust_remote_code=True, dtype=torch.bfloat16\n")
c8.append(")\n")
c8.append("print(f'Model loaded. Vocab size: {len(tokenizer)}')\n")
c8.append("\n")
c8.append("for name, mod in sys.modules.items():\n")
c8.append("    if 'modeling_nemotron_h' in name:\n")
c8.append("        mod.is_fast_path_available = False\n")
c8.append("        print(f'Patched {name}: is_fast_path_available = False')\n")
c8.append("\n")
c8.append("# === LoRA Layer Analysis (for team review) ===\n")
c8.append("print('\\n' + '='*60)\n")
c8.append("print('MODEL ARCHITECTURE - LoRA TARGET ANALYSIS')\n")
c8.append("print('='*60)\n")
c8.append("from collections import Counter\n")
c8.append("layer_types = Counter()\n")
c8.append("total_params = 0\n")
c8.append("for name, param in model.named_parameters():\n")
c8.append("    total_params += param.numel()\n")
c8.append("    parts = name.split('.')\n")
c8.append("    for i, p in enumerate(parts):\n")
c8.append("        if p in ('weight', 'bias'):\n")
c8.append("            key = parts[i-1] if i > 0 else name\n")
c8.append("            layer_types[key] += param.numel()\n")
c8.append("            break\n")
c8.append("\n")
c8.append("print(f'\\nTotal parameters: {total_params:,}')\n")
c8.append("print('\\nParameter distribution by layer type:')\n")
c8.append("for lt, count in layer_types.most_common(15):\n")
c8.append("    pct = count / total_params * 100\n")
c8.append("    print(f'  {lt:30s}: {count:>12,} ({pct:5.1f}%)')\n")
c8.append("\n")
c8.append("# Apply LoRA\n")
c8.append("lora_config = LoraConfig(\n")
c8.append("    r=LORA_RANK,\n")
c8.append("    lora_alpha=LORA_ALPHA,\n")
c8.append("    target_modules='all-linear',\n")
c8.append("    lora_dropout=LORA_DROPOUT,\n")
c8.append("    bias='none',\n")
c8.append("    task_type=TaskType.CAUSAL_LM,\n")
c8.append(")\n")
c8.append("model = get_peft_model(model, lora_config)\n")
c8.append("model.print_trainable_parameters()\n")
c8.append("\n")
c8.append("# List LoRA'd modules\n")
c8.append("lora_modules = set()\n")
c8.append("for name, _ in model.named_parameters():\n")
c8.append("    if 'lora_' in name:\n")
c8.append("        parts = name.split('.')\n")
c8.append("        for i, p in enumerate(parts):\n")
c8.append("            if p.startswith('lora_'):\n")
c8.append("                lora_modules.add(parts[i-1])\n")
c8.append("                break\n")
c8.append("print(f'\\nLoRA on {len(lora_modules)} module types: {sorted(lora_modules)}')")
cells.append(code(c8))

# Cell 9: Training header
cells.append(md(["## Training"]))

# Cell 10: Training
c10 = []
c10.append("import triton.backends.nvidia.compiler as nv_compiler\n")
c10.append("os.environ['TRITON_PTXAS_BLACKWELL_PATH'] = '/tmp/ptxas-blackwell'\n")
c10.append("nv_compiler.get_ptxas_version = lambda arch: '12.0'\n")
c10.append("\n")
c10.append("training_args = SFTConfig(\n")
c10.append("    output_dir=OUTPUT_DIR,\n")
c10.append("    per_device_train_batch_size=1,\n")
c10.append("    gradient_accumulation_steps=GRAD_ACCUM,\n")
c10.append("    num_train_epochs=NUM_EPOCHS,\n")
c10.append("    learning_rate=LR,\n")
c10.append("    logging_steps=5,\n")
c10.append("    bf16=True,\n")
c10.append("    max_grad_norm=1.0,\n")
c10.append("    optim='adamw_torch',\n")
c10.append("    lr_scheduler_type='cosine',\n")
c10.append("    warmup_ratio=0.1,\n")
c10.append("    save_strategy='no',\n")
c10.append("    report_to='none',\n")
c10.append("    dataset_text_field='text',\n")
c10.append("    max_length=MAX_SEQ_LEN,\n")
c10.append("    packing=False,\n")
c10.append("    gradient_checkpointing=True,\n")
c10.append("    gradient_checkpointing_kwargs={'use_reentrant': True},\n")
c10.append(")\n")
c10.append("\n")
c10.append("trainer = SFTTrainer(\n")
c10.append("    model=model,\n")
c10.append("    train_dataset=hf_dataset,\n")
c10.append("    processing_class=tokenizer,\n")
c10.append("    args=training_args,\n")
c10.append(")\n")
c10.append("\n")
c10.append("total_steps = len(hf_dataset) // GRAD_ACCUM * NUM_EPOCHS\n")
c10.append("print(f'Training: {len(hf_dataset)} x {NUM_EPOCHS} epoch(s)')\n")
c10.append("print(f'Effective batch: {1 * GRAD_ACCUM} | Steps: ~{total_steps}')\n")
c10.append("print(f'LR: {LR} | Max seq: {MAX_SEQ_LEN}')\n")
c10.append("print('Starting training...')\n")
c10.append("trainer.train()")
cells.append(code(c10))

# Cell 11: Save header
cells.append(md(["## Save & Package Submission"]))

# Cell 12: Save
c12 = []
c12.append("trainer.model.save_pretrained(OUTPUT_DIR)\n")
c12.append("print(f'Adapter saved to {OUTPUT_DIR}:')\n")
c12.append("for f in os.listdir(OUTPUT_DIR):\n")
c12.append("    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))\n")
c12.append("    print(f'  {f} ({size/1024:.1f} KB)')\n")
c12.append("\n")
c12.append("zip_path = '/kaggle/working/submission.zip'\n")
c12.append("with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:\n")
c12.append("    for fname in os.listdir(OUTPUT_DIR):\n")
c12.append("        fpath = os.path.join(OUTPUT_DIR, fname)\n")
c12.append("        zf.write(fpath, fname)\n")
c12.append("\n")
c12.append("print(f'Created {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)')\n")
c12.append("\n")
c12.append("with zipfile.ZipFile(zip_path, 'r') as zf:\n")
c12.append("    names = zf.namelist()\n")
c12.append("    print(f'Contents: {names}')\n")
c12.append("    assert 'adapter_config.json' in names\n")
c12.append("    assert 'adapter_model.safetensors' in names\n")
c12.append("    print('submission.zip is valid and ready!')")
cells.append(code(c12))

# === Assemble notebook ===
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, '..', 'nvidia-nemotron-sfttrainer-v32.ipynb')
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

ct = sum(1 for c in cells if c['cell_type'] == 'code')
mt = sum(1 for c in cells if c['cell_type'] == 'markdown')
print(f"Generated: {out_path}")
print(f"  {len(cells)} cells ({ct} code, {mt} markdown)")
