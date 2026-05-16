#!/usr/bin/env python3
"""Generate v33 notebook: Pure E1 replica (0.68 reproduction attempt).
Exact same config as the original E1 SFT from kaggle_scripts/sft_old/sfttrainer-training.ipynb.
Only difference: no GRPO phase (pure SFT only).
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
    "# v33: Pure E1 Replica (target: reproduce 0.68)\n",
    "\n",
    "**Exact E1 config** from `kaggle_scripts/sft_old/sfttrainer-training.ipynb`:\n",
    "- Data: train.csv, stratified 100/type = 600 samples, seed=42\n",
    "- SUFFIX: Official METRIC_SUFFIX (same as E1)\n",
    "- LR: 2e-4, dropout: 0.05, 1 epoch, seq_len=1024\n",
    "- LoRA: rank=32, alpha=16, all-linear\n",
    "- Loss: Standard full-text (dataset_text_field)\n",
    "- NO GRPO (pure SFT only, unlike E1 which had GRPO after but saved SFT as fallback)"
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

# Cell 4: Imports + Triton + Config (EXACT E1)
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
c4.append("src = '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell'\n")
c4.append("dst = '/tmp/ptxas-blackwell'\n")
c4.append("if os.path.exists(src):\n")
c4.append("    shutil.copy2(src, dst)\n")
c4.append("    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)\n")
c4.append("    import triton.backends.nvidia as nv_backend\n")
c4.append("    src_bin = os.path.join(os.path.dirname(nv_backend.__file__), 'bin')\n")
c4.append("    dst_bin = '/tmp/triton_nvidia_bin'\n")
c4.append("    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)\n")
c4.append("    for f in os.listdir(dst_bin):\n")
c4.append("        fp = os.path.join(dst_bin, f)\n")
c4.append("        if os.path.isfile(fp):\n")
c4.append("            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)\n")
c4.append("    nv_backend.__file__ = os.path.join(dst_bin, '..', '__init__.py')\n")
c4.append("    os.environ['TRITON_PTXAS_PATH'] = dst\n")
c4.append("\n")
c4.append("# ================================================================\n")
c4.append("#   v33 = EXACT E1 REPLICA (pure SFT, no GRPO)\n")
c4.append("# ================================================================\n")
c4.append("SFT_SAMPLES_PER_TYPE = 100  # 6 * 100 = 600 (E1 exact)\n")
c4.append("LORA_RANK = 32\n")
c4.append("LORA_ALPHA = 16\n")
c4.append("LORA_DROPOUT = 0.05       # E1 exact\n")
c4.append("MAX_SEQ_LEN = 1024\n")
c4.append("NUM_EPOCHS = 1\n")
c4.append("GRAD_ACCUM = 4\n")
c4.append("LR = 2e-4                 # E1 exact\n")
c4.append("\n")
c4.append("OUTPUT_DIR = '/kaggle/working/adapter'\n")
c4.append("os.makedirs(OUTPUT_DIR, exist_ok=True)\n")
c4.append("print(f'Config: samples/type={SFT_SAMPLES_PER_TYPE}, lr={LR}, epochs={NUM_EPOCHS}, '\n")
c4.append("      f'rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}')")
cells.append(code(c4))

# Cell 5: Data header
cells.append(md(["## Data Loading & Formatting (E1 exact replica)"]))

# Cell 6: Data loading + formatting (EXACT E1 logic)
c6 = []
c6.append("MODEL_PATH = kagglehub.model_download(\n")
c6.append("    'metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')\n")
c6.append("\n")
c6.append("# Load original competition data (same as E1)\n")
c6.append("train_df = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')\n")
c6.append("print(f'Total training samples: {len(train_df)}')\n")
c6.append("\n")
c6.append("# --- Type classification (E1 exact) ---\n")
c6.append("def classify_type(prompt_text):\n")
c6.append("    p = prompt_text.lower()\n")
c6.append("    if 'bit manipulation' in p or '8-bit binary' in p: return 'bit_ops'\n")
c6.append("    elif 'encrypt' in p or 'decrypt' in p: return 'cipher'\n")
c6.append("    elif 'gravitational' in p or 'falling distance' in p: return 'gravity'\n")
c6.append("    elif 'numeral system' in p: return 'numeral'\n")
c6.append("    elif 'transformation rules' in p: return 'symbol'\n")
c6.append("    elif 'unit conversion' in p or 'convert the following measurement' in p: return 'unit_conv'\n")
c6.append("    return 'unknown'\n")
c6.append("\n")
c6.append("train_df = train_df.with_columns(\n")
c6.append("    pl.col('prompt').map_elements(classify_type, return_dtype=pl.Utf8).alias('qtype')\n")
c6.append(")\n")
c6.append("\n")
c6.append("# --- Stratified sampling (E1 exact: 100/type, seed=42) ---\n")
c6.append("def stratified_sample(df, n_per_type, seed):\n")
c6.append("    dfs = []\n")
c6.append("    for qtype in df['qtype'].unique().to_list():\n")
c6.append("        subset = df.filter(pl.col('qtype') == qtype)\n")
c6.append("        n = min(n_per_type, len(subset))\n")
c6.append("        dfs.append(subset.sample(n=n, seed=seed))\n")
c6.append("    return pl.concat(dfs)\n")
c6.append("\n")
c6.append("sft_df = stratified_sample(train_df, SFT_SAMPLES_PER_TYPE, seed=42)\n")
c6.append("print(f'SFT samples: {len(sft_df)}')\n")
c6.append("for row in sft_df['qtype'].value_counts().sort('qtype').iter_rows():\n")
c6.append("    print(f'  {row[0]}: {row[1]}')\n")
c6.append("\n")
c6.append("hf_dataset = Dataset.from_pandas(sft_df.drop('qtype').to_pandas())\n")
c6.append("\n")
c6.append("tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)\n")
c6.append("if tokenizer.pad_token is None:\n")
c6.append("    tokenizer.pad_token = tokenizer.eos_token\n")
c6.append("\n")
c6.append("# Official METRIC_SUFFIX (same as E1 and evaluation)\n")
c6.append("METRIC_SUFFIX = '\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`'\n")
c6.append("\n")
c6.append("def build_sft_text(example):\n")
c6.append("    user_msg = example['prompt'] + METRIC_SUFFIX\n")
c6.append("    assistant_msg = f'\\\\boxed{{{example[\"answer\"]}}}'\n")
c6.append("    messages = [\n")
c6.append("        {'role': 'user', 'content': user_msg},\n")
c6.append("        {'role': 'assistant', 'content': assistant_msg},\n")
c6.append("    ]\n")
c6.append("    for kwargs in [{'enable_thinking': True}, {}]:\n")
c6.append("        try:\n")
c6.append("            text = tokenizer.apply_chat_template(\n")
c6.append("                messages, tokenize=False, add_generation_prompt=False, **kwargs\n")
c6.append("            )\n")
c6.append("            return {'text': text}\n")
c6.append("        except Exception:\n")
c6.append("            continue\n")
c6.append("    return {'text': f'<|im_start|>user\\n{user_msg}<|im_end|>\\n<|im_start|>assistant\\n{assistant_msg}<|im_end|>'}\n")
c6.append("\n")
c6.append("# Template verification\n")
c6.append("_sample = build_sft_text({'prompt': 'What is 2+2?', 'answer': '4'})\n")
c6.append("print('Template preview:')\n")
c6.append("print(_sample['text'])\n")
c6.append("\n")
c6.append("hf_dataset = hf_dataset.map(\n")
c6.append("    build_sft_text,\n")
c6.append("    remove_columns=hf_dataset.column_names,\n")
c6.append(")\n")
c6.append("print(f'\\nSFT dataset: {len(hf_dataset)} samples')")
cells.append(code(c6))

# Cell 7: Model header
cells.append(md(["## Model Loading & LoRA Configuration"]))

# Cell 8: Model + LoRA (E1 exact)
c8 = []
c8.append("from unittest.mock import MagicMock\n")
c8.append("_mock_modules = [\n")
c8.append("    'cutlass', 'cutlass.cute', 'cutlass.utils',\n")
c8.append("    'mamba_ssm.ops.cute', 'mamba_ssm.ops.cute.mamba3',\n")
c8.append("    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',\n")
c8.append("    'mamba_ssm.ops.tilelang', 'mamba_ssm.ops.tilelang.mamba3',\n")
c8.append("    'mamba_ssm.ops.tilelang.mamba3.mamba3_mimo',\n")
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
c8.append("for name, mod in list(sys.modules.items()):\n")
c8.append("    if hasattr(mod, 'rmsnorm_fn'):\n")
c8.append("        mod.rmsnorm_fn = _pure_rmsnorm_fn\n")
c8.append("\n")
c8.append("# LoRA (E1 exact: rank=32, alpha=16, dropout=0.05)\n")
c8.append("lora_config = LoraConfig(\n")
c8.append("    r=LORA_RANK,\n")
c8.append("    lora_alpha=LORA_ALPHA,\n")
c8.append("    target_modules='all-linear',\n")
c8.append("    lora_dropout=LORA_DROPOUT,\n")
c8.append("    bias='none',\n")
c8.append("    task_type=TaskType.CAUSAL_LM,\n")
c8.append(")\n")
c8.append("model = get_peft_model(model, lora_config)\n")
c8.append("model.print_trainable_parameters()")
cells.append(code(c8))

# Cell 9: Training header
cells.append(md(["## Training (E1 exact config)"]))

# Cell 10: Training
c10 = []
c10.append("import triton.backends.nvidia.compiler as nv_compiler\n")
c10.append("os.environ['TRITON_PTXAS_BLACKWELL_PATH'] = '/tmp/ptxas-blackwell'\n")
c10.append("nv_compiler.get_ptxas_version = lambda arch: '12.0'\n")
c10.append("\n")
c10.append("sft_args = SFTConfig(\n")
c10.append("    output_dir=OUTPUT_DIR,\n")
c10.append("    per_device_train_batch_size=1,\n")
c10.append("    gradient_accumulation_steps=GRAD_ACCUM,\n")
c10.append("    num_train_epochs=NUM_EPOCHS,\n")
c10.append("    learning_rate=LR,\n")
c10.append("    logging_steps=10,\n")
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
c10.append("    args=sft_args,\n")
c10.append(")\n")
c10.append("\n")
c10.append("total_steps = len(hf_dataset) // GRAD_ACCUM * NUM_EPOCHS\n")
c10.append("print(f'SFT: {len(hf_dataset)} samples x {NUM_EPOCHS} epoch = ~{total_steps} steps')\n")
c10.append("print(f'LR: {LR} | Max seq: {MAX_SEQ_LEN} | Dropout: {LORA_DROPOUT}')\n")
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
out_path = os.path.join(script_dir, '..', 'nvidia-nemotron-sfttrainer-v33.ipynb')
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

ct = sum(1 for c in cells if c['cell_type'] == 'code')
mt = sum(1 for c in cells if c['cell_type'] == 'markdown')
print(f"Generated: {out_path}")
print(f"  {len(cells)} cells ({ct} code, {mt} markdown)")
