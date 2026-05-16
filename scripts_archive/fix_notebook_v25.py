#!/usr/bin/env python3
"""Fix notebook for v25: Two-stage curriculum training.

Strategy:
- Stage 1: All 7741 verified samples, answer-only, LR=1e-5, 1 epoch
  → Gentle pattern learning across all 5 solvable types
- Stage 2: E1's 600 balanced samples, answer-only, LR=2e-4, 1 epoch
  → Format refinement (same as best E1 config)

This tests the "先学规律，再学格式" hypothesis.
"""
import json
import copy

NB_PATH = "nvidia-nemotron-sfttrainer-training.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

# ============================================================
# Fix 1: Repair broken Cell 6 (build_training_text function)
# Lines 77-79 have return mixed into apply_chat_template args
# ============================================================
cell6_src = nb['cells'][6]['source']
fixed_lines = []
skip_next = False
for i, line in enumerate(cell6_src):
    if skip_next:
        skip_next = False
        continue
    if 'messages, tokenize=False, add_generation_prompt=False,    return {"text": text}' in line:
        fixed_lines.append('        messages, tokenize=False, add_generation_prompt=False,\n')
    elif '        enable_thinking=True    )' in line:
        fixed_lines.append('        enable_thinking=True\n')
        fixed_lines.append('    )\n')
        fixed_lines.append('    return {"text": text}')
    else:
        fixed_lines.append(line)
nb['cells'][6]['source'] = fixed_lines

# Verify the fix
joined = ''.join(fixed_lines)
assert 'apply_chat_template(' in joined
assert 'return {"text": text}' in joined
assert 'enable_thinking=True' in joined
print("✓ Cell 6 syntax fixed")

# ============================================================
# Fix 2: Update config cell (Cell 4) - add two-stage params
# ============================================================
cell4_src = ''.join(nb['cells'][4]['source'])

# Replace DATA_SOURCE line and add two-stage config
old_config = 'DATA_SOURCE = "original"'
new_config = '''# --- Training Mode ---
# "standard"   : single-stage training (E1 baseline)
# "two_stage"  : Stage 1 (pattern learning) → Stage 2 (format polishing)
TRAINING_MODE = "two_stage"

# Stage 1 config (pattern learning with all verified data)
STAGE1_LR = 1e-5        # Very conservative - preserve base capabilities
STAGE1_EPOCHS = 1
STAGE1_MAX_SEQ = 1024

# Stage 2 config (format refinement with E1's curated 600)
STAGE2_LR = 2e-4        # Standard E1 learning rate
STAGE2_EPOCHS = 1

DATA_SOURCE = "original"'''

cell4_src = cell4_src.replace(old_config, new_config)
nb['cells'][4]['source'] = [cell4_src]
print("✓ Cell 4 config updated with two-stage params")

# ============================================================
# Fix 3: Update Cell 6 - also load Stage 1 data
# Add stage1 data loading after main data loading
# ============================================================
cell6_joined = ''.join(nb['cells'][6]['source'])

# Add Stage 1 data loading after the main data loading block
old_print = 'print(f"Data source: {DATA_SOURCE}, samples: {len(train_df)}")'
new_print = '''print(f"Data source: {DATA_SOURCE}, samples: {len(train_df)}")

# Load Stage 1 data (all verified samples, answer-only) if two-stage
if TRAINING_MODE == "two_stage":
    stage1_df = pl.read_csv(f'{COT_DATA}/sft_full_cot.csv')
    # Drop thinking column for answer-only training
    if 'thinking' in stage1_df.columns:
        stage1_df = stage1_df.drop('thinking')
    print(f"Stage 1 data: {len(stage1_df)} verified samples (answer-only)")'''

cell6_joined = cell6_joined.replace(old_print, new_print)
nb['cells'][6]['source'] = [cell6_joined]
print("✓ Cell 6 Stage 1 data loading added")

# ============================================================
# Fix 4: Add Stage 1 dataset mapping after Cell 7
# ============================================================
stage1_map_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "stage1_data_prep",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Prepare Stage 1 dataset (if two-stage mode)\n",
        "if TRAINING_MODE == \"two_stage\":\n",
        "    stage1_hf = Dataset.from_pandas(stage1_df.to_pandas())\n",
        "    stage1_hf = stage1_hf.map(\n",
        "        build_training_text,\n",
        "        remove_columns=stage1_hf.column_names\n",
        "    )\n",
        "    print(f\"Stage 1 dataset: {len(stage1_hf)} samples\")\n",
        "    print(f\"Stage 1 example:\\n{stage1_hf[0]['text'][:300]}\")\n",
    ]
}

# Insert after Cell 7 (the hf_dataset.map cell)
nb['cells'].insert(8, stage1_map_cell)
print("✓ Stage 1 dataset mapping cell inserted after Cell 7")

# ============================================================
# Fix 5: Replace training cell with two-stage logic
# Cell 12 now (was 11, shifted by insertion)
# ============================================================
training_cell_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'SFTConfig(' in src and 'trainer = SFTTrainer(' in src:
        training_cell_idx = i
        break

assert training_cell_idx is not None, "Could not find training cell!"
print(f"✓ Found training cell at index {training_cell_idx}")

new_training_src = [
    "import os\n",
    "import triton.backends.nvidia.compiler as nv_compiler\n",
    "\n",
    "# Tell Triton's environment parser where the writable Blackwell binary is\n",
    'os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"\n',
    'nv_compiler.get_ptxas_version = lambda arch: "12.0"\n',
    "\n",
    "if TRAINING_MODE == \"two_stage\":\n",
    "    # ==========================================\n",
    "    # Stage 1: Pattern learning (all verified data, low LR)\n",
    "    # ==========================================\n",
    "    print(\"=\"*60)\n",
    "    print(\"STAGE 1: Pattern learning\")\n",
    "    print(f\"  Data: {len(stage1_hf)} verified samples\")\n",
    "    print(f\"  LR: {STAGE1_LR}, Epochs: {STAGE1_EPOCHS}\")\n",
    "    print(\"=\"*60)\n",
    "\n",
    "    stage1_args = SFTConfig(\n",
    "        output_dir=OUTPUT_DIR + \"/stage1\",\n",
    "        per_device_train_batch_size=1,\n",
    "        gradient_accumulation_steps=GRAD_ACCUM,\n",
    "        num_train_epochs=STAGE1_EPOCHS,\n",
    "        learning_rate=STAGE1_LR,\n",
    "        logging_steps=10,\n",
    "        bf16=True,\n",
    "        max_grad_norm=1.0,\n",
    "        optim=\"adamw_torch\",\n",
    "        lr_scheduler_type=\"cosine\",\n",
    "        warmup_ratio=0.1,\n",
    "        save_strategy=\"no\",\n",
    "        report_to=\"none\",\n",
    "        dataset_text_field=\"text\",\n",
    "        max_length=STAGE1_MAX_SEQ,\n",
    "        packing=False,\n",
    "        gradient_checkpointing=True,\n",
    "        gradient_checkpointing_kwargs={\"use_reentrant\": True},\n",
    "    )\n",
    "\n",
    "    stage1_trainer = SFTTrainer(\n",
    "        model=model,\n",
    "        train_dataset=stage1_hf,\n",
    "        processing_class=tokenizer,\n",
    "        args=stage1_args\n",
    "    )\n",
    "\n",
    "    stage1_trainer.train()\n",
    "    print(\"Stage 1 complete.\")\n",
    "\n",
    "    # Free Stage 1 trainer memory\n",
    "    del stage1_trainer\n",
    "    import gc; gc.collect()\n",
    "    torch.cuda.empty_cache()\n",
    "\n",
    "    # ==========================================\n",
    "    # Stage 2: Format refinement (E1 data, normal LR)\n",
    "    # ==========================================\n",
    "    print(\"=\"*60)\n",
    "    print(\"STAGE 2: Format refinement\")\n",
    "    print(f\"  Data: {len(hf_dataset)} curated samples\")\n",
    "    print(f\"  LR: {STAGE2_LR}, Epochs: {STAGE2_EPOCHS}\")\n",
    "    print(\"=\"*60)\n",
    "\n",
    "    stage2_args = SFTConfig(\n",
    "        output_dir=OUTPUT_DIR,\n",
    "        per_device_train_batch_size=1,\n",
    "        gradient_accumulation_steps=GRAD_ACCUM,\n",
    "        num_train_epochs=STAGE2_EPOCHS,\n",
    "        learning_rate=STAGE2_LR,\n",
    "        logging_steps=5,\n",
    "        bf16=True,\n",
    "        max_grad_norm=1.0,\n",
    "        optim=\"adamw_torch\",\n",
    "        lr_scheduler_type=\"cosine\",\n",
    "        warmup_ratio=0.1,\n",
    "        save_strategy=\"no\",\n",
    "        report_to=\"none\",\n",
    "        dataset_text_field=\"text\",\n",
    "        max_length=MAX_SEQ_LEN,\n",
    "        packing=False,\n",
    "        gradient_checkpointing=True,\n",
    "        gradient_checkpointing_kwargs={\"use_reentrant\": True},\n",
    "    )\n",
    "\n",
    "    trainer = SFTTrainer(\n",
    "        model=model,\n",
    "        train_dataset=hf_dataset,\n",
    "        processing_class=tokenizer,\n",
    "        args=stage2_args\n",
    "    )\n",
    "\n",
    "    trainer.train()\n",
    "    print(\"Stage 2 complete. Two-stage training finished!\")\n",
    "\n",
    "else:\n",
    "    # ==========================================\n",
    "    # Standard single-stage training (E1 baseline)\n",
    "    # ==========================================\n",
    "    training_args = SFTConfig(\n",
    "        output_dir=OUTPUT_DIR,\n",
    "        per_device_train_batch_size=1,\n",
    "        gradient_accumulation_steps=GRAD_ACCUM,\n",
    "        num_train_epochs=NUM_EPOCHS,\n",
    "        learning_rate=LR,\n",
    "        logging_steps=5,\n",
    "        bf16=True,\n",
    "        max_grad_norm=1.0,\n",
    "        optim=\"adamw_torch\",\n",
    "        lr_scheduler_type=\"cosine\",\n",
    "        warmup_ratio=0.1,\n",
    "        save_strategy=\"no\",\n",
    "        report_to=\"none\",\n",
    "        dataset_text_field=\"text\",\n",
    "        max_length=MAX_SEQ_LEN,\n",
    "        packing=False,\n",
    "        gradient_checkpointing=True,\n",
    "        gradient_checkpointing_kwargs={\"use_reentrant\": True},\n",
    "    )\n",
    "\n",
    "    trainer = SFTTrainer(\n",
    "        model=model,\n",
    "        train_dataset=hf_dataset,\n",
    "        processing_class=tokenizer,\n",
    "        args=training_args\n",
    "    )\n",
    "\n",
    "    print(\"Starting training...\")\n",
    "    trainer.train()\n",
]

nb['cells'][training_cell_idx]['source'] = new_training_src
print(f"✓ Training cell {training_cell_idx} replaced with two-stage logic")

# ============================================================
# Fix 6: Update save cell - handle two-stage case
# ============================================================
save_cell_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'trainer.model.save_pretrained' in src:
        save_cell_idx = i
        break

if save_cell_idx:
    save_src = ''.join(nb['cells'][save_cell_idx]['source'])
    # The trainer variable exists in both paths, so this should work as-is
    print(f"✓ Save cell at index {save_cell_idx} (no change needed)")

# ============================================================
# Fix 7: Remove DataCollatorForCompletionOnlyLM import (not needed for two-stage)
# ============================================================
cell4_src = ''.join(nb['cells'][4]['source'])
cell4_src = cell4_src.replace(
    'from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM',
    'from trl import SFTTrainer, SFTConfig'
)
nb['cells'][4]['source'] = [cell4_src]
print("✓ Removed DataCollatorForCompletionOnlyLM import")

# ============================================================
# Save
# ============================================================
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n{'='*60}")
print("Notebook fixed and updated for v25 two-stage training!")
print("  Stage 1: 7741 verified samples, LR=1e-5, 1 epoch")
print("  Stage 2: 600 E1 samples, LR=2e-4, 1 epoch")
print(f"{'='*60}")
