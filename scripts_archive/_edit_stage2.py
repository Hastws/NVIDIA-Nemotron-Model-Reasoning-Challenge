"""Edit notebook: Modify Stage 2 to use thinking rows with masked loss."""
import json

with open('nvidia-nemotron-2stage-sft.ipynb') as f:
    nb = json.load(f)

# ============================================================
# 1. Modify Cell 8: Replace build_stage2_text + verification
# ============================================================
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'build_stage2_text' in src and 'build_stage1_text' in src:
            lines = src.split('\n')
            
            # Find the Stage 2 builder section start
            stage2_start = None
            stage2_verif_start = None
            stage2_verif_end = None
            s1_verif_start = None
            
            for j, line in enumerate(lines):
                if '#  Stage 2 builder:' in line:
                    stage2_start = j - 1  # include the # ==== line above
                if 'STAGE 1 FORMAT VERIFICATION' in line:
                    s1_verif_start = j
                if 'STAGE 2 FORMAT VERIFICATION' in line:
                    stage2_verif_start = j
                if "All format checks passed!" in line:
                    stage2_verif_end = j
            
            print(f"Cell {i}: stage2_builder={stage2_start}, s1_verif={s1_verif_start}, s2_verif={stage2_verif_start}, end={stage2_verif_end}")
            
            new_stage2_builder = [
                '# =============================================',
                '#  Stage 2 builder: thinking + boxed (thinking masked in loss)',
                '# =============================================',
                'THINK_END_TOKEN_ID = 13  # </think> token in Nemotron tokenizer',
                '',
                'def build_stage2_text(example):',
                '    """Stage 2: Unified format with thinking + \\\\boxed{}.',
                '    Thinking content will be masked in loss computation."""',
                '    prompt = example["prompt"]',
                '    answer = str(example["answer"])',
                '    thinking = example.get("thinking", "")',
                '    user_msg = prompt + PROMPT_SUFFIX',
                '    ',
                '    if _has_thinking(thinking):',
                '        text = (',
                '            f"<|im_start|>user\\n{user_msg}<|im_end|>\\n"',
                '            f"<|im_start|>assistant\\n<think>\\n{str(thinking).strip()}\\n</think>\\n\\\\boxed{{{answer}}}<|im_end|>"',
                '        )',
                '    else:',
                '        text = (',
                '            f"<|im_start|>user\\n{user_msg}<|im_end|>\\n"',
                '            f"<|im_start|>assistant\\n<think></think>\\\\boxed{{{answer}}}<|im_end|>"',
                '        )',
                '    return {"text": text}',
                '',
                'def tokenize_and_mask_thinking(example):',
                '    """Tokenize and mask everything up to </think> (inclusive) in labels."""',
                "    text = example['text']",
                '    encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=STAGE2_MAX_SEQ)',
                "    input_ids = encoded['input_ids']",
                '    labels = list(input_ids)',
                '    ',
                '    # Find </think> token and mask everything up to and including it',
                '    try:',
                '        think_end_pos = input_ids.index(THINK_END_TOKEN_ID)',
                '        for i in range(think_end_pos + 1):',
                '            labels[i] = -100',
                '    except ValueError:',
                "        pass  # no </think> found, don't mask",
                '    ',
                '    return {',
                "        'input_ids': input_ids,",
                "        'attention_mask': [1] * len(input_ids),",
                "        'labels': labels,",
                '    }',
            ]
            
            new_stage2_verif = [
                'print("\\n=== STAGE 2 FORMAT VERIFICATION ===")',
                'if len(think_rows) > 0:',
                '    row = think_rows.iloc[0].to_dict()',
                '    result = build_stage2_text(row)',
                "    text = result['text']",
                '    print(f"\\n--- STAGE 2: THINKING ROW (id={row[\'id\']}) ---")',
                '    print(text[:500])',
                '    if len(text) > 500:',
                '        print(f"... ({len(text)} chars total)")',
                "    assert PROMPT_SUFFIX.lstrip('\\n') in text, \"Missing prompt suffix\"",
                "    assert '\\\\boxed{' in text, \"Missing \\\\boxed{}\"",
                "    assert '<think>\\n' in text, \"Missing thinking content\"",
                '    print("✅ Stage 2 row: has suffix, has boxed, has thinking")',
                '    ',
                '    # Verify masking',
                '    enc = tokenizer(text, add_special_tokens=False)',
                "    ids = enc['input_ids']",
                '    try:',
                '        pos = ids.index(THINK_END_TOKEN_ID)',
                '        total_tokens = len(ids)',
                '        masked_tokens = pos + 1',
                '        trained_tokens = total_tokens - masked_tokens',
                '        print(f"  Masking: {masked_tokens}/{total_tokens} tokens masked, {trained_tokens} tokens trained")',
                '    except ValueError:',
                '        print("  WARNING: </think> token not found!")',
            ]
            
            # Build new cell content
            new_lines = lines[:stage2_start]
            new_lines.extend(new_stage2_builder)
            new_lines.append('')
            # Keep Stage 1 verification (from s1_verif to s2_verif)
            new_lines.extend(lines[s1_verif_start:stage2_verif_start])
            # Add new Stage 2 verification
            new_lines.extend(new_stage2_verif)
            new_lines.append('')
            new_lines.append('print("\\n✅ All format checks passed!")')
            
            new_src = '\n'.join(new_lines)
            # Convert to notebook source format
            src_lines = new_src.split('\n')
            cell['source'] = [line + '\n' for line in src_lines[:-1]]
            if src_lines[-1]:
                cell['source'].append(src_lines[-1])
            
            print(f"Cell {i} modified. Old: {len(src)} chars, New: {len(new_src)} chars")
            break

# ============================================================
# 2. Modify Cell 15: Replace Stage 2 training
# ============================================================
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'STAGE2_ENABLED' in src and 'stage2_trainer' in src:
            new_stage2_training = r'''if STAGE2_ENABLED:
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    
    print(f"{'='*60}")
    print(f"  STAGE 2: Format Polish (thinking masked)")
    print(f"{'='*60}")
    
    # Prepare Stage 2 dataset from THINKING rows (not answer-only)
    full_df = train_df.to_pandas()
    think_mask = full_df['thinking'].apply(_has_thinking)
    thinking_df = full_df[think_mask]
    
    n_sample = min(STAGE2_N_SAMPLES, len(thinking_df))
    stage2_df = thinking_df.sample(n=n_sample, random_state=42)
    
    eff_batch2 = STAGE2_BATCH * STAGE2_GRAD_ACCUM
    total_steps2 = (n_sample // eff_batch2) * STAGE2_EPOCHS
    print(f"  Thinking pool: {len(thinking_df)}")
    print(f"  Sampled for Stage 2: {n_sample}")
    print(f"  LR: {STAGE2_LR}, Epochs: {STAGE2_EPOCHS}, Max Seq: {STAGE2_MAX_SEQ}")
    print(f"  Batch: {STAGE2_BATCH}, Grad Accum: {STAGE2_GRAD_ACCUM}")
    print(f"  Loss: thinking MASKED, only \\boxed{{answer}} trained")
    print(f"  Estimated steps: ~{total_steps2}")
    
    # Build text then tokenize with thinking masked
    stage2_raw = Dataset.from_pandas(stage2_df)
    stage2_raw = stage2_raw.map(
        build_stage2_text,
        remove_columns=stage2_raw.column_names,
    )
    stage2_tokenized = stage2_raw.map(
        tokenize_and_mask_thinking,
        remove_columns=['text'],
    )
    
    # Show a Stage 2 sample with masking info
    print(f"\n--- Stage 2 sample ---")
    sample_ids = stage2_tokenized[0]['input_ids']
    sample_labels = stage2_tokenized[0]['labels']
    n_masked = sum(1 for l in sample_labels if l == -100)
    n_trained = sum(1 for l in sample_labels if l != -100)
    print(f"  Tokens: {len(sample_ids)}, masked: {n_masked}, trained: {n_trained}")
    # Decode the trained portion
    trained_ids = [i for i, l in zip(sample_ids, sample_labels) if l != -100]
    print(f"  Trained text: {tokenizer.decode(trained_ids)[:200]}")
    
    # Use Trainer (not SFTTrainer) for pre-tokenized data with custom labels
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    
    stage2_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=STAGE2_BATCH,
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    
    stage2_trainer = Trainer(
        model=model,
        train_dataset=stage2_tokenized,
        data_collator=collator,
        args=stage2_args,
    )
    
    t0 = time.time()
    stage2_result = stage2_trainer.train()
    stage2_time = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"  STAGE 2 COMPLETE")
    print(f"  Time: {stage2_time/60:.1f} min")
    print(f"  Final loss: {stage2_result.training_loss:.4f}")
    print(f"{'='*60}")
else:
    print("Stage 2 SKIPPED (STAGE2_ENABLED=False)")'''
            
            src_lines = new_stage2_training.split('\n')
            cell['source'] = [line + '\n' for line in src_lines[:-1]]
            if src_lines[-1]:
                cell['source'].append(src_lines[-1])
            
            print(f"Cell {i} modified (Stage 2 training). Old: {len(src)} chars, New: {len(new_stage2_training)} chars")
            break

# ============================================================
# 3. Save
# ============================================================
with open('nvidia-nemotron-2stage-sft.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("\n✅ Notebook saved successfully")
