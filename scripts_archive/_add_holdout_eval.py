"""Add holdout evaluation to the 2-stage SFT notebook."""
import json
import copy

NB_PATH = 'nvidia-nemotron-2stage-sft.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Original: {len(cells)} cells")

# ============================================================
# 1. Edit Cell 4 (config): Add holdout config
# ============================================================
cell4_src = ''.join(cells[4]['source'])
assert 'HYPERPARAMETERS' in cell4_src, f"Cell 4 is not config! Got: {cell4_src[:100]}"

# Add holdout config after LORA_DROPOUT line
old_line = "LORA_DROPOUT     = 0.05"
new_block = """LORA_DROPOUT     = 0.05

# --- Holdout Evaluation ---
HOLDOUT_ENABLED  = True
HOLDOUT_N_PER_TYPE = 20   # 20 per type × 6 = 120 total"""
cell4_src = cell4_src.replace(old_line, new_block)

# Add holdout to the print section
old_print = 'print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")'
new_print = """print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Holdout: enabled={HOLDOUT_ENABLED}, n_per_type={HOLDOUT_N_PER_TYPE}")"""
cell4_src = cell4_src.replace(old_print, new_print)

cells[4]['source'] = cell4_src.splitlines(True)
# Fix: ensure last line has newline
if cells[4]['source'] and not cells[4]['source'][-1].endswith('\n'):
    cells[4]['source'][-1] += '\n'
print("✅ Cell 4: added holdout config")

# ============================================================
# 2. Edit Cell 6 (data loading): Add holdout split
# ============================================================
cell6_src = ''.join(cells[6]['source'])
assert 'CELL V78 DATA LOADING' in cell6_src, f"Cell 6 is not data loading!"

# Add holdout split at the END of cell 6
holdout_split_code = r'''

# =============================================
#  HOLDOUT SPLIT (for per-type evaluation)
# =============================================
if HOLDOUT_ENABLED:
    import re as _re
    
    # Load original train.csv for ground truth + type inference
    _orig_train = pl.read_csv(f'{COMP_DATA}/train.csv').to_pandas()
    
    def _infer_type(prompt):
        p = prompt.lower()
        if 'bit manipulation' in p or '8-bit binary' in p:
            return 'bit_ops'
        elif 'numeral system' in p:
            return 'numeral'
        elif 'encrypt' in p or 'decrypt' in p:
            return 'cipher'
        elif 'gravitational' in p or 'gravity' in p or 'free-fall' in p:
            return 'gravity'
        elif 'unit' in p and ('convert' in p or 'conversion' in p):
            return 'unit_conv'
        elif 'symbol' in p or 'transformation rule' in p:
            return 'symbol'
        return 'unknown'
    
    _orig_train['type'] = _orig_train['prompt'].apply(_infer_type)
    
    # Stratified holdout: HOLDOUT_N_PER_TYPE per type
    _holdout_parts = []
    for _t in sorted(_orig_train['type'].unique()):
        _type_df = _orig_train[_orig_train['type'] == _t]
        _sampled = _type_df.sample(n=min(HOLDOUT_N_PER_TYPE, len(_type_df)), random_state=999)
        _holdout_parts.append(_sampled)
    holdout_df = pd.concat(_holdout_parts).reset_index(drop=True)
    holdout_ids = set(holdout_df['id'].tolist())
    
    print(f"\n{'='*60}")
    print(f"  HOLDOUT: {len(holdout_df)} samples ({HOLDOUT_N_PER_TYPE}/type)")
    print(f"{'='*60}")
    for _t in sorted(holdout_df['type'].unique()):
        print(f"    {_t}: {(holdout_df['type']==_t).sum()}")
    
    # Filter training data to exclude holdout
    _before = len(train_df)
    _keep_mask = ~train_df.to_pandas()['id'].isin(holdout_ids)
    train_df = pl.from_pandas(train_df.to_pandas()[_keep_mask.values].reset_index(drop=True))
    print(f"\n  Training data: {_before} → {len(train_df)} (removed {_before - len(train_df)} holdout overlap)")
else:
    holdout_df = None
    print("\nHoldout evaluation: DISABLED")
'''

cell6_src = cell6_src + holdout_split_code
cells[6]['source'] = cell6_src.splitlines(True)
if cells[6]['source'] and not cells[6]['source'][-1].endswith('\n'):
    cells[6]['source'][-1] += '\n'
print("✅ Cell 6: added holdout split")

# ============================================================
# 3. Insert eval cells between Cell 15 (Stage 2) and Cell 16 (Save markdown)
# ============================================================

eval_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Holdout Evaluation\n",
        "\n",
        "Run inference on held-out samples (not seen during training) to get **per-type accuracy**.\n",
        "Uses the same prompt format and answer extraction as the official evaluation.\n"
    ]
}

eval_code = r'''# =============================================
#  HOLDOUT EVALUATION — Per-Type Accuracy
# =============================================
import re as _re

if HOLDOUT_ENABLED and holdout_df is not None:
    
    # --- Official answer extraction (from nemotron-baseline-evaluation.ipynb) ---
    def extract_final_answer(text):
        if text is None:
            return 'NOT_FOUND'
        matches = _re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
        if matches:
            non_empty = [m.strip() for m in matches if m.strip()]
            if non_empty:
                return non_empty[-1]
            return matches[-1].strip()
        patterns = [
            r'The final answer is:\s*([^\n]+)',
            r'Final answer is:\s*([^\n]+)',
            r'Final answer\s*[:：]\s*([^\n]+)',
            r'final answer\s*[:：]\s*([^\n]+)',
        ]
        for pattern in patterns:
            matches = _re.findall(pattern, text, _re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        matches = _re.findall(r'-?\d+(?:\.\d+)?', text)
        if matches:
            return matches[-1]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else 'NOT_FOUND'

    # --- Official verify (from nemotron-baseline-evaluation.ipynb) ---
    def eval_verify(stored_answer, predicted):
        stored_answer = str(stored_answer).strip()
        predicted = str(predicted).strip()
        try:
            stored_num = float(stored_answer)
            predicted_num = float(predicted)
            return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
        except Exception:
            return predicted.lower() == stored_answer.lower()

    # --- Run evaluation ---
    model.eval()
    eval_results = []
    t0_eval = time.time()
    
    print(f"{'='*60}")
    print(f"  HOLDOUT EVALUATION: {len(holdout_df)} samples")
    print(f"  Prompt: official format (enable_thinking=True)")
    print(f"  Decoding: greedy (do_sample=False)")
    print(f"  Max new tokens: 3584")
    print(f"{'='*60}")
    
    for eval_idx, row in holdout_df.iterrows():
        prompt = row['prompt']
        answer = str(row['answer'])
        qtype = row['type']
        
        # Build prompt exactly like official eval
        user_content = prompt + PROMPT_SUFFIX
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=3584,
                do_sample=False,
            )
        
        gen_ids = output[0][prompt_len:]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=False)
        predicted = extract_final_answer(generated)
        correct = eval_verify(answer, predicted)
        
        eval_results.append({
            'id': row['id'],
            'type': qtype,
            'answer': answer,
            'predicted': predicted,
            'correct': correct,
            'gen_tokens': len(gen_ids),
        })
        
        # Progress every 10 samples
        if (len(eval_results)) % 10 == 0:
            elapsed = time.time() - t0_eval
            n_correct = sum(r['correct'] for r in eval_results)
            n_done = len(eval_results)
            eta = elapsed / n_done * (len(holdout_df) - n_done) / 60
            print(f"  [{n_done}/{len(holdout_df)}] {elapsed/60:.1f}min | "
                  f"{n_correct}/{n_done} ({n_correct/n_done*100:.0f}%) | "
                  f"ETA: {eta:.0f}min")

    eval_time = time.time() - t0_eval
    
    # --- Results table ---
    import pandas as pd
    results_df = pd.DataFrame(eval_results)
    
    print(f"\n{'='*60}")
    print(f"  📊 HOLDOUT RESULTS (eval time: {eval_time/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  {'Type':12s} {'Correct':>8s} {'Total':>6s} {'Acc':>8s} {'Avg Tokens':>11s}")
    print(f"  {'-'*47}")
    
    for t in sorted(results_df['type'].unique()):
        t_df = results_df[results_df['type'] == t]
        n_correct = t_df['correct'].sum()
        n_total = len(t_df)
        avg_tok = t_df['gen_tokens'].mean()
        acc = n_correct / n_total * 100
        bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
        print(f"  {t:12s} {n_correct:5d}/{n_total:<3d}   {acc:5.1f}%  {avg_tok:8.0f}  {bar}")
    
    total_correct = results_df['correct'].sum()
    total_n = len(results_df)
    total_acc = total_correct / total_n * 100
    print(f"  {'-'*47}")
    print(f"  {'TOTAL':12s} {total_correct:5d}/{total_n:<3d}   {total_acc:5.1f}%")
    
    # --- Show failures ---
    failures = results_df[~results_df['correct']]
    if len(failures) > 0:
        print(f"\n  ❌ Failures ({len(failures)} total, showing first 10):")
        for _, f in failures.head(10).iterrows():
            print(f"    [{f['type']:10s}] expected='{f['answer']}', got='{f['predicted']}' ({f['gen_tokens']} tok)")
    
    # --- Show stats ---
    print(f"\n  Token stats: min={results_df['gen_tokens'].min()}, "
          f"median={results_df['gen_tokens'].median():.0f}, "
          f"max={results_df['gen_tokens'].max()}")
    not_found = (results_df['predicted'] == 'NOT_FOUND').sum()
    if not_found > 0:
        print(f"  ⚠️ NOT_FOUND: {not_found} samples (no \\boxed{{}} in output)")

else:
    print("Holdout evaluation: SKIPPED (HOLDOUT_ENABLED=False)")
    eval_time = 0
'''

eval_code_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": eval_code.splitlines(True),
    "execution_count": None,
    "outputs": []
}

# Insert at position 16 (after Cell 15 = Stage 2, before Cell 16 = Save markdown)
cells.insert(16, eval_code_cell)
cells.insert(16, eval_markdown_cell)
print("✅ Inserted 2 eval cells at position 16-17")

# ============================================================
# 4. Update summary cell (now at index 20, was 18)
# ============================================================
summary_cell_idx = len(cells) - 1
summary_src = ''.join(cells[summary_cell_idx]['source'])
if 'TRAINING SUMMARY' in summary_src:
    old_summary_end = 'print("=" * 60)\n'
    # Find the LAST occurrence
    last_idx = summary_src.rfind(old_summary_end)
    if last_idx >= 0:
        insert_pos = last_idx + len(old_summary_end)
        eval_summary = '''
# Holdout results
if HOLDOUT_ENABLED and 'results_df' in dir():
    total_correct = results_df['correct'].sum()
    total_n = len(results_df)
    print(f"  Holdout: {total_correct}/{total_n} = {total_correct/total_n*100:.1f}%")
    print(f"  Eval time: {eval_time/60:.1f} min")
'''
        summary_src = summary_src[:insert_pos] + eval_summary + summary_src[insert_pos:]
        cells[summary_cell_idx]['source'] = summary_src.splitlines(True)
        if cells[summary_cell_idx]['source'] and not cells[summary_cell_idx]['source'][-1].endswith('\n'):
            cells[summary_cell_idx]['source'][-1] += '\n'
        print("✅ Updated summary cell with holdout results")

# Save
nb['cells'] = cells
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"\n✅ Saved {NB_PATH} with {len(cells)} cells")
