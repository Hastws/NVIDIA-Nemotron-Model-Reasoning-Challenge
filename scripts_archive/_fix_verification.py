"""Fix corrupted verification section in Cell 8."""
import json

with open('nvidia-nemotron-2stage-sft.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'build_stage2_text' in src and 'build_stage1_text' in src:
            lines = src.split('\n')
            
            # Keep lines 0-118 (everything up to and including first assert)
            # Then replace lines 119-176 with clean verification
            good_part = lines[:118]  # lines 0-117
            
            # Add the missing assert from line 118
            # line 118: assert '<think>\n' in text, "Missing <think> tag"
            # That line is actually already in good_part as lines[118]
            # Wait, good_part is lines[:118] = lines 0..117
            # Line 118 is: assert '<think>\n' in text, "Missing <think> tag"
            # We need that too
            good_part = lines[:119]  # lines 0-118 inclusive
            
            clean_verification = [
                "    assert '\\n</think>\\n' in text, \"Missing </think> tag\"",
                "    assert '\\\\boxed{' not in text, \"❌ Stage 1 should NOT contain \\\\boxed{}!\"",
                "    assert 'boxed' not in text.split('<|im_start|>user')[1].split('<|im_end|>')[0], \"❌ Stage 1 user msg should NOT mention boxed!\"",
                "    print(\"✅ Stage 1 thinking row: no boxed, has thinking\")",
                "",
                "# Test Stage 1 with an answer-only row",
                "ao_rows = sample_df[~sample_df['thinking'].apply(_has_thinking)]",
                "if len(ao_rows) > 0:",
                "    row = ao_rows.iloc[0].to_dict()",
                "    result = build_stage1_text(row)",
                "    text = result['text']",
                "    print(f\"\\n--- STAGE 1: ANSWER-ONLY ROW (id={row['id']}) ---\")",
                "    print(text[:500])",
                "    assert '\\\\boxed{' not in text, \"❌ Stage 1 should NOT contain \\\\boxed{}!\"",
                "    assert '<think></think>' in text, \"Missing empty think tags\"",
                "    print(\"✅ Stage 1 answer-only row: no boxed, empty think\")",
                "",
                "print(\"\\n=== STAGE 2 FORMAT VERIFICATION ===\")",
                "if len(think_rows) > 0:",
                "    row = think_rows.iloc[0].to_dict()",
                "    result = build_stage2_text(row)",
                "    text = result['text']",
                "    print(f\"\\n--- STAGE 2: THINKING ROW (id={row['id']}) ---\")",
                "    print(text[:500])",
                "    if len(text) > 500:",
                "        print(f\"... ({len(text)} chars total)\")",
                "    assert PROMPT_SUFFIX.lstrip('\\n') in text, \"Missing prompt suffix\"",
                "    assert '\\\\boxed{' in text, \"Missing \\\\boxed{}\"",
                "    assert '<think>\\n' in text, \"Missing thinking content\"",
                "    print(\"✅ Stage 2 row: has suffix, has boxed, has thinking\")",
                "    ",
                "    # Verify masking",
                "    enc = tokenizer(text, add_special_tokens=False)",
                "    ids = enc['input_ids']",
                "    try:",
                "        pos = ids.index(THINK_END_TOKEN_ID)",
                "        total_tokens = len(ids)",
                "        masked_tokens = pos + 1",
                "        trained_tokens = total_tokens - masked_tokens",
                "        print(f\"  Masking: {masked_tokens}/{total_tokens} tokens masked, {trained_tokens} tokens trained\")",
                "    except ValueError:",
                "        print(\"  WARNING: </think> token not found!\")",
                "",
                "print(\"\\n✅ All format checks passed!\")",
            ]
            
            new_lines = good_part + clean_verification
            new_src = '\n'.join(new_lines)
            
            # Convert to notebook format
            src_lines = new_src.split('\n')
            cell['source'] = [line + '\n' for line in src_lines[:-1]]
            if src_lines[-1]:
                cell['source'].append(src_lines[-1])
            
            print(f"Cell {i} fixed. {len(lines)} -> {len(new_lines)} lines")
            break

with open('nvidia-nemotron-2stage-sft.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Saved. Verifying...")

# Quick verify
with open('nvidia-nemotron-2stage-sft.ipynb') as f:
    nb2 = json.load(f)
for i, cell in enumerate(nb2['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'build_stage2_text' in src and 'build_stage1_text' in src:
            lines = src.split('\n')
            print(f"\nVerification section (lines 105+):")
            for k in range(105, len(lines)):
                print(f'{k:3d}: {lines[k]}')
            break
