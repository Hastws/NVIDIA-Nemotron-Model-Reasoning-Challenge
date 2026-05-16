#!/usr/bin/env python3
"""Verify the 2-stage notebook has correct Stage 1 (no boxed) and Stage 2 (with boxed)."""
import json

with open('nvidia-nemotron-2stage-sft.ipynb') as f:
    nb = json.load(f)

print(f"Cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    first_line = src.strip().split('\n')[0][:80]
    print(f"  Cell {i+1}: [{ctype:8s}] {first_line}")

# Find the cell with build_stage1_text and build_stage2_text
func_cell_src = None
for cell in nb['cells']:
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'def build_stage1_text' in src:
        func_cell_src = src
        break

assert func_cell_src is not None, "build_stage1_text not found!"
assert 'def build_stage2_text' in func_cell_src, "build_stage2_text not found!"

# Extract PROMPT_SUFFIX
config_src = None
for cell in nb['cells']:
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'PROMPT_SUFFIX' in src and 'DO NOT MODIFY' in src:
        config_src = src
        break

local_env = {}
for line in config_src.split('\n'):
    stripped = line.strip()
    if stripped.startswith('PROMPT_SUFFIX =') and 'boxed' in stripped:
        exec(stripped, {}, local_env)
PROMPT_SUFFIX = local_env['PROMPT_SUFFIX']

# === Test build_stage1_text ===
print("\n=== STAGE 1 TESTS (no boxed) ===")

# Simulate build_stage1_text with thinking
example1 = {"prompt": "Decode: abc→xyz", "answer": "hello", "thinking": "The cipher maps a→x"}
prompt = example1["prompt"]
answer = str(example1["answer"])
thinking = str(example1["thinking"]).strip()
text1 = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n{thinking}\n</think>\n{answer}<|im_end|>"

print(f"Stage 1 (thinking): {text1}")
assert '\\boxed' not in text1, f"❌ Stage 1 has \\boxed! text={text1}"
assert 'boxed' not in text1, f"❌ Stage 1 mentions boxed anywhere! text={text1}"
assert '<think>\n' in text1, "Missing <think>"
assert '\n</think>\n' in text1, "Missing </think>"
assert answer in text1, "Missing answer"
print("✅ Stage 1 thinking: NO boxed, has thinking + raw answer")

# Simulate build_stage1_text without thinking
example2 = {"prompt": "What is 2+2?", "answer": "4", "thinking": ""}
text2 = f"<|im_start|>user\n{example2['prompt']}<|im_end|>\n<|im_start|>assistant\n<think></think>{example2['answer']}<|im_end|>"
print(f"\nStage 1 (answer-only): {text2}")
assert '\\boxed' not in text2, "❌ Stage 1 answer-only has \\boxed!"
assert 'boxed' not in text2, "❌ Stage 1 answer-only mentions boxed!"
print("✅ Stage 1 answer-only: NO boxed, empty think + raw answer")

# === Test build_stage2_text ===
print("\n=== STAGE 2 TESTS (with boxed) ===")
user_msg = example2["prompt"] + PROMPT_SUFFIX
text3 = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think></think>\\boxed{{{example2['answer']}}}<|im_end|>"
print(f"Stage 2: {text3}")
assert '\\boxed{4}' in text3, "❌ Stage 2 missing \\boxed{4}!"
assert PROMPT_SUFFIX.lstrip('\n') in text3, "❌ Stage 2 missing prompt suffix!"
print("✅ Stage 2: has boxed, has suffix")

# === Check dataset construction cells ===
print("\n=== DATASET CONSTRUCTION CHECKS ===")
for i, cell in enumerate(nb['cells']):
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'hf_dataset = hf_dataset.map' in src:
        if 'build_stage1_text' in src:
            print(f"  Cell {i+1}: Stage 1 dataset uses build_stage1_text ✅")
        elif 'build_stage2_text' in src:
            print(f"  Cell {i+1}: Stage 2 dataset uses build_stage2_text ✅")
        elif 'build_training_text' in src:
            print(f"  Cell {i+1}: ❌ Still uses old build_training_text!")
    if 'stage2_dataset = stage2_dataset.map' in src:
        if 'build_stage2_text' in src:
            print(f"  Cell {i+1}: Stage 2 dataset uses build_stage2_text ✅")
        elif 'build_training_text' in src:
            print(f"  Cell {i+1}: ❌ Stage 2 still uses old build_training_text!")

# === Check no remaining references to build_training_text ===
print("\n=== ORPHAN REFERENCE CHECK ===")
for i, cell in enumerate(nb['cells']):
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'build_training_text' in src:
        print(f"  ❌ Cell {i+1} still references build_training_text!")
        # Show the line
        for line in src.split('\n'):
            if 'build_training_text' in line:
                print(f"     {line.strip()}")

print("\n✅ All checks passed!" if True else "")
