#!/usr/bin/env python3
"""Verify the 2-stage notebook prompt suffix and format."""
import json

with open('nvidia-nemotron-2stage-sft.ipynb') as f:
    nb = json.load(f)

print(f"Cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    first_line = src.strip().split('\n')[0][:80]
    print(f"  Cell {i+1}: [{ctype:8s}] {first_line}")

# Find and execute PROMPT_SUFFIX
print("\n=== PROMPT_SUFFIX VERIFICATION ===")
for i, cell in enumerate(nb['cells']):
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'PROMPT_SUFFIX' in src and 'DO NOT MODIFY' in src:
        for line in src.split('\n'):
            stripped = line.strip()
            if stripped.startswith('PROMPT_SUFFIX =') and 'boxed' in stripped:
                print(f"Cell {i+1}, raw code: {stripped}")
                local_ns = {}
                exec(stripped, {}, local_ns)
                val = local_ns['PROMPT_SUFFIX']
                print(f"Evaluated value: {repr(val)}")
                
                # Official value from nemotron-baseline-evaluation.ipynb
                official = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
                print(f"Official value:  {repr(official)}")
                
                if val == official:
                    print("✅ MATCH!")
                else:
                    print("❌ MISMATCH!")
                    # Show diff
                    for j, (a, b) in enumerate(zip(val, official)):
                        if a != b:
                            print(f"  First diff at index {j}: ours={repr(a)}, official={repr(b)}")
                            break
                break
        break

# Check build_training_text for boxed wrapping
print("\n=== TRAINING TEXT CHECKS ===")
for i, cell in enumerate(nb['cells']):
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if 'def build_training_text' in src:
        print(f"build_training_text found in Cell {i+1}")
        # Check key patterns
        if 'PROMPT_SUFFIX' in src:
            print("  ✅ Uses PROMPT_SUFFIX variable")
        if '\\\\boxed{{{answer}}}' in src:
            print("  ✅ Uses \\boxed{answer} in assistant response")
        if 'enable_thinking=False' in src:
            print("  ✅ Uses enable_thinking=False for answer-only")
        if 'enable_thinking=True' not in src:
            print("  ✅ No enable_thinking=True (correct: thinking rows use manual ChatML)")
        break
