#!/usr/bin/env python3
"""Verify v28 fixes."""
import json
nb = json.load(open('nvidia-nemotron-sfttrainer-training.ipynb'))
ok = True
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'THINK_CLOSE_ID' in src:
        print(f'Cell {i}: Has THINK_CLOSE_ID ✓')
    if 'build_training_example' in src and 'hf_dataset.map' in src:
        print(f'Cell {i}: Calls build_training_example ✓')
    if 'skip_prepare_dataset' in src:
        print(f'Cell {i}: Has skip_prepare_dataset ✓')
    if 'DataCollatorForSeq2Seq' in src:
        print(f'Cell {i}: Has DataCollatorForSeq2Seq ✓')
    if 'CompletionOnlyCollator' in src:
        print(f'Cell {i}: ✗ STILL HAS CompletionOnlyCollator!')
        ok = False
    if 'DataCollatorForCompletionOnlyLM' in src:
        print(f'Cell {i}: ✗ STILL HAS DataCollatorForCompletionOnlyLM!')
        ok = False
    if 'build_training_text' in src and 'def build_training_text' not in src:
        print(f'Cell {i}: ✗ STILL CALLS build_training_text!')
        ok = False
    if 'dataset_text_field' in src:
        print(f'Cell {i}: ✗ STILL HAS dataset_text_field (should be removed)!')
        ok = False
print()
print('ALL CHECKS PASSED ✓' if ok else 'SOME CHECKS FAILED ✗')
