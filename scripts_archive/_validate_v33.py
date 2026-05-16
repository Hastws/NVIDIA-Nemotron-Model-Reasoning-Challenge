#!/usr/bin/env python3
"""Validate v33 notebook."""
import json, ast

with open('nvidia-nemotron-sfttrainer-v33.ipynb') as f:
    nb = json.load(f)

print(f'Cells: {len(nb["cells"])}')
errors = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if src.startswith('!'):
            print(f'  Cell {i+1} [code]: shell command (skip syntax check)')
            continue
        try:
            ast.parse(src)
            print(f'  Cell {i+1} [code]: OK ({len(src)} chars)')
        except SyntaxError as e:
            print(f'  Cell {i+1} [code]: SYNTAX ERROR: {e}')
            errors += 1
    else:
        print(f'  Cell {i+1} [markdown]: {len("".join(cell["source"]))} chars')

if errors:
    print(f'\nFOUND {errors} ERRORS!')
else:
    print('\nAll code cells pass syntax check.')

src_all = ''.join(''.join(c['source']) for c in nb['cells'])
checks = [
    ('stratified_sample', 'stratified_sample' in src_all),
    ('SFT_SAMPLES_PER_TYPE=100', 'SFT_SAMPLES_PER_TYPE = 100' in src_all),
    ('seed=42', 'seed=42' in src_all),
    ('METRIC_SUFFIX', 'METRIC_SUFFIX' in src_all),
    ('enable_thinking', 'enable_thinking' in src_all),
    ('LR=2e-4', 'LR = 2e-4' in src_all),
    ('LORA_DROPOUT=0.05', 'LORA_DROPOUT = 0.05' in src_all),
    ('build_sft_text', 'build_sft_text' in src_all),
    ('classify_type', 'classify_type' in src_all),
    ('all-linear', "'all-linear'" in src_all),
    ('save_pretrained', 'save_pretrained' in src_all),
    ('submission.zip', 'submission.zip' in src_all),
    ('SFTTrainer', 'SFTTrainer' in src_all),
    ('train.csv', 'train.csv' in src_all),
    ('logging_steps=10', 'logging_steps=10' in src_all),
]
print('\nCritical element checks:')
all_pass = True
for name, ok in checks:
    print(f'  {"PASS" if ok else "FAIL"}: {name}')
    if not ok: all_pass = False

print(f'\n{"ALL PASSED" if all_pass else "SOME FAILED"}')
