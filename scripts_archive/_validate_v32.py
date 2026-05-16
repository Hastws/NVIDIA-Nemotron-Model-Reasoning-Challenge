#!/usr/bin/env python3
"""Validate v32 notebook: JSON structure, Python syntax, critical elements."""
import json, ast, os

nb_path = os.path.join(os.path.dirname(__file__), '..', 'nvidia-nemotron-sfttrainer-v32.ipynb')
with open(nb_path) as f:
    nb = json.load(f)

print(f'nbformat: {nb["nbformat"]}.{nb["nbformat_minor"]}')
print(f'cells: {len(nb["cells"])}')

errors = []
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])
    if ct == 'code':
        try:
            ast.parse(src)
            status = 'OK'
        except SyntaxError as e:
            status = f'SYNTAX ERROR: {e}'
            errors.append((i, e))
        has_boxed = '\\boxed' in src
        extra = ' | has \\boxed' if has_boxed else ''
        print(f'  Cell {i+1} [{ct}]: {len(src):>5} chars | syntax: {status}{extra}')
    else:
        print(f'  Cell {i+1} [{ct}]: {len(src):>5} chars')

if errors:
    print(f'\nFOUND {len(errors)} ERRORS!')
    for idx, err in errors:
        print(f'  Cell {idx+1}: {err}')
else:
    print('\nAll code cells pass syntax check.')

# Check critical elements
src_all = ''.join(''.join(c['source']) for c in nb['cells'])
checks = [
    ('DATA_SOURCE=cot_v2_hybrid', 'cot_v2_hybrid' in src_all),
    ('SUFFIX with boxed', 'boxed{}' in src_all),
    ('build_training_text', 'build_training_text' in src_all),
    ('enable_thinking=True', 'enable_thinking=True' in src_all),
    ('LoRA r=LORA_RANK', 'r=LORA_RANK' in src_all),
    ('target all-linear', "'all-linear'" in src_all),
    ('LR=1e-4', 'LR = 1e-4' in src_all),
    ('save_pretrained', 'save_pretrained' in src_all),
    ('submission.zip', 'submission.zip' in src_all),
    ("dataset_text_field='text'", "dataset_text_field='text'" in src_all),
    ('SFTTrainer', 'SFTTrainer' in src_all),
    ('adapter_config.json check', 'adapter_config.json' in src_all),
    ('adapter_model.safetensors check', 'adapter_model.safetensors' in src_all),
]
print('\nCritical element checks:')
all_pass = True
for name, ok in checks:
    print(f'  {"PASS" if ok else "FAIL"}: {name}')
    if not ok:
        all_pass = False

if all_pass:
    print('\nAll checks PASSED. Notebook is ready.')
else:
    print('\nSome checks FAILED. Fix before deploying.')
