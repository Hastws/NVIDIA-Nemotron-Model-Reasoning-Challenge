#!/usr/bin/env python3
"""Extract E1 build function from old notebook."""
import json

with open('kaggle_scripts/sft_old/sfttrainer-training.ipynb') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'sft_dataset' in src and ('map' in src or 'apply_chat_template' in src):
        print(f'=== Cell {i} ===')
        print(src[:3000])
        print()
