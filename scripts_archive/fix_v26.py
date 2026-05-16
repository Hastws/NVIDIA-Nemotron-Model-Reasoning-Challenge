#!/usr/bin/env python3
"""Fix v26 notebook: ensure alpha=64, dropout=0, standard mode, curated_700."""
import json
import re

with open('nvidia-nemotron-sfttrainer-v26.ipynb') as f:
    nb = json.load(f)

# Fix ALL cells that contain lora params
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    changed = False
    if 'lora_alpha=16,' in src:
        src = src.replace('lora_alpha=16,', 'lora_alpha=64,')
        changed = True
    if 'lora_dropout=0.05,' in src:
        src = src.replace('lora_dropout=0.05,', 'lora_dropout=0.0,')
        changed = True
    if changed:
        nb['cells'][i]['source'] = [src]
        print(f"Fixed Cell {i}: alpha=64, dropout=0.0")

# Ensure TRAINING_MODE = "standard" exists in Cell 4
c4 = ''.join(nb['cells'][4]['source'])
if 'TRAINING_MODE' not in c4:
    c4 = c4.replace(
        'DATA_SOURCE = "curated_700"',
        'TRAINING_MODE = "standard"\nDATA_SOURCE = "curated_700"'
    )
    nb['cells'][4]['source'] = [c4]
    print("Added TRAINING_MODE = standard")

# Save
with open('nvidia-nemotron-sfttrainer-v26.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

# Final verification
print("\n=== Final Configuration ===")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    for pat, label in [
        (r'lora_alpha=(\d+)', 'lora_alpha'),
        (r'lora_dropout=([0-9.]+)', 'lora_dropout'),
        (r'TRAINING_MODE\s*=\s*"([^"]+)"', 'TRAINING_MODE'),
        (r'^DATA_SOURCE\s*=\s*"([^"]+)"', 'DATA_SOURCE'),
    ]:
        for m in re.finditer(pat, src, re.MULTILINE):
            print(f"  Cell {i}: {label} = {m.group(1)}")
