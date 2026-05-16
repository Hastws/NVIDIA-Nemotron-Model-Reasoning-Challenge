#!/usr/bin/env python3
"""Create v26 notebook: alpha=64, dropout=0, curated_700 data, standard mode."""
import json, copy, re

with open('nvidia-nemotron-sfttrainer-training.ipynb') as f:
    nb = json.load(f)

nb26 = copy.deepcopy(nb)

# Cell 4: config changes
c4 = ''.join(nb26['cells'][4]['source'])
c4 = c4.replace('TRAINING_MODE = "two_stage"', 'TRAINING_MODE = "standard"')
c4 = c4.replace('DATA_SOURCE = "original"', 'DATA_SOURCE = "curated_700"')
nb26['cells'][4]['source'] = [c4]

# Cell 6: add curated_700 data source
c6 = ''.join(nb26['cells'][6]['source'])
if 'curated_700' not in c6:
    insertion = 'elif DATA_SOURCE == "medium_cot":\n    train_df = pl.read_csv(f\'{COT_DATA}/sft_medium_cot.csv\')'
    replacement = insertion + '\nelif DATA_SOURCE == "curated_700":\n    train_df = pl.read_csv(f\'{COT_DATA}/sft_curated_700.csv\')'
    c6 = c6.replace(insertion, replacement)
nb26['cells'][6]['source'] = [c6]

# Cell 10: LoRA config - alpha=64, dropout=0
c10 = ''.join(nb26['cells'][10]['source'])
c10 = c10.replace('lora_alpha=16,', 'lora_alpha=64,')
c10 = c10.replace('lora_dropout=0.05,', 'lora_dropout=0.0,')
nb26['cells'][10]['source'] = [c10]

with open('nvidia-nemotron-sfttrainer-v26.ipynb', 'w') as f:
    json.dump(nb26, f, indent=1)

# Verify
print("v26 config:")
for i, cell in enumerate(nb26['cells']):
    src = ''.join(cell['source'])
    for pat, label in [
        (r'TRAINING_MODE\s*=\s*"([^"]+)"', 'TRAINING_MODE'),
        (r'DATA_SOURCE\s*=\s*"([^"]+)"', 'DATA_SOURCE'),
        (r'lora_alpha=(\d+)', 'lora_alpha'),
        (r'lora_dropout=([0-9.]+)', 'lora_dropout'),
    ]:
        m = re.search(pat, src)
        if m:
            print(f"  Cell {i}: {label}={m.group(1)}")
print("Done!")
