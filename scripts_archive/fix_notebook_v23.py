#!/usr/bin/env python3
"""Modify notebook for v23: completion-only loss + answer-only E1 data."""
import json
import sys

path = 'nvidia-nemotron-sfttrainer-training.ipynb'

with open(path, 'r') as f:
    nb = json.load(f)

# 1) Fix cell 4: DATA_SOURCE = "original"
cell4 = nb['cells'][4]
new_src = []
for line in cell4['source']:
    if 'DATA_SOURCE = ' in line and not line.strip().startswith('#'):
        line = 'DATA_SOURCE = "original"\n'
    new_src.append(line)
cell4['source'] = new_src
print("Fixed DATA_SOURCE = original")

# 2) Add DataCollatorForCompletionOnlyLM import
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'from trl import SFTTrainer, SFTConfig' in src and 'DataCollatorForCompletionOnlyLM' not in src:
        new_src = []
        for line in cell['source']:
            line = line.replace(
                'from trl import SFTTrainer, SFTConfig',
                'from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM'
            )
            new_src.append(line)
        nb['cells'][i]['source'] = new_src
        print(f"Fixed import in cell {i}")

# 3) Add data_collator to SFTTrainer
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'trainer = SFTTrainer(' in src and 'data_collator' not in src:
        new_src = []
        for line in cell['source']:
            if line.strip() == 'args=training_args':
                new_src.append('    data_collator=DataCollatorForCompletionOnlyLM(\n')
                new_src.append('        response_template="<|im_start|>assistant\\n",\n')
                new_src.append('        tokenizer=tokenizer,\n')
                new_src.append('    ),\n')
            new_src.append(line)
        nb['cells'][i]['source'] = new_src
        print(f"Fixed trainer in cell {i}")

with open(path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open(path, 'r') as f:
    nb2 = json.load(f)

for i, cell in enumerate(nb2['cells']):
    src = ''.join(cell['source'])
    if 'DATA_SOURCE =' in src:
        for line in src.split('\n'):
            if 'DATA_SOURCE =' in line and not line.strip().startswith('#'):
                print(f"Verified cell {i}: {line.strip()}")
    if 'DataCollatorForCompletionOnlyLM' in src:
        print(f"Cell {i} has DataCollator")
    if 'data_collator' in src:
        print(f"Cell {i} has data_collator param")

print("\nDone!")
