#!/usr/bin/env python3
"""Extract key configuration from all available notebooks."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

notebooks = [
    'kaggle_scripts/sft_old/sfttrainer-training.ipynb',
    'kaggle_scripts/sft/nvidia-nemotron-sfttrainer-training.ipynb',
    'nvidia-nemotron-sfttrainer-training.ipynb',
    'nvidia-nemotron-sfttrainer-v26.ipynb',
    'nvidia-nemotron-sfttrainer-v27.ipynb',
    'nvidia-nemotron-sfttrainer-v32.ipynb',
    'nvidia-nemotron-sfttrainer-v33.ipynb',
    'nvidia-nemotron-grpotrainer-training.ipynb',
]

keywords = [
    'SFT_SAMPLES_PER_TYPE', 'learning_rate', 'lora_dropout', 'lora_alpha',
    'enable_thinking', 'max_seq_length', 'num_train_epochs',
    'SUFFIX', 'data_file', 'csv_file', '.csv', '.jsonl',
    'boxed_only', 'response_template', 'DataCollator', 'METRIC_SUFFIX',
    'logging_steps', 'per_device_train_batch_size', 'gradient_accumulation',
]

for nb_path in notebooks:
    full_path = os.path.join(ROOT, nb_path)
    try:
        with open(full_path) as f:
            nb = json.load(f)
        cells = nb.get('cells', [])
        code_cells = [c for c in cells if c['cell_type'] == 'code']
        all_text = '\n'.join(''.join(c['source']) for c in code_cells)
        
        print(f'=== {nb_path} ===')
        print(f'  Code cells: {len(code_cells)}')
        
        seen = set()
        for kw in keywords:
            lines = [l.strip() for l in all_text.split('\n') 
                     if kw in l and not l.strip().startswith('#') and l.strip() not in seen]
            for l in lines[:2]:
                seen.add(l.strip())
                if len(l) > 130:
                    l = l[:130] + '...'
                print(f'  {l}')
        print()
    except Exception as e:
        print(f'=== {nb_path} === ERROR: {e}\n')
