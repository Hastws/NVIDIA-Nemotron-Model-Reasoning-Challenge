#!/usr/bin/env python3
"""Extract key cells from downloaded Kaggle notebooks."""
import json
import sys

def extract_cells(path, keywords):
    nb = json.load(open(path))
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell['source'])
        if any(kw in src for kw in keywords):
            print(f"=== Cell {i} ({cell['cell_type']}) ===")
            print(src)
            print()

# SFT notebook - build_training_text
print("=" * 70)
print("SFT NOTEBOOK - build_training_text + config")
print("=" * 70)
extract_cells(
    'kaggle_scripts/sft/nvidia-nemotron-sfttrainer-training.ipynb',
    ['def build_training_text']
)

# GRPO notebook - build_training_text + reward
print("=" * 70)
print("GRPO NOTEBOOK - build_training_text + reward")
print("=" * 70)
extract_cells(
    'kaggle_scripts/grpo/nvidia-nemotron-grpotrainer-training.ipynb',
    ['def build_training_text', 'def reward', 'def format_reward']
)
