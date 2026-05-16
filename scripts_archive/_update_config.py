#!/usr/bin/env python3
"""Update notebook config from smoke test to full training."""
import json

NB_PATH = "nvidia-nemotron-2stage-sft.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

# Find config cell
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "STAGE1_N_SAMPLES" in src and "STAGE2_N_SAMPLES" in src and "HYPERPARAMETERS" in src:
        print(f"Config is in cell index {i}")
        new_lines = []
        for line in cell["source"]:
            if "STAGE1_N_SAMPLES" in line and "=" in line and "print" not in line:
                line = "STAGE1_N_SAMPLES = None     # Full training - use all thinking rows\n"
            elif "STAGE2_N_SAMPLES" in line and "=" in line and "print" not in line:
                line = "STAGE2_N_SAMPLES = 4000     # All answer-only rows\n"
            new_lines.append(line)
        cell["source"] = new_lines
        print("Updated config:")
        for l in new_lines:
            if "N_SAMPLES" in l and "print" not in l:
                print(f"  {l.strip()}")
        break

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
