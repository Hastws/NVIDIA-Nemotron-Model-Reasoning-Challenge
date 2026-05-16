"""Modify notebook for v30: 7741 answer-only, LR=5e-5, 2 epochs."""
import json

NB_PATH = "nvidia-nemotron-sfttrainer-training.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

changes = 0

for cell in nb["cells"]:
    src = cell.get("source", [])
    joined = "".join(src)
    
    # Cell 4: Hyperparameters
    if "SUBSAMPLE_SIZE" in joined and "DATA_SOURCE" in joined and "NUM_EPOCHS" in joined:
        new_lines = []
        for line in src:
            if line.startswith("NUM_EPOCHS = "):
                new_lines.append("NUM_EPOCHS = 2          # 2 epochs: let model see all variants twice\n")
                changes += 1
            elif line.startswith("LR = "):
                new_lines.append("LR = 5e-5               # Lower LR: 7741 samples (13x more data)\n")
                changes += 1
            elif line.startswith("DATA_SOURCE = "):
                new_lines.append('DATA_SOURCE = "ao_7741"   # v30: 7741 solver-verified answer-only, 5 types, no symbol\n')
                changes += 1
            else:
                new_lines.append(line)
        cell["source"] = new_lines
    
    # Cell 6: Data loading - add ao_7741 branch
    if "DATA_SOURCE ==" in joined and "curated_700" in joined and "ao_7741" not in joined:
        new_lines = []
        for line in src:
            new_lines.append(line)
            if 'elif DATA_SOURCE == "curated_700":' in line:
                # After the curated_700 line, find and add ao_7741 after its body
                pass
        
        # Rebuild more carefully: insert ao_7741 branch before the else
        new_lines = []
        for i, line in enumerate(src):
            if line.strip().startswith("else:") and "raise ValueError" in "".join(src[i:i+2]):
                new_lines.append('elif DATA_SOURCE == "ao_7741":\n')
                new_lines.append("    train_df = pl.read_csv(f'{COT_DATA}/sft_ao_7741.csv')\n")
                changes += 1
            new_lines.append(line)
        cell["source"] = new_lines

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Applied {changes} changes to {NB_PATH}")
