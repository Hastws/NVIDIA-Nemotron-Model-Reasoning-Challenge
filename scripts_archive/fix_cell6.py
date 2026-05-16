"""Fix cell 6 data loading section."""
import json

NB_PATH = "nvidia-nemotron-sfttrainer-training.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = cell.get("source", [])
    joined = "".join(src)
    
    if "DATA_SOURCE ==" in joined and "ao_7741" in joined:
        # Rebuild the data loading section properly
        new_lines = []
        skip_next = False
        for i, line in enumerate(src):
            if skip_next:
                skip_next = False
                continue
            
            # Fix: the raise ValueError line lost its else: prefix
            if line.strip() == "raise ValueError(f\"Unknown DATA_SOURCE: {DATA_SOURCE}\")":
                new_lines.append("else:\n")
                new_lines.append("    raise ValueError(f\"Unknown DATA_SOURCE: {DATA_SOURCE}\")\n")
                continue
            
            new_lines.append(line)
        
        cell["source"] = new_lines
        print("Fixed cell 6")
        
        # Verify
        for line in new_lines:
            if "DATA_SOURCE" in line or "ao_7741" in line or "raise" in line:
                print(f"  {line.rstrip()}")
        break

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("Done")
