"""Remove duplicate else: line in cell 6."""
import json

NB_PATH = "nvidia-nemotron-sfttrainer-training.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = cell.get("source", [])
    joined = "".join(src)
    if "DATA_SOURCE ==" in joined and "ao_7741" in joined:
        # Remove the duplicate else: at line 14
        new_src = []
        prev_was_else = False
        for line in src:
            if line == "else:\n" and prev_was_else:
                continue  # skip duplicate
            prev_was_else = (line == "else:\n")
            new_src.append(line)
        cell["source"] = new_src
        print(f"Fixed: {len(src)} -> {len(new_src)} lines")
        # Verify the data loading section
        for line in new_src[7:17]:
            print(f"  {line.rstrip()}")
        break

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
print("Saved")
