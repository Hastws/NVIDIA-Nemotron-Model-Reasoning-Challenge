#!/usr/bin/env python3
"""Patch V2 notebook with cutlass/mamba3 mocks for current Kaggle env."""
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('nvidia-nemotron-sfttrainer-training_0.68.ipynb') as f:
    nb = json.load(f)

# Cell 9 is the model loading cell
cell9_src = ''.join(nb['cells'][9]['source'])
print("=== BEFORE (first 200 chars) ===")
print(cell9_src[:200])

# Prepend the cutlass mock code before "# Load Model"
mock_code = (
    "# --- Mock missing optional deps (cutlass/mamba3) ---\n"
    "from unittest.mock import MagicMock\n"
    "_mock_modules = [\n"
    "    'cutlass', 'cutlass.cute', 'cutlass.utils',\n"
    "    'mamba_ssm.ops.cute', 'mamba_ssm.ops.cute.mamba3',\n"
    "    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',\n"
    "    'mamba_ssm.ops.tilelang', 'mamba_ssm.ops.tilelang.mamba3',\n"
    "    'mamba_ssm.ops.tilelang.mamba3.mamba3_mimo',\n"
    "]\n"
    "for mod_name in _mock_modules:\n"
    "    if mod_name not in sys.modules:\n"
    "        sys.modules[mod_name] = MagicMock()\n"
    "\n"
)

old_start = "# Load Model"
new_src = cell9_src.replace(old_start, mock_code + old_start, 1)

# Add rmsnorm re-patch after model load
old_patch_end = '        print(f"Patched {name}: is_fast_path_available = False")'
rmsnorm_repatch = (
    '\n\n'
    '# Re-apply rmsnorm patch after model load\n'
    'for name, mod in list(sys.modules.items()):\n'
    '    if hasattr(mod, \'rmsnorm_fn\'):\n'
    '        mod.rmsnorm_fn = _pure_rmsnorm_fn'
)
new_src = new_src.replace(old_patch_end, old_patch_end + rmsnorm_repatch, 1)

# Convert back to cell source format
lines = new_src.split('\n')
nb['cells'][9]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

print("\n=== AFTER (first 500 chars) ===")
print(''.join(nb['cells'][9]['source'])[:500])

with open('nvidia-nemotron-sfttrainer-training_0.68.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\n✓ Notebook saved")
