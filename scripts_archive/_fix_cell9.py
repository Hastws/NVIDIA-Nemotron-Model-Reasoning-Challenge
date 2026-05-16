#!/usr/bin/env python3
"""Fix Cell 9 to have exactly one copy of the mock + load + patch code."""
import json, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('nvidia-nemotron-sfttrainer-training_0.68.ipynb') as f:
    nb = json.load(f)

# Replace Cell 9 completely with clean version
new_cell9 = '''# --- Mock missing optional deps (cutlass/mamba3) ---
from unittest.mock import MagicMock
_mock_modules = [
    'cutlass', 'cutlass.cute', 'cutlass.utils',
    'mamba_ssm.ops.cute', 'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
    'mamba_ssm.ops.tilelang', 'mamba_ssm.ops.tilelang.mamba3',
    'mamba_ssm.ops.tilelang.mamba3.mamba3_mimo',
]
for mod_name in _mock_modules:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto", 
    trust_remote_code=True, 
    dtype=torch.bfloat16
)
print(f"Model loaded. Vocab size: {len(tokenizer)}")

# Force slow path — bypass the broken CUDA kernels
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

# Re-apply rmsnorm patch after model load
for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn

# Setup LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()'''

lines = new_cell9.split('\n')
nb['cells'][9]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open('nvidia-nemotron-sfttrainer-training_0.68.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
with open('nvidia-nemotron-sfttrainer-training_0.68.ipynb') as f:
    nb2 = json.load(f)
src = ''.join(nb2['cells'][9]['source'])
print(src)
print("\n✓ Clean Cell 9 written")
