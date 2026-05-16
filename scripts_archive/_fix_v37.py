#!/usr/bin/env python3
"""Fix v37 notebook: replace cell 13 (GRPO data prep) to add _classify_type()."""
import json

NB_PATH = 'nvidia-nemotron-cot-grpo-v37.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

# The new cell source code
new_source = r'''# --- Free SFT trainer memory ---
del trainer
gc.collect()
torch.cuda.empty_cache()
print(f"GPU memory after SFT cleanup: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

# --- Mock missing optional deps that TRL imports at module level ---
import types as _types

def _create_mock_module(name, attrs=None):
    mod = _types.ModuleType(name)
    mod.__path__ = []
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod

_mock_class = type("_Mock", (), {})

_all_mocks = {
    "mergekit": {},
    "mergekit.config": {"MergeConfiguration": _mock_class},
    "mergekit.merge": {"MergeOptions": _mock_class, "run_merge": lambda *a, **kw: None},
    "mergekit.architecture": {},
    "mergekit.io": {},
    "mergekit.io.tasks": {},
    "mergekit.io.lazy_tensor_loader": {},
    "mergekit.common": {},
    "mergekit.graph": {},
    "mergekit.merge_methods": {},
    "mergekit.options": {},
    "mergekit.plan": {},
    "mergekit.sparsify": {},
    "llm_blender": {"Blender": _mock_class},
    "weave": {"EvaluationLogger": _mock_class},
    "weave.trace": {},
    "weave.trace.context": {"weave_client_context": _mock_class},
    "liger_kernel": {},
    "liger_kernel.transformers": {},
}

for pkg_name, attrs in _all_mocks.items():
    if pkg_name not in sys.modules:
        sys.modules[pkg_name] = _create_mock_module(pkg_name, attrs)

sys.modules["weave"].trace = sys.modules["weave.trace"]
sys.modules["weave.trace"].context = sys.modules["weave.trace.context"]
sys.modules["mergekit"].config = sys.modules["mergekit.config"]
sys.modules["mergekit"].merge = sys.modules["mergekit.merge"]
sys.modules["mergekit"].io = sys.modules["mergekit.io"]
sys.modules["mergekit.io"].tasks = sys.modules["mergekit.io.tasks"]
sys.modules["mergekit.io"].lazy_tensor_loader = sys.modules["mergekit.io.lazy_tensor_loader"]

print(f"✓ Mocked {len(_all_mocks)} optional TRL dependencies")

# --- GRPO imports ---
import re
from trl import GRPOTrainer, GRPOConfig

# --- Prepare GRPO dataset ---
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

grpo_df = pl.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')

# --- Classify type from prompt text (train.csv has no "type" column) ---
_type_keywords = {
    'bit_ops': ['bitwise', 'bit operation', 'bit shift', 'XOR', 'AND, OR, NOT'],
    'gravity': ['gravitational', 'gravity', 'celestial', 'planet', 'gravitational constant'],
    'unit_conv': ['unit conversion', 'convert the following measurement', 'secret unit'],
    'cipher': ['encryption', 'cipher', 'encrypt', 'decrypt', 'encoded', 'secret code'],
    'numeral': ['numeral system', 'Roman numeral', 'ancient numeral', 'number system'],
    'symbol': ['symbol', 'symbolic', 'equation', 'transformation rule', 'symbol manipulation'],
}

def _classify_type(prompt):
    p_lower = prompt.lower()
    for t, kws in _type_keywords.items():
        for kw in kws:
            if kw.lower() in p_lower:
                return t
    return "unknown"

grpo_df = grpo_df.with_columns(
    pl.col("prompt").map_elements(_classify_type, return_dtype=pl.Utf8).alias("type")
)
print("Type distribution in full dataset:")
print(grpo_df.group_by("type").len().sort("type"))

# Strategic sampling: focus on improvable types, skip near-perfect ones
type_quotas = {
    "numeral": 50,        # already near-perfect, minimal GRPO
    "gravity": 200,       # decent headroom
    "unit_conv": 200,     # decent headroom
    "cipher": 200,        # biggest opportunity
    "bit_ops": 175,       # hardest but RL can discover patterns
    "symbol": 175,        # hard, RL exploration may help
}
grpo_frames = []
for t, n in type_quotas.items():
    subset = grpo_df.filter(pl.col("type") == t)
    actual_n = min(n, len(subset))
    grpo_frames.append(subset.sample(n=actual_n, seed=42))
grpo_df = pl.concat(grpo_frames).sample(fraction=1.0, seed=42)  # shuffle

print(f"\nGRPO dataset: {len(grpo_df)} samples")
for t in type_quotas:
    cnt = len(grpo_df.filter(pl.col("type") == t))
    print(f"  {t}: {cnt}")

# Pre-format prompts with enable_thinking=True
def format_grpo_prompt(example):
    user_msg = example["prompt"] + METRIC_SUFFIX
    messages = [{"role": "user", "content": user_msg}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
    except Exception:
        text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    return {"prompt": text, "answer": example["answer"]}

grpo_dataset = Dataset.from_pandas(grpo_df.to_pandas())
grpo_dataset = grpo_dataset.map(format_grpo_prompt, remove_columns=[c for c in grpo_dataset.column_names if c not in ["prompt", "answer"]])
print(f"✓ GRPO prompts formatted with enable_thinking=True")
print(f"Example prompt (first 200 chars):\n{grpo_dataset[0]['prompt'][:200]}")

# --- Unified reward function ---
def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'\\boxed\{([^}]*?)$', text)
    if m:
        return m.group(1).strip()
    return None

def answers_match(pred, gold):
    if pred is None:
        return False
    p, g = pred.strip().lower(), gold.strip().lower()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) <= 1e-2
    except (ValueError, OverflowError):
        return False

_debug_count = 0

def reward_func(completions, answer, **kwargs):
    """Single unified reward: correct=+1.0, has_format_wrong=-0.5, no_format=-1.0"""
    global _debug_count
    rewards = []
    for comp, gold in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        pred = extract_boxed_answer(text)

        if _debug_count < 8:
            print(f"[GRPO debug #{_debug_count}] pred={pred} | gold={gold} | text_tail=...{text[-150:]}")
            _debug_count += 1

        if pred is None:
            rewards.append(-1.0)  # no format at all
        elif answers_match(pred, gold):
            rewards.append(1.0)   # correct!
        else:
            rewards.append(-0.5)  # format ok but wrong answer
    return rewards

print("✓ GRPO data and reward function ready")
'''

# Convert to notebook source format (list of lines with \n)
lines = new_source.split('\n')
source_lines = []
for i, line in enumerate(lines):
    if i < len(lines) - 1:
        source_lines.append(line + '\n')
    else:
        source_lines.append(line)  # last line no trailing newline

# Replace cell 13
nb['cells'][13]['source'] = source_lines

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

# Verify
with open(NB_PATH) as f:
    nb2 = json.load(f)
src = ''.join(nb2['cells'][13]['source'])
print(f"✓ Fixed cell 13:")
print(f"  Has _classify_type: {'_classify_type' in src}")
print(f"  Has type_col bug: {'type_col = grpo_df' in src}")
print(f"  Source lines: {len(nb2['cells'][13]['source'])}")
