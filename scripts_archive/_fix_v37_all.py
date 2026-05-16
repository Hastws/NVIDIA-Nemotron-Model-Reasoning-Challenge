#!/usr/bin/env python3
"""Fix v37 notebook: fix cell 4 (GRPO hyperparams) and cell 6 (data loading)."""
import json

NB_PATH = 'nvidia-nemotron-cot-grpo-v37.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

# =========================================================
# Fix Cell 4: GRPO hyperparameters (reduce for OOM)
# =========================================================
cell4_src = ''.join(nb['cells'][4]['source'])

# Replace GRPO_NUM_GEN
cell4_src = cell4_src.replace(
    'GRPO_NUM_GEN = 4            # 4 generations per prompt (was 2)',
    'GRPO_NUM_GEN = 2             # 2 generations (was 4 -> OOM)'
)
cell4_src = cell4_src.replace(
    'GRPO_MAX_COMPLETION = 1024  # let model think fully (was 384)',
    'GRPO_MAX_COMPLETION = 512    # 512 tokens max (was 1024 -> OOM)'
)

lines4 = cell4_src.split('\n')
source4 = []
for i, line in enumerate(lines4):
    if i < len(lines4) - 1:
        source4.append(line + '\n')
    else:
        source4.append(line)
nb['cells'][4]['source'] = source4

# =========================================================
# Fix Cell 6: Data loading -> use typed-CoT data
# =========================================================
new_cell6 = r'''MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

# ===== Stage 1: Load typed-CoT training data =====
# This CSV has columns: id, prompt, answer, thinking
# - 5 types have concise programmatic CoT in "thinking" column (300-600 chars)
# - symbol type has empty "thinking" (answer-only format, same as V2)
cot_df = pl.read_csv('/kaggle/input/prog-cot-training-data/sft_typed_cot_600.csv')
print(f"CoT training data: {len(cot_df)} samples")
print(f"Columns: {cot_df.columns}")
print(f"Has thinking: {cot_df.filter(pl.col('thinking').str.len_chars() > 0).shape[0]}")
print(f"Answer-only:  {cot_df.filter(pl.col('thinking').str.len_chars() == 0).shape[0]}")

# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(cot_df.to_pandas())

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SUFFIX = "\nPut your final answer inside \\boxed{}."

def build_training_text(example):
    """Build training text with type-specific CoT in <think> blocks."""
    prompt = example["prompt"]
    answer = example["answer"]
    thinking = example.get("thinking", "") or ""
    
    user_msg = prompt + SUFFIX
    
    if thinking.strip():
        # CoT types: put concise reasoning in <think> block
        assistant_msg = f"<think>\n{thinking}\n</think>\n\\boxed{{{answer}}}"
    else:
        # Symbol / answer-only: empty thinking (same as V2)
        assistant_msg = f"\\boxed{{{answer}}}"

    try:
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True
        )
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{thinking}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
        )
    return {"text": text}
'''

lines6 = new_cell6.split('\n')
source6 = []
for i, line in enumerate(lines6):
    if i < len(lines6) - 1:
        source6.append(line + '\n')
    else:
        source6.append(line)
nb['cells'][6]['source'] = source6

# =========================================================
# Save
# =========================================================
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

# Verify
with open(NB_PATH) as f:
    nb2 = json.load(f)

cell4_check = ''.join(nb2['cells'][4]['source'])
cell6_check = ''.join(nb2['cells'][6]['source'])
cell13_check = ''.join(nb2['cells'][13]['source'])

print("=== Cell 4 (Hyperparams) ===")
for line in nb2['cells'][4]['source']:
    if 'GRPO_NUM_GEN' in line or 'GRPO_MAX_COMPLETION' in line:
        print(f"  {line.rstrip()}")

print("\n=== Cell 6 (Data Loading) ===")
print(f"  Has typed-CoT: {'sft_typed_cot_600' in cell6_check}")
print(f"  Has old random: {'train_df.sample' in cell6_check}")
print(f"  Has thinking: {'thinking' in cell6_check}")

print("\n=== Cell 13 (GRPO Data) ===")
print(f"  Has _classify_type: {'_classify_type' in cell13_check}")
print(f"  Has type_col bug: {'type_col = grpo_df' in cell13_check}")

print("\n✓ All fixes applied and verified!")
