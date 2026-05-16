#!/usr/bin/env python3
"""
Modify SFT notebook to use reasoning_content field for CoT training.
Key changes:
1. Cell 4: DATA_SOURCE = "e1_hybrid_cot" 
2. Cell 6: Add data source, rewrite build_training_text to use reasoning_content
"""
import json
import copy

INPUT_NB = 'kaggle_scripts/sft/nvidia-nemotron-sfttrainer-training.ipynb'
OUTPUT_NB = 'nvidia-nemotron-sfttrainer-training.ipynb'

with open(INPUT_NB, 'r') as f:
    nb = json.load(f)

# --- Modify Cell 4: Change DATA_SOURCE ---
cell4_src = ''.join(nb['cells'][4]['source'])
cell4_src = cell4_src.replace(
    'DATA_SOURCE = "e1_cipher100"',
    'DATA_SOURCE = "e1_hybrid_cot"'
)
nb['cells'][4]['source'] = [cell4_src]

# --- Modify Cell 6: Rewrite data loading and build_training_text ---
new_cell6 = '''# Download model
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

# Load data based on DATA_SOURCE config
COMP_DATA = '/kaggle/input/nvidia-nemotron-3-reasoning-challenge'
COT_DATA = '/kaggle/input/prog-cot-training-data'

if DATA_SOURCE == "original":
    train_df = pl.read_csv(f'{COMP_DATA}/train.csv')
    train_df = train_df.sample(n=min(SUBSAMPLE_SIZE, len(train_df)), seed=42)
elif DATA_SOURCE == "e1_hybrid_cot":
    train_df = pl.read_csv(f'{COT_DATA}/sft_e1_hybrid_cot.csv')
elif DATA_SOURCE == "prog_cot_ao":
    train_df = pl.read_csv(f'{COT_DATA}/sft_prog_answer_only.csv')
elif DATA_SOURCE == "prog_cot":
    train_df = pl.read_csv(f'{COT_DATA}/sft_prog_with_cot.csv')
elif DATA_SOURCE == "prog_cot_200":
    train_df = pl.read_csv(f'{COT_DATA}/sft_prog_cot_200.csv')
elif DATA_SOURCE == "e1_enhanced":
    train_df = pl.read_csv(f'{COT_DATA}/sft_e1_enhanced.csv')
elif DATA_SOURCE == "e1_cipher100":
    train_df = pl.read_csv(f'{COT_DATA}/sft_e1_plus_cipher100.csv')
elif DATA_SOURCE == "e1_cipher200":
    train_df = pl.read_csv(f'{COT_DATA}/sft_e1_plus_cipher200.csv')
elif DATA_SOURCE == "enhanced_600":
    train_df = pl.read_csv(f'{COT_DATA}/sft_enhanced_600.csv')
elif DATA_SOURCE == "balanced_100":
    train_df = pl.read_csv(f'{COT_DATA}/sft_balanced_100.csv')
else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}")

print(f"Data source: {DATA_SOURCE}, samples: {len(train_df)}")

# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(train_df.to_pandas())

# Initialize tokenizer to build the text
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# METRIC_SUFFIX (aligned with evaluation)
SUFFIX = "\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`"

def build_training_text(example):
    prompt = example["prompt"]
    answer = example["answer"]
    thinking = example.get("thinking", None)
    
    user_msg = prompt + SUFFIX
    assistant_msg = f"\\\\boxed{{{answer}}}"

    messages = [{"role": "user", "content": user_msg}]
    
    if thinking:
        # CoT: use reasoning_content field for proper template formatting
        messages.append({
            "role": "assistant",
            "content": assistant_msg,
            "reasoning_content": thinking,
        })
    else:
        # Answer-only
        messages.append({
            "role": "assistant",
            "content": assistant_msg,
        })

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=True
    )
    return {"text": text}
'''

nb['cells'][6]['source'] = [new_cell6]

# Clear all outputs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(OUTPUT_NB, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Modified notebook saved to {OUTPUT_NB}")
print(f"Changes:")
print(f"  Cell 4: DATA_SOURCE = 'e1_hybrid_cot'")
print(f"  Cell 6: Added e1_hybrid_cot data source + reasoning_content build_training_text")
print(f"  SUFFIX aligned with evaluation metric suffix")
