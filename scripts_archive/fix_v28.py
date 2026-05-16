#!/usr/bin/env python3
"""Fix v28 training notebook: replace collator approach with pre-built labels."""
import json

NB_PATH = "nvidia-nemotron-sfttrainer-training.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

# Identify cells by content fingerprints
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    
    # Cell with build_training_text or build_training_example (data prep)
    if 'build_training_text' in src or 'build_training_example' in src:
        if 'apply_chat_template' in src:
            print(f"Found data prep cell at index {i}")
            # Replace with new version
            new_src = '''# Download model
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

# Load data based on DATA_SOURCE config
COMP_DATA = '/kaggle/input/nvidia-nemotron-3-reasoning-challenge'
COT_DATA = '/kaggle/input/prog-cot-training-data'

if DATA_SOURCE == "original":
    train_df = pl.read_csv(f'{COMP_DATA}/train.csv')
    train_df = train_df.sample(n=min(SUBSAMPLE_SIZE, len(train_df)), seed=42)
elif DATA_SOURCE == "curated_700":
    train_df = pl.read_csv(f'{COT_DATA}/sft_curated_700.csv')
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

# </think> special token ID — used to find where boxed answer starts
THINK_CLOSE_ID = tokenizer.convert_tokens_to_ids("</think>")
print(f"</think> token ID: {THINK_CLOSE_ID}")

def build_training_example(example):
    """Pre-tokenize with labels: mask everything up to and including </think>."""
    prompt = example["prompt"]
    answer = example["answer"]
    
    user_msg = prompt + SUFFIX
    assistant_msg = f"\\\\boxed{{{answer}}}"

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=True,
    )
    
    # Tokenize
    input_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)
    
    # Find </think> token and mask everything up to and including it
    prefix_len = len(input_ids)  # default: mask all (safety)
    for i, tid in enumerate(input_ids):
        if tid == THINK_CLOSE_ID:
            prefix_len = i + 1
            break
    
    labels = [-100] * prefix_len + input_ids[prefix_len:]
    
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }

# Verify on a sample
_sample = build_training_example({"prompt": "test", "answer": "42"})
_n_loss = sum(1 for x in _sample["labels"] if x != -100)
print(f"Sample: {len(_sample['input_ids'])} tokens, {_n_loss} loss tokens")
print(f"Loss tokens: {tokenizer.decode([t for t in _sample['labels'] if t != -100])}")
'''
            cell['source'] = [line + '\n' for line in new_src.rstrip('\n').split('\n')]
            print(f"  ✓ Replaced data prep cell")
    
    # Cell with hf_dataset.map (dataset mapping)
    if 'hf_dataset.map' in src and ('build_training_text' in src or 'build_training_example' in src):
        print(f"Found dataset map cell at index {i}")
        new_src = '''hf_dataset = hf_dataset.map(
    build_training_example,
    remove_columns=hf_dataset.column_names,
)

# Verify dataset
print(f"Dataset ready: {len(hf_dataset)} examples")
print(f"Keys: {list(hf_dataset[0].keys())}")
n_loss = sum(1 for x in hf_dataset[0]["labels"] if x != -100)
print(f"Example 0: {len(hf_dataset[0]['input_ids'])} tokens, {n_loss} loss tokens")
'''
        cell['source'] = [line + '\n' for line in new_src.rstrip('\n').split('\n')]
        print(f"  ✓ Replaced dataset map cell")
    
    # Cell with SFTTrainer (training cell)
    if 'SFTTrainer(' in src and 'trainer.train()' in src:
        print(f"Found training cell at index {i}")
        new_src = '''import os
import triton.backends.nvidia.compiler as nv_compiler

# Tell Triton's environment parser where the writable Blackwell binary is
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

from transformers import DataCollatorForSeq2Seq

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,           
    gradient_accumulation_steps=GRAD_ACCUM,  
    num_train_epochs=NUM_EPOCHS,             
    learning_rate=LR,                        
    logging_steps=5,                         
    bf16=True,                               
    max_grad_norm=1.0,                       
    optim="adamw_torch",                     
    lr_scheduler_type="cosine",              
    warmup_ratio=0.1,                        
    save_strategy="no",                      
    report_to="none",
    max_length=MAX_SEQ_LEN,              
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    dataset_kwargs={"skip_prepare_dataset": True},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=None,
    ),
    args=training_args,
)

print("Starting training...")
trainer.train()
'''
        cell['source'] = [line + '\n' for line in new_src.rstrip('\n').split('\n')]
        print(f"  ✓ Replaced training cell")

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✓ Saved {NB_PATH}")
