"""Fix v27 notebook for trl 0.24.0 compatibility.

Problem: DataCollatorForCompletionOnlyLM was removed from trl 0.24.0
Solution: Implement a minimal replacement inline
"""
import json
import sys

NB_PATH = "nvidia-nemotron-sfttrainer-v27.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

# --- Fix Cell 4: Remove DataCollatorForCompletionOnlyLM from import ---
cell4 = nb['cells'][4]
assert cell4['cell_type'] == 'code'
old_import = "from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM\n"
new_import = "from trl import SFTTrainer, SFTConfig\n"

found = False
for i, line in enumerate(cell4['source']):
    if line == old_import:
        cell4['source'][i] = new_import
        found = True
        break

if not found:
    print("ERROR: Could not find import line in Cell 4!")
    sys.exit(1)

print("✓ Fixed Cell 4: Removed DataCollatorForCompletionOnlyLM from imports")

# --- Fix Cell 11: Add manual implementation and update SFTTrainer call ---
cell11 = nb['cells'][11]
assert cell11['cell_type'] == 'code'
src = ''.join(cell11['source'])

# New cell 11 content with inline DataCollatorForCompletionOnlyLM
new_cell11 = '''import os
import triton.backends.nvidia.compiler as nv_compiler

# Tell Triton's environment parser where the writable Blackwell binary is
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
nv_compiler.get_ptxas_version = lambda arch: "12.0"

# --- Manual DataCollatorForCompletionOnlyLM (removed in trl 0.24) ---
from dataclasses import dataclass, field
from transformers import DataCollatorForLanguageModeling

@dataclass
class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """Masks labels for all tokens before (and including) the response template."""
    response_template_ids: list = field(default_factory=list)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            ids = batch["labels"][i].tolist()
            tpl = self.response_template_ids
            found = False
            for j in range(len(ids) - len(tpl) + 1):
                if ids[j:j+len(tpl)] == tpl:
                    batch["labels"][i][:j+len(tpl)] = -100
                    found = True
                    break
            if not found:
                batch["labels"][i][:] = -100  # template not found -> no loss
        return batch

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
    dataset_text_field="text",               
    max_length=MAX_SEQ_LEN,              
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)

# KEY CHANGE: response_template = "</think>\\n"
# This means loss is computed ONLY on \\boxed{answer}<|im_end|>
# NOT on <think></think> — so we never teach the model "don't think"
_resp_tpl = "</think>\\n"
_resp_ids = tokenizer.encode(_resp_tpl, add_special_tokens=False)
print(f"Response template: {repr(_resp_tpl)} -> token IDs: {_resp_ids}")

# Double-check: find it in a sample
_sample_text = hf_dataset[0]["text"]
_sample_ids = tokenizer.encode(_sample_text, add_special_tokens=False)
_found = False
for i in range(len(_sample_ids) - len(_resp_ids) + 1):
    if _sample_ids[i:i+len(_resp_ids)] == _resp_ids:
        _found = True
        print(f"✓ Response template found at token position {i}")
        print(f"  Loss will be on tokens {i+len(_resp_ids)}..{len(_sample_ids)} ({len(_sample_ids)-i-len(_resp_ids)} tokens)")
        break
if not _found:
    print("⚠ WARNING: response_template not found in tokenized sample!")

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    data_collator=CompletionOnlyCollator(
        tokenizer=tokenizer,
        mlm=False,
        response_template_ids=_resp_ids,
    ),
    args=training_args
)

print("Starting training...")
trainer.train()
'''

cell11['source'] = [line + '\n' for line in new_cell11.split('\n')]
# Remove trailing extra newline
if cell11['source'][-1] == '\n':
    cell11['source'] = cell11['source'][:-1]

print("✓ Fixed Cell 11: Added CompletionOnlyCollator implementation")

# Save
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✓ Saved to {NB_PATH}")
