#!/usr/bin/env python3
"""Test: find </think> token position reliably and build correct labels."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/tmp/nemotron_tokenizer', trust_remote_code=True)

SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
THINK_CLOSE_ID = 13  # </think> token

messages = [
    {"role": "user", "content": "What is 2+2?" + SUFFIX},
    {"role": "assistant", "content": "\\boxed{4}"},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False,
    enable_thinking=True,
)

full_ids = tokenizer.encode(text, add_special_tokens=False)

# Method: find </think> token ID (13) in full_ids, mask everything up to and including it
for i, tid in enumerate(full_ids):
    if tid == THINK_CLOSE_ID:
        prefix_len = i + 1  # include </think> itself
        break

labels = [-100] * prefix_len + full_ids[prefix_len:]
print(f"Total tokens: {len(full_ids)}")
print(f"Prefix (masked): {prefix_len} tokens")
print(f"Loss tokens: {len(full_ids) - prefix_len} tokens")
print()
print("Labels:")
for i, (tid, lbl) in enumerate(zip(full_ids, labels)):
    d = repr(tokenizer.decode([tid]))
    print(f"  {i}: tid={tid:>6} label={lbl:>6}  {d}")
