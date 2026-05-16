#!/usr/bin/env python3
"""
Deep investigation: WHY does CoT training always hurt?
Test exactly what apply_chat_template(enable_thinking=True) produces
for different assistant content formats.
"""
from transformers import AutoTokenizer
import kagglehub

# Download model (just for tokenizer)
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("=" * 80)
print("TEST 1: V2 format — answer-only + enable_thinking=True")
print("=" * 80)
messages_v2 = [
    {"role": "user", "content": "What is 2+2?\nPut your final answer inside \\boxed{}."},
    {"role": "assistant", "content": "\\boxed{4}"},
]
text_v2 = tokenizer.apply_chat_template(
    messages_v2, tokenize=False, add_generation_prompt=False, enable_thinking=True
)
print(repr(text_v2))
print()
print(text_v2)
print()

print("=" * 80)
print("TEST 2: CoT format — <think> in assistant + enable_thinking=True")
print("=" * 80)
messages_cot = [
    {"role": "user", "content": "What is 2+2?\nPut your final answer inside \\boxed{}."},
    {"role": "assistant", "content": "<think>\n2+2=4\n</think>\n\\boxed{4}"},
]
text_cot = tokenizer.apply_chat_template(
    messages_cot, tokenize=False, add_generation_prompt=False, enable_thinking=True
)
print(repr(text_cot))
print()
print(text_cot)
print()

print("=" * 80)
print("TEST 3: CoT format — enable_thinking=FALSE")
print("=" * 80)
text_cot_nothink = tokenizer.apply_chat_template(
    messages_cot, tokenize=False, add_generation_prompt=False, enable_thinking=False
)
print(repr(text_cot_nothink))
print()
print(text_cot_nothink)
print()

print("=" * 80)
print("TEST 4: What does inference-time prompt look like?")
print("=" * 80)
messages_infer = [
    {"role": "user", "content": "What is 2+2?\nPut your final answer inside \\boxed{}."},
]
text_infer = tokenizer.apply_chat_template(
    messages_infer, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
print(repr(text_infer))
print()
print(text_infer)
print()

print("=" * 80)
print("TEST 5: Token-level comparison")
print("=" * 80)
# Tokenize both and show token counts
tokens_v2 = tokenizer(text_v2, return_tensors=None)["input_ids"]
tokens_cot = tokenizer(text_cot, return_tensors=None)["input_ids"]
print(f"V2 tokens: {len(tokens_v2)}")
print(f"CoT tokens: {len(tokens_cot)}")

# Find the <think> and </think> token IDs
think_open = tokenizer.encode("<think>", add_special_tokens=False)
think_close = tokenizer.encode("</think>", add_special_tokens=False)
print(f"\n<think> token IDs: {think_open}")
print(f"</think> token IDs: {think_close}")

# Show V2 tokens around assistant start
print(f"\nV2 first 30 tokens: {tokens_v2[:30]}")
print(f"V2 decoded first 30: {tokenizer.decode(tokens_v2[:30])}")

# Find assistant marker in V2
assistant_marker = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
print(f"\nassistant marker token IDs: {assistant_marker}")

# Show how many <think> tokens appear in each
v2_think_count = sum(1 for t in tokens_v2 if t == 12)  # <think> = 12
v2_think_close_count = sum(1 for t in tokens_v2 if t == 13)  # </think> = 13
cot_think_count = sum(1 for t in tokens_cot if t == 12)
cot_think_close_count = sum(1 for t in tokens_cot if t == 13)

print(f"\nV2: <think>(12) x{v2_think_count}, </think>(13) x{v2_think_close_count}")
print(f"CoT: <think>(12) x{cot_think_count}, </think>(13) x{cot_think_close_count}")

if cot_think_count > 1:
    print("\n⚠️⚠️⚠️ DOUBLE <think> DETECTED! ⚠️⚠️⚠️")
    print("This means CoT training teaches model: <think></think><think>cot</think>\\boxed{}")
    print("This is BROKEN — model learns two think blocks instead of one!")
