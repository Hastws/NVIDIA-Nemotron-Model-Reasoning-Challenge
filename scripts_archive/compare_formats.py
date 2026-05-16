#!/usr/bin/env python3
"""Compare manual ChatML vs template-based reasoning_content format."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/tmp/nemotron_tokenizer', trust_remote_code=True)

user_msg = 'What is 2+2?\nPut your final answer inside \\boxed{}.'
thinking = 'Step 1: 2+2=4'
answer = '4'

# Method 1: OLD manual ChatML (what we used before in CoT training)
manual = (
    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
    f"<|im_start|>assistant\n<think>\n{thinking}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
)

# Method 2: NEW reasoning_content field
messages_cot = [
    {"role": "user", "content": user_msg},
    {"role": "assistant", "content": f"\\boxed{{{answer}}}", "reasoning_content": thinking},
]
template_cot = tokenizer.apply_chat_template(
    messages_cot, tokenize=False, add_generation_prompt=False, enable_thinking=True
)

# Method 3: answer-only + enable_thinking=True (E1 format)
messages_ao = [
    {"role": "user", "content": user_msg},
    {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
]
template_ao = tokenizer.apply_chat_template(
    messages_ao, tokenize=False, add_generation_prompt=False, enable_thinking=True
)

print("=== Method 1: Manual ChatML (OLD CoT) ===")
print(repr(manual))
print()
print("=== Method 2: reasoning_content (NEW CoT) ===")
print(repr(template_cot))
print()
print("=== Method 3: answer-only + thinking=True (E1) ===")
print(repr(template_ao))
print()

# Diff
print("=== EXACT DIFF: Manual vs Template ===")
if manual == template_cot:
    print("IDENTICAL!")
else:
    print("DIFFERENT!")
    for i, (a, b) in enumerate(zip(manual, template_cot)):
        if a != b:
            print(f"  First diff at pos {i}: manual={repr(a)} vs template={repr(b)}")
            print(f"  Manual context: ...{repr(manual[max(0,i-20):i+20])}...")
            print(f"  Template context: ...{repr(template_cot[max(0,i-20):i+20])}...")
            break
    if len(manual) != len(template_cot):
        print(f"  Length: manual={len(manual)} vs template={len(template_cot)}")
        # Show endings
        print(f"  Manual ending: {repr(manual[-50:])}")
        print(f"  Template ending: {repr(template_cot[-50:])}")

# Also compare tokenization
print()
print("=== TOKENIZATION COMPARISON ===")
manual_tokens = tokenizer.encode(manual)
template_tokens = tokenizer.encode(template_cot)
ao_tokens = tokenizer.encode(template_ao)
print(f"Manual ChatML: {len(manual_tokens)} tokens")
print(f"Template CoT:  {len(template_tokens)} tokens")
print(f"E1 answer-only: {len(ao_tokens)} tokens")

if manual_tokens == template_tokens:
    print("Token sequences IDENTICAL!")
else:
    print("Token sequences DIFFERENT!")
    for i, (a, b) in enumerate(zip(manual_tokens, template_tokens)):
        if a != b:
            print(f"  First diff at token pos {i}: manual={a}({repr(tokenizer.decode([a]))}) vs template={b}({repr(tokenizer.decode([b]))})")
            break
