#!/usr/bin/env python3
"""Compare enable_thinking=True vs False template outputs."""
import os

paths = [
    os.path.expanduser('~/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1'),
    os.path.expanduser('~/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default'),
]
MODEL_PATH = None
for p in paths:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if not MODEL_PATH:
    print("Model not found locally")
    exit()

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
messages = [
    {'role': 'user', 'content': 'What is 2+2?' + SUFFIX},
    {'role': 'assistant', 'content': '\\boxed{4}'},
]

# Test enable_thinking=True (v33 approach)
print('=== enable_thinking=True (v33/E1 approach) ===')
try:
    text_think = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
    )
    print(repr(text_think))
    print()
    print("Readable:")
    print(text_think)
except Exception as e:
    print(f"ERROR: {e}")
    text_think = None

print()
print('=== enable_thinking=False (baseline 0.66 approach) ===')
try:
    text_nothink = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    print(repr(text_nothink))
    print()
    print("Readable:")
    print(text_nothink)
except Exception as e:
    print(f"ERROR: {e}")
    text_nothink = None

print()
print('=== No enable_thinking kwarg at all ===')
try:
    text_default = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    print(repr(text_default))
except Exception as e:
    print(f"ERROR: {e}")
    text_default = None

if text_think and text_nothink:
    print()
    ids_think = tokenizer.encode(text_think)
    ids_nothink = tokenizer.encode(text_nothink)
    print(f'enable_thinking=True:  {len(ids_think)} tokens')
    print(f'enable_thinking=False: {len(ids_nothink)} tokens')
    print(f'Difference: {len(ids_think) - len(ids_nothink)} extra tokens')
    
    # Show the extra tokens
    print()
    print('=== Token-level diff ===')
    for i, (a, b) in enumerate(zip(ids_think, ids_nothink)):
        if a != b:
            print(f'  Position {i}: thinking={a} ({tokenizer.decode([a])!r}) vs nothink={b} ({tokenizer.decode([b])!r})')
            # Show surrounding context
            remain_think = ids_think[i:i+10]
            remain_nothink = ids_nothink[i:i+10]
            print(f'    thinking continues: {tokenizer.decode(remain_think)!r}')
            print(f'    nothink continues:  {tokenizer.decode(remain_nothink)!r}')
            break
