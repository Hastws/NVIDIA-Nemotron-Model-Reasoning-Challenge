#!/usr/bin/env python3
"""Test Nemotron chat template to understand exact format for CoT training."""
import json
from jinja2 import BaseLoader, Environment

# Load the chat template
cfg = json.load(open('/tmp/nemotron_tokenizer/tokenizer_config.json'))
template_str = cfg['chat_template']

env = Environment(loader=BaseLoader())
# Add tojson filter
env.filters['tojson'] = lambda x: json.dumps(x)
template = env.from_string(template_str)

print("=" * 70)
print("TEST 1: Answer-only, enable_thinking=True (E1 GRPO style)")
print("=" * 70)
messages = [
    {"role": "user", "content": "What is 2+2?\nPut your final answer inside \\boxed{}."},
    {"role": "assistant", "content": "\\boxed{4}"},
]
result = template.render(
    messages=messages,
    add_generation_prompt=False,
    enable_thinking=True,
)
print(repr(result))
print()

print("=" * 70)
print("TEST 2: Answer-only, enable_thinking=False (E1 SFT style)")
print("=" * 70)
result = template.render(
    messages=messages,
    add_generation_prompt=False,
    enable_thinking=False,
)
print(repr(result))
print()

print("=" * 70)
print("TEST 3: With reasoning_content field, enable_thinking=True")
print("=" * 70)
messages_cot = [
    {"role": "user", "content": "What is 2+2?\nPut your final answer inside \\boxed{}."},
    {"role": "assistant", "content": "\\boxed{4}", "reasoning_content": "2+2=4, simple addition."},
]
result = template.render(
    messages=messages_cot,
    add_generation_prompt=False,
    enable_thinking=True,
)
print(repr(result))
print()

print("=" * 70)
print("TEST 4: With <think> in content directly, enable_thinking=True")
print("=" * 70)
messages_think = [
    {"role": "user", "content": "What is 2+2?\nPut your final answer inside \\boxed{}."},
    {"role": "assistant", "content": "<think>\n2+2=4, simple addition.\n</think>\n\\boxed{4}"},
]
result = template.render(
    messages=messages_think,
    add_generation_prompt=False,
    enable_thinking=True,
)
print(repr(result))
print()

print("=" * 70)
print("TEST 5: Generation prompt, enable_thinking=True")
print("=" * 70)
messages_gen = [
    {"role": "user", "content": "What is 2+2?\nPut your final answer inside \\boxed{}."},
]
result = template.render(
    messages=messages_gen,
    add_generation_prompt=True,
    enable_thinking=True,
)
print(repr(result))
print()

print("=" * 70)
print("TEST 6: Generation prompt, enable_thinking=False")
print("=" * 70)
result = template.render(
    messages=messages_gen,
    add_generation_prompt=True,
    enable_thinking=False,
)
print(repr(result))
