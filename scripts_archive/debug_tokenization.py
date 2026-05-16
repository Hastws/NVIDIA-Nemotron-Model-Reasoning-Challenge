#!/usr/bin/env python3
"""Debug why </think>\n response_template is not found in tokenized text."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/tmp/nemotron_tokenizer', trust_remote_code=True)

# Test 1: Encode '</think>\n' standalone
tpl = '</think>\n'
tpl_ids = tokenizer.encode(tpl, add_special_tokens=False)
print(f'Standalone encode of {repr(tpl)}: {tpl_ids}')
print(f'  Decoded: {[tokenizer.decode([t]) for t in tpl_ids]}')

# Test 2: Build a full training example (same as v28 build_training_text)
SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
messages = [
    {"role": "user", "content": "What is 2+2?" + SUFFIX},
    {"role": "assistant", "content": "\\boxed{4}"},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False,
    enable_thinking=True,
)
print(f'\nFull text:\n{repr(text)}')

# Test 3: Tokenize full text and show all tokens
full_ids = tokenizer.encode(text, add_special_tokens=False)
print(f'\nFull token IDs ({len(full_ids)}):')
for i, tid in enumerate(full_ids):
    d = repr(tokenizer.decode([tid]))
    print(f'  {i}: {tid} -> {d}')

# Test 4: Search for tpl_ids in full_ids
print(f'\n--- Searching for standalone-encoded {tpl_ids} in full_ids ---')
found = False
for i in range(len(full_ids) - len(tpl_ids) + 1):
    if full_ids[i:i+len(tpl_ids)] == tpl_ids:
        print(f'FOUND at position {i}!')
        found = True
        break
if not found:
    print('NOT FOUND!')
    
# Test 5: Check if </think> is a special token
print(f'\n--- Special token analysis ---')
think_close = '</think>'
think_close_id = tokenizer.convert_tokens_to_ids(think_close)
print(f'{repr(think_close)} token ID: {think_close_id}')

think_open = '<think>'
think_open_id = tokenizer.convert_tokens_to_ids(think_open)
print(f'{repr(think_open)} token ID: {think_open_id}')

# Check all_special_tokens
print(f'\nSpecial tokens with think:')
for tok in tokenizer.all_special_tokens:
    if 'think' in tok.lower():
        print(f'  {repr(tok)} -> {tokenizer.convert_tokens_to_ids(tok)}')

# Test 6: Try tokenize=True to see what the template produces
full_ids_from_template = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    enable_thinking=True,
)
print(f'\nTemplate tokenize=True ({len(full_ids_from_template)} tokens):')
for i, tid in enumerate(full_ids_from_template):
    d = repr(tokenizer.decode([tid]))
    print(f'  {i}: {tid} -> {d}')

# Compare
print(f'\nMatch between encode(text) and template(tokenize=True):')
print(f'  encode: {len(full_ids)} tokens')
print(f'  template: {len(full_ids_from_template)} tokens')
print(f'  Equal: {full_ids == full_ids_from_template}')
if full_ids != full_ids_from_template:
    for i in range(max(len(full_ids), len(full_ids_from_template))):
        a = full_ids[i] if i < len(full_ids) else None
        b = full_ids_from_template[i] if i < len(full_ids_from_template) else None
        if a != b:
            print(f'  DIFF at {i}: encode={a} vs template={b}')
