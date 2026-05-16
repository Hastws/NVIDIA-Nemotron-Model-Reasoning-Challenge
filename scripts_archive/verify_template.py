#!/usr/bin/env python3
"""Verify response_template tokenization for DataCollatorForCompletionOnlyLM."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/tmp/nemotron_tokenizer', trust_remote_code=True)

# Check what tokens response_template produces
template = '<|im_start|>assistant\n'
token_ids = tokenizer.encode(template, add_special_tokens=False)
print(f'response_template tokens: {token_ids}')
print(f'decoded: {[tokenizer.decode([t]) for t in token_ids]}')

# Now check a full answer-only example
messages = [{'role': 'user', 'content': 'test prompt'}]
messages.append({'role': 'assistant', 'content': '\\boxed{42}'})
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
print(f'\nFull text:\n{repr(text)}')
full_ids = tokenizer.encode(text, add_special_tokens=False)
print(f'\nFull tokens ({len(full_ids)}): {full_ids}')

# Find where response_template appears
found = False
for i in range(len(full_ids) - len(token_ids) + 1):
    if full_ids[i:i+len(token_ids)] == token_ids:
        print(f'\nTemplate found at position {i}')
        resp_ids = full_ids[i+len(token_ids):]
        print(f'Response tokens: {resp_ids}')
        print(f'Response text: {repr(tokenizer.decode(resp_ids))}')
        found = True
        break

if not found:
    print('\nWARNING: Template NOT found in tokenized text!')
    print('Trying individual token search...')
    # Show what each token decodes to
    for i, tid in enumerate(full_ids):
        decoded = repr(tokenizer.decode([tid]))
        print(f'  {i}: {tid} -> {decoded}')
    
    # Try token-by-token matching  
    template_first = token_ids[0]
    for i, tid in enumerate(full_ids):
        if tid == template_first:
            match_len = 0
            for j in range(len(token_ids)):
                if i+j < len(full_ids) and full_ids[i+j] == token_ids[j]:
                    match_len += 1
                else:
                    break
            if match_len > 0:
                print(f'  Partial match at {i}: {match_len}/{len(token_ids)} tokens')

# Also try with token IDs directly (sometimes encoding differs in context)
print('\n\n--- Alternative: use token ID list as response_template ---')
# In TRL, you can pass token IDs directly
im_start_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
print(f'<|im_start|> token ID: {im_start_id}')
assistant_ids = tokenizer.encode('assistant\n', add_special_tokens=False)
print(f'"assistant\\n" token IDs: {assistant_ids}')
full_template_ids = [im_start_id] + assistant_ids
print(f'Full template IDs: {full_template_ids}')

# Check if this matches
for i in range(len(full_ids) - len(full_template_ids) + 1):
    if full_ids[i:i+len(full_template_ids)] == full_template_ids:
        print(f'Token ID template found at position {i}!')
        break
else:
    print('Token ID template NOT found either')
