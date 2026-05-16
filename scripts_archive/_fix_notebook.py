"""Fix the corrupted build_training_text function in the notebook"""
import json

with open('nvidia-nemotron-sfttrainer-training.ipynb') as f:
    nb = json.load(f)

# Find the cell with build_training_text
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source'])
    if 'build_training_text' in source and cell['cell_type'] == 'code':
        print(f"Fixing cell {i}")
        
        old_lines = cell['source']
        
        # Print lines 56-end to see where corruption starts
        print("Current lines 56+:")
        for j in range(56, len(old_lines)):
            print(f"  {j:3d}: {repr(old_lines[j])}")
        
        # Keep lines 0-61 (up through the messages list closing bracket)
        # Lines 0-61 are: everything up through "]"
        good_lines = old_lines[:62]
        
        # Rebuild the correct tail
        new_tail = [
            "            text = tokenizer.apply_chat_template(\n",
            "                messages, tokenize=False, add_generation_prompt=False,\n",
            "                enable_thinking=True\n",
            "            )\n",
            "        except Exception:\n",
            "            text = (\n",
            "                f\"<|im_start|>user\\n{user_msg}<|im_end|>\\n\"\n",
            "                f\"<|im_start|>assistant\\n{assistant_msg}<|im_end|>\"\n",
            "            )\n",
            "    return {\"text\": text}\n",
        ]
        
        cell['source'] = good_lines + new_tail
        break

with open('nvidia-nemotron-sfttrainer-training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nNotebook saved. Verifying syntax...")

# Verify
with open('nvidia-nemotron-sfttrainer-training.ipynb') as f:
    nb2 = json.load(f)

for cell in nb2['cells']:
    source = ''.join(cell['source'])
    if 'build_training_text' in source and cell['cell_type'] == 'code':
        try:
            compile(source, '<cell>', 'exec')
            print("PASS: Syntax check passed")
        except SyntaxError as e:
            print(f"FAIL: Syntax error: {e}")
        
        # Show the fixed function
        lines = source.split('\n')
        in_func = False
        for line in lines:
            if 'def build_training_text' in line:
                in_func = True
            if in_func:
                print(line)
                if line.startswith('    return'):
                    break
        break
