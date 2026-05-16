"""Verify notebook edits for Stage 2."""
import json

with open('nvidia-nemotron-2stage-sft.ipynb') as f:
    nb = json.load(f)

# Verify Cell 8 (format builders)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'build_stage2_text' in src and 'build_stage1_text' in src:
            print(f'=== Cell {i} Stage 2 builder + verification ===')
            lines = src.split('\n')
            for j, line in enumerate(lines):
                if 'Stage 2 builder' in line or 'THINK_END' in line:
                    for k in range(j, min(len(lines), j+50)):
                        print(f'{k:3d}: {lines[k]}')
                    break
            print()
            for j, line in enumerate(lines):
                if 'STAGE 2 FORMAT VERIFICATION' in line:
                    for k in range(j, min(len(lines), j+30)):
                        print(f'{k:3d}: {lines[k]}')
                    break
            break

print()

# Verify Cell 15 (Stage 2 training)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'STAGE2_ENABLED' in src and 'stage2_trainer' in src:
            print(f'=== Cell {i} Stage 2 training ===')
            lines = src.split('\n')
            for k in range(min(30, len(lines))):
                print(f'{k:3d}: {lines[k]}')
            print('...')
            for k in range(max(0, len(lines)-15), len(lines)):
                print(f'{k:3d}: {lines[k]}')
            break

# Check JSON validity
print()
try:
    json.dumps(nb)
    print("JSON valid")
except Exception as e:
    print(f"JSON ERROR: {e}")
