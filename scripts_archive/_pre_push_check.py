#!/usr/bin/env python3
"""Pre-push verification: check all critical fixes are in the notebook on disk."""
import json, hashlib, ast

with open('nvidia-nemotron-2stage-sft.ipynb', 'rb') as f:
    content = f.read()
    md5 = hashlib.md5(content).hexdigest()
    print(f'File: {len(content)} bytes, MD5: {md5}')

nb = json.loads(content)
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f'Cells: {len(nb["cells"])} total, {len(code_cells)} code\n')

errors = []

for i, cell in enumerate(code_cells):
    src = ''.join(cell['source'])
    
    # Syntax check
    lines = [l for l in src.split('\n') if not l.strip().startswith('!')]
    try:
        ast.parse('\n'.join(lines))
    except SyntaxError as e:
        errors.append(f'Code cell {i}: SyntaxError at line {e.lineno}')
        print(f'❌ Code cell {i}: SyntaxError at line {e.lineno}: {e.msg}')

# Check _has_thinking exists
full_src = '\n'.join(''.join(c['source']) for c in code_cells)
checks = {
    'def _has_thinking': '_has_thinking function defined',
    'import math': 'math module imported',
    'math.isnan': 'NaN check with math.isnan',
    '_has_thinking(thinking)': 'build_stage1_text uses _has_thinking',
    "apply(_has_thinking)": 'verification uses _has_thinking',
    'STAGE1_MAX_SEQ   = 512': 'STAGE1_MAX_SEQ is 512',
}

for pattern, desc in checks.items():
    if pattern in full_src:
        print(f'✅ {desc}')
    else:
        errors.append(desc)
        print(f'❌ {desc} — NOT FOUND!')

# Check NO old vulnerable code
bad_patterns = {
    'if thinking and str(thinking).strip():': 'Old NaN-vulnerable thinking check',
    'STAGE1_MAX_SEQ   = 2048': 'Old max_seq=2048',
}
for pattern, desc in bad_patterns.items():
    if pattern in full_src:
        errors.append(desc)
        print(f'❌ Still has: {desc}')
    else:
        print(f'✅ Removed: {desc}')

if errors:
    print(f'\n❌ {len(errors)} PROBLEMS FOUND — DO NOT PUSH')
    for e in errors:
        print(f'  - {e}')
else:
    print(f'\n✅ ALL CHECKS PASSED — safe to push')
