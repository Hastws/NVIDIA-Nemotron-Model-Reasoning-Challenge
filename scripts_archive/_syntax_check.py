import json, ast

with open('nvidia-nemotron-2stage-sft.ipynb') as f:
    nb = json.load(f)

errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        lines = [l for l in src.split('\n') if not l.strip().startswith('!')]
        clean_src = '\n'.join(lines)
        try:
            ast.parse(clean_src)
        except SyntaxError as e:
            errors.append((i, e))
            print(f'Cell {i} (cell #{i+1}): SyntaxError at line {e.lineno}: {e.msg}')
            code_lines = clean_src.split('\n')
            start = max(0, e.lineno - 3)
            end = min(len(code_lines), e.lineno + 2)
            for j in range(start, end):
                marker = '>>>' if j == e.lineno - 1 else '   '
                print(f'  {marker} {j+1}: {repr(code_lines[j])}')

if not errors:
    n_code = len([c for c in nb['cells'] if c['cell_type'] == 'code'])
    print(f'All {n_code} code cells pass syntax check')
else:
    print(f'\n{len(errors)} cells with syntax errors')
