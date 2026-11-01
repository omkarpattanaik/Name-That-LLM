import json
import ast
import sys

nb_path = 'kaggle_v8/name-that-llm-v8-apex-10gb.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Verifying {nb_path} with {len(nb["cells"])} cells...')

errors = 0
for idx, cell in enumerate(nb['cells']):
    ct = cell.get('cell_type')
    src = ''.join(cell.get('source', []))
    if ct == 'code':
        # Check if cell uses %%writefile
        lines = src.split('\n')
        if lines and lines[0].startswith('%%writefile'):
            fname = lines[0].split()[1].strip()
            py_code = '\n'.join(lines[1:])
            try:
                ast.parse(py_code)
                print(f'  [PASS] Cell {idx}: %%writefile {fname} ({len(py_code.splitlines())} lines) valid Python syntax')
            except SyntaxError as e:
                print(f'  [FAIL] Cell {idx}: %%writefile {fname} SyntaxError: {e}')
                errors += 1
        else:
            try:
                ast.parse(src)
                print(f'  [PASS] Cell {idx}: Code cell ({len(lines)} lines) valid Python syntax')
            except SyntaxError as e:
                print(f'  [FAIL] Cell {idx}: Code cell SyntaxError: {e}')
                errors += 1

if errors == 0:
    print('ALL CELLS PASSED SYNTAX AND AST PARSING VERIFICATION PERFECTLY!')
else:
    print(f'FOUND {errors} SYNTAX ERRORS!')
    sys.exit(1)

