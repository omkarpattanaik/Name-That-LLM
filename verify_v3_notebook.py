import json
import ast
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('kaggle_v3/name-that-llm-v3-sota-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Notebook loaded: {len(nb['cells'])} cells.")
errors = 0

for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = "".join(cell.get('source', []))
    print(f"Cell {i:2d} [{ctype}]: {len(source.splitlines())} lines")
    if ctype == 'code':
        # strip ipython magics for ast parsing
        clean_lines = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('%%') or stripped.startswith('%') or stripped.startswith('!'):
                clean_lines.append('# ' + line)
            else:
                clean_lines.append(line)
        clean_code = '\n'.join(clean_lines)
        try:
            ast.parse(clean_code)
            print(f"    --> Syntax check PASSED")
        except SyntaxError as e:
            print(f"    --> Syntax Error at line {e.lineno}: {e.msg}")
            errors += 1

print(f"\nVerification finished with {errors} errors.")

