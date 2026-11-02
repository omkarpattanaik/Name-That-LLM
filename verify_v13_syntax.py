import json
import ast

with open('kaggle_v13/name-that-llm-v13-apex-grandmaster-sovereign.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # Strip magic lines like % or %%writefile
        code_lines = []
        for line in source.split('\n'):
            if line.strip().startswith('%'):
                continue
            code_lines.append(line)
        cleaned_code = '\n'.join(code_lines)
        try:
            ast.parse(cleaned_code)
        except SyntaxError as e:
            print(f"Cell {idx} SYNTAX ERROR: {e}")
            raise e

print("All code cells in V13 notebook parsed successfully with ZERO syntax errors!")
