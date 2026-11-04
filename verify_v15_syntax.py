import json
import ast

with open('kaggle_v15/name-that-llm-v15-apex-grandmaster-sovereign.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in V15: {len(nb['cells'])}")
errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        cleaned_lines = []
        for line in source.splitlines():
            if line.strip().startswith('%%') or line.strip().startswith('!'):
                continue
            cleaned_lines.append(line)
        cleaned_code = "\n".join(cleaned_lines)
        try:
            ast.parse(cleaned_code)
            print(f"Cell {i:02d}: Syntax OK ({len(cell['source'])} lines)")
        except SyntaxError as e:
            print(f"Cell {i:02d}: SYNTAX ERROR: {e}")
            errors.append((i, e))

if not errors:
    print("\nALL V15 CODE CELLS PASSED SYNTAX VALIDATION!")
else:
    print(f"\nFOUND {len(errors)} SYNTAX ERRORS!")
