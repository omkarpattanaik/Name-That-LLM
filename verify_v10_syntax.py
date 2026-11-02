import ast
import json

with open('kaggle_v10/name-that-llm-v10-apex-grandmaster.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in V10 notebook: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        # Strip cell magic
        lines = [l for l in source.splitlines() if not l.strip().startswith('%') and not l.strip().startswith('!')]
        cleaned_source = '\n'.join(lines)
        try:
            ast.parse(cleaned_source)
            print(f"Cell {idx}: syntax OK (length {len(source)} chars)")
        except SyntaxError as e:
            print(f"Cell {idx}: SYNTAX ERROR: {e}")
            raise

print("\nAll code cells in V10 notebook validated successfully with ast.parse!")

