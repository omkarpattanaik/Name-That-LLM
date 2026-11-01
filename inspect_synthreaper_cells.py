import json

with open('reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = cell.get('source', [])
    txt = "".join(source)
    lines = txt.splitlines()
    first_line = lines[0] if lines else ""
    print(f"Cell {i:2d} [{ctype:8s}] ({len(lines):4d} lines): {first_line[:80]}")

