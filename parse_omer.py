import json

with open('reference_kernels/omer_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    full_text = "".join(cell.get('source', []))
    lines = full_text.splitlines()
    first_line = lines[0] if lines else "<empty>"
    print(f"Cell {i} [{cell['cell_type']}]: {len(lines)} lines | Starts with: {first_line[:90]}")
    for line in lines[:8]:
        print("    ", line[:100])
