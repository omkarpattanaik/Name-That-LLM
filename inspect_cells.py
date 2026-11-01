import json

with open('reference_kernels/omer_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [2, 4, 6, 8, 10, 12]:
    cell = nb['cells'][i]
    code = "".join(cell.get('source', []))
    lines = code.splitlines()
    print(f"=== Cell {i} ({len(lines)} lines) ===")
    for line in lines[:20]:
        print("  ", line)
    print("   ...")
    for line in lines[-10:]:
        print("  ", line)

