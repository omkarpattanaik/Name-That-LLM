import json

with open('reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell2 = "".join(nb['cells'][2]['source'])
for i, line in enumerate(cell2.splitlines()):
    print(f"L{i:2d}: {line}")

