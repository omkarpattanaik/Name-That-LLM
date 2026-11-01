import json

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c['source'])
    print(f"Cell {i:2d} ({c['cell_type']:8s}): {src[:70].replace(chr(10), ' ')}")

