import json

with open('reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Synthreaper cells:", len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    txt = "".join(cell.get('source', []))
    lines = txt.splitlines()
    print(f"Cell {i} [{cell['cell_type']}]: {len(lines)} lines | {lines[0] if lines else ''}")
    for l in lines[:15]:
        if any(w in l for w in ['W_', 'MODELS', 'sinkhorn', 'cmd', 'infer.py', 'LB', '0.6', '0.7', 'atlas_w']):
            print("   *", l.strip()[:100])

