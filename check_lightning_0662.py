import json

with open('reference_kernels/lightning_0662/0-662-qwen3-8-27b-lora-weighted-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Cells in lightning_0662:", len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    txt = "".join(cell.get('source', []))
    lines = txt.splitlines()
    first = lines[0] if lines else ""
    if any(k in txt.lower() for k in ['score', 'lb', 'weight', 'models =', 'cmd =', 'sinkhorn', 'spectral']):
        print(f"Cell {i} [{cell['cell_type']}]: {first[:80]}")
        for l in lines[:10]:
            if any(k in l.lower() for k in ['lb', 'score', 'sinkhorn', 'spectral', 'cmd =', 'weight']):
                print("   *", l.strip()[:100])

