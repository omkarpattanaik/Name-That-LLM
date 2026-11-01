import json

with open('reference_kernels/lightning_0661/0-661-qwen3-8-27b-lora-spectral-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [2, 4, 12, 14, 20, 30, 40, 43, 44]:
    if i < len(nb['cells']):
        c = nb['cells'][i]
        txt = "".join(c.get('source', []))
        lines = txt.splitlines()
        print(f"=== Lightning Cell {i} [{c['cell_type']}] ({len(lines)} lines) ===")
        for l in lines[:10]:
            print("  ", l[:100])

