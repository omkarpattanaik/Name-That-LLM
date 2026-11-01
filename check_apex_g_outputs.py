import json

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    for out in c.get('outputs', []):
        if 'text' in out:
            txt = "".join(out['text'])
            if any(k in txt for k in ['score', '0.6', 'ENSEMBLE', 'accuracy']):
                print(f"Cell {i} output: {txt[:300].strip()}\n")

