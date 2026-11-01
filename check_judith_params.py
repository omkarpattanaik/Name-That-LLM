import json

with open('reference_kernels/judith_0664/0-664-hyperparameter-tuned-who-are-u.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [2, 10, 12]:
    if i < len(nb['cells']):
        c = nb['cells'][i]
        code = "".join(c.get('source', []))
        print(f"=== Cell {i} ===")
        for line in code.splitlines():
            if any(k in line for k in ['MODELS', 'cmd =', 'sinkhorn', 'atlas_w', 'twin', 'gauss', 'comp_w', 'adapt_w']):
                print("   ", line.strip())

