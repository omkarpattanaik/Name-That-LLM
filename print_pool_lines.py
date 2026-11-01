import json

with open('reference_kernels/omer_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_code = "".join(nb['cells'][6]['source'])
lines = infer_code.splitlines()
for i in range(195, 230):
    print(f"L{i:4d}: {lines[i]}")

