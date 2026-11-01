import json

with open('reference_kernels/omer_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_code = "".join(nb['cells'][6]['source'])
for i, line in enumerate(infer_code.splitlines()):
    if 'pool' in line.lower() and ('read' in line.lower() or 'load' in line.lower() or 'df' in line.lower() or 'parquet' in line.lower()):
        print(f"L{i:4d}: {line}")

