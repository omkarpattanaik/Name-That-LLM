import json

with open('reference_kernels/lightning_0662/0-662-qwen3-8-27b-lora-weighted-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell6 = "".join(nb['cells'][6]['source'])
for i, line in enumerate(cell6.splitlines()):
    print(f"L{i:3d}: {line}")

