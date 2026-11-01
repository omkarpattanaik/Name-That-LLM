import json

with open('reference_kernels/lightning_0662/0-662-qwen3-8-27b-lora-weighted-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell41 = "".join(nb['cells'][41]['source'])
for i, line in enumerate(cell41.splitlines()):
    print(f"L{i:3d}: {line}")

