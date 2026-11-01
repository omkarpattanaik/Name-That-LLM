import json

with open('reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_code = "".join(nb['cells'][8]['source'])
for i, line in enumerate(infer_code.splitlines()):
    if 660 <= i <= 703:
        print(f"L{i:4d}: {line}")

