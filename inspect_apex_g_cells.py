import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [3, 4, 11, 17]:
    print(f"\n=================== CELL {idx} ===================")
    print("".join(nb['cells'][idx]['source']))

