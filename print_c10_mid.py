import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c10 = "".join(nb['cells'][10]['source'])
lines = c10.splitlines()
for i in range(630, 655):
    print(f"{i:3d}: {lines[i]}")

