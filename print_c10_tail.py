import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c10 = "".join(nb['cells'][10]['source'])
lines = c10.splitlines()
print(f"Total lines in infer.py: {len(lines)}")
for i in range(len(lines) - 50, len(lines)):
    print(f"{i:3d}: {lines[i]}")

