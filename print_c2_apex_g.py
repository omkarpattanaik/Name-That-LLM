import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 2 of ntl-apex-g ===")
print("".join(nb['cells'][2]['source']))

