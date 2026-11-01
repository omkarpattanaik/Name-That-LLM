import json

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("\n=== Cell 0 ===")
print("".join(nb['cells'][0]['source'])[:800])

for i, c in enumerate(nb['cells']):
    src = "".join(c['source'])
    for kw in ['POOL2', 'pool2', '0.670', '0.669', '0.668', 'svc_w', 'open_pool']:
        if kw in src:
            print(f"Cell {i} ({c['cell_type']}) has '{kw}':")
            print(src[:300].replace(chr(10), ' '))
            print("-" * 40)
            break

