import json

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb_zb = json.load(f)

for i in range(8, len(nb_zb['cells'])):
    c = nb_zb['cells'][i]
    src = "".join(c['source'])
    print(f"=== Zb Cell {i} ({c['cell_type']}) len {len(src)} ===")
    print(src[:300])
    print("...\n")

