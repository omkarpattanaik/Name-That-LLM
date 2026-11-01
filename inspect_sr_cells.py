import json

with open("reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb", encoding="utf-8") as f:
    nb_sr = json.load(f)

print(f"Synthreaper cell count: {len(nb_sr['cells'])}")
for i, c in enumerate(nb_sr['cells']):
    src = "".join(c['source'])
    print(f"Cell {i:2d} ({c['cell_type']}): {src[:80].replace(chr(10), ' ')}")

