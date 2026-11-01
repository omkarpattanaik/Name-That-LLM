import json

with open("reference_kernels/judith_0664/0-664-hyperparameter-tuned-who-are-u.ipynb", encoding="utf-8") as f:
    nb_j = json.load(f)

print(f"Judith cell count: {len(nb_j['cells'])}")
for i, c in enumerate(nb_j['cells']):
    src = "".join(c['source'])
    print(f"Cell {i:2d} ({c['cell_type']}): {src[:60].replace(chr(10), ' ')}")

