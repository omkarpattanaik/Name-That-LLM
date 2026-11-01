import json, os, sys

# Load V2 notebook as base (which scored 0.662 cleanly)
with open('kaggle_v2/name-that-llm-v2-sota-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Base V2 notebook loaded with {len(nb['cells'])} cells.")
for i, c in enumerate(nb['cells']):
    src = "".join(c['source'])
    print(f"Cell {i:2d} ({c['cell_type']:8s}): {src[:60].replace(chr(10), ' ')}")

