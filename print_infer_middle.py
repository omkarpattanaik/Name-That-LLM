import json

with open('kaggle_v2/name-that-llm-v2-sota-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_code = "".join(nb['cells'][6]['source'])
lines = infer_code.splitlines()
for i in range(540, min(650, len(lines))):
    print(f"L{i:4d}: {lines[i]}")

