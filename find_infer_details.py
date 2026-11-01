import json

with open('kaggle_v2/name-that-llm-v2-sota-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 6 contains infer.py
infer_code = "".join(nb['cells'][6]['source'])
lines = infer_code.splitlines()

# Search for hungarian, sinkhorn, cluster, etc.
for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['hungarian', 'linear_sum_assignment', 'sinkhorn', 'cluster', 'twin', 'claude', 'adapt']):
        print(f"L{i:4d}: {line[:100]}")

