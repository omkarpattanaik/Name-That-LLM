import json

with open('kaggle_v2/name-that-llm-v2-sota-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

comp_code = "".join(nb['cells'][8]['source'])
lines = comp_code.splitlines()

for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['claude', 'gate_lam', 'conditional_family_gate', 'specialist', 'seyitkaan']):
        print(f"L{i:4d}: {line[:100]}")

