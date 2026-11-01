import json

with open('kaggle_v2/name-that-llm-v2-sota-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [2, 4, 6, 8, 10, 12]:
    cell = nb['cells'][i]
    code = "".join(cell.get('source', []))
    lines = code.splitlines()
    print(f"=== V2 Cell {i} ({len(lines)} lines) ===")
    for line in lines[:15]:
        print("  ", line)
    print("   ...")
    for line in lines[-8:]:
        print("  ", line)

