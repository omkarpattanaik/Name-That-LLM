import json, difflib, sys
sys.stdout.reconfigure(encoding='utf-8')

with open("kaggle_v2/name-that-llm-v2-sota-ensemble.ipynb", encoding="utf-8") as f:
    nb_v2 = json.load(f)

with open("reference_kernels/judith_0664/0-664-hyperparameter-tuned-who-are-u.ipynb", encoding="utf-8") as f:
    nb_j = json.load(f)

for i in [2, 4, 6, 8, 10, 12]:
    c_v2 = "".join(nb_v2['cells'][i]['source'])
    c_j = "".join(nb_j['cells'][i]['source'])
    if c_v2 != c_j:
        print(f"=== Diff in Judith Cell {i} ===")
        for line in difflib.unified_diff(c_v2.splitlines(keepends=True), c_j.splitlines(keepends=True), fromfile='v2', tofile='judith'):
            print(line, end='')

