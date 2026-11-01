import json, difflib, sys
sys.stdout.reconfigure(encoding='utf-8')

with open("kaggle_v2/name-that-llm-v2-sota-ensemble.ipynb", encoding="utf-8") as f:
    nb_v2 = json.load(f)

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb_zb = json.load(f)

for i in range(len(nb_v2['cells'])):
    c_v2 = "".join(nb_v2['cells'][i]['source'])
    c_zb = "".join(nb_zb['cells'][i]['source'])
    if c_v2 != c_zb and nb_v2['cells'][i]['cell_type'] == 'code':
        print(f"=== Diff in Code Cell {i} ===")
        for line in difflib.unified_diff(c_v2.splitlines(keepends=True), c_zb.splitlines(keepends=True), fromfile='v2', tofile='zb'):
            print(line, end='')

