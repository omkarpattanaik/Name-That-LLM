import json, difflib

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb_zb = json.load(f)

with open("kaggle_v5/name-that-llm-v5-sota-apex-ensemble.ipynb", encoding="utf-8") as f:
    nb_v5 = json.load(f)

print("=== CELL 2 DIFF ===")
c2_zb = "".join(nb_zb['cells'][2]['source'])
c2_v5 = "".join(nb_v5['cells'][2]['source'])
for line in difflib.unified_diff(c2_zb.splitlines(keepends=True), c2_v5.splitlines(keepends=True), fromfile='zb_c2', tofile='v5_c2'):
    print(line, end='')

print("\n=== CELL 14/12 DIFF (CMD EXECUTION) ===")
# In Zbigniew it's cell 12, in V5 it's cell 14
c_cmd_zb = "".join(nb_zb['cells'][12]['source'])
c_cmd_v5 = "".join(nb_v5['cells'][14]['source'])
for line in difflib.unified_diff(c_cmd_zb.splitlines(keepends=True), c_cmd_v5.splitlines(keepends=True), fromfile='zb_cmd', tofile='v5_cmd'):
    print(line, end='')

