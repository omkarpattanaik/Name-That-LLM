import json, difflib

with open("kaggle_v2/name-that-llm-v2-sota-ensemble.ipynb", encoding="utf-8") as f:
    nb_v2 = json.load(f)

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb_zb = json.load(f)

c_v2 = "".join(nb_v2['cells'][8]['source'])
c_zb = "".join(nb_zb['cells'][8]['source'])

diff = list(difflib.unified_diff(c_v2.splitlines(keepends=True), c_zb.splitlines(keepends=True), fromfile='v2_comp', tofile='zb_comp'))
print(f"Diff lines count between V2 and Zbigniew comp_stack.py: {len(diff)}")
for line in diff:
    print(line, end='')

