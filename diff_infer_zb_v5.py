import json, difflib

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb_zb = json.load(f)

with open("kaggle_v5/name-that-llm-v5-sota-apex-ensemble.ipynb", encoding="utf-8") as f:
    nb_v5 = json.load(f)

c_infer_zb = "".join(nb_zb['cells'][6]['source'])
c_infer_v5 = "".join(nb_v5['cells'][8]['source'])

diff = list(difflib.unified_diff(
    c_infer_zb.splitlines(keepends=True),
    c_infer_v5.splitlines(keepends=True),
    fromfile='zb_infer',
    tofile='v5_infer',
    n=2
))

print(f"Diff lines count: {len(diff)}")
for line in diff:
    print(line, end='')

