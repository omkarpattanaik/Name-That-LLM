import json

with open("kaggle_v5/name-that-llm-v5-sota-apex-ensemble.ipynb", encoding="utf-8") as f:
    nb_v5 = json.load(f)

with open("reference_kernels/zbigniew_0664/0-664-one-swap-198k-clean-pool-fully-public.ipynb", encoding="utf-8") as f:
    nb_zb = json.load(f)

print(f"V5 cell count: {len(nb_v5['cells'])}, Zbigniew cell count: {len(nb_zb['cells'])}")

for i in range(min(len(nb_v5['cells']), len(nb_zb['cells']))):
    c_v5 = "".join(nb_v5['cells'][i]['source'])
    c_zb = "".join(nb_zb['cells'][i]['source'])
    diff = (c_v5 != c_zb)
    print(f"Cell {i:2d} | type: {nb_v5['cells'][i]['cell_type']:5s} | identical: {not diff} | V5 len: {len(c_v5)} | Zb len: {len(c_zb)}")

