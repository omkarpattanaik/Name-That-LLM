import json

with open("reference_kernels/synthreaper_0670/ntl-gold-apex-champion-0-670.ipynb", encoding="utf-8") as f:
    nb_sr = json.load(f)

with open("kaggle_v5/name-that-llm-v5-sota-apex-ensemble.ipynb", encoding="utf-8") as f:
    nb_v5 = json.load(f)

for i in [2, 6, 8, 12, 14]:
    sr_c = "".join(nb_sr['cells'][i]['source'])
    v5_c = "".join(nb_v5['cells'][i]['source'])
    print(f"=== Cell {i} identical: {sr_c == v5_c} ===")
    if sr_c != v5_c:
        import difflib
        diff = list(difflib.unified_diff(
            sr_c.splitlines(keepends=True),
            v5_c.splitlines(keepends=True),
            fromfile=f'synthreaper_cell_{i}',
            tofile=f'v5_cell_{i}',
            n=3
        ))
        print("".join(diff[:80]))
        if len(diff) > 80:
            print(f"... ({len(diff) - 80} more diff lines)")

