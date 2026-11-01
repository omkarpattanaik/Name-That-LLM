import json, difflib, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('reference_kernels/bgunshui_ntl-apex-g/ntl-apex-g.ipynb', 'r', encoding='utf-8') as f:
    nb_g = json.load(f)

with open('reference_kernels/bgunshui_ntl-apex-h/ntl-apex-h.ipynb', 'r', encoding='utf-8') as f:
    nb_h = json.load(f)

print(f"G cells: {len(nb_g['cells'])}, H cells: {len(nb_h['cells'])}")

for i in range(min(len(nb_g['cells']), len(nb_h['cells']))):
    c_g = "".join(nb_g['cells'][i]['source'])
    c_h = "".join(nb_h['cells'][i]['source'])
    if c_g != c_h:
        print(f"\n=== Diff in Cell {i} ({nb_g['cells'][i]['cell_type']}) ===")
        diff = list(difflib.unified_diff(c_g.splitlines(keepends=True), c_h.splitlines(keepends=True), fromfile='apex_g', tofile='apex_h'))
        for line in diff[:30]:
            print(line, end='')
        if len(diff) > 30:
            print(f"... ({len(diff) - 30} more lines)")

