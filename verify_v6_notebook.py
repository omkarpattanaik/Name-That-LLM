import json, ast

with open('kaggle_v6/name-that-llm-v6-apex-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if cell['cell_type'] == 'code':
        test_src = src
        if test_src.startswith('%%writefile'):
            test_src = test_src.split('\n', 1)[1]
        lines = []
        for l in test_src.splitlines():
            if l.strip().startswith('!'):
                indent = len(l) - len(l.lstrip())
                lines.append(' ' * indent + 'pass #' + l.strip())
            else:
                lines.append(l)
        ast.parse('\n'.join(lines))
        print(f"Cell {i:2d} (code) AST syntax: VALID ({len(src)} chars)")

# Check specific features in Cell 6 (infer.py)
c6 = "".join(nb['cells'][6]['source'])
assert "pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)" in c6, "pred assignment missing in infer.py"
assert "\\u2019" in c6, "Unicode marks missing in infer.py"
assert "1.2 * _evidence" in c6, "Judith evidence scaling missing in infer.py"

# Check Cell 2 (inputs)
c2 = "".join(nb['cells'][2]['source'])
assert "ntl_open_pool.parquet" in c2, "open pool missing in Cell 2"
assert "flag_train_prompt_regen" in c2, "flag_train_prompt_regen filter missing in Cell 2"
assert "KAGGLE_IS_COMPETITION_RERUN" in c2, "bulletproof FAST_COMMIT missing in Cell 2"

# Check Cell 12 (run)
c12 = "".join(nb['cells'][12]['source'])
assert "'--svc_w', '12'" in c12, "svc_w 12 missing in Cell 12"
assert "'--sinkhorn', '0.50'" in c12, "sinkhorn 0.50 missing in Cell 12"
assert "candidate_reference_0660.parquet" in c12, "fallback reference missing in Cell 12"

print("\nALL VERIFICATIONS PASSED 100%!")
