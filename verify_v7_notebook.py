import json, ast

with open('kaggle_v7/name-that-llm-v7-apex-sovereign.ipynb', 'r', encoding='utf-8') as f:
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

# Check specific features
c2 = "".join(nb['cells'][2]['source'])
assert "KAGGLE_IS_COMPETITION_RERUN" in c2, "bulletproof FAST_COMMIT missing in Cell 2"

c3 = "".join(nb['cells'][3]['source'])
assert "flag_train_prompt_regen" in c3, "flag_train_prompt_regen missing in Cell 3"
assert "flag_gemini37_routing_risk" in c3, "flag_gemini37_routing_risk missing in Cell 3"

c4 = "".join(nb['cells'][4]['source'])
assert "POOL2" in c4, "POOL2 missing in Cell 4"

c8 = "".join(nb['cells'][8]['source'])
assert "MARKERS" in c8, "MARKERS missing in Cell 8 (sibling_style.py)"

c10 = "".join(nb['cells'][10]['source'])
assert "pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)" in c10, "pred assignment missing in infer.py"

c11 = "".join(nb['cells'][11]['source'])
assert "pool2" in c11, "pool2 patch missing in Cell 11"

c17 = "".join(nb['cells'][17]['source'])
assert "'--svc_w', '12'" in c17, "svc_w 12 missing in Cell 17"
assert "'--sinkhorn', '0.60'" in c17, "sinkhorn 0.60 missing in Cell 17"
assert "'--pool2_tfidf', '0.35'" in c17, "pool2_tfidf 0.35 missing in Cell 17"
assert "candidate_reference_0660.parquet" in c17, "fallback reference missing in Cell 17"

print("\nALL V7 VERIFICATIONS PASSED 100%!")

