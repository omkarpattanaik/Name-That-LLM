import json
import ast
import sys

sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'kaggle_v5/name-that-llm-v5-sota-apex-ensemble.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"=== Verifying {nb_path} ===")
print(f"Total cells: {len(nb['cells'])}")

errors = []
for idx, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    src = "".join(cell.get('source', []))
    print(f"Cell {idx:2d} ({ctype}, {len(src)} chars): {src.splitlines()[0] if src.splitlines() else 'EMPTY'}")
    
    if ctype == 'code':
        clean_lines = []
        for line in src.splitlines():
            sline = line.strip()
            if sline.startswith('%') or sline.startswith('!'):
                clean_lines.append('# ' + line)
            else:
                clean_lines.append(line)
        clean_code = '\n'.join(clean_lines)
        try:
            ast.parse(clean_code)
        except SyntaxError as e:
            errors.append(f"Syntax error in Cell {idx}: {e}")

# Detailed check on Cell 2
c2_src = "".join(nb['cells'][2]['source'])
assert "KAGGLE_IS_COMPETITION_RERUN" in c2_src, "Missing KAGGLE_IS_COMPETITION_RERUN guard in Cell 2"
assert "open_pool.parquet" in c2_src, "Missing open pool setup in Cell 2"

# Detailed check on Cell 6 (sibling_style.py)
c6_src = "".join(nb['cells'][6]['source'])
assert '\\u2019' in c6_src or '\u2019' in c6_src, "Missing curly apostrophe in sibling_style.py"
assert '\\u2014' in c6_src or '\u2014' in c6_src, "Missing em-dash in sibling_style.py"
assert '\ufffd' not in c6_src, "Corrupted \\ufffd character found in sibling_style.py"

# Detailed check on Cell 8 (infer.py)
c8_src = "".join(nb['cells'][8]['source'])
assert '_pool_weight = 0.35 * (len(train) / max(len(_P), 1))' in c8_src, "Pool weight normalization formula missing in infer.py"
assert '\ufffd' not in c8_src, "Corrupted \\ufffd character found in infer.py"
assert '_marks = [' in c8_src, "Missing _marks in infer.py"
assert '\\u2019' in c8_src or '\u2019' in c8_src, "Missing curly apostrophe in infer.py _marks"
assert '\\u2014' in c8_src or '\u2014' in c8_src, "Missing em-dash in infer.py _marks"

# Detailed check on Cell 14 (execution command)
c14_src = "".join(nb['cells'][14]['source'])
assert "'--distort_w', '0.15'" in c14_src, "distort_w 0.15 missing in Cell 14"
assert "'--svc_w', '6.0'" in c14_src, "svc_w 6.0 missing in Cell 14"
assert "'--sinkhorn', '0.45'" in c14_src, "sinkhorn 0.45 missing in Cell 14"
assert "'--sinkhorn_target_mix', '0.35'" in c14_src, "sinkhorn_target_mix 0.35 missing in Cell 14"
assert 'FAST_COMMIT' in c14_src, "FAST_COMMIT check missing in Cell 14"

if errors:
    print("VERIFICATION FAILED WITH ERRORS:")
    for err in errors:
        print("  -", err)
    sys.exit(1)
else:
    print(">>> ALL V5 VERIFICATION CHECKS PASSED PERFECTLY! <<<")

