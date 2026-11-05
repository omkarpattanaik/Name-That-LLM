import json
import ast
import symtable
import builtins
import sys

nb_path = 'kaggle_v18/name-that-llm-v18-apex-sovereign-calibrated.ipynb'
meta_path = 'kaggle_v18/kernel-metadata.json'

print("--- 1. Validating kernel-metadata.json ---")
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
assert meta['id'] == 'omkarpattanaik123/name-that-llm-v18-apex-sovereign-calibrated'
assert meta['code_file'] == 'name-that-llm-v18-apex-sovereign-calibrated.ipynb'
assert meta['enable_gpu'] is True
print(f"Metadata valid! Datasets: {len(meta['dataset_sources'])}")

print("\n--- 2. Validating Notebook Syntax (ast.parse) ---")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        clean_lines = []
        for line in src.split('\n'):
            if line.strip().startswith('!') or line.strip().startswith('%'):
                clean_lines.append('# ' + line)
            else:
                clean_lines.append(line)
        clean_src = '\n'.join(clean_lines)
        try:
            ast.parse(clean_src)
        except SyntaxError as se:
            print(f"ERROR: SyntaxError in Cell {idx}: {se}")
            sys.exit(1)
print(f"All {len(nb['cells'])} cells parsed with zero syntax errors!")

print("\n--- 3. Verifying Scope and Undefined Variables (symtable) ---")
scripts = {
    'gauss.py': nb['cells'][7]['source'],
    'sibling_style.py': nb['cells'][9]['source'],
    'v34_pairwise.py': nb['cells'][11]['source'],
    'infer.py': nb['cells'][14]['source'],
    'comp_stack.py': nb['cells'][16]['source'],
}

global_defs = set(dir(builtins))

def check_undefined(code_str, filename):
    if code_str.startswith('%%writefile'):
        code_str = code_str.split('\n', 1)[1]
    top = symtable.symtable(code_str, filename, 'exec')
    file_defs = set(global_defs)
    for s in top.get_symbols():
        if s.is_assigned() or s.is_imported() or s.is_namespace():
            file_defs.add(s.get_name())
    
    undefined = set()
    def check_table(t, path):
        for s in t.get_symbols():
            name = s.get_name()
            if s.is_global() and not (s.is_assigned() or s.is_imported() or s.is_namespace()):
                if name not in file_defs:
                    undefined.add((path, name))
        for child in t.get_children():
            check_table(child, f"{path}.{child.get_name()}")
    check_table(top, 'global')
    return undefined

all_clean = True
for name, src in scripts.items():
    code = ''.join(src)
    undef = check_undefined(code, name)
    undef = {u for u in undef if u[1] != '__file__'}
    if undef:
        print(f"WARNING: {name} has unresolved globals: {undef}")
        all_clean = False
    else:
        print(f"PASS: {name} has ZERO undefined variables!")

assert all_clean, "One or more scripts contain undefined variables!"

print("\n--- 4. Checking Cell 20 Parameters ---")
c20 = ''.join(nb['cells'][20]['source'])
assert "'--sk_div', '0'" in c20, "--sk_div 0 not found in Cell 20"
assert "'--distort_w', '0.0'" in c20, "--distort_w 0.0 not found in Cell 20"
assert "'--pool2_w', '0.0'" in c20, "--pool2_w 0.0 not found in Cell 20"
print("PASS: Cell 20 calibrated temperature and clean pool parameters verified!")

print("\n=== V18 VERIFICATION COMPLETE: ALL CHECKS PASSED ===")
