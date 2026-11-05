"""
Verification script for Kaggle Version 16:
1. Validates JSON structure of notebook and kernel-metadata.json.
2. Parses every code cell with ast.parse to ensure zero syntax errors.
3. Performs symbol scope analysis with symtable on all generated python scripts:
   - gauss.py (Cell 7)
   - sibling_style.py (Cell 9)
   - v34_pairwise.py (Cell 11)
   - infer.py (Cell 14)
   - comp_stack.py (Cell 16)
4. Asserts that logsumexp and pathlib are correctly resolved in infer.py.
"""

import json
import ast
import symtable
import builtins
import sys

nb_path = 'kaggle_v16/name-that-llm-v16-apex-grandmaster-sovereign.ipynb'
meta_path = 'kaggle_v16/kernel-metadata.json'

print("--- 1. Validating kernel-metadata.json ---")
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)
assert meta['id'] == 'omkarpattanaik123/name-that-llm-v16-apex-grandmaster-sovereign'
assert meta['code_file'] == 'name-that-llm-v16-apex-grandmaster-sovereign.ipynb'
assert meta['enable_gpu'] is True
assert len(meta['dataset_sources']) >= 15
print(f"Metadata valid! Datasets: {len(meta['dataset_sources'])}, Kernels: {len(meta['kernel_sources'])}, Models: {len(meta['model_sources'])}")

print("\n--- 2. Validating Notebook Syntax (ast.parse) ---")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        # Strip cell magics for ast parsing
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
    # Filter __file__ which is provided by Python runtime for scripts
    undef = {u for u in undef if u[1] != '__file__'}
    if undef:
        print(f"WARNING: {name} has unresolved globals: {undef}")
        all_clean = False
    else:
        print(f"PASS: {name} has ZERO undefined variables!")

assert all_clean, "One or more scripts contain undefined variables!"

print("\n--- 4. Checking infer.py specific symbols ---")
infer_code = ''.join(nb['cells'][14]['source'])
assert 'from scipy.special import logsumexp' in infer_code, "logsumexp not imported in infer.py"
assert 'pathlib' in infer_code, "pathlib not imported in infer.py"
assert '_pairwise_path = pathlib.Path' in infer_code, "V34.1 Pairwise path not found in infer.py"
assert 'log_col_mean = logsumexp(log_P, axis=0)' in infer_code, "Damped Sinkhorn logsumexp not found in infer.py"
print("PASS: infer.py has explicit logsumexp and pathlib imports verified!")

print("\n=== V16 VERIFICATION COMPLETE: ALL CHECKS PASSED ===")
